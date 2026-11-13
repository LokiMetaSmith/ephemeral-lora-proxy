import logging
import time
import math
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("advisor")


@dataclass
class AdvisorDecision:
    recommended_action: str  # "retain", "defer", "consolidate", "force_consolidate"
    per_turn_savings_sec: float
    break_even_turns: float
    expected_turns: int
    estimated_t_comp_sec: float
    current_keep_recent_tokens: int
    rework_count: int
    context_tokens: int
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_action": self.recommended_action,
            "per_turn_savings_sec": round(self.per_turn_savings_sec, 6),
            "break_even_turns": round(self.break_even_turns, 2) if not math.isinf(self.break_even_turns) else "inf",
            "expected_turns": self.expected_turns,
            "estimated_t_comp_sec": round(self.estimated_t_comp_sec, 4),
            "current_keep_recent_tokens": self.current_keep_recent_tokens,
            "rework_count": self.rework_count,
            "context_tokens": self.context_tokens,
            "reason": self.reason,
        }


class ConsolidationAdvisor:
    """
    Finite-horizon latency break-even advisor tailored for AMD Strix Halo (gfx1151).
    Replaces static context length thresholds with empirical memory bus / wall-clock
    latency economics.
    """

    REWORK_KEYWORDS = {
        "no", "wrong", "incorrect", "not right", "wait", "fix", "revert",
        "you forgot", "remember", "where is", "recall", "repeat", "correction"
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        thresholds = config.get("thresholds", {})
        advisor_cfg = config.get("advisor", {})

        self.soft_ctx_tokens = int(thresholds.get("soft_ctx_tokens", 4096))
        self.hard_max_ctx_tokens = int(thresholds.get("hard_max_ctx_tokens", 8192))
        self.base_keep_recent_tokens = int(thresholds.get("keep_recent_tokens", 2048))
        self.max_keep_recent_tokens = int(thresholds.get("max_keep_recent_tokens", 4096))
        self.rework_token_penalty = int(advisor_cfg.get("rework_token_penalty", 512))
        self.default_expected_turns = int(advisor_cfg.get("default_expected_turns", 10))
        self.min_break_even_savings = float(advisor_cfg.get("min_break_even_savings", 0.0001))

        # Rolling consolidation latency T_comp (self-distillation + autograd train + reload)
        self._rolling_t_comp: float = float(advisor_cfg.get("default_t_comp_sec", 0.85))
        self._t_comp_samples: List[float] = []

        # Telemetry curves: context_length -> (prefill_tok_per_sec, decode_tok_per_sec)
        # Seed with empirical AMD Strix Halo gfx1151 baseline curves (LPDDR5X 256 GB/s)
        self._prefill_samples: List[Tuple[int, float]] = []  # (H, tokens/sec)
        self._decode_samples: List[Tuple[int, float]] = []   # (H, tokens/sec)

        # Per-session state: session_id -> metadata
        self._session_rework_count: Dict[str, int] = {}
        self._session_keep_tokens: Dict[str, int] = {}
        self._session_last_eviction_time: Dict[str, float] = {}

    def record_consolidation_telemetry(self, duration_sec: float):
        """Records wall-clock duration of a completed consolidation cycle."""
        if duration_sec <= 0:
            return
        self._t_comp_samples.append(duration_sec)
        if len(self._t_comp_samples) > 20:
            self._t_comp_samples.pop(0)
        self._rolling_t_comp = sum(self._t_comp_samples) / len(self._t_comp_samples)
        logger.info(f"[ConsolidationAdvisor] Updated rolling T_comp: {self._rolling_t_comp:.4f}s across {len(self._t_comp_samples)} samples")

    def record_inference_telemetry(self, prompt_n: int, prompt_ms: float,
                                   predicted_n: int, predicted_ms: float,
                                   prompt_per_second: float = 0.0,
                                   predicted_per_second: float = 0.0):
        """Records inference throughput telemetry parsed from llama-server timings."""
        if prompt_n > 0 and prompt_ms > 0:
            speed_u = prompt_per_second if prompt_per_second > 0 else (prompt_n / (prompt_ms / 1000.0))
            self._prefill_samples.append((prompt_n, speed_u))
            if len(self._prefill_samples) > 100:
                self._prefill_samples.pop(0)

        if predicted_n > 0 and predicted_ms > 0:
            speed_o = predicted_per_second if predicted_per_second > 0 else (predicted_n / (predicted_ms / 1000.0))
            self._decode_samples.append((prompt_n, speed_o))
            if len(self._decode_samples) > 100:
                self._decode_samples.pop(0)

    def record_eviction_event(self, session_id: str):
        """Marks timestamp when an eviction was triggered for rework tracking."""
        self._session_last_eviction_time[session_id] = time.time()

    def get_session_keep_tokens(self, session_id: str) -> int:
        """Returns the current target keep_recent_tokens window for the session."""
        return self._session_keep_tokens.get(session_id, self.base_keep_recent_tokens)

    def check_rework_penalty(self, session_id: str, user_message: str):
        """
        Detects if the user sends an immediate corrective follow-up right after an eviction.
        Widens keep_recent_tokens by rework_token_penalty (+512 tokens) to protect verbatim recall.
        """
        last_evict = self._session_last_eviction_time.get(session_id, 0.0)
        now = time.time()
        # Check within 90 seconds of an eviction
        if now - last_evict < 90.0:
            msg_lower = user_message.lower().strip()
            is_rework = any(kw in msg_lower for kw in self.REWORK_KEYWORDS) or len(msg_lower) < 25
            if is_rework:
                curr_rework = self._session_rework_count.get(session_id, 0) + 1
                self._session_rework_count[session_id] = curr_rework
                current_keep = self.get_session_keep_tokens(session_id)
                new_keep = min(current_keep + self.rework_token_penalty, self.max_keep_recent_tokens)
                self._session_keep_tokens[session_id] = new_keep
                logger.warning(
                    f"[ConsolidationAdvisor] Immediate rework follow-up detected for session {session_id}! "
                    f"Rework count: {curr_rework}. Expanded keep_recent_tokens: {current_keep} -> {new_keep}"
                )

    def _estimate_throughput(self, context_tokens: int) -> Tuple[float, float]:
        """
        Estimates prefill throughput v_u(H) and decode throughput v_o(H) at context length H.
        Uses nearest recorded samples or Strix Halo LPDDR5X empirical degradation curve.
        """
        # Baseline AMD Strix Halo gfx1151 speeds
        base_prefill = 2600.0  # tok/s
        base_decode = 75.0     # tok/s

        # Quadratic / linear attention degradation factors
        gamma_u = 0.00008
        gamma_o = 0.00004

        est_u = base_prefill / (1.0 + gamma_u * context_tokens)
        est_o = base_decode / (1.0 + gamma_o * context_tokens)

        # Blend with empirical observations near this context window if available
        nearby_u = [s for h, s in self._prefill_samples if abs(h - context_tokens) < 1024]
        if nearby_u:
            est_u = (est_u + sum(nearby_u) / len(nearby_u)) / 2.0

        nearby_o = [s for h, s in self._decode_samples if abs(h - context_tokens) < 1024]
        if nearby_o:
            est_o = (est_o + sum(nearby_o) / len(nearby_o)) / 2.0

        return max(50.0, est_u), max(5.0, est_o)

    def evaluate_turn(self, session_id: str, context_tokens: int,
                      expected_turns: Optional[int] = None,
                      user_message: str = "") -> AdvisorDecision:
        """
        Evaluates the latency break-even economics for the active turn:
        - If H < soft_ctx_tokens: 'retain'
        - If H >= hard_max_ctx_tokens: 'force_consolidate'
        - Between soft and hard: calculates per-turn latency savings B and N_break_even.
          Triggers 'consolidate' only if B > 0 and N_est > N_break_even; otherwise 'defer'.
        """
        if user_message:
            self.check_rework_penalty(session_id, user_message)

        current_keep = self.get_session_keep_tokens(session_id)
        rework_count = self._session_rework_count.get(session_id, 0)
        n_est = expected_turns if expected_turns is not None and expected_turns > 0 else self.default_expected_turns
        t_comp = self._rolling_t_comp

        # 1. Below soft threshold: retain context cleanly
        if context_tokens < self.soft_ctx_tokens:
            return AdvisorDecision(
                recommended_action="retain",
                per_turn_savings_sec=0.0,
                break_even_turns=float("inf"),
                expected_turns=n_est,
                estimated_t_comp_sec=t_comp,
                current_keep_recent_tokens=current_keep,
                rework_count=rework_count,
                context_tokens=context_tokens,
                reason=f"Context tokens ({context_tokens}) below soft threshold ({self.soft_ctx_tokens})."
            )

        # 2. Hard threshold: force consolidation to prevent context window overflow
        if context_tokens >= self.hard_max_ctx_tokens:
            return AdvisorDecision(
                recommended_action="force_consolidate",
                per_turn_savings_sec=0.05,
                break_even_turns=1.0,
                expected_turns=n_est,
                estimated_t_comp_sec=t_comp,
                current_keep_recent_tokens=current_keep,
                rework_count=rework_count,
                context_tokens=context_tokens,
                reason=f"Context tokens ({context_tokens}) reached hard ceiling ({self.hard_max_ctx_tokens}). Forced eviction."
            )

        # 3. Soft-to-Hard Range: Compute empirical break-even latency economics
        # H = current context tokens, S = target post-eviction window (current_keep)
        H = context_tokens
        S = min(current_keep, H - 256)

        v_u_H, v_o_H = self._estimate_throughput(H)
        v_u_S, v_o_S = self._estimate_throughput(S)

        # Average fresh input tokens G and output tokens O per turn
        G = 80.0
        O = 60.0

        # Per-turn savings B:
        # B = G * (1/v_u(H) - 1/v_u(S)) + O * (1/v_o(H) - 1/v_o(S))
        prefill_diff = (1.0 / v_u_H) - (1.0 / v_u_S)
        decode_diff = (1.0 / v_o_H) - (1.0 / v_o_S)
        B = (G * prefill_diff) + (O * decode_diff)

        # Additional attention evaluation savings across surviving prefix difference
        # (H - S) / v_eval
        attention_savings = (H - S) * 0.00003
        B += attention_savings

        if B <= self.min_break_even_savings:
            n_break_even = float("inf")
            action = "defer"
            reason = f"Per-turn latency savings ({B:.6f}s) insufficient to break even."
        else:
            n_break_even = t_comp / B
            if n_est > n_break_even:
                action = "consolidate"
                reason = (
                    f"Consolidation is latency break-even favorable: expected turns ({n_est}) > "
                    f"break-even turns ({n_break_even:.1f}) with B={B:.4f}s/turn savings."
                )
            else:
                action = "defer"
                reason = (
                    f"Consolidation deferred: expected remaining turns ({n_est}) <= "
                    f"break-even turns ({n_break_even:.1f}). Preserving KV cache."
                )

        return AdvisorDecision(
            recommended_action=action,
            per_turn_savings_sec=B,
            break_even_turns=n_break_even,
            expected_turns=n_est,
            estimated_t_comp_sec=t_comp,
            current_keep_recent_tokens=current_keep,
            rework_count=rework_count,
            context_tokens=context_tokens,
            reason=reason
        )

    def get_metrics(self, session_id: str) -> Dict[str, Any]:
        """Returns live decision metrics for /health and /advisor endpoints."""
        current_keep = self.get_session_keep_tokens(session_id)
        rework_count = self._session_rework_count.get(session_id, 0)
        v_u_curr, v_o_curr = self._estimate_throughput(self.soft_ctx_tokens)
        return {
            "estimated_t_comp_sec": round(self._rolling_t_comp, 4),
            "soft_ctx_tokens": self.soft_ctx_tokens,
            "hard_max_ctx_tokens": self.hard_max_ctx_tokens,
            "current_keep_recent_tokens": current_keep,
            "rework_count": rework_count,
            "prefill_throughput_estimate": round(v_u_curr, 1),
            "decode_throughput_estimate": round(v_o_curr, 1),
            "telemetry_samples_count": len(self._prefill_samples),
        }

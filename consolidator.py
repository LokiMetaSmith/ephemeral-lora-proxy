import os
import re
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx

from session_store import SessionStore
from backends import TrainerBackend, LlamaCppNativeBackend, PyTorchSidecarBackend

logger = logging.getLogger("consolidator")


class ConsolidationItem:
    def __init__(self, session_id: str, evicted_messages: List[Dict[str, Any]]):
        self.session_id = session_id
        self.evicted_messages = evicted_messages


class EphemeralConsolidator:
    def __init__(
        self,
        config: Dict[str, Any],
        session_store: SessionStore,
        inference_idle_event: asyncio.Event,
        advisor: Optional[Any] = None,
    ):
        self.config = config
        self.session_store = session_store
        self.inference_idle_event = inference_idle_event
        self.advisor = advisor
        self.queue: asyncio.Queue[ConsolidationItem] = asyncio.Queue()
        self.running = False
        self._task: Optional[asyncio.Task] = None

        backend_type = config.get("backend", "llamacpp_native")
        if backend_type == "pytorch_sidecar":
            self.backend: TrainerBackend = PyTorchSidecarBackend()
        else:
            self.backend: TrainerBackend = LlamaCppNativeBackend()

        self.replay_buffer: List[Dict[str, str]] = self._load_replay_buffer()

    def _load_replay_buffer(self) -> List[Dict[str, str]]:
        path = self.config.get("replay_buffer", {}).get("path", "/home/loki/ephemeral-lora-proxy/replay_buffer.jsonl")
        buffer = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            buffer.append(json.loads(line))
                        except Exception:
                            pass
        logger.info(f"[Consolidator] Loaded {len(buffer)} anchor items from replay buffer")
        return buffer

    def start(self):
        self.running = True
        self._task = asyncio.create_task(self._worker_loop())
        logger.info("[Consolidator] Background worker started")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def enqueue(self, item: ConsolidationItem):
        self.queue.put_nowait(item)
        logger.info(f"[Consolidator] Enqueued {len(item.evicted_messages)} messages for session {item.session_id}. Queue depth: {self.queue.qsize()}")

    async def _worker_loop(self):
        while self.running:
            try:
                # 1. Wait for an item in the queue
                item = await self.queue.get()

                # Constraint 4: Coalesce pending chunks if multiple thresholds were hit
                coalesced_items = [item]
                while not self.queue.empty():
                    next_item = self.queue.get_nowait()
                    if next_item.session_id == item.session_id:
                        coalesced_items.append(next_item)
                    else:
                        # Put back if for a different session
                        self.queue.put_nowait(next_item)
                        break

                all_messages = []
                for it in coalesced_items:
                    all_messages.extend(it.evicted_messages)

                session_id = item.session_id
                logger.info(f"[Consolidator] Processing consolidation for session {session_id} ({len(all_messages)} messages)")

                # Strictly await inference_idle_event so 100% memory bus is granted to foreground inference
                if not self.inference_idle_event.is_set():
                    logger.info("[Consolidator] Active inference in-flight. Yielding memory bus and waiting for idle...")
                    await self.inference_idle_event.wait()
                    logger.info("[Consolidator] Inference bus is now idle. Proceeding with consolidation work.")

                await self._process_consolidation(session_id, all_messages)
                self.queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Consolidator] Error in worker loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    async def _process_consolidation(self, session_id: str, messages: List[Dict[str, Any]]):
        t_start = time.time()
        upstream_host = self.config.get("upstream", {}).get("host", "127.0.0.1")
        upstream_port = self.config.get("upstream", {}).get("port", 8081)
        upstream_url = f"http://{upstream_host}:{upstream_port}"

        # Step 1: Self-Distillation using clean base weights (lora: [])
        new_distilled_pairs = await self._self_distill(upstream_url, messages)
        if not new_distilled_pairs:
            logger.warning("[Consolidator] Distillation produced no pairs; falling back to direct extractive turns")
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content")
                    if content:
                        new_distilled_pairs.append({"q": str(content), "a": "Acknowledged and retained."})

        # Maintain rolling cumulative distilled facts for this session across multiple cycles
        sessions_dir = Path("/home/loki/ephemeral-lora-proxy/sessions")
        sessions_dir.mkdir(parents=True, exist_ok=True)
        session_file = sessions_dir / f"{session_id}_distilled.jsonl"

        with open(session_file, "a", encoding="utf-8") as f:
            for pair in new_distilled_pairs:
                f.write(json.dumps(pair) + "\n")

        # Load all accumulated distilled pairs for this session
        cumulative_distilled: List[Dict[str, str]] = []
        with open(session_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        cumulative_distilled.append(json.loads(line))
                    except Exception:
                        pass

        # Step 2: Replay Buffer Mix (~15% ratio of cumulative pairs)
        distilled_pairs = list(cumulative_distilled)
        ratio = self.config.get("replay_buffer", {}).get("ratio", 0.15)
        n_replay = max(1, int(len(distilled_pairs) * ratio))
        if self.replay_buffer:
            selected_replay = self.replay_buffer[:n_replay]
            for item in selected_replay:
                distilled_pairs.append({"q": item.get("instruction", ""), "a": item.get("response", "")})

        logger.info(f"[Consolidator] Total training pairs (cumulative distilled {len(cumulative_distilled)} + replay): {len(distilled_pairs)}")

        # Step 3: Fast Weight Update via TrainerBackend
        # Double check inference is still idle before GPU weight writes
        if not self.inference_idle_event.is_set():
            logger.info("[Consolidator] Inference interrupted before weight update. Pausing for idle...")
            await self.inference_idle_event.wait()

        # Scale Lock: Set scale to 0.0 before weight mutation so running queries do not read uncommitted weights
        await self._reload_lora_adapter(upstream_url, 0.0)

        success = await asyncio.to_thread(self.backend.train, distilled_pairs, session_id, self.config)
        msg_ids = [m["id"] for m in messages if "id" in m]
        if not success:
            logger.error("[Consolidator] Weight update failed in TrainerBackend; resetting pending messages")
            self.session_store.reset_pending_messages(session_id, msg_ids)
            scale = self.config.get("lora", {}).get("scale", 1.0)
            await self._reload_lora_adapter(upstream_url, scale)
            return

        # Step 4: Hot-Reload & Restore Scale
        scale = self.config.get("lora", {}).get("scale", 1.0)
        await self._reload_lora_adapter(upstream_url, scale)

        # Mark messages consolidated in session store
        self.session_store.mark_messages_consolidated(session_id, msg_ids)
        new_gen = self.session_store.increment_generation(session_id)
        duration_sec = time.time() - t_start
        if self.advisor:
            self.advisor.record_consolidation_telemetry(duration_sec)
        logger.info(f"[Consolidator] Session {session_id} consolidated successfully in {duration_sec:.3f}s! Generation: {new_gen}")

    def reset_session_distilled(self, session_id: str):
        """Removes the rolling cumulative distilled facts file for a session."""
        session_file = Path(f"/home/loki/ephemeral-lora-proxy/sessions/{session_id}_distilled.jsonl")
        if session_file.exists():
            try:
                session_file.unlink()
            except OSError:
                pass

    async def _self_distill(self, upstream_url: str, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        dialogue_lines = []
        for m in messages:
            role = m.get("role", "user").capitalize()
            c = m.get("content")
            if c:
                dialogue_lines.append(f"{role}: {c}")
            if "tool_calls" in m and m["tool_calls"]:
                try:
                    dialogue_lines.append(f"{role} ToolCalls: {json.dumps(m['tool_calls'])}")
                except Exception:
                    pass
        dialogue_text = "\n".join(dialogue_lines)
        extraction_prompt = (
            "You are an expert knowledge distillation system. Extract 5 to 10 high-density, factual Q&A pairs "
            "and rule-enforcement examples from the following dialogue. Preserve all specific facts, user preferences, "
            "instructions, decisions, code snippets, and rules verbatim without narrative conversational fluff.\n\n"
            "Dialogue:\n"
            f"{dialogue_text}\n\n"
            "Format each pair strictly as:\n"
            "Q: <question or context rule>\n"
            "A: <exact fact, answer, or code>\n"
            "---"
        )

        payload = {
            "messages": [
                {"role": "user", "content": extraction_prompt}
            ],
            "lora": [],  # Strictly use clean base weights for distillation
            "temperature": 0.1,
            "max_tokens": 512,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{upstream_url}/v1/chat/completions", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return self._parse_distilled_qa(content)
                else:
                    logger.warning(f"[Consolidator] Self-distillation returned HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"[Consolidator] Self-distillation request failed: {e}")

        return []

    def _parse_distilled_qa(self, text: str) -> List[Dict[str, str]]:
        pairs = []
        blocks = text.split("---")
        for block in blocks:
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            q, a = None, None
            for line in lines:
                if re.match(r"^(Q|Question):\s*", line, re.IGNORECASE):
                    q = re.sub(r"^(Q|Question):\s*", "", line, flags=re.IGNORECASE).strip()
                elif re.match(r"^(A|Answer):\s*", line, re.IGNORECASE):
                    a = re.sub(r"^(A|Answer):\s*", "", line, flags=re.IGNORECASE).strip()
            if q and a:
                pairs.append({"q": q, "a": a})
        return pairs

    async def _reload_lora_adapter(self, upstream_url: str, scale: float):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{upstream_url}/lora-adapters",
                    json=[{"id": 0, "scale": scale}]
                )
                if resp.status_code == 200:
                    logger.info(f"[Consolidator] Successfully hot-reloaded LoRA adapter scale={scale} in llama-server")
                else:
                    logger.warning(f"[Consolidator] POST /lora-adapters returned {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"[Consolidator] Failed to call POST /lora-adapters: {e}")

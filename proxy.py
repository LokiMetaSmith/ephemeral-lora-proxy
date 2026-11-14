import os
import time
import glob
import json
import shutil
from pathlib import Path
import asyncio
import logging
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

import yaml
import httpx
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse

from session_store import SessionStore
from consolidator import EphemeralConsolidator, ConsolidationItem
from advisor import ConsolidationAdvisor, AdvisorDecision

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("proxy")

# Global state
config: Dict[str, Any] = {}
session_store: SessionStore = None
consolidator: EphemeralConsolidator = None
active_inference_streams: int = 0
inference_idle_event = asyncio.Event()
_grace_timer_task: Optional[asyncio.Task] = None
_lock = asyncio.Lock()


def load_config(path: str = "/home/loki/ephemeral-lora-proxy/config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_gtt_memory() -> Dict[str, Any]:
    total_bytes = 0
    used_bytes = 0
    try:
        total_files = glob.glob("/sys/class/drm/card*/device/mem_info_gtt_total")
        used_files = glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used")
        if total_files and os.path.exists(total_files[0]):
            with open(total_files[0], "r") as f:
                total_bytes = int(f.read().strip())
        if used_files and os.path.exists(used_files[0]):
            with open(used_files[0], "r") as f:
                used_bytes = int(f.read().strip())
    except Exception as e:
        logger.warning(f"Failed to read GTT memory: {e}")

    free_bytes = max(0, total_bytes - used_bytes)
    return {
        "gtt_total_bytes": total_bytes,
        "gtt_used_bytes": used_bytes,
        "gtt_free_bytes": free_bytes,
        "gtt_total_gb": round(total_bytes / (1024**3), 2),
        "gtt_used_gb": round(used_bytes / (1024**3), 2),
        "gtt_free_gb": round(free_bytes / (1024**3), 2),
    }


async def _on_stream_start():
    global active_inference_streams, _grace_timer_task
    async with _lock:
        active_inference_streams += 1
        if _grace_timer_task and not _grace_timer_task.done():
            _grace_timer_task.cancel()
            _grace_timer_task = None
        inference_idle_event.clear()
        logger.debug(f"Inference stream started. Active: {active_inference_streams}")


async def _on_stream_end():
    global active_inference_streams, _grace_timer_task
    async with _lock:
        active_inference_streams = max(0, active_inference_streams - 1)
        logger.debug(f"Inference stream ended. Active: {active_inference_streams}")
        if active_inference_streams == 0:
            grace_ms = config.get("strix_halo", {}).get("idle_grace_ms", 400)
            if _grace_timer_task and not _grace_timer_task.done():
                _grace_timer_task.cancel()
            _grace_timer_task = asyncio.create_task(_delayed_set_idle(grace_ms))


async def _delayed_set_idle(grace_ms: int):
    try:
        await asyncio.sleep(grace_ms / 1000.0)
        async with _lock:
            if active_inference_streams == 0:
                inference_idle_event.set()
                logger.debug(f"Inference bus reached idle state after {grace_ms}ms grace period")
    except asyncio.CancelledError:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, session_store, consolidator, advisor
    config = load_config()
    inference_idle_event.set()  # Initial state is idle

    # Check GTT memory and log pre-flight warning if capped
    gtt_info = read_gtt_memory()
    min_expected_gb = config.get("strix_halo", {}).get("gtt_min_gb_expected", 100)
    if gtt_info["gtt_total_gb"] > 0 and gtt_info["gtt_total_gb"] < min_expected_gb:
        logger.warning(
            f"[WARNING] AMD Strix Halo GTT shared memory is capped at {gtt_info['gtt_total_gb']} GB "
            f"(expected >= {min_expected_gb} GB). Remediation: Add 'amdgpu.gttsize=131072 ttm.pages_limit=33554432 amd_iommu=off' "
            "to /etc/default/grub GRUB_CMDLINE_LINUX_DEFAULT and run update-grub."
        )
    else:
        logger.info(f"[Strix Halo GTT Check] GTT Total: {gtt_info['gtt_total_gb']} GB (Verified >= {min_expected_gb} GB)")

    session_store = SessionStore(config.get("database", {}).get("path", "/home/loki/ephemeral-lora-proxy/sessions.db"))
    advisor = ConsolidationAdvisor(config)
    consolidator = EphemeralConsolidator(config, session_store, inference_idle_event, advisor=advisor)
    consolidator.start()
    logger.info("Ephemeral LoRA Proxy initialized successfully with ConsolidationAdvisor")

    yield

    await consolidator.stop()
    logger.info("Ephemeral LoRA Proxy shutdown completed")


app = FastAPI(title="Ephemeral LoRA Proxy for llama.cpp", lifespan=lifespan)


@app.get("/health")
async def health():
    gtt = read_gtt_memory()
    q_depth = consolidator.queue.qsize() if consolidator else 0
    return {
        "status": "ok",
        "active_inference_streams": active_inference_streams,
        "is_bus_idle": inference_idle_event.is_set(),
        "consolidation_queue_depth": q_depth,
        "gtt_memory": gtt,
        "backend": config.get("backend", "llamacpp_native"),
        "advisor": advisor.get_metrics("default") if advisor else {},
    }


@app.get("/v1/models")
async def list_models():
    """Returns OpenAI-compatible model listing aliasing 'ephemeral-local' alongside loaded model."""
    upstream_host = config.get("upstream", {}).get("host", "127.0.0.1")
    upstream_port = config.get("upstream", {}).get("port", 8081)
    upstream_url = f"http://{upstream_host}:{upstream_port}"

    models_data = []
    created_ts = int(time.time())

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{upstream_url}/v1/models")
            if resp.status_code == 200:
                res_json = resp.json()
                models_data = res_json.get("data", [])
    except Exception as e:
        logger.warning(f"[Proxy] Could not fetch models from upstream: {e}")

    cfg_model_path = config.get("model", {}).get("path", "ephemeral-local")

    # Add canonical "ephemeral-local" model
    ephemeral_entry = {
        "id": "ephemeral-local",
        "object": "model",
        "created": created_ts,
        "owned_by": "ephemeral-lora-proxy",
        "meta": {"base_model": cfg_model_path}
    }
    models_data = [m for m in models_data if m.get("id") != "ephemeral-local"]
    models_data.insert(0, ephemeral_entry)

    # Ensure configured base model ID is also in the list
    if not any(m.get("id") == cfg_model_path for m in models_data):
        models_data.append({
            "id": cfg_model_path,
            "object": "model",
            "created": created_ts,
            "owned_by": "ephemeral-lora-proxy"
        })

    return {
        "object": "list",
        "data": models_data
    }


@app.get("/v1/sessions/{session_id}/advisor")
async def get_session_advisor(session_id: str):
    """Exposes the ConsolidationAdvisor live decision metrics and break-even calculations."""
    upstream_host = config.get("upstream", {}).get("host", "127.0.0.1")
    upstream_port = config.get("upstream", {}).get("port", 8081)
    upstream_url = f"http://{upstream_host}:{upstream_port}"
    active_msgs = session_store.get_active_messages(session_id)
    ctx_tokens = await _tokenize_messages(active_msgs, upstream_url) if active_msgs else 0
    decision = advisor.evaluate_turn(session_id, ctx_tokens)
    return {
        "session_id": session_id,
        "decision": decision.to_dict(),
        "metrics": advisor.get_metrics(session_id),
    }


def _normalize_content_for_comparison(c: Any) -> str:
    """Safely normalizes content for string comparison, token counting, and logging."""
    if c is None:
        return ""
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):
        texts = []
        for part in c:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", "").strip())
            else:
                texts.append(str(part).strip())
        return " ".join(texts)
    try:
        return json.dumps(c, sort_keys=True)
    except Exception:
        return str(c).strip()


def _messages_equal(m1: Dict[str, Any], m2: Dict[str, Any]) -> bool:
    """Compares two messages across role, normalized content, tool_calls, and tool IDs."""
    if m1.get("role") != m2.get("role"):
        return False
    if _normalize_content_for_comparison(m1.get("content")) != _normalize_content_for_comparison(m2.get("content")):
        return False
    if m1.get("tool_calls") != m2.get("tool_calls"):
        return False
    if m1.get("tool_call_id") != m2.get("tool_call_id"):
        return False
    if m1.get("name") != m2.get("name"):
        return False
    return True


def _group_atomic_turns(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Groups messages into atomic units so that an assistant turn with tool_calls
    is never separated from its immediately following role: 'tool' response messages.
    """
    groups: List[List[Dict[str, Any]]] = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        group = [m]
        # If assistant turn with tool_calls, bind all consecutive succeeding 'tool' messages
        if m.get("role") == "assistant" and m.get("tool_calls"):
            j = i + 1
            while j < n and messages[j].get("role") == "tool":
                group.append(messages[j])
                j += 1
            i = j
        else:
            i += 1
        groups.append(group)
    return groups


async def _tokenize_messages(messages: List[Dict[str, Any]], upstream_url: str) -> int:
    """Estimate token count by calling upstream /tokenize or token length fallback."""
    parts = []
    for m in messages:
        c = m.get("content")
        if c is not None:
            parts.append(_normalize_content_for_comparison(c))
        if "tool_calls" in m and m["tool_calls"]:
            try:
                parts.append(json.dumps(m["tool_calls"]))
            except Exception:
                pass
        if "name" in m and m["name"]:
            parts.append(str(m["name"]))
    text = " ".join([p for p in parts if p])
    if not text:
        return 1
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{upstream_url}/tokenize", json={"content": text})
            if resp.status_code == 200:
                data = resp.json()
                return len(data.get("tokens", []))
    except Exception:
        pass
    # Rough estimate fallback: 1 token ~= 4 characters
    return max(1, len(text) // 4)


def _insert_message_into_store(session_id: str, m: Dict[str, Any], existing_msgs: List[Dict[str, Any]]):
    role = m.get("role", "user")
    content = m.get("content")
    extra = {}
    if "tool_calls" in m and m["tool_calls"]:
        extra["tool_calls"] = m["tool_calls"]
    if "tool_call_id" in m and m["tool_call_id"]:
        extra["tool_call_id"] = m["tool_call_id"]
    if "name" in m and m["name"]:
        extra["name"] = m["name"]
    if isinstance(content, (list, dict)):
        extra["content"] = content

    text_content = _normalize_content_for_comparison(content) if content is not None else None
    msg_id = session_store.add_message(session_id, role, text_content, extra=extra if extra else None)
    stored_item = {"id": msg_id, "role": role, "content": content, "is_consolidated": 0}
    stored_item.update(extra)
    existing_msgs.append(stored_item)


def _sync_client_messages(session_id: str, incoming_messages: List[Dict[str, Any]]):
    """
    Deduplicates incoming client messages against stored session history.
    OpenAI-compatible clients send the entire conversation history on every turn.
    We identify already-recorded messages and insert only net-new turns,
    preserving tool_calls, tool_call_id, name, and structured content.
    """
    existing_msgs = session_store.get_all_messages(session_id)
    if not existing_msgs:
        for m in incoming_messages:
            _insert_message_into_store(session_id, m, existing_msgs)
        return

    # Check prefix match from start
    prefix_len = 0
    while prefix_len < len(incoming_messages) and prefix_len < len(existing_msgs):
        if _messages_equal(incoming_messages[prefix_len], existing_msgs[prefix_len]):
            prefix_len += 1
        else:
            break

    if prefix_len > 0:
        net_new = incoming_messages[prefix_len:]
    else:
        # Check suffix overlap (sliding window client)
        best_overlap = 0
        for k in range(min(len(incoming_messages), len(existing_msgs)), 0, -1):
            sub_inc = incoming_messages[:k]
            sub_ex = existing_msgs[-k:]
            if all(_messages_equal(i, e) for i, e in zip(sub_inc, sub_ex)):
                best_overlap = k
                break
        net_new = incoming_messages[best_overlap:]

    for m in net_new:
        if existing_msgs and _messages_equal(existing_msgs[-1], m):
            continue
        _insert_message_into_store(session_id, m, existing_msgs)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Resolve session_id with priority order:
    # 1) X-Session-ID header
    # 2) body.get("session_id")
    # 3) Authorization: Bearer <token> (if token not empty / "none" / "sk-no-key") -> f"bearer:{token}"
    # 4) Fallback: "default"
    session_id = request.headers.get("X-Session-ID") or body.get("session_id")
    if not session_id:
        auth_header = request.headers.get("Authorization", "").strip()
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            if token and token.lower() not in ("none", "sk-no-key", "null", ""):
                session_id = f"bearer:{token}"
    if not session_id:
        session_id = "default"

    messages: List[Dict[str, Any]] = body.get("messages", [])
    is_stream = bool(body.get("stream", False))

    upstream_host = config.get("upstream", {}).get("host", "127.0.0.1")
    upstream_port = config.get("upstream", {}).get("port", 8081)
    upstream_url = f"http://{upstream_host}:{upstream_port}"

    session_state = session_store.get_or_create_session(session_id)

    # 1. Deduplicate and record net-new incoming messages in session store
    _sync_client_messages(session_id, messages)

    # 2. Get active messages (unconsolidated + pending consolidation)
    active_msgs = session_store.get_active_messages(session_id)
    if not active_msgs:
        active_msgs = messages

    # 3. ConsolidationAdvisor Break-Even Evaluation
    raw_exp = request.headers.get("X-Expected-Turns") or body.get("expected_turns")
    expected_turns = int(raw_exp) if raw_exp is not None else None

    last_user_text = next((_normalize_content_for_comparison(m.get("content")) for m in reversed(messages) if m.get("role") == "user"), "")
    total_tokens = await _tokenize_messages(active_msgs, upstream_url)

    decision = advisor.evaluate_turn(session_id, total_tokens, expected_turns=expected_turns, user_message=last_user_text)
    logger.info(f"[Advisor] Session {session_id} (H={total_tokens}): Action={decision.recommended_action} | {decision.reason}")

    # Eviction triggering:
    explicit_max = body.get("max_ctx_tokens") or os.environ.get("MAX_CTX_TOKENS")
    force_by_threshold = bool(explicit_max and total_tokens > int(explicit_max))

    if decision.recommended_action in ("consolidate", "force_consolidate") or force_by_threshold:
        candidates = session_store.get_unconsolidated_candidates(session_id)
        evict_chunk_tokens = body.get("evict_chunk_tokens") or int(os.environ.get("EVICT_CHUNK_TOKENS", config.get("thresholds", {}).get("evict_chunk_tokens", 2048)))
        evicted = []
        tokens_evicted = 0
        candidate_groups = _group_atomic_turns(candidates)
        grp_idx = 0
        while grp_idx < len(candidate_groups) and tokens_evicted < evict_chunk_tokens:
            grp = candidate_groups[grp_idx]
            evicted.extend(grp)
            for chunk in grp:
                tokens_evicted += max(1, len(_normalize_content_for_comparison(chunk.get("content"))) // 4)
            grp_idx += 1

        if evicted:
            evicted_ids = [m["id"] for m in evicted if "id" in m]
            logger.info(f"[Context Eviction] Enqueueing {len(evicted)} turns ({tokens_evicted} tokens, ids={evicted_ids}) for session {session_id}")
            session_store.mark_messages_pending(session_id, evicted_ids)
            advisor.record_eviction_event(session_id)
            consolidator.enqueue(ConsolidationItem(session_id, evicted))
    elif decision.recommended_action == "defer":
        logger.info(f"[Advisor] Deferring consolidation for session {session_id}: expected turns ({decision.expected_turns}) <= break-even ({decision.break_even_turns})")

    # Trimming: Ensure only system prompt + last keep_recent_tokens are sent in prompt payload
    # Dynamic target keep window adapted by advisor (expanded by rework penalty if detected)
    session_keep_tokens = body.get("keep_recent_tokens") or advisor.get_session_keep_tokens(session_id)
    slot_safety_limit = min(session_keep_tokens, 1200)

    trimmed_payload_msgs = []
    has_system = active_msgs and active_msgs[0].get("role") == "system"
    if has_system:
        system_msg = active_msgs[0]
        dialogue = active_msgs[1:]
    else:
        system_msg = None
        dialogue = active_msgs

    # Group dialogue into atomic units so tool calls and results are never split
    dialogue_groups = _group_atomic_turns(dialogue)
    curr_tokens = 0
    recent_kept_groups = []
    for grp in reversed(dialogue_groups):
        toks = sum(max(1, int(len(_normalize_content_for_comparison(m.get("content"))) / 1.5)) for m in grp)
        if curr_tokens + toks <= slot_safety_limit or not recent_kept_groups:
            recent_kept_groups.append(grp)
            curr_tokens += toks
        else:
            break

    recent_kept_groups.reverse()
    if system_msg:
        trimmed_payload_msgs.append(system_msg)
    for grp in recent_kept_groups:
        trimmed_payload_msgs.extend(grp)

    # Double check total token count with upstream tokenize to guarantee context limit
    total_trimmed_tokens = await _tokenize_messages(trimmed_payload_msgs, upstream_url)
    while total_trimmed_tokens > 1500 and len(trimmed_payload_msgs) > 2:
        if has_system and len(trimmed_payload_msgs) > 1:
            dropped = trimmed_payload_msgs[1]
            if dropped.get("role") == "assistant" and dropped.get("tool_calls"):
                idx_drop = 2
                while idx_drop < len(trimmed_payload_msgs) and trimmed_payload_msgs[idx_drop].get("role") == "tool":
                    idx_drop += 1
                trimmed_payload_msgs = [trimmed_payload_msgs[0]] + trimmed_payload_msgs[idx_drop:]
            else:
                trimmed_payload_msgs = [trimmed_payload_msgs[0]] + trimmed_payload_msgs[2:]
        else:
            dropped = trimmed_payload_msgs[0]
            if dropped.get("role") == "assistant" and dropped.get("tool_calls"):
                idx_drop = 1
                while idx_drop < len(trimmed_payload_msgs) and trimmed_payload_msgs[idx_drop].get("role") == "tool":
                    idx_drop += 1
                trimmed_payload_msgs = trimmed_payload_msgs[idx_drop:]
            else:
                trimmed_payload_msgs = trimmed_payload_msgs[1:]
        total_trimmed_tokens = await _tokenize_messages(trimmed_payload_msgs, upstream_url)

    # 4. Inject LoRA configuration dynamically into upstream request, preserving tool fields
    forward_body = dict(body)
    f_messages = []
    for m in trimmed_payload_msgs:
        f_msg = {"role": m["role"], "content": m.get("content")}
        if "tool_calls" in m and m["tool_calls"]:
            f_msg["tool_calls"] = m["tool_calls"]
        if "tool_call_id" in m and m["tool_call_id"]:
            f_msg["tool_call_id"] = m["tool_call_id"]
        if "name" in m and m["name"]:
            f_msg["name"] = m["name"]
        f_messages.append(f_msg)
    forward_body["messages"] = f_messages
    scale = session_state.active_lora_scale
    forward_body["lora"] = [{"id": 0, "scale": scale}]

    # 5. Inference execution under Memory Bus Lock
    await _on_stream_start()

    if is_stream:
        async def event_generator():
            try:
                timeout = httpx.Timeout(config.get("upstream", {}).get("timeout_seconds", 120.0), connect=10.0)
                client = httpx.AsyncClient(timeout=timeout)
                async with client:
                    async with client.stream("POST", f"{upstream_url}/v1/chat/completions", json=forward_body) as resp:
                        accumulated_content = []
                        accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}
                        async for line in resp.aiter_lines():
                            if line:
                                yield f"{line}\n\n"
                                if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                                    try:
                                        chunk_json = json.loads(line[6:])
                                        timings = chunk_json.get("timings")
                                        if timings and advisor:
                                            advisor.record_inference_telemetry(
                                                prompt_n=timings.get("prompt_n", 0),
                                                prompt_ms=timings.get("prompt_ms", 0.0),
                                                predicted_n=timings.get("predicted_n", 0),
                                                predicted_ms=timings.get("predicted_ms", 0.0),
                                                prompt_per_second=timings.get("prompt_per_second", 0.0),
                                                predicted_per_second=timings.get("predicted_per_second", 0.0),
                                            )
                                        choices = chunk_json.get("choices", [])
                                        if choices:
                                            delta = choices[0].get("delta", {})
                                            c_delta = delta.get("content")
                                            if c_delta:
                                                accumulated_content.append(c_delta)
                                            tc_deltas = delta.get("tool_calls")
                                            if tc_deltas:
                                                for tc in tc_deltas:
                                                    idx = tc.get("index", 0)
                                                    if idx not in accumulated_tool_calls:
                                                        accumulated_tool_calls[idx] = {
                                                            "id": tc.get("id", ""),
                                                            "type": tc.get("type", "function"),
                                                            "function": {
                                                                "name": tc.get("function", {}).get("name", ""),
                                                                "arguments": tc.get("function", {}).get("arguments", "")
                                                            }
                                                        }
                                                    else:
                                                        if tc.get("id"):
                                                            accumulated_tool_calls[idx]["id"] += tc["id"]
                                                        if tc.get("function", {}).get("name"):
                                                            accumulated_tool_calls[idx]["function"]["name"] += tc["function"]["name"]
                                                        if tc.get("function", {}).get("arguments"):
                                                            accumulated_tool_calls[idx]["function"]["arguments"] += tc["function"]["arguments"]
                                    except Exception:
                                        pass
                        # Save full assistant response
                        full_resp = "".join(accumulated_content) if accumulated_content else None
                        tool_calls_list = list(accumulated_tool_calls.values()) if accumulated_tool_calls else None
                        if full_resp is not None or tool_calls_list:
                            extra = {}
                            if tool_calls_list:
                                extra["tool_calls"] = tool_calls_list
                            session_store.add_message(session_id, "assistant", full_resp, extra=extra if extra else None)
            finally:
                await _on_stream_end()

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        try:
            timeout = httpx.Timeout(config.get("upstream", {}).get("timeout_seconds", 120.0), connect=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{upstream_url}/v1/chat/completions", json=forward_body)
                if resp.status_code == 200:
                    data = resp.json()
                    msg_obj = data["choices"][0]["message"]
                    content = msg_obj.get("content")
                    tool_calls = msg_obj.get("tool_calls")
                    extra = {}
                    if tool_calls:
                        extra["tool_calls"] = tool_calls
                    session_store.add_message(session_id, "assistant", content, extra=extra if extra else None)
                    timings = data.get("timings") or {}
                    if timings and advisor:
                        advisor.record_inference_telemetry(
                            prompt_n=timings.get("prompt_n", 0),
                            prompt_ms=timings.get("prompt_ms", 0.0),
                            predicted_n=timings.get("predicted_n", 0),
                            predicted_ms=timings.get("predicted_ms", 0.0),
                            prompt_per_second=timings.get("prompt_per_second", 0.0),
                            predicted_per_second=timings.get("predicted_per_second", 0.0),
                        )
                    elif "usage" in data and advisor:
                        usage = data["usage"]
                        advisor.record_inference_telemetry(
                            prompt_n=usage.get("prompt_tokens", 0),
                            prompt_ms=0.0,
                            predicted_n=usage.get("completion_tokens", 0),
                            predicted_ms=0.0,
                        )
                    return JSONResponse(status_code=resp.status_code, content=data)
                else:
                    logger.error(f"[Proxy] Upstream HTTP {resp.status_code}: {resp.text} | forward_body keys: {list(forward_body.keys())} | messages count: {len(forward_body.get('messages', []))}")
                    return JSONResponse(status_code=resp.status_code, content=resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"detail": resp.text})
        finally:
            await _on_stream_end()


@app.get("/v1/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Returns session state, message count, and stored messages."""
    session_state = session_store.get_or_create_session(session_id)
    messages = session_store.get_all_messages(session_id)
    return {
        "session_id": session_id,
        "generation": session_state.generation,
        "active_lora_scale": session_state.active_lora_scale,
        "message_count": len(messages),
        "messages": messages,
    }


@app.post("/v1/sessions/{session_id}/export")
async def export_session(session_id: str):
    """Persists current ephemeral weights permanently."""
    adapter_path = config.get("lora", {}).get("adapter_path", "/dev/shm/ephemeral_session.gguf")
    if not os.path.exists(adapter_path):
        raise HTTPException(status_code=404, detail="No ephemeral adapter found on /dev/shm")

    out_dir = Path("/home/loki/ephemeral-lora-proxy/persisted_adapters")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{session_id}.gguf"

    shutil.copyfile(adapter_path, out_file)
    session_state = session_store.get_or_create_session(session_id)
    all_msgs = session_store.get_all_messages(session_id)
    return {
        "status": "exported",
        "session_id": session_id,
        "generation": session_state.generation,
        "active_lora_scale": session_state.active_lora_scale,
        "exported_path": str(out_file),
        "size_bytes": os.path.getsize(out_file),
        "message_count": len(all_msgs),
    }


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """Clears session history and resets the ephemeral LoRA weights."""
    session_store.reset_session(session_id)
    if consolidator and consolidator.backend:
        consolidator.backend.reset_adapter(config)
    upstream_host = config.get("upstream", {}).get("host", "127.0.0.1")
    upstream_port = config.get("upstream", {}).get("port", 8081)
    await consolidator._reload_lora_adapter(f"http://{upstream_host}:{upstream_port}", 0.0)
    return {
        "status": "reset",
        "session_id": session_id,
        "detail": "Session reset and ephemeral weights cleared"
    }

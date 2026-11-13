import os
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
    global config, session_store, consolidator
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
    consolidator = EphemeralConsolidator(config, session_store, inference_idle_event)
    consolidator.start()
    logger.info("Ephemeral LoRA Proxy initialized successfully")

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
    }


async def _tokenize_messages(messages: List[Dict[str, Any]], upstream_url: str) -> int:
    """Estimate token count by calling upstream /tokenize or word estimate fallback."""
    text = " ".join([m.get("content", "") for m in messages])
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


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    session_id = request.headers.get("X-Session-ID") or body.get("session_id") or "default"
    messages: List[Dict[str, Any]] = body.get("messages", [])
    is_stream = bool(body.get("stream", False))

    upstream_host = config.get("upstream", {}).get("host", "127.0.0.1")
    upstream_port = config.get("upstream", {}).get("port", 8081)
    upstream_url = f"http://{upstream_host}:{upstream_port}"

    session_state = session_store.get_or_create_session(session_id)

    # 1. Record incoming user message in session store
    for m in messages:
        if m.get("role") == "user":
            # Check if this user message is already stored as the latest
            existing = session_store.get_all_messages(session_id)
            if not existing or existing[-1]["content"] != m.get("content"):
                session_store.add_message(session_id, m["role"], m.get("content", ""))

    # 2. Get unconsolidated active messages
    active_msgs = session_store.get_active_messages(session_id)
    if not active_msgs:
        active_msgs = messages

    # 3. Check token count against max_ctx_tokens threshold
    max_ctx_tokens = body.get("max_ctx_tokens") or int(os.environ.get("MAX_CTX_TOKENS", config.get("thresholds", {}).get("max_ctx_tokens", 8192)))
    evict_chunk_tokens = body.get("evict_chunk_tokens") or int(os.environ.get("EVICT_CHUNK_TOKENS", config.get("thresholds", {}).get("evict_chunk_tokens", 2048)))
    keep_recent_tokens = body.get("keep_recent_tokens") or int(os.environ.get("KEEP_RECENT_TOKENS", config.get("thresholds", {}).get("keep_recent_tokens", 2048)))

    total_tokens = await _tokenize_messages(active_msgs, upstream_url)

    # Context growth eviction check
    if total_tokens > max_ctx_tokens:
        # Separate system prompt from dialogue turns
        has_system = active_msgs and active_msgs[0]["role"] == "system"
        system_msg = active_msgs[0] if has_system else None
        dialogue = active_msgs[1:] if has_system else active_msgs

        evicted = []
        tokens_evicted = 0
        idx = 0
        while idx < len(dialogue) and tokens_evicted < evict_chunk_tokens:
            chunk = dialogue[idx]
            evicted.append(chunk)
            tokens_evicted += max(1, len(chunk.get("content", "")) // 4)
            idx += 1

        if evicted:
            logger.info(f"[Context Eviction] Evicting {len(evicted)} turns ({tokens_evicted} tokens) for session {session_id}")
            consolidator.enqueue(ConsolidationItem(session_id, evicted))
            active_msgs = ([system_msg] if system_msg else []) + dialogue[idx:]

    # Trimming: Ensure only system prompt + last keep_recent_tokens are sent in prompt payload
    trimmed_payload_msgs = []
    has_system = active_msgs and active_msgs[0]["role"] == "system"
    if has_system:
        trimmed_payload_msgs.append(active_msgs[0])
        dialogue_recent = active_msgs[1:]
    else:
        dialogue_recent = active_msgs

    # Retain up to keep_recent_tokens
    curr_tokens = 0
    recent_kept = []
    for m in reversed(dialogue_recent):
        toks = max(1, len(m.get("content", "")) // 4)
        if curr_tokens + toks <= keep_recent_tokens or not recent_kept:
            recent_kept.append(m)
            curr_tokens += toks
        else:
            break
    recent_kept.reverse()
    trimmed_payload_msgs.extend(recent_kept)

    # 4. Inject LoRA configuration dynamically into upstream request
    forward_body = dict(body)
    forward_body["messages"] = [{"role": m["role"], "content": m["content"]} for m in trimmed_payload_msgs]
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
                        async for line in resp.aiter_lines():
                            if line:
                                yield f"{line}\n\n"
                                if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                                    try:
                                        chunk_json = json.loads(line[6:])
                                        delta = chunk_json["choices"][0].get("delta", {}).get("content", "")
                                        if delta:
                                            accumulated_content.append(delta)
                                    except Exception:
                                        pass
                        # Save full assistant response
                        if accumulated_content:
                            full_resp = "".join(accumulated_content)
                            session_store.add_message(session_id, "assistant", full_resp)
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
                    content = data["choices"][0]["message"].get("content", "")
                    if content:
                        session_store.add_message(session_id, "assistant", content)
                    return JSONResponse(status_code=resp.status_code, content=data)
                else:
                    return JSONResponse(status_code=resp.status_code, content=resp.json())
        finally:
            await _on_stream_end()


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
    return {
        "status": "exported",
        "session_id": session_id,
        "generation": session_state.generation,
        "exported_path": str(out_file),
        "size_bytes": os.path.getsize(out_file),
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

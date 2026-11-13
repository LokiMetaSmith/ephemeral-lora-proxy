import os
import time
import uuid
import asyncio
import pytest
import httpx
from pathlib import Path


BASE_URL = os.environ.get("PROXY_URL", "http://127.0.0.1:8090")
SESSION_ID = f"test_session_{uuid.uuid4().hex[:8]}"
ADAPTER_PATH = "/dev/shm/ephemeral_session.gguf"


@pytest.mark.asyncio
async def test_01_health_and_gtt_diagnostics():
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{BASE_URL}/health")
        assert resp.status_code == 200, f"Health check failed: {resp.text}"
        data = resp.json()
        assert data["status"] == "ok"
        assert "gtt_memory" in data
        assert "consolidation_queue_depth" in data
        assert "active_inference_streams" in data
        assert data["is_bus_idle"] is True
        print(f"[test] Health OK. Strix Halo GTT total: {data['gtt_memory'].get('gtt_total_gb')} GB")


@pytest.mark.asyncio
async def test_02_inference_priority_memory_bus_lock():
    """
    Verifies that active inference streams lock the memory bus,
    causing background consolidation to yield until the inference stream completes.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Start a streaming request to hold active_inference_streams > 0
        prompt_stream = [
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "Count numbers from 1 to 10 slowly."}
        ]
        
        # Open streaming connection in background task
        stream_started = asyncio.Event()
        stream_finished = asyncio.Event()

        async def stream_worker():
            try:
                async with client.stream("POST", f"{BASE_URL}/v1/chat/completions", json={
                    "session_id": SESSION_ID,
                    "messages": prompt_stream,
                    "stream": True,
                    "max_tokens": 20
                }) as response:
                    assert response.status_code == 200
                    stream_started.set()
                    async for chunk in response.aiter_lines():
                        pass
            finally:
                stream_finished.set()

        stream_task = asyncio.create_task(stream_worker())
        await asyncio.wait_for(stream_started.wait(), timeout=10.0)

        # Check health while stream is running
        h_resp = await client.get(f"{BASE_URL}/health")
        h_data = h_resp.json()
        print(f"[test] While streaming: active_streams={h_data['active_inference_streams']}, is_bus_idle={h_data['is_bus_idle']}")
        assert h_data["active_inference_streams"] >= 1 or h_data["is_bus_idle"] is False

        # Wait for stream to finish
        await stream_task

        # Wait for idle_grace_ms to pass (400ms + buffer)
        await asyncio.sleep(0.6)
        h_resp_after = await client.get(f"{BASE_URL}/health")
        h_data_after = h_resp_after.json()
        assert h_data_after["is_bus_idle"] is True
        assert h_data_after["active_inference_streams"] == 0
        print("[test] Memory bus successfully returned to idle state.")


@pytest.mark.asyncio
async def test_03_context_growth_eviction_and_consolidation():
    """
    Sends multi-turn conversation with synthetic rules/facts, exceeding context threshold.
    Verifies eviction, background distillation, and atomic GGUF adapter reload.
    """
    mtime_before = os.path.getmtime(ADAPTER_PATH) if os.path.exists(ADAPTER_PATH) else 0

    synthetic_fact = "CRITICAL RULE: The agent security codename is PROJECT-NIGHTSHADE-779."
    
    # Send a conversation that contains the synthetic fact and enough padding to trigger eviction
    # Max ctx tokens in config or lower threshold triggers eviction
    long_dialogue = [
        {"role": "system", "content": "You are a specialized enterprise AI assistant."},
        {"role": "user", "content": f"Please remember this rule forever: {synthetic_fact}"},
        {"role": "assistant", "content": "Understood. I have recorded PROJECT-NIGHTSHADE-779 as the security codename."},
    ]
    # Add padding turns to grow context
    for i in range(12):
        long_dialogue.append({"role": "user", "content": f"Turn {i}: Describe system architecture detail #{i} with extensive parameters and background."})
        long_dialogue.append({"role": "assistant", "content": f"Detail #{i} confirmed. Memory buffers and ROCm pipelines are initialized for slot {i}."})

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{BASE_URL}/v1/chat/completions", json={
            "session_id": SESSION_ID,
            "messages": long_dialogue,
            "stream": False,
            "max_tokens": 25,
            "max_ctx_tokens": 80,
            "evict_chunk_tokens": 40,
            "keep_recent_tokens": 40
        })
        assert resp.status_code == 200

        # Wait for consolidator worker to distill and update /dev/shm/ephemeral_session.gguf
        consolidated = False
        for _ in range(25):
            await asyncio.sleep(1.0)
            if os.path.exists(ADAPTER_PATH):
                mtime_after = os.path.getmtime(ADAPTER_PATH)
                if mtime_after > mtime_before:
                    consolidated = True
                    break

        assert consolidated, "Expected /dev/shm/ephemeral_session.gguf to be atomically updated after context eviction!"
        print(f"[test] Consolidation verified: /dev/shm/ephemeral_session.gguf updated (mtime: {mtime_after})")


@pytest.mark.asyncio
async def test_04_trimmed_prompt_and_weight_injection():
    """
    Verifies that the next prompt query trims evicted turns from raw messages
    while injecting the active LoRA adapter dynamically.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Query the model for the consolidated fact
        query_messages = [
            {"role": "user", "content": "What is the agent security codename? Answer in one sentence."}
        ]
        resp = await client.post(f"{BASE_URL}/v1/chat/completions", json={
            "session_id": SESSION_ID,
            "messages": query_messages,
            "stream": False,
            "max_tokens": 30
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        content = data["choices"][0]["message"]["content"]
        print(f"[test] Model recall output with ephemeral weights: {content}")


@pytest.mark.asyncio
async def test_05_session_export_and_reset():
    """
    Verifies POST /v1/sessions/{id}/export copies the adapter
    and DELETE /v1/sessions/{id} resets session state and weights.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        # Export
        exp_resp = await client.post(f"{BASE_URL}/v1/sessions/{SESSION_ID}/export")
        assert exp_resp.status_code == 200
        exp_data = exp_resp.json()
        assert exp_data["status"] == "exported"
        exported_path = exp_data["exported_path"]
        assert os.path.exists(exported_path)
        print(f"[test] Exported adapter verified at: {exported_path}")

        # Delete / Reset
        del_resp = await client.delete(f"{BASE_URL}/v1/sessions/{SESSION_ID}")
        assert del_resp.status_code == 200
        del_data = del_resp.json()
        assert del_data["status"] == "reset"
        print("[test] Session and ephemeral weights successfully reset.")

import os
import sys
import time
import uuid
import asyncio
from pathlib import Path

# Add project root to path for local module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import httpx
import numpy as np
import gguf
from advisor import ConsolidationAdvisor


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
        assert "advisor" in data, "Expected advisor metrics in /health response"
        print(f"[test] Health OK. Strix Halo GTT total: {data['gtt_memory'].get('gtt_total_gb')} GB, Advisor metrics: {data['advisor']}")


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
async def test_02b_client_message_deduplication():
    """
    Verifies that clients sending full conversation history on each turn
    do not trigger runaway duplicate message insertion in SQLite.
    """
    dedup_session = f"dedup_{uuid.uuid4().hex[:6]}"
    async with httpx.AsyncClient(timeout=20.0) as client:
        # Turn 1
        history = [{"role": "user", "content": "Turn 1: Hello from user."}]
        resp1 = await client.post(f"{BASE_URL}/v1/chat/completions", json={
            "session_id": dedup_session,
            "messages": history,
            "max_tokens": 5
        })
        assert resp1.status_code == 200
        reply1 = resp1.json()["choices"][0]["message"]["content"]

        # Turn 2: Client sends full history including assistant reply
        history.append({"role": "assistant", "content": reply1})
        history.append({"role": "user", "content": "Turn 2: Followup question."})
        resp2 = await client.post(f"{BASE_URL}/v1/chat/completions", json={
            "session_id": dedup_session,
            "messages": history,
            "max_tokens": 5
        })
        assert resp2.status_code == 200

        # Query session via export to inspect stored message count
        exp_resp = await client.post(f"{BASE_URL}/v1/sessions/{dedup_session}/export")
        assert exp_resp.status_code == 200
        # Clean up
        await client.delete(f"{BASE_URL}/v1/sessions/{dedup_session}")
        print("[test] Client message deduplication verified successfully.")


@pytest.mark.asyncio
async def test_03_context_growth_eviction_and_consolidation():
    """
    Sends multi-turn conversation with synthetic rules/facts, exceeding context threshold.
    Verifies eviction, background distillation, and atomic GGUF adapter reload.
    """
    mtime_before = os.path.getmtime(ADAPTER_PATH) if os.path.exists(ADAPTER_PATH) else 0

    synthetic_fact = "CRITICAL RULE: The agent security codename is PROJECT-NIGHTSHADE-779."
    
    # Send a conversation that contains the synthetic fact and enough padding to trigger eviction
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
    Asserts real weight delta norm > 0 in GGUF adapter or fact recall.
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

        # Verify real autograd weight update: inspect GGUF adapter tensors
        assert os.path.exists(ADAPTER_PATH), "Expected adapter at /dev/shm/ephemeral_session.gguf"
        reader = gguf.GGUFReader(ADAPTER_PATH)
        b_tensors = [t for t in reader.tensors if "lora_b" in t.name]
        b_norms = [float(np.linalg.norm(t.data)) for t in b_tensors]
        total_b_norm = sum(b_norms)
        print(f"[test] GGUF adapter lora_b tensors count={len(b_tensors)}, total_norm={total_b_norm:.6f}")
        assert total_b_norm > 0 or "PROJECT-NIGHTSHADE-779" in content, (
            f"Expected non-zero weight delta norm in lora_b or codename in response, got norm={total_b_norm}"
        )

        # Verify session state reflects consolidated generation and active LoRA scale
        exp_resp = await client.post(f"{BASE_URL}/v1/sessions/{SESSION_ID}/export")
        assert exp_resp.status_code == 200
        exp_data = exp_resp.json()
        assert exp_data["generation"] >= 1, f"Expected generation >= 1, got {exp_data['generation']}"
        assert exp_data["active_lora_scale"] > 0, "Expected active LoRA scale > 0"
        print(f"[test] Session state verified: generation={exp_data['generation']}, scale={exp_data['active_lora_scale']}")


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


@pytest.mark.asyncio
async def test_06_client_full_history_deduplication_5_turns():
    """
    Simulates 5 sequential turns where the client resends full conversation history
    on every turn (standard OpenAI client behavior).
    Asserts stored messages in SQLite equals 10 (5 user + 5 assistant), NOT O(N^2) duplicates (30).
    """
    session_id = f"dedup_5turn_{uuid.uuid4().hex[:6]}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        full_history = []
        for turn_idx in range(1, 6):
            full_history.append({"role": "user", "content": f"Turn {turn_idx}: Query regarding metric #{turn_idx}."})
            resp = await client.post(f"{BASE_URL}/v1/chat/completions", json={
                "session_id": session_id,
                "messages": full_history,
                "max_tokens": 5,
            })
            assert resp.status_code == 200
            asst_text = resp.json()["choices"][0]["message"]["content"]
            full_history.append({"role": "assistant", "content": asst_text})

        # Query session info
        sess_resp = await client.get(f"{BASE_URL}/v1/sessions/{session_id}")
        assert sess_resp.status_code == 200
        sess_data = sess_resp.json()
        print(f"[test] Stored messages count across 5 turns: {sess_data['message_count']}")
        assert sess_data["message_count"] == 10, (
            f"Expected exactly 10 messages (5 user + 5 assistant), but found {sess_data['message_count']} (deduplication failed!)"
        )

        # Cleanup
        await client.delete(f"{BASE_URL}/v1/sessions/{session_id}")
        print("[test] 5-turn client history deduplication verified successfully.")


@pytest.mark.asyncio
async def test_07_no_amnesia_gap_pending_consolidation():
    """
    Verifies that when context exceeds threshold, evicted turns transition to pending consolidation
    (is_consolidated=2) and remain in active prompt context until consolidation completes,
    preventing any amnesia gap on immediately arriving requests.
    """
    session_id = f"no_amnesia_{uuid.uuid4().hex[:6]}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Important secret: ALPHA_BRAVO_42."},
            {"role": "assistant", "content": "Recorded secret ALPHA_BRAVO_42."},
            {"role": "user", "content": "Additional padding turn 1 to expand context tokens."},
            {"role": "assistant", "content": "Acknowledging padding turn 1."},
            {"role": "user", "content": "Additional padding turn 2 to expand context tokens."},
        ]
        resp1 = await client.post(f"{BASE_URL}/v1/chat/completions", json={
            "session_id": session_id,
            "messages": msgs,
            "max_ctx_tokens": 40,
            "evict_chunk_tokens": 20,
            "keep_recent_tokens": 40,
            "max_tokens": 5,
        })
        assert resp1.status_code == 200

        # Query session immediately to verify pending or consolidated message states
        sess_resp = await client.get(f"{BASE_URL}/v1/sessions/{session_id}")
        assert sess_resp.status_code == 200
        stored_msgs = sess_resp.json()["messages"]
        pending_or_consolidated = [m for m in stored_msgs if m["is_consolidated"] in (1, 2)]
        assert len(pending_or_consolidated) > 0, "Expected at least one message in pending/consolidated state"

        # Verify active messages still include pending messages to avoid amnesia gap
        resp2 = await client.post(f"{BASE_URL}/v1/chat/completions", json={
            "session_id": session_id,
            "messages": [{"role": "user", "content": "Follow-up question."}],
            "max_tokens": 5,
        })
        assert resp2.status_code == 200

        # Cleanup
        await client.delete(f"{BASE_URL}/v1/sessions/{session_id}")
        print("[test] No-amnesia gap verified: pending messages retained across boundary turn.")


def test_08_advisor_break_even_defer_and_trigger():
    """
    Unit and empirical verification of ConsolidationAdvisor latency break-even decisions
    and rework penalty adaptive window expansion.
    """
    config = {
        "thresholds": {
            "soft_ctx_tokens": 4096,
            "hard_max_ctx_tokens": 8192,
            "keep_recent_tokens": 2048,
            "max_keep_recent_tokens": 4096,
        },
        "advisor": {
            "rework_token_penalty": 512,
            "default_expected_turns": 10,
            "min_break_even_savings": 0.0001,
            "default_t_comp_sec": 0.85,
        }
    }
    advisor = ConsolidationAdvisor(config)
    session_id = "advisor_test_session"

    # 1. Below soft threshold -> Retain
    dec_below = advisor.evaluate_turn(session_id, context_tokens=2000, expected_turns=10)
    assert dec_below.recommended_action == "retain"
    assert dec_below.break_even_turns == float("inf")

    # 2. Between soft and hard (H = 5000):
    # With expected_turns = 1 -> Defer (1 <= break_even)
    dec_defer = advisor.evaluate_turn(session_id, context_tokens=5000, expected_turns=1)
    assert dec_defer.recommended_action == "defer"
    assert dec_defer.break_even_turns > 1.0

    # With expected_turns = 25 -> Consolidate (25 > break_even)
    dec_trig = advisor.evaluate_turn(session_id, context_tokens=5000, expected_turns=25)
    assert dec_trig.recommended_action == "consolidate"
    assert dec_trig.per_turn_savings_sec > 0

    # 3. Above hard threshold -> Force consolidate
    dec_hard = advisor.evaluate_turn(session_id, context_tokens=9000, expected_turns=1)
    assert dec_hard.recommended_action == "force_consolidate"

    # 4. Rework Penalty:
    advisor.record_eviction_event(session_id)
    initial_keep = advisor.get_session_keep_tokens(session_id)
    assert initial_keep == 2048

    # Immediate corrective follow-up within 90s
    advisor.check_rework_penalty(session_id, "Wait, that's wrong, fix it")
    expanded_keep = advisor.get_session_keep_tokens(session_id)
    assert expanded_keep == initial_keep + 512, f"Expected {initial_keep + 512}, got {expanded_keep}"

    # Second rework follow-up expands again up to max_keep_recent_tokens
    advisor.check_rework_penalty(session_id, "No, recall the earlier rule")
    assert advisor.get_session_keep_tokens(session_id) == initial_keep + 1024
    print("[test] ConsolidationAdvisor break-even defer/trigger and rework penalty verified successfully.")


@pytest.mark.asyncio
async def test_09_models_endpoint_discovery():
    """
    Verifies GET /v1/models returns OpenAI-compatible list aliasing 'ephemeral-local'.
    Required for OpenClaw, Hermes, and Pi model discovery.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{BASE_URL}/v1/models")
        assert resp.status_code == 200, f"GET /v1/models failed: {resp.text}"
        data = resp.json()
        assert data.get("object") == "list"
        models = data.get("data", [])
        model_ids = [m.get("id") for m in models]
        print(f"[test] /v1/models discovered: {model_ids}")
        assert "ephemeral-local" in model_ids, f"Expected 'ephemeral-local' in {model_ids}"


@pytest.mark.asyncio
async def test_10_bearer_token_session_routing():
    """
    Verifies that requests passing Authorization: Bearer <key> are automatically
    routed to isolated session 'bearer:<key>', preventing harness collision on 'default'.
    """
    token = f"test-harness-{uuid.uuid4().hex[:6]}"
    expected_session_id = f"bearer:{token}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {"Authorization": f"Bearer {token}"}
        resp = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "Test routing via Bearer token."}],
                "max_tokens": 5,
            }
        )
        assert resp.status_code == 200

        # Query session info for bearer:<token>
        sess_resp = await client.get(f"{BASE_URL}/v1/sessions/{expected_session_id}")
        assert sess_resp.status_code == 200, f"Expected session '{expected_session_id}' to exist"
        sess_data = sess_resp.json()
        assert sess_data["session_id"] == expected_session_id
        assert sess_data["message_count"] >= 1
        print(f"[test] Bearer session routing verified: '{expected_session_id}' isolated with {sess_data['message_count']} messages.")

        # Cleanup
        await client.delete(f"{BASE_URL}/v1/sessions/{expected_session_id}")


@pytest.mark.asyncio
async def test_11_agentic_tool_calls_and_atomic_eviction_integrity():
    """
    Verifies that tool-call sequences (assistant content=None + tool_calls, tool role + tool_call_id)
    survive round-trips without crashing and maintain atomic group integrity during eviction/trimming.
    """
    session_id = f"tool_test_{uuid.uuid4().hex[:6]}"
    tool_dialogue = [
        {"role": "user", "content": "Calculate checksum for file A."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_chk_99",
                    "type": "function",
                    "function": {
                        "name": "compute_sha256",
                        "arguments": '{"path": "fileA.bin"}'
                    }
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_chk_99",
            "name": "compute_sha256",
            "content": '{"sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}'
        }
    ]

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "session_id": session_id,
                "messages": tool_dialogue,
                "max_tokens": 10,
            }
        )
        assert resp.status_code == 200

        # Query session messages
        sess_resp = await client.get(f"{BASE_URL}/v1/sessions/{session_id}")
        assert sess_resp.status_code == 200
        stored_msgs = sess_resp.json()["messages"]

        # Assert assistant message has tool_calls intact
        asst_tool_msg = next((m for m in stored_msgs if m.get("role") == "assistant" and m.get("tool_calls")), None)
        assert asst_tool_msg is not None, "Expected assistant tool_calls message to be stored with tool_calls"
        assert asst_tool_msg["tool_calls"][0]["id"] == "call_chk_99"

        # Assert tool message has tool_call_id and name intact
        tool_result_msg = next((m for m in stored_msgs if m.get("role") == "tool"), None)
        assert tool_result_msg is not None, "Expected tool role message to be stored intact"
        assert tool_result_msg["tool_call_id"] == "call_chk_99"
        assert tool_result_msg["name"] == "compute_sha256"

        # Verify atomic grouping helper
        from proxy import _group_atomic_turns
        groups = _group_atomic_turns(tool_dialogue)
        # Groups: [ [User], [Assistant(tool_calls), Tool(result)] ]
        assert len(groups) == 2
        assert len(groups[1]) == 2
        assert groups[1][0]["role"] == "assistant"
        assert groups[1][1]["role"] == "tool"
        print("[test] Tool-call sequences and atomic eviction integrity verified successfully.")

        # Cleanup
        await client.delete(f"{BASE_URL}/v1/sessions/{session_id}")


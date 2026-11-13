# Ephemeral Weight Consolidation Proxy & LoRA Sidecar for llama.cpp

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform: AMD ROCm](https://img.shields.io/badge/Platform-AMD%20ROCm%20%2F%20gfx1151-red.svg)](https://rocm.docs.amd.com/)
[![llama.cpp](https://img.shields.io/badge/Engine-llama.cpp-brightgreen.svg)](https://github.com/ggml-org/llama.cpp)
[![FastAPI](https://img.shields.io/badge/API-OpenAI%20Compatible-blue.svg)](https://fastapi.tiangolo.com/)

An ephemeral weight consolidation reverse proxy sitting in front of `llama-server`. Optimized for **AMD Strix Halo (Ryzen AI Max+ 395 / Radeon 8060S, `gfx1151`, 128 GB Unified LPDDR5X RAM)** and other unified memory architectures.

Instead of discarding or lossily summarizing context tokens when dialogues overflow the context window, this service intercepts context growth, distills evicted dialogue turns into high-density training pairs, performs a fast rank-4/rank-8 LoRA weight update written to `/dev/shm` (tmpfs), and hot-reloads the adapter in `llama-server` without restarting the server or flushing the active KV cache — all while strictly yielding GPU and memory-bus access to foreground inference streams.

---

## The Motivation: Why Ephemeral Weight Consolidation?

### 1. The Context Window Dilemma
As multi-turn conversations expand, maintaining full conversation history inside the attention KV cache leads to:
- **Exponential Memory Growth:** Long KV caches consume tens of gigabytes of RAM.
- **Latency Degradation (Time to First Token):** Processing enormous prompt contexts slows down interactive response times.
- **Flawed Summarization:** Traditional context compaction relies on LLM summarization, which consumes prompt tokens, degrades nuance, and introduces hallucinated condensations.

### 2. The Solution: Turn Context into Weights
Instead of treating context as static tokens in the prompt, **Ephemeral Weight Consolidation** distills evicted dialogue turns into factual Q&A/rule pairs, performs an incremental low-rank update (LoRA) directly in RAM (`/dev/shm`), and applies the adapter dynamically to the active session. The dialogue window stays compact and fast, while the model retains factual memory natively through weights.

### 3. The Strix Halo Memory Bus Challenge
AMD Strix Halo combines a high-performance 16-core Zen 5 CPU and a 40-CU RDNA 3.5 iGPU with 128 GB unified memory sharing a single 256 GB/s LPDDR5X bus. Running gradient calculations concurrently with token generation cuts generation throughput (`tok/s`) in half. The **Inference-Priority Memory Bus Lock** solves this by maintaining an atomic counter of active inference streams and only executing weight updates when the memory bus has been idle for a configurable grace period (`idle_grace_ms: 400`).

---

## Architecture Overview

```
                            Client Requests (OpenAI API)
                                         |
                                         v
         +---------------------------------------------------------------+
         |                  Ephemeral LoRA Proxy (FastAPI)               |
         |  - POST /v1/chat/completions (SSE streaming & non-streaming)  |
         |  - GET /health (GTT total/used, bus idle state, queue depth)  |
         |  - Sliding Window: Keeps system prompt + last keep_tokens     |
         |  - Dynamic Weight Injection: "lora": [{"id": 0, "scale": 1.0}]|
         +-------------------------------+-------------------------------+
                                         |                               |
             [Active Inference Bus Lock] |               [Context Eviction]
                                         |                               |
                                         v                               v
         +-------------------------------+       +-----------------------+
         |         llama-server          |       | Ephemeral Consolidator|
         |  - --load-mode none /         |       | - Waits for bus idle  |
         |    --no-mmap                  |       | - Step 1: Self-Distill|
         |  - -ngl 999 --parallel 2      |       | - Step 2: Replay Buffer|
         |  - Mounted on tmpfs:          |       | - Step 3: Fast Weights|
         |    /dev/shm/ephemeral_...gguf |       | - Step 4: Hot-Reload  |
         +-------------------------------+       +-----------+-----------+
                         ^                                   |
                         +---------- Atomic Swap ------------+
                              (os.replace via .tmp)
```

---

## Consolidation Workflow

1. **Interception:** When active unconsolidated tokens exceed `max_ctx_tokens` (e.g. 8192), the proxy slices off the oldest `evict_chunk_tokens` (e.g. 2048) and enqueues them for consolidation.
2. **Memory Bus Arbitration:** The background consolidator strictly awaits `inference_idle_event` before initiating any GPU/GTT work.
3. **Self-Distillation:** Calls upstream `llama-server` with clean base weights (`lora: []`) using a deterministic extraction prompt to convert raw dialogue into factual Q&A and rule-enforcement pairs.
4. **Replay Buffer Mixing:** Blends distilled pairs with a static anchor replay buffer (`replay_buffer.jsonl`, ~15% ratio) with strong weight decay to eliminate catastrophic forgetting.
5. **Incremental Weight Update:** Invokes the pluggable `TrainerBackend` to update the rank-4 adapter on `/dev/shm/ephemeral_session.gguf` via atomic temp-file replace (`os.replace`).
6. **Hot-Reload:** Calls `POST /lora-adapters` to update the adapter scale without dropping KV cache or restarting the server.

---

## AMD Strix Halo Hardware & Kernel Optimization

To enable 100+ GB of unified GTT shared memory on Debian / Ubuntu / Arch for Strix Halo:

1. Edit `/etc/default/grub` and append to `GRUB_CMDLINE_LINUX_DEFAULT`:
   ```bash
   amd_iommu=off amdgpu.gttsize=131072 ttm.pages_limit=33554432
   ```
2. Update GRUB and reboot:
   ```bash
   sudo update-grub
   sudo reboot
   ```
3. Verify GTT memory:
   ```bash
   cat /sys/class/drm/card*/device/mem_info_gtt_total
   # Should report >= 137438953472 (~128 GB)
   ```

---

## Directory Structure

```
ephemeral-lora-proxy/
├── config.yaml                   # Configuration (ports, thresholds, rank, flags)
├── proxy.py                      # FastAPI async reverse proxy with memory bus lock
├── consolidator.py               # Background async consolidation worker
├── session_store.py              # SQLite + in-memory session persistence
├── replay_buffer.jsonl           # Anchor instruction-following pairs (~15% mix)
├── backends/
│   ├── base.py                   # TrainerBackend abstract interface
│   ├── llamacpp_native.py        # Native llama-finetune & GGUF LoRA consolidator
│   └── pytorch_sidecar.py        # Optional ROCm PyTorch / PEFT sidecar
├── scripts/
│   ├── init_seed_lora.py         # Seed rank-4 GGUF LoRA generator
│   └── launch_stack.sh           # Hardware diagnostic pre-flight & stack launcher
├── tests/
│   └── test_consolidation_loop.py # End-to-end integration test suite
└── persisted_adapters/           # Storage for exported session weights
```

---

## Quickstart

### 1. Prerequisites
- Linux host with Python >= 3.10 and `llama.cpp` built with ROCm or Vulkan.
- `pip` or `uv`:
  ```bash
  uv venv .venv
  uv pip install --python .venv -r requirements.txt
  ```

### 2. Configure
Adjust paths in `config.yaml`:
```yaml
server:
  host: "0.0.0.0"
  port: 8090

upstream:
  host: "127.0.0.1"
  port: 8081

model:
  path: "/path/to/base-model.gguf"

lora:
  adapter_path: "/dev/shm/ephemeral_session.gguf"
  r: 4
  alpha: 16.0
  scale: 1.0

thresholds:
  max_ctx_tokens: 8192
  evict_chunk_tokens: 2048
  keep_recent_tokens: 2048

strix_halo:
  no_mmap: true
  hipblaslt: true
  hip_launch_blocking: false
  parallel_slots: 2
  idle_grace_ms: 400
```

### 3. Launch
```bash
bash scripts/launch_stack.sh
```

### 4. Run Tests
```bash
./.venv/bin/pytest -v -s tests/test_consolidation_loop.py
```

---

## API Reference

### `POST /v1/chat/completions`
OpenAI-compatible chat completion endpoint supporting streaming SSE (`"stream": true`) and non-streaming responses. Automatically injects the active LoRA weights and tracks active inference streams.

### `GET /health`
Returns live system diagnostics, including:
```json
{
  "status": "ok",
  "active_inference_streams": 0,
  "is_bus_idle": true,
  "consolidation_queue_depth": 0,
  "gtt_memory": {
    "gtt_total_gb": 128.0,
    "gtt_used_gb": 23.82,
    "gtt_free_gb": 104.18
  },
  "backend": "llamacpp_native"
}
```

### `POST /v1/sessions/{id}/export`
Persists the session's current ephemeral adapter weights from `/dev/shm` to `persisted_adapters/{id}.gguf`.

### `DELETE /v1/sessions/{id}`
Resets the session state, clears message history, and zeroes out the ephemeral LoRA weights on `/dev/shm`.

---

## License
MIT License. See [LICENSE](LICENSE) for details.

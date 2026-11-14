#!/usr/bin/env bash
# ==============================================================================
# Ephemeral LoRA Proxy & llama.cpp Stack Launcher (AMD Strix Halo gfx1151 Optimized)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "===================================================================="
echo " AMD Strix Halo (Ryzen AI Max+ 395 / Radeon 8060S, gfx1151) Launcher "
echo "===================================================================="

# 1. Strix Halo Pre-Flight Checks
echo "[1/5] Running Pre-Flight Hardware & Kernel Diagnostics..."

GTT_TOTAL_FILE=$(ls /sys/class/drm/card*/device/mem_info_gtt_total 2>/dev/null | head -n 1 || true)
GTT_USED_FILE=$(ls /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | head -n 1 || true)

if [[ -f "$GTT_TOTAL_FILE" ]]; then
  GTT_TOTAL_BYTES=$(cat "$GTT_TOTAL_FILE")
  GTT_TOTAL_GB=$(( GTT_TOTAL_BYTES / 1024 / 1024 / 1024 ))
  GTT_USED_BYTES=$(cat "$GTT_USED_FILE" 2>/dev/null || echo 0)
  GTT_USED_GB=$(( GTT_USED_BYTES / 1024 / 1024 / 1024 ))
  echo "  - GTT Shared Memory: Total ${GTT_TOTAL_GB} GB | Currently Used: ${GTT_USED_GB} GB"

  if (( GTT_TOTAL_GB < 100 )); then
    echo "  [WARNING] GTT Memory is only ${GTT_TOTAL_GB} GB (< 100 GB)!"
    echo "  Remediation: Set GRUB_CMDLINE_LINUX_DEFAULT=\"amd_iommu=off amdgpu.gttsize=131072 ttm.pages_limit=33554432\" and reboot."
  else
    echo "  - GTT Memory Verified: Unified Memory Pool >= 100 GB OK"
  fi
else
  echo "  [WARNING] GTT total file not found at /sys/class/drm/card*/device/mem_info_gtt_total"
fi

# rocminfo / device check
if command -v rocminfo >/dev/null 2>&1; then
  GFX_DEV=$(rocminfo 2>/dev/null | grep -m 1 "gfx1151" || true)
  if [[ -n "$GFX_DEV" ]]; then
    echo "  - ROCm Device Check: Found gfx1151 agent"
  else
    echo "  [WARNING] gfx1151 not found in rocminfo output"
  fi
fi

# /dev/shm check
SHM_FREE_GB=$(df -BG /dev/shm | awk 'NR==2 {print $4}' | tr -d 'G')
echo "  - /dev/shm tmpfs Free: ${SHM_FREE_GB} GB"
if (( SHM_FREE_GB < 2 )); then
  echo "  [ERROR] /dev/shm has less than 2 GB free! Aborting."
  exit 1
fi

# 2. Export Environment Variables for Strix Halo Unified Memory
echo "[2/5] Exporting Strix Halo ROCm Unified-Memory Environment Variables..."
export ROCM_HOME="${ROCM_HOME:-$HOME/rocm}"
export LD_LIBRARY_PATH="$HOME/.local/llama.cpp:$ROCM_HOME/lib:$ROCM_HOME/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export ROCBLAS_USE_HIPBLASLT=1
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
# Stability toggle if gfx1151 async batching shows artifacts:
# export HIP_LAUNCH_BLOCKING=1

# Extract configuration from config.yaml
MODEL_PATH=$(./.venv/bin/python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['model']['path'])")
UPSTREAM_PORT=$(./.venv/bin/python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['upstream']['port'])")
PROXY_PORT=$(./.venv/bin/python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['server']['port'])")
ADAPTER_PATH=$(./.venv/bin/python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['lora']['adapter_path'])")
RANK=$(./.venv/bin/python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['lora']['r'])")

# 3. Initialize Seed Rank-4 LoRA Adapter
echo "[3/5] Checking / Initializing Seed LoRA Adapter at ${ADAPTER_PATH}..."
./.venv/bin/python3 scripts/init_seed_lora.py   --model "$MODEL_PATH"   --output "$ADAPTER_PATH"   --rank "$RANK"

# 4. Start Upstream llama-server
echo "[4/5] Starting upstream llama-server on port ${UPSTREAM_PORT}..."
# Kill any lingering process on upstream port
PID_TO_KILL=$(lsof -ti tcp:"${UPSTREAM_PORT}" 2>/dev/null || true)
if [[ -n "$PID_TO_KILL" ]]; then
  kill -9 $PID_TO_KILL 2>/dev/null || true
  sleep 1
fi

LLAMA_SERVER_BIN="/home/loki/.local/bin/llama-server"
if [[ ! -x "$LLAMA_SERVER_BIN" ]]; then
  LLAMA_SERVER_BIN="/usr/local/bin/llama-server"
fi

# Determine whether the binary expects --no-mmap or --load-mode none
if "$LLAMA_SERVER_BIN" --help 2>&1 | grep -q -- '--load-mode'; then
  MMAP_FLAG="--load-mode none"
else
  MMAP_FLAG="--no-mmap"
fi

nohup "$LLAMA_SERVER_BIN"   -m "$MODEL_PATH"   --host 127.0.0.1   --port "${UPSTREAM_PORT}"   $MMAP_FLAG   -ngl 999   --parallel 2   --lora-init-without-apply   --lora "$ADAPTER_PATH"   --ctx-size 8192   --slots > "$PROJECT_DIR/upstream_llama_server.log" 2>&1 &
SERVER_PID=$!
echo "Upstream llama-server launched (PID: ${SERVER_PID})"

# 5. Start Ephemeral LoRA Proxy
echo "[5/5] Starting Ephemeral Proxy on port ${PROXY_PORT}..."
PROXY_PID_TO_KILL=$(lsof -ti tcp:"${PROXY_PORT}" 2>/dev/null || true)
if [[ -n "$PROXY_PID_TO_KILL" ]]; then
  kill -9 $PROXY_PID_TO_KILL 2>/dev/null || true
  sleep 1
fi

nohup ./.venv/bin/python3 -m uvicorn proxy:app   --host 0.0.0.0   --port "${PROXY_PORT}"   --log-level info > "$PROJECT_DIR/proxy.log" 2>&1 &
PROXY_PID=$!
echo "Proxy launched (PID: ${PROXY_PID})"

# Wait and tail health check
echo "Waiting for services to become healthy..."
for i in {1..60}; do
  if curl -s "http://127.0.0.1:${UPSTREAM_PORT}/health" >/dev/null 2>&1 &&      curl -s "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null 2>&1; then
    echo "Both upstream llama-server and Ephemeral Proxy are READY!"
    curl -s "http://127.0.0.1:${PROXY_PORT}/health" | python3 -m json.tool
    exit 0
  fi
  sleep 1
done

echo "[ERROR] Timed out waiting for services to initialize! Check upstream_llama_server.log and proxy.log."
exit 1

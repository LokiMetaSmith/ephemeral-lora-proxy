#!/usr/bin/env python3
"""
Seed LoRA Initializer for llama.cpp on AMD Strix Halo (gfx1151).
Generates an empty/seed rank-4 (or specified rank) GGUF LoRA adapter
at /dev/shm/ephemeral_session.gguf if one does not already exist.
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path
import numpy as np

# Ensure gguf-py is accessible
llama_gguf_path = Path("/home/loki/Projects/llama.cpp/gguf-py")
if llama_gguf_path.exists() and str(llama_gguf_path) not in sys.path:
    sys.path.insert(0, str(llama_gguf_path))

import gguf


def create_seed_lora(model_path: str, output_path: str, rank: int = 4, alpha: float = 16.0, overwrite: bool = False):
    out = Path(output_path)
    if out.exists() and not overwrite:
        print(f"[init_seed_lora] File already exists at {output_path}, skipping initialization.")
        return

    print(f"[init_seed_lora] Inspecting base model: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Base model not found: {model_path}")

    reader = gguf.GGUFReader(model_path)
    arch_field = reader.get_field("general.architecture")
    if not arch_field:
        arch = "llama"
    else:
        arch = arch_field.parts[-1].tobytes().decode("utf-8")
    print(f"[init_seed_lora] Model architecture: {arch}")

    # Find attention tensors to target with LoRA
    # Target attn_q and attn_v across all blocks
    target_tensors = []
    for t in reader.tensors:
        name = t.name
        if (".attn_q.weight" in name or ".attn_v.weight" in name) and len(t.shape) == 2:
            target_tensors.append(t)

    # If no attn_q/v found, fallback to any 2D weight in blk.0
    if not target_tensors:
        for t in reader.tensors:
            if t.name.startswith("blk.0.") and t.name.endswith(".weight") and len(t.shape) == 2:
                target_tensors.append(t)
                break

    if not target_tensors:
        raise ValueError(f"No suitable 2D attention projection tensors found in {model_path}")

    print(f"[init_seed_lora] Found {len(target_tensors)} target projection tensors for LoRA rank={rank}")

    tmp_out = f"{output_path}.tmp.{os.getpid()}"
    writer = gguf.GGUFWriter(tmp_out, arch=arch)
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, alpha)

    for t in target_tensors:
        # In GGUF reader, shape is [in_features, out_features] or ggml dimension ordering
        in_dim = int(t.shape[0])
        out_dim = int(t.shape[1])

        # Standard LoRA initialization (Hu et al. 2021):
        # lora_a is initialized with normal distribution N(0, 1/sqrt(in_dim))
        # lora_b is initialized to zeros so initial Delta W = 0
        rng = np.random.RandomState(42)
        lora_a = rng.normal(0.0, 1.0 / np.sqrt(in_dim), size=(rank, in_dim)).astype(np.float32)
        lora_b = np.zeros((out_dim, rank), dtype=np.float32)

        writer.add_tensor(f"{t.name}.lora_a", lora_a)
        writer.add_tensor(f"{t.name}.lora_b", lora_b)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    os.replace(tmp_out, output_path)
    print(f"[init_seed_lora] Seed LoRA successfully created at {output_path} (size: {os.path.getsize(output_path)} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Generate seed GGUF LoRA adapter for llama-server")
    parser.add_argument("--model", required=True, help="Path to base model .gguf")
    parser.add_argument("--output", default="/dev/shm/ephemeral_session.gguf", help="Output path for adapter")
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank (default: 4)")
    parser.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha (default: 16.0)")
    parser.add_argument("--force", action="store_true", help="Overwrite if exists")
    args = parser.parse_args()

    create_seed_lora(args.model, args.output, rank=args.rank, alpha=args.alpha, overwrite=args.force)


if __name__ == "__main__":
    main()

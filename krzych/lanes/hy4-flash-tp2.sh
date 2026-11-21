#!/usr/bin/env bash
# Tencent Hy4 Flash (tencent/Hy4-flash-FP8), TP2 on GPUs 4,7: 77.7 GiB of weights per card
# (Engram tables in pinned host RAM) leave 10.75 GiB of KV cache, 525,573 tokens.
# Usage: hy4-flash-tp2.sh [PORT]   (default 11112)
# Image and constraints as in hy4-flash.sh (TP4 lane on GPUs 0-3; both can run together).
: "${IMAGE:=localhost/lil-krzych:krzych-20261121.5}"
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

NAME=hy4-flash-tp2
GPUS=4,7
CACHE_VOL=karmic-hy4f-tp2-runtime
MODEL=tencent/Hy4-flash-FP8
MODEL_REVISION=bd0e136e89a9ecd5ea17668cd89cf54fa127e795
SERVE_CMD=(vllm serve "$MODEL" --revision "$MODEL_REVISION")
# Compile caches (Triton, FlashInfer, vLLM, DeepGEMM) persist in the lane volume.
ENV=(
  VLLM_CACHE_ROOT=/cache/vllm TRITON_CACHE_DIR=/cache/triton FLASHINFER_WORKSPACE_BASE=/cache
  VLLM_DETERMINISM_TOPK_SORT=0 VLLM_DETERMINISM_PADZERO=0
  VLLM_DETERMINISM_PAD_INPUT_IDS=0 B12X_DETERMINISM_ROUTE_PACK=0
  HF_HOME=/root/.cache/huggingface XDG_CACHE_HOME=/cache/xdg
  B12X_COMPILE_CACHE_DIR=/cache/b12x/compile
  B12X_CUTE_COMPILE_CACHE_DIR=/cache/b12x/cute CUTE_DSL_CACHE_DIR=/cache/cute-dsl
  # TP all-reduce of [tokens, 3072] bf16 through FlashInfer PCIe IPC (decode sizes); larger
  # (prefill) reductions fall back to the default backend.
  VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1
)
ARGS=(
  --served-model-name Hy4-flash-tp2
  --tensor-parallel-size 2
  --kv-cache-dtype fp8_ds_mla
  --max-model-len 262144
  --max-num-seqs 4
  --gpu-memory-utilization 0.97
  --reasoning-parser hy_v4
  --tool-call-parser hy_v4 --enable-auto-tool-choice
  --host 0.0.0.0
)
ARGS+=(--port "${1:-11112}")
lane_run "${1:-11112}"

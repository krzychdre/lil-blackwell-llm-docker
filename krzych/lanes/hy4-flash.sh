#!/usr/bin/env bash
# Tencent Hy4 Flash (tencent/Hy4-flash-FP8, blockwise FP8, text + image + video), TP4.
# Usage: hy4-flash.sh [PORT]   (default 11110)
#
# Needs an image with the vLLM branch feat/hy4-flash (vllm-project/vllm#68296 ported to
# base/kk-20261020 with SM12x attention), i.e. krzych-20261121.4 or later. The other lanes stay
# on lib.sh's image. The image launcher has no Hy4 profile, so the lane runs `vllm serve`
# directly (SERVE_CMD).
# TP4 only among 4/6: 32 sparse-attention heads do not divide by 6.
# Required: --kv-cache-dtype fp8_ds_mla (FlashInfer SM120 sparse MLA). Engram tables (~4 x 15 GB
# FP8) live in pinned host RAM. No speculative decoding: the DFly drafter is not ported.
: "${IMAGE:=localhost/lil-krzych:krzych-20261121.4}"
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

NAME=hy4-flash
GPUS=0,1,2,3
CACHE_VOL=karmic-hy4f-runtime
MODEL=tencent/Hy4-flash-FP8
MODEL_REVISION=bd0e136e89a9ecd5ea17668cd89cf54fa127e795
SERVE_CMD=(vllm serve "$MODEL" --revision "$MODEL_REVISION")
# Compile caches (Triton, FlashInfer, vLLM, DeepGEMM) persist in the lane volume.
ENV=(
  VLLM_CACHE_ROOT=/cache/vllm TRITON_CACHE_DIR=/cache/triton FLASHINFER_WORKSPACE_BASE=/cache
  VLLM_DETERMINISM_TOPK_SORT=0 VLLM_DETERMINISM_PADZERO=0
  VLLM_DETERMINISM_PAD_INPUT_IDS=0 B12X_DETERMINISM_ROUTE_PACK=0
)
ARGS=(
  --served-model-name Hy4-flash
  --tensor-parallel-size 4
  --kv-cache-dtype fp8_ds_mla
  --max-model-len 262144
  --max-num-seqs 8
  --gpu-memory-utilization 0.92
  --reasoning-parser hy_v4
  --tool-call-parser hy_v4 --enable-auto-tool-choice
  --host 0.0.0.0
)
ARGS+=(--port "${1:-11110}")
lane_run "${1:-11110}"

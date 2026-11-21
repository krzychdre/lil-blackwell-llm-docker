#!/usr/bin/env bash
# GLM-5.3 Flash (CSF-QAD qad2) on two GPUs, 4 and 7: preset glm53-tp2, TP2/DCP2, MTP3.
# Can run alongside a lane on GPUs 0-3.
# Usage: glm53-flash-tp2.sh [PORT]   (default 11112)
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

NAME=glm53-flash-tp2
GPUS=4,7
CACHE_VOL=karmic-53-flash-tp2-runtime
ENV=(
  PRESET=glm53-tp2
  HARDWARE_PROFILE=rtx-pro-6000-pcie
  MODEL=local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD
  MODEL_REVISION=96f27f8c4e443be5bc486f8a978e3b9e0d3f7125
  SERVED_MODEL_NAME=GLM-5.3-Flash-NVFP4-tp2
  TP=2 DCP=2
  MAX_MODEL_LEN=1048576
  CACHE_MODE=vram
  VLLM_DETERMINISM_TOPK_SORT=0 VLLM_DETERMINISM_PADZERO=0
  VLLM_DETERMINISM_PAD_INPUT_IDS=0 B12X_DETERMINISM_ROUTE_PACK=0
)
ARGS=(
  --mode mtp --draft-tokens 3
  --default-chat-template-kwargs '{"reasoning_effort":"max","clear_thinking":false}'
)
lane_run "${1:-11112}"

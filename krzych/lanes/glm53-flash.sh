#!/usr/bin/env bash
# GLM-5.3 Flash (CSF-QAD qad2 checkpoint), TP4/DCP1, MTP3, context 1,048,576.
# Usage: glm53-flash.sh [PORT]   (default 11110)
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

NAME=glm53-flash
GPUS=0,1,2,3
CACHE_VOL=karmic-53-flash-runtime
ENV=(
  PROFILE=glm53-flash
  HARDWARE_PROFILE=rtx-pro-6000-pcie
  MODEL=local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD
  MODEL_REVISION=96f27f8c4e443be5bc486f8a978e3b9e0d3f7125
  SERVED_MODEL_NAME=GLM-5.3-Flash-NVFP4
  TP=4 DCP=1
  MAX_MODEL_LEN=1048576
  CACHE_MODE=vram
  # qad2 keeps lm_head in MXFP8: the draft shares that head instead of building its own NVFP4 one
  VLLM_GLM53_MTP_DRAFT_HEAD=bf16
  VLLM_GLM53_KDA_MXFP8_ACTIVATION_MODE=prefill
  # determinism patches (from the full lane) off: Flash has always run without them
  VLLM_DETERMINISM_TOPK_SORT=0 VLLM_DETERMINISM_PADZERO=0
  VLLM_DETERMINISM_PAD_INPUT_IDS=0 B12X_DETERMINISM_ROUTE_PACK=0
)
ARGS=(
  --mode mtp --draft-tokens 3
  --default-chat-template-kwargs '{"reasoning_effort":"max","clear_thinking":false}'
)
lane_run "${1:-11110}"

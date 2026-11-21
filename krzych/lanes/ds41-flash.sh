#!/usr/bin/env bash
# DeepSeek-V4.1-Flash-1017 (oryginalny checkpoint MXFP4), TP4, DSpark, Engram w RAM.
# Użycie: ds41-flash.sh [PORT]   (domyślnie 11110)
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

NAME=ds41-flash
GPUS=0,1,2,3
CACHE_VOL=karmic-ds41f-runtime
DOCKER_OPTS=(--security-opt seccomp=unconfined)   # io_uring (INSTANTTENSOR_BACKEND)
ENV=(
  PROFILE=ds41-flash
  HARDWARE_PROFILE=rtx-pro-6000-pcie
  MODEL=deepseek-ai/DeepSeek-V4.1-Flash-1017
  MODEL_REVISION=1bae0e1088993b9b82de08f59de5be5a34e42170
  CHECKPOINT=original
  SERVED_MODEL_NAME=DeepSeek-V4.1-Flash-1017
  ENGRAM_TABLE_MEMORY=ram
  TP=4 DCP=1
  CACHE_MODE=vram
  VLLM_ENFORCE_STRICT_TOOL_CALLING=0   # xgrammar bez budowniczego V4.1: strict -> 400
  VLLM_ENABLE_STARTUP_PLAN=1
  INSTANTTENSOR_BACKEND=URING,AIO
  VLLM_DETERMINISM_TOPK_SORT=0 VLLM_DETERMINISM_PADZERO=0
  VLLM_DETERMINISM_PAD_INPUT_IDS=0 B12X_DETERMINISM_ROUTE_PACK=0
)
ARGS=(
  --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"max"}'
  --override-generation-config '{"temperature":1.0,"top_p":1.0}'
)
lane_run "${1:-11110}"

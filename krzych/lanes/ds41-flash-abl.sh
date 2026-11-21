#!/usr/bin/env bash
# DeepSeek-V4.1-Flash-1017 abliterated (jcartu, lossless CSF), TP4, DSpark, Engram tables in host RAM.
# Prompt encoder and tokenizer come from the 1017 checkpoint (ds41-abl/): the image's version strips
# reasoning from tool-call turns. Both files are identical in the 20261011 and 20261020 bases.
# Usage: ds41-flash-abl.sh [PORT]   (default 11110)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"

NAME=ds41-flash-abl
GPUS=0,1,2,3
CACHE_VOL=karmic-ds41f-runtime
TOK=/opt/venv/lib/python3.12/site-packages/vllm/tokenizers
DOCKER_OPTS=(
  --security-opt seccomp=unconfined
  -v "$HERE/ds41-abl/deepseek_v41_encoding_1017.py:$TOK/deepseek_v41_encoding.py:ro"
  -v "$HERE/ds41-abl/deepseek_v41_tokenizer_1017.py:$TOK/deepseek_v41.py:ro"
)
ENV=(
  PROFILE=ds41-flash
  HARDWARE_PROFILE=rtx-pro-6000-pcie
  MODEL=jcartu/DeepSeek-V4.1-Flash-1017-abliterated-lossless-CSF
  MODEL_REVISION=0010a7a9d0f2257b6bcc4463e7d724361ee90c9f
  CHECKPOINT=csf
  SERVED_MODEL_NAME=DeepSeek-V4.1-Flash-1017-abliterated
  ENGRAM_TABLE_MEMORY=ram
  TP=4 DCP=1
  CACHE_MODE=vram
  VLLM_ENFORCE_STRICT_TOOL_CALLING=0
  VLLM_ENABLE_STARTUP_PLAN=1
  VLLM_DETERMINISM_TOPK_SORT=0 VLLM_DETERMINISM_PADZERO=0
  VLLM_DETERMINISM_PAD_INPUT_IDS=0 B12X_DETERMINISM_ROUTE_PACK=0
)
ARGS=(
  --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"max"}'
  --override-generation-config '{"temperature":1.0,"top_p":1.0}'
)
lane_run "${1:-11110}"

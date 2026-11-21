#!/usr/bin/env bash
# GLM-5.3 full (744B NVFP4-CSF), TP6/DCP3, MTP3, context 262,144.
# BF16 attention (empty quantization-config), MTP draft with a quantised layer 78, 15 GiB KV
# (pool 882,385 tokens). Determinism built into the image (on), default scheduler.
# Usage: glm53-full.sh [PORT]   (default 10004)
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

NAME=glm53-full
GPUS=0,1,2,3,4,7
CACHE_VOL=karmic-53-full-runtime
ENV=(
  PRESET=glm53-csf-tp6
  MODEL=local-inference-lab/GLM-5.3-NVFP4-CSF
  MODEL_REVISION=a4d406932f2db96db8e852867c7939f448e0cd6f
  SERVED_MODEL_NAME=GLM-5.3-NVFP4
  TP=6 DCP=3
  MAX_MODEL_LEN=262144
  B12X_PCIE_DMA_FP8=0 VLLM_PCIE_DMA_FP8=0
)
ARGS=(
  --mode mtp --draft-tokens 3
  --dcp-comm-backend ag_rs
  --default-chat-template-kwargs '{"reasoning_effort":"max","clear_thinking":false}'
  --override-generation-config '{"temperature":1.0,"top_p":0.95}'
  --quantization-config '{"targets":{}}'
  --speculative-config '{"method":"mtp","draft_sample_method":"probabilistic","moe_backend":"b12x","attention_backend":"B12X","num_speculative_tokens":3,"quantization_config":{"targets":{"model.layers.78.mlp.experts":"nvfp4_a16","model.layers.78.self_attn.*":"mxfp8","model.layers.78.mlp.shared_experts.*":"mxfp8"}}}'
  --kv-cache-memory-bytes 16106162176
)
lane_run "${1:-10004}"

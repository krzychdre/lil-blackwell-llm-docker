# Changelog: localhost/lil-krzych (KK-derived images from krzychdre/lil-* forks)

Every release is a tag of the same name in all three forks:
`krzychdre/lil-vllm`, `krzychdre/lil-flashinfer` and `krzychdre/lil-blackwell-llm-docker`.
The image is built by `krzych/build-layer-image.sh`: the Python files that differ between
`base/kk-<REL>` and `integration/krzych-kk-<REL>` are laid over
`ghcr.io/local-inference-lab/vllm:karmic-kraken-<REL>`. Each change below lives on its own
feature branch, merged `--no-ff` into `integration/krzych-kk-20261020`.

Measurements are on 6x RTX PRO 6000 Blackwell Workstation (SM120, PCIe), driver 610.57.04.
"Coherence" is `check-concurrency-coherence-53.py` (C1-C8, 30 replies); "prose bench" is
`bench-prose-53.py` (29k-token prompt, 512 tokens, median of 3 runs).

## krzych-20261122

Image `localhost/lil-krzych:krzych-20261122`; vLLM `integration/krzych-kk-20261020` @ 5a02be3ad8,
FlashInfer `integration/krzych-kk-20261020` @ e1b86b7ba (unchanged). Only the Hy4 lanes use this image;
the other lanes keep `krzych-20261121.2` (`lib.sh`).

### Added

- **DFly speculative decoding for Hy4 Flash** (vLLM, branch `feat/hy4-flash-dfly`), ported from
  vllm-project/vllm#68296 onto this tree's speculator framework (upstream's speculator-config
  framework is not imported). The checkpoint's 5 draft layers have no KV cache or indexer of their
  own: each aliases the KV of target layer 25/29/33/37/41 and attends through that layer's top-k
  plus the causal query block; one FULL CUDA graph runs the draft step. N is given explicitly (5),
  since the checkpoint declares no `dfly_block_size`. Shared fixes: KV-sharing aliases now cover
  MLA layers and are not duplicated in KV groups; HYV4 MLA sinks use FlashInfer SM120 sparse MLA on
  SM12x instead of falling back to FlashMLA (which disabled the sink).
- **Hy4 lanes run DFly** (blackwell-llm-docker, branch `feat/krzych-hy4-flash-dfly`):
  `--speculative-config '{"method":"dfly","num_speculative_tokens":5}'` in both lanes.

### Verified

Context 262,144, N=5, FlashInfer PCIe IPC all-reduce; "mixed" = 16 prompts (code, JSON, SQL,
translation, explanations) at temperature 0, single stream including prefill; "prose" = 600-token
essays at temperature 0.7.

| Lane | KV pool | Mixed | Prose, 1 stream | Prose, batch |
|---|---|---|---|---|
| `hy4-flash` (TP4), `krzych-20261121.5` | 2,688,315 | 63.7 | 64.2 | 338 (8) |
| `hy4-flash` (TP4), DFly | 2,636,935 | 114.0 | 82.4 | 345 (8) |
| `hy4-flash-tp2` (TP2), `krzych-20261121.5` | 657,340 | 55.1 | 56.1 | 170 (4) |
| `hy4-flash-tp2` (TP2), DFly | 594,309 | 99.8 | 71.7 | 180 (4) |

A 100-problem multi-step arithmetic set at temperature 0 is 100/100 on both lanes with and without
DFly, at 3.2 tokens per target step. Greedy outputs are not token-identical with and without DFly
(15 of 16 prompts diverge in the reasoning text); two boots of the same image without speculation
also diverge on 8 of 16 prompts, at top-2 logprob gaps up to 0.64.

## krzych-20261121.5

Image `localhost/lil-krzych:krzych-20261121.5`; vLLM `integration/krzych-kk-20261020` @ 97bca9f4d6,
FlashInfer `integration/krzych-kk-20261020` @ e1b86b7ba (unchanged). Only the Hy4 lanes use this image;
the other lanes keep `krzych-20261121.2` (`lib.sh`).

### Changed

- **Hy4 Flash: one TP all-reduce per MoE layer** (vLLM, branch `feat/hy4-flash-speed`). The latent
  experts' output projection is a bias-free BF16 linear, so it commutes with the tensor-parallel
  sum; the MoE runner now applies it to the rank-local latent output and reduces routed and shared
  outputs together once. TP all-reduces per decode step drop from 146 to 105, all
  `[tokens, 3072]`. The embedding module gets its prefix, which B12X PCIe preparation needs.
- **Hy4 lanes on FlashInfer PCIe IPC all-reduce** (blackwell-llm-docker, branch
  `feat/krzych-hy4-flash-speed`): `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` in both lanes, which
  replaces PyNCCL (TP4) and vLLM CUSTOM (TP2) for decode-size reductions. `HF_HOME` is pinned and
  the b12x / CuTe compile caches sit in the lane volume.

### Verified

TP4 on GPUs 0-3, 600-token decode, two runs per arm (tok/s, single stream / 8 streams):

| Arm | 1 stream | 8 streams |
|---|---|---|
| `krzych-20261121.4` (PyNCCL, two MoE reduces) | 59.2 | 305.7 |
| one MoE reduce, PyNCCL | 61.3-61.5 | 298-302 |
| one MoE reduce, B12X PCIe (`comm.pcie` preparation 6 s) | 63.8-64.0 | 317-320 |
| one MoE reduce, FlashInfer PCIe IPC | 64.3-64.5 | 322-326 |
| this release, lane as shipped | 64.2 | 338.3 |

TP2 on GPUs 4,7 (single stream / 4 streams): `.4` 54.9 / 164, CUSTOM with one MoE reduce
55.5 / 154-168, this release 56.1 / 169.9.

A 20-problem arithmetic set at temperature 0 (final answer checked, max 3000 tokens) is 20/20 with
every reply finished, on both lanes of this release and on `.4`; the 23k-token needle is retrieved.
One reply in an intermediate TP2 CUSTOM arm came back with empty content; five later tries in that
configuration, including as the first request after a fresh boot, answered correctly.

## krzych-20261121.4

Image `localhost/lil-krzych:krzych-20261121.4`; vLLM `integration/krzych-kk-20261020` @ ce08fdbd56,
FlashInfer `integration/krzych-kk-20261020` @ e1b86b7ba (unchanged). Only the Hy4 lanes use this image;
the other lanes keep `krzych-20261121.2` (`lib.sh`) until they are verified on it.

### Added

- **Tencent Hy4 Flash** (vLLM, branch `feat/hy4-flash`): vllm-project/vllm#68296 (upstream
  b067b6f6fa, "Add HY4 Flash and multimodal support") ported to `base/kk-20261020`, so
  `tencent/Hy4-flash-FP8` (`HYV4FlashVLForConditionalGeneration`: text, image, video) serves on
  SM120. Imported: the `hy_v4_flash` package (latent-expert MoE, sliding-window and IndexPool
  sparse attention, Flash Engram), the HYV4 vision tower, processor, renderer, M-RoPE and the
  `hy_v4` tool-parser update. Adapted to this tree:
  - sparse and sliding-window MLA run on FlashInfer's SM120 sparse MLA kernel, with the per-head
    learnable sink forwarded (`hy_v4_flash/nvidia/sm120.py`); requires `--kv-cache-dtype fp8_ds_mla`;
  - the IndexPool pool and tail caches are expressed as KV specs this allocator accepts (pool
    padded to the MLA page, tail as a circular buffer);
  - IndexPool scoring uses the bundled DeepGEMM paged MQA logits, with a Triton fallback;
  - Flash Engram uses this tree's DeepSeek V4.1 n-gram hashing; tables sit in pinned host RAM;
  - Transformers 5.15 adds flat RoPE defaults next to Hy4's per-layer RoPE; they are dropped.

  Not ported: DFly speculative decoding (Hy4 Flash rejects any speculative config), P/D M-RoPE
  delta transfer, iHC sequence parallelism. TP6 is impossible (32 sparse-attention heads).
- **Hy4 lanes** (blackwell-llm-docker, branch `feat/krzych-hy4-flash-lane`): `hy4-flash.sh`
  (TP4, GPUs 0-3, context 262,144) and `hy4-flash-tp2.sh` (TP2, GPUs 4,7, context 262,144), both
  with the hy_v4 reasoning and tool-call parsers and the determinism switches off. `lib.sh` gains
  an optional `SERVE_CMD` (a command after the image, here `vllm serve MODEL`) because the image
  launcher has no Hy4 profile; lanes without it produce the same `docker run` as before.

### Changed

- **Lanes in English** (blackwell-llm-docker, branch `docs/krzych-lanes-english`): comments,
  messages and README of `krzych/lanes/`. Verified: `DRY_RUN=1` output of all five existing
  lanes is byte-identical before and after.

### Verified

Both Hy4 lanes running together on this image (GPUs 0-3 and 4,7):

| Lane | Weights/GPU | KV pool | 1 stream | Batch | Needle 23k | Tool call | Image |
|---|---|---|---|---|---|---|---|
| `hy4-flash` (TP4) | 40.5 GiB | 2,688,315 tok | 59.2 tok/s | 305.7 tok/s (8) | pass | pass | pass |
| `hy4-flash-tp2` (TP2) | 77.81 GiB | 657,340 tok | 54.9 tok/s | 164.4 tok/s (4) | pass | pass | pass |

Decode 600 tokens per request (`ignore_eos`); needle = a vault code at record 700 of 1,500
(23,331 prompt tokens). The TP2 row's weights, KV pool and 1-stream figures come from the lane at
context 262,144 (run alone); its batch figure and checks from the same lane at context 65,536.
At 262,144 the TP2 lane also retrieved a needle from a 153,015-token prompt (prefill 36 s).
First boot with both lanes reading the checkpoint at once took ~8.5 min (weights 5.3 min); with
warm compile caches the engine starts in ~30 s after loading.
Host shared memory (`free`) grows by ~128 GB per running Hy4 lane (pinned Engram tables).

## krzych-20261121.3

Image unchanged (`localhost/lil-krzych:krzych-20261121.2`); vLLM and FlashInfer unchanged.

### Added

- **Serving lanes** (blackwell-llm-docker, branch `feat/krzych-lanes`): `krzych/lanes/` with one
  short script per model (`glm53-full.sh`, `glm53-flash.sh`, `glm53-flash-tp2.sh`,
  `ds41-flash.sh`, `ds41-flash-abl.sh`), shared `lib.sh` and `stop.sh <name>`. Options are fixed
  in each script; the only argument is the port (llama-swap `${PORT}`), the only overrides
  `IMAGE` and `DRY_RUN=1`. `lib.sh` pins the release image, waits up to 90 s for the lane's GPUs,
  runs in the foreground (the container stops with the process on SIGTERM) and logs to
  `/home/krzych/glm-5.2/lane-logs/`. They replace the karmic_53_lil / karmic_ds41f /
  karmic_ds41f_abliterated launchers (`run-*-kk.sh`, `serve-*.sh`), whose switches were not used.
  The abliterated lane carries its 1017 prompt encoder and tokenizer in `krzych/lanes/ds41-abl/`
  (sha256 13231c59…, ce3a05ce…, identical to the old lane's files).
  Verified: with a stub `docker` recording the real `docker run` arguments, every new script
  produces exactly the old launcher's options, environment and post-image arguments, except
  `--rm` (the old scripts kept a stopped container for their stop scripts to remove) and, for the
  abliterated lane, the four determinism switches set to 0 (the old lane ran on KK 20261011
  without that code).

### Verified

- DeepSeek-V4.1-Flash-1017 abliterated (CSF) now runs on this image (it ran on LIL KK 20261011
  before): KV pool 12,818,310 tokens, gate PASS, Polish arithmetic answer correct, tool call
  parsed. First boot on the image took ~19 min (B12X tuning from scratch, once per image/model);
  the 20261011-only csf-prefetch overlay is not applied.
- `glm53-flash-tp2.sh`: coherence 30/30; `stop.sh` stops the container and frees the GPUs.

## krzych-20261121.2

Image `localhost/lil-krzych:krzych-20261121.2`; vLLM `integration/krzych-kk-20261020` @ b48b578219
(unchanged), FlashInfer `integration/krzych-kk-20261020` @ e1b86b7ba.

### Added

- **Deterministic DSA top-k set under score ties** (flashinfer, branch `det/dsa-topk-tie-set-pr6736`;
  commits 0465ddf1b, 10aea2de9, e1db27618, d6764164a by João Sequeira, cherry-picked from the open
  upstream PR flashinfer-ai/flashinfer#6736).
  The indexer's radix-select gave pivot-equal candidates their output slots through shared-memory
  atomics, so on exactly tied scores (e.g. verbatim repeated text) the selected index *set* changed
  from run to run. Ties are now filled in logical-column order. Complements our
  `det/padding-topk` (vLLM), which canonicalises the *order* of the returned set; the PR states that
  order is not part of its contract.
  Verified: the PR's 6 GPU determinism tests fail on base/kk-20261020 (20 distinct sets in 20 runs)
  and pass with the change.

### Fixed

- **Process-independent fused-indexer variant names** (flashinfer, branch
  `det/dsa-topk-tie-set-pr6736`, commit 4301acc56).
  PR #6736 named each fused-indexer variant `DSAFusedIndexerKernel_variant<N>` from a per-process
  counter. The name is part of the CuTe compile key, and b12x preparation plans programs in the
  serving process but compiles them in fresh workers, so GLM-5.3 full failed at boot with
  "factory program keys changed between planning and compilation". The tag is now a hash of the
  variant's parameters. Verified: names no longer depend on build order across processes; the
  PR's determinism tests still pass.

### Verified (GLM-5.3 full TP6/DCP3, bf16-attention + quantised-draft variant)

- Boot PASS, KV pool 882,385 tokens (unchanged), coherence 30/30.
- Prose bench A/B against krzych-20261121 on the same box: prefill 2986 vs 2993 tok/s, decode
  153.1 vs 154.6 tok/s (within run-to-run spread, decode runs ranged 143-158).
- First boot after the change recompiles the b12x CuTe kernels once (package fingerprint changed).

## krzych-20261121

Image `localhost/lil-krzych:krzych-20261121` (sha256:0d1b1905…); vLLM @ b48b578219,
FlashInfer @ 525ca47d1, base KK 20261020 (vLLM 8244d1c, FlashInfer 6023d03).

### Fixed

- **GLM-5.3-NVFP4-CSF flat fixed planes** (vllm, `fix/glm53-csf-flat-fixed`, e1d9f61b5e).
  The full checkpoint (local-inference-lab/GLM-5.3-NVFP4-CSF a4d40693) stores each expert
  projection's `nvfp4_csf_fixed` flat; KK 20261020's CsfSource validation expects
  (segments, rows/16, 16·(1+columns/2)) and every rank stopped at load. The plane is viewed as
  16-row groups. Verified on CPU against the real checkpoint: 108 projections (TP6 ranks 0-5,
  layers 3/40/77, experts 0/255) stage and validate; the unpatched loader fails on the first.
- **Slab CSF planes staged in host memory** (vllm, `fix/glm53-csf-host-staging`, 7b8c51c00e).
  Each rank staged every projection's whole compressed plane on the GPU until
  process_weights_after_loading (after the last layer): 21.7 GiB per GPU at TP6 on top of 59 GiB of
  expert weights. Both the LIL preset and our bf16-attention variant ran out of memory at 95% of
  the weights. Planes now wait in host RAM and are copied per layer during preparation.
  `VLLM_NVFP4_CSF_HOST_STAGING=0` restores GPU staging. Result: 71.98 GiB per GPU after load
  (KK 20261011: 72.74), KV pool 882,385 (same as 20261011).
- **DeepSeek MXFP4-CSF contract reader** (vllm, `fix/csf-contract-family-kk20261020`;
  353ac5eef2, f57f90293f by festr, cherry-picked from LIL `fix/csf-contract-family`).
  Model family from the top-level model_type; slab scales handed over as one segment of 16-row
  groups.

### Added

- **Run-to-run determinism** (vllm `det/padding-topk`: 153b920881, b14c7e96e2; flashinfer
  `det/route-pack-stable`: 3c887f444, 54134e1b4). Port of the karmic_53_lil overlay (patches
  01-04, identical code): canonical DSA top-k order, zeroed attention-output and input_ids padding
  rows (target and MTP drafter), stable W4A16 route ranks without atomics. On by default in code;
  switches `VLLM_DETERMINISM_TOPK_SORT`, `VLLM_DETERMINISM_PADZERO`,
  `VLLM_DETERMINISM_PAD_INPUT_IDS`, `B12X_DETERMINISM_ROUTE_PACK` (0 = off). The Flash and DS4.1
  launchers set them to 0 unless `DETERMINISM=1`. Verified: overlay CPU tests 9/9, route-pack
  test under the Triton interpreter.
- **Isolated scheduler** (vllm `feat/isolated-scheduler`, bf78dbe804): opt-in
  `--scheduler-cls=vllm.v1.core.sched.lil_isolated_scheduler.IsolatedAsyncScheduler`, one request
  per model step; `VLLM_ISOLATED_SCHED_DECODE_QUANTUM` (default 16). Verified: 84/84 CPU checks;
  on GPU max 1 request per step, 0 violations. Off by default in the GLM-5.3 launcher since
  2026-11-21 (`ISOLATION=1` enables).
- **DeepSeek-V4.1 fast boot** (vllm `perf/ds41-fast-boot`: 49087c15f2; flashinfer
  `perf/ds41-mapped-host-tables`: bed0eeb2d). Port of the karmic_ds41f overlay: Engram tables
  read concurrently, mapped with cudaHostRegister instead of cudaHostAlloc, DSpark cost curves
  cached on /cache.
- **Layer image build** (blackwell-llm-docker `feat/krzych-layer-image`, fe78b27):
  `krzych/build-layer-image.sh`, guards that the base image's originals equal the base blobs and
  refuses compiled sources.

### Not ported

- `csf-prep-threads` overlay: from KK 20261020 the CSF scales are decoded by a GPU kernel, the host
  numpy loop it parallelised no longer exists.

### Verified on GPUs

- GLM-5.3 full TP6/DCP3 bf16-attention + quantised draft: gate PASS, coherence 30/30.
- GLM-5.3 Flash TP2 (glm53-tp2 preset): pool 1,844,162 (unchanged), coherence 30/30.
- DeepSeek-V4.1-Flash-1017 TP4: model load 57 s, pool 9,462,537, gate PASS.

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

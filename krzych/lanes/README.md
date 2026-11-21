# Serving lanes on localhost/lil-krzych

One bash script per model with fixed options; the only argument is the port (that is how
llama-swap calls it: `<lane>.sh ${PORT}`). Each script starts one Docker container with the image
built by `krzych/build-layer-image.sh` and serves an OpenAI-compatible vLLM API on that port.

The shared part is `lib.sh`: the release image, waiting up to 90 s for the lane's GPUs to be free,
a foreground run (the container stops together with the script on SIGTERM/SIGINT), and a log file
per start.

| Script | API model name | GPUs | Default port | What runs |
|---|---|---|---|---|
| `glm53-full.sh` | `GLM-5.3-NVFP4` | 0,1,2,3,4,7 | 10004 | GLM-5.3 full NVFP4-CSF, TP6/DCP3, MTP3, context 262,144, KV 15 GiB |
| `glm53-flash.sh` | `GLM-5.3-Flash-NVFP4` | 0-3 | 11110 | GLM-5.3 Flash CSF-QAD qad2, TP4, MTP3, context 1M |
| `glm53-flash-tp2.sh` | `GLM-5.3-Flash-NVFP4-tp2` | 4,7 | 11112 | GLM-5.3 Flash qad2, preset `glm53-tp2` (TP2/DCP2) |
| `ds41-flash.sh` | `DeepSeek-V4.1-Flash-1017` | 0-3 | 11110 | DeepSeek-V4.1-Flash-1017, original checkpoint, TP4 |
| `ds41-flash-abl.sh` | `DeepSeek-V4.1-Flash-1017-abliterated` | 0-3 | 11110 | DeepSeek-V4.1-Flash-1017 abliterated (CSF), TP4, 1017 prompt encoder |

Lanes that share GPUs cannot run together; `glm53-flash-tp2` (GPUs 4,7) can run next to one of the
GPU 0-3 lanes. The checkpoint and its revision are `MODEL=` and `MODEL_REVISION=` in each script.

## Quick start

```bash
# 1. build the image the lanes expect (forks checked out side by side in ~/lil)
TAG=localhost/lil-krzych:krzych-20261121.2 krzych/build-layer-image.sh

# 2. download the checkpoint at the pinned revision (the containers run offline)
hf download local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD \
  --revision 96f27f8c4e443be5bc486f8a978e3b9e0d3f7125

# 3. edit HF_CACHE and LOG_DIR in lib.sh (and GPUS= in the lane if your numbering differs)

# 4. run; the first boot on a new image compiles kernels and tunes B12X (up to ~20 min)
krzych/lanes/glm53-flash.sh 11110
curl -s http://localhost:11110/v1/models

# 5. stop (name = script name without .sh); waits until the lane's GPUs are free
krzych/lanes/stop.sh glm53-flash
```

llama-swap entry:

```yaml
models:
  "GLM-5.3-Flash-NVFP4":
    cmd: /path/to/blackwell-llm-docker/krzych/lanes/glm53-flash.sh ${PORT}
    cmdStop: /path/to/blackwell-llm-docker/krzych/lanes/stop.sh glm53-flash
    unloadTimeout: 120
```

Use a long `healthCheckTimeout` (e.g. 1500 s) for first boots.

## Variables read from outside (only these two)

- `IMAGE=...` a different image, e.g. the previous release for comparison;
- `DRY_RUN=1` prints the full `docker run` and starts nothing.

## Notes

- Determinism (the patches from the full lane) is on by default in the image; the Flash and DS4.1
  lanes turn it off with the four switches in `ENV`. The isolated scheduler is not used (it is
  `--scheduler-cls=vllm.v1.core.sched.lil_isolated_scheduler.IsolatedAsyncScheduler` in `ARGS`).
- Each lane mounts a Docker volume (`CACHE_VOL`) on `/cache`; it keeps compiled kernels and B12X
  tuning between runs. Docker creates it on first use.
- `ds41-abl/` holds the 1017 prompt encoder and tokenizer that the abliterated lane mounts over the
  image's copies (the image's encoder strips reasoning from tool-call turns).

# Shared part of the lanes: image, waiting for GPUs, foreground run for llama-swap, log.
# A lane script sets NAME, GPUS, CACHE_VOL, ENV=(), ARGS=() (optionally DOCKER_OPTS=())
# and calls lane_run PORT. The only variables taken from outside:
#   IMAGE=...   a different image (e.g. for an A/B comparison)
#   DRY_RUN=1   print the docker run command and start nothing
set -euo pipefail

IMAGE="${IMAGE:-localhost/lil-krzych:krzych-20261121.2}"
HF_CACHE=/home/krzych/.cache/huggingface
LOG_DIR=/home/krzych/glm-5.2/lane-logs
GPU_WAIT=90        # s: how long to wait for the previous model to free the GPUs (llama-swap)
STOP_TIMEOUT=60    # s: graceful container stop
DOCKER_OPTS=()

_gpus_busy() {
  local i
  for i in ${GPUS//,/ }; do
    [[ -n "$(nvidia-smi -i "$i" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] && return 0
  done
  return 1
}

lane_run() {
  local port="$1"
  local cmd=(docker run --rm --name "$NAME" --init --gpus "\"device=$GPUS\""
    --network host --ipc host --shm-size 32g
    --ulimit memlock=-1 --ulimit stack=67108864 "${DOCKER_OPTS[@]}"
    -v "$CACHE_VOL":/cache -v "$HF_CACHE":/root/.cache/huggingface
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NVIDIA_VISIBLE_DEVICES="$GPUS"
    -e HF_HUB_OFFLINE=1 -e PORT="$port")
  local kv
  for kv in "${ENV[@]}"; do cmd+=(-e "$kv"); done
  cmd+=("$IMAGE" --enable-prompt-tokens-details "${ARGS[@]}")

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"; echo; return 0
  fi

  local waited=0
  while _gpus_busy; do
    (( waited >= GPU_WAIT )) && { echo "ERROR: GPUs $GPUS still busy after ${GPU_WAIT}s" >&2; exit 2; }
    (( waited == 0 )) && echo "GPUs $GPUS busy, waiting up to ${GPU_WAIT}s..."
    sleep 1; waited=$((waited + 1))
  done
  if ss -tln "( sport = :$port )" 2>/dev/null | grep -q LISTEN; then
    echo "ERROR: port $port already in use" >&2; exit 2
  fi
  docker image inspect "$IMAGE" >/dev/null
  docker rm -f "$NAME" >/dev/null 2>&1 || true

  mkdir -p "$LOG_DIR"
  local log="$LOG_DIR/$NAME-$(date +%Y%m%d-%H%M%S).log"
  echo "start $NAME: image $IMAGE, GPUs $GPUS, port $port, log $log"

  # Foreground: the container must die together with this process (llama-swap sends SIGTERM).
  # Bash does not run traps while waiting on a foreground pipeline, so the pipeline goes to
  # the background and we wait for it.
  trap 'docker stop -t "$STOP_TIMEOUT" "$NAME" >/dev/null 2>&1 || true' TERM INT HUP EXIT
  "${cmd[@]}" 2>&1 | tee "$log" &
  local pipeline=$! rc=0
  while kill -0 "$pipeline" 2>/dev/null; do
    wait "$pipeline" && rc=0 || rc=$?
  done
  exit "$rc"
}

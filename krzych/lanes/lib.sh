# Wspólna część torów: obraz, czekanie na karty, pierwszy plan pod llama-swap, log.
# Skrypt toru ustawia NAME, GPUS, CACHE_VOL, ENV=(), ARGS=() (opcjonalnie DOCKER_OPTS=())
# i woła lane_run PORT. Jedyne zmienne z zewnątrz:
#   IMAGE=...   inny obraz (np. do porównania A/B)
#   DRY_RUN=1   wypisz docker run i nic nie uruchamiaj
set -euo pipefail

IMAGE="${IMAGE:-localhost/lil-krzych:krzych-20261121.2}"
HF_CACHE=/home/krzych/.cache/huggingface
LOG_DIR=/home/krzych/glm-5.2/lane-logs
GPU_WAIT=90        # s: tyle czekamy, aż poprzedni model zwolni karty (llama-swap)
STOP_TIMEOUT=60    # s: łagodne zatrzymanie kontenera
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
    (( waited >= GPU_WAIT )) && { echo "ERROR: karty $GPUS zajęte po ${GPU_WAIT}s" >&2; exit 2; }
    (( waited == 0 )) && echo "karty $GPUS zajęte, czekam do ${GPU_WAIT}s..."
    sleep 1; waited=$((waited + 1))
  done
  if ss -tln "( sport = :$port )" 2>/dev/null | grep -q LISTEN; then
    echo "ERROR: port $port zajęty" >&2; exit 2
  fi
  docker image inspect "$IMAGE" >/dev/null
  docker rm -f "$NAME" >/dev/null 2>&1 || true

  mkdir -p "$LOG_DIR"
  local log="$LOG_DIR/$NAME-$(date +%Y%m%d-%H%M%S).log"
  echo "start $NAME: obraz $IMAGE, karty $GPUS, port $port, log $log"

  # Pierwszy plan: kontener musi umrzeć razem z procesem (llama-swap wysyła SIGTERM).
  # Bash nie wykonuje pułapek, czekając na potok, więc potok idzie w tło + wait.
  trap 'docker stop -t "$STOP_TIMEOUT" "$NAME" >/dev/null 2>&1 || true' TERM INT HUP EXIT
  "${cmd[@]}" 2>&1 | tee "$log" &
  local pipeline=$! rc=0
  while kill -0 "$pipeline" 2>/dev/null; do
    wait "$pipeline" && rc=0 || rc=$?
  done
  exit "$rc"
}

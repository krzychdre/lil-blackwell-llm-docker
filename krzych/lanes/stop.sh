#!/usr/bin/env bash
# Stops a lane by name and waits until its GPUs free their memory (at most 5 min).
# Usage: stop.sh glm53-full | glm53-flash | glm53-flash-tp2 | ds41-flash | ds41-flash-abl
set -euo pipefail
NAME="${1:?give the lane name}"
docker inspect "$NAME" >/dev/null 2>&1 || { echo "$NAME is not running"; exit 0; }
GPUS="$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$NAME" \
  | sed -n 's/^NVIDIA_VISIBLE_DEVICES=//p')"
docker stop -t 60 "$NAME" >/dev/null 2>&1 || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "stopped $NAME, waiting for GPUs $GPUS to free"
for _ in $(seq 1 60); do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" | sort -rn | head -1)"
  (( used < 500 )) && { echo "GPUs free"; exit 0; }
  sleep 5
done
echo "WARNING: after 5 min GPUs $GPUS still use ${used} MiB" >&2
exit 1

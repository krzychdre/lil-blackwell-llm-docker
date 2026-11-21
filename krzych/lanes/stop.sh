#!/usr/bin/env bash
# Zatrzymuje tor po nazwie i czeka, aż jego karty zwolnią pamięć (najwyżej 5 min).
# Użycie: stop.sh glm53-full | glm53-flash | glm53-flash-tp2 | ds41-flash | ds41-flash-abl
set -euo pipefail
NAME="${1:?podaj nazwę toru}"
docker inspect "$NAME" >/dev/null 2>&1 || { echo "$NAME nie działa"; exit 0; }
GPUS="$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$NAME" \
  | sed -n 's/^NVIDIA_VISIBLE_DEVICES=//p')"
docker stop -t 60 "$NAME" >/dev/null 2>&1 || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "zatrzymany $NAME, czekam na zwolnienie kart $GPUS"
for _ in $(seq 1 60); do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" | sort -rn | head -1)"
  (( used < 500 )) && { echo "karty wolne"; exit 0; }
  sleep 5
done
echo "UWAGA: po 5 min karty $GPUS nadal zajmują ${used} MiB" >&2
exit 1

#!/usr/bin/env bash
# Build a KK-derived image with our vLLM and FlashInfer branches laid over the
# published wheels. Only Python sources change, so no wheel is rebuilt: files
# that differ between the base commit and the branch are copied into
# site-packages of the KK image the base commits came from.
#
# Usage: krzych/build-layer-image.sh [--dry-run]
#   ENV (defaults in brackets):
#     LIL_ROOT      checkouts of the forks            [~/lil]
#     KK_REL        KK release the bases belong to    [20261020]
#     BASE_REF      base branch in both forks         [base/kk-$KK_REL]
#     VLLM_REF      vLLM branch to lay over           [integration/krzych-kk-$KK_REL]
#     FI_REF        FlashInfer branch to lay over     [integration/krzych-kk-$KK_REL]
#     TAG           image name                        [localhost/lil-krzych:kk$KK_REL-<vllm>-<fi>]
#
# Guards (the build stops):
#   - the base image's copy of every replaced file must equal the base commit's
#     blob (the image really is that commit);
#   - only added/modified files under vllm/ and flashinfer/ are allowed (a
#     delete or rename would need explicit handling);
#   - files that need compilation (*.cu, *.cpp, *.h, CMake) are refused:
#     those need a real wheel build.
set -euo pipefail

DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
LIL_ROOT="${LIL_ROOT:-$HOME/lil}"
KK_REL="${KK_REL:-20261020}"
BASE_REF="${BASE_REF:-base/kk-$KK_REL}"
VLLM_REF="${VLLM_REF:-integration/krzych-kk-$KK_REL}"
FI_REF="${FI_REF:-integration/krzych-kk-$KK_REL}"
BASE_IMAGE="ghcr.io/local-inference-lab/vllm:karmic-kraken-$KK_REL"
SP=/opt/venv/lib/python3.12/site-packages
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

V="$LIL_ROOT/vllm"; F="$LIL_ROOT/flashinfer"
vrev="$(git -C "$V" rev-parse --short=10 "$VLLM_REF")"
frev="$(git -C "$F" rev-parse --short=9 "$FI_REF")"
TAG="${TAG:-localhost/lil-krzych:kk$KK_REL-v$vrev-f$frev}"

CTX="$(mktemp -d)"; trap 'rm -rf "$CTX"' EXIT
mkdir -p "$CTX/files"
manifest="$CTX/files/LIL_KRZYCH_MANIFEST"
: > "$CTX/expected.sha256"

collect() {  # repo ref prefix
  local repo=$1 ref=$2 prefix=$3 status path
  while IFS=$'\t' read -r status path; do
    case "$status" in A|M) ;; *) echo "ERROR: $repo $ref: $status $path (only A/M handled)" >&2; exit 2;; esac
    case "$path" in
      "$prefix"/*) ;;
      *) echo "skip (outside $prefix/): $path"; continue;;
    esac
    case "$path" in
      *.cu|*.cuh|*.cpp|*.cc|*.h|*.hpp|*CMakeLists.txt|*.cmake)
        echo "ERROR: $path needs a compiled wheel, not a layer" >&2; exit 2;;
    esac
    mkdir -p "$CTX/files/$(dirname "$path")"
    git -C "$repo" show "$ref:$path" > "$CTX/files/$path"
    if [[ "$status" == M ]]; then
      echo "$(git -C "$repo" show "$BASE_REF:$path" | sha256sum | cut -d' ' -f1)  $path" >> "$CTX/expected.sha256"
    fi
    echo "$status $(sha256sum < "$CTX/files/$path" | cut -c1-12) $path" >> "$manifest"
  done < <(git -C "$repo" diff --no-renames --name-status "$BASE_REF" "$ref")
}
collect "$V" "$VLLM_REF" vllm
collect "$F" "$FI_REF" flashinfer
{
  echo "base_image $BASE_IMAGE"
  echo "vllm $(git -C "$V" rev-parse "$VLLM_REF") ($VLLM_REF over $(git -C "$V" rev-parse --short=10 "$BASE_REF"))"
  echo "flashinfer $(git -C "$F" rev-parse "$FI_REF") ($FI_REF over $(git -C "$F" rev-parse --short=9 "$BASE_REF"))"
} >> "$manifest"

echo "== files laid over $BASE_IMAGE"; cat "$manifest"

docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || docker pull "$BASE_IMAGE"
echo "== base image check ($(wc -l < "$CTX/expected.sha256") modified files)"
docker run --rm --network none --entrypoint sh -v "$CTX/expected.sha256:/x.sha256:ro" \
  "$BASE_IMAGE" -c "cd $SP && sha256sum -c --quiet /x.sha256" \
  || { echo "ERROR: base image differs from $BASE_REF" >&2; exit 2; }
echo "   base image == $BASE_REF"

cp "$HERE/Dockerfile.layer" "$CTX/Dockerfile"
if (( DRY )); then echo "DRY_RUN: would build $TAG"; exit 0; fi
docker build -q -t "$TAG" -t "localhost/lil-krzych:kk$KK_REL" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --label "org.krzych.lil.vllm=$(git -C "$V" rev-parse "$VLLM_REF")" \
  --label "org.krzych.lil.flashinfer=$(git -C "$F" rev-parse "$FI_REF")" \
  --label "org.krzych.lil.base=$BASE_IMAGE" \
  "$CTX"
echo "built $TAG (also localhost/lil-krzych:kk$KK_REL)"
docker image inspect --format '{{.Id}}' "$TAG"

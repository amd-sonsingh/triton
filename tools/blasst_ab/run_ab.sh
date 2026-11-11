#!/bin/bash
# AITER's unified_attention kernel (block-skip path) on stock upstream Triton vs
# the same Triton with the chain-dot patch. Needs a GPU and Docker; see README.md.
#
#   AITER_DIR=/path/to/aiter bash run_ab.sh                       # defaults
#   AITER_DIR=/path/to/aiter bash run_ab.sh --seqlens 8192,16384  # args pass through
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="${OUT:-$REPO/.blasst-out}"
RESULTS="${RESULTS:-$OUT/blasst_ab}"

STOCK="${STOCK:-triton-blasst-ab:stock}"
PATCHED="${PATCHED:-triton-blasst-ab:patched}"

AITER_SHA="${AITER_SHA:-3fc0ebd668030b3d12a94ce034deb87704092899}"
TARBALL="$HERE/aiter-${AITER_SHA:0:9}.tar.gz"
# aiter checked out at AITER_SHA. If unset, a tarball next to this script is used.
AITER_DIR="${AITER_DIR:-$OUT/aiter-${AITER_SHA:0:9}}"

# The shape the published numbers were measured at.
BENCH_ARGS=("$@")
[ ${#BENCH_ARGS[@]} -eq 0 ] && BENCH_ARGS=(--shapes 32x8 --seqlens 16384)

command -v docker >/dev/null || {
  echo "ERROR: no docker here. Are you on a login node? This must run on a compute node." >&2
  exit 1
}
[ -e /dev/kfd ] || {
  echo "ERROR: no /dev/kfd -- this needs a real GPU allocation, not --cpu." >&2
  exit 1
}

mkdir -p "$RESULTS"

# Kernel, wrapper and benchmark must all come from the same aiter commit.
if [ ! -d "$AITER_DIR" ]; then
  [ -f "$TARBALL" ] || {
    echo "ERROR: no aiter tree at $AITER_DIR and no tarball at $TARBALL." >&2
    echo "" >&2
    echo "Point AITER_DIR at a clone pinned to the commit above:" >&2
    echo "  git clone https://github.com/amd-sonsingh/aiter.git /tmp/aiter" >&2
    echo "  git -C /tmp/aiter checkout $AITER_SHA" >&2
    echo "  AITER_DIR=/tmp/aiter bash \$0" >&2
    exit 1
  }
  echo "=== extracting aiter @ ${AITER_SHA:0:9} ==="
  mkdir -p "$AITER_DIR"
  tar -xzf "$TARBALL" -C "$AITER_DIR"
fi

BENCH=/aiter/blasst_repro/bench_blasst_ua2d.py
[ -f "$AITER_DIR/blasst_repro/bench_blasst_ua2d.py" ] || {
  echo "ERROR: benchmark not found in the extracted aiter tree." >&2; exit 1; }

# Rebuild if an image is missing, was built from a different base commit, or
# (patched) from a different patch. /opt/triton-src-commit.txt holds the base
# commit on line 1 and, for the patched image, the patch's sha256 on line 2.
PATCH_BASE=$(head -1 "$HERE/patch-base.txt" 2>/dev/null || echo "")
PATCH_SHA=$(sha256sum "$HERE/0001-force-chain-dot-across-scf-if.patch" | cut -d' ' -f1)
for img in "$STOCK" "$PATCHED"; do
  prov=$(docker run --rm "$img" cat /opt/triton-src-commit.txt 2>/dev/null || echo "")
  img_base=$(printf '%s\n' "$prov" | head -1)
  if [ -z "$img_base" ]; then
    echo "=== $img missing -- building both images ==="
    bash "$HERE/build.sh"
    break
  fi
  if [ -n "$PATCH_BASE" ] && [ "$img_base" != "$PATCH_BASE" ]; then
    echo "=== $img was built from ${img_base:0:9}, patch is cut from ${PATCH_BASE:0:9}"
    echo "=== rebuilding both images ==="
    bash "$HERE/build.sh"
    break
  fi
  if [ "$img" = "$PATCHED" ] && ! printf '%s\n' "$prov" | grep -q "sha256=$PATCH_SHA"; then
    echo "=== $img was built from a different patch than the one shipped here"
    echo "=== rebuilding both images (stock is a Docker cache hit) ==="
    bash "$HERE/build.sh"
    break
  fi
done

# One arm, with its own Triton cache and dump dir. aiter is mounted read-only and
# used via PYTHONPATH: `pip install -e` would replace the image's Triton.
run_arm() {
  local name="$1" image="$2"
  local cache="$OUT/tcache_$name" dump="$OUT/dump_$name"

  echo
  echo "############################################################"
  echo "# arm: $name"
  echo "#   image: $image"
  echo "############################################################"

  # Start each arm from empty dirs. Deleted inside a container because a
  # previous run's files are root-owned.
  docker run --rm -v "$OUT":/out "$image" \
    rm -rf "/out/tcache_$name" "/out/dump_$name"
  mkdir -p "$cache" "$dump"

  local vid ren
  vid=$(getent group video | cut -d: -f3)
  ren=$(getent group render | cut -d: -f3 || echo "$vid")

  docker run --rm -i \
    --device=/dev/kfd --device=/dev/dri --group-add "$vid" --group-add "$ren" \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --network=host --ipc=host --shm-size=64g \
    -e ROCR_VISIBLE_DEVICES \
    -e PYTHONUNBUFFERED=1 \
    -e AITER_SRC=/aiter \
    -e PYTHONPATH=/aiter \
    -e AITER_TRITON_ONLY=1 \
    -e TRITON_CACHE_DIR=/out/tcache_"$name" \
    -e TRITON_KERNEL_DUMP=1 \
    -e TRITON_DUMP_DIR=/out/dump_"$name" \
    -v "$AITER_DIR":/aiter:ro \
    -v "$OUT":/out \
    -v "$RESULTS":/results \
    -w /aiter \
    "$image" bash -c "
      set -e
      echo '--- provenance ---'
      cat /opt/triton-src-commit.txt
      python3 -c 'import triton; print(\"triton:\", triton.__version__, \"->\", triton.__file__)'
      echo '--- bench ---'
      python3 $BENCH --csv /results/bench_$name.csv ${BENCH_ARGS[*]}
    " 2>&1 | tee "$RESULTS/bench_$name.log"
}

# Under Slurm, use the allocated GPU's global index: --device=/dev/kfd exposes
# all GPUs to the container, so Slurm's cgroup-relative index would pick the
# wrong one.
if [ -n "${SLURM_STEP_GPUS:-}" ]; then
  ROCR_VISIBLE_DEVICES="$SLURM_STEP_GPUS"
elif [ -n "${SLURM_JOB_GPUS:-}" ]; then
  ROCR_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
fi
export ROCR_VISIBLE_DEVICES
echo "GPU visibility: ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset:all>}"

run_arm stock   "$STOCK"
run_arm patched "$PATCHED"

echo
echo "############################################################"
echo "# layout check -- does the recovered warpsPerCTA differ?"
echo "############################################################"
# Checked before the timings: if both arms chose the same layout, the patch had
# no effect and the timings mean nothing.
python3 "$HERE/layout_check.py" "$OUT/dump_stock" "$OUT/dump_patched" \
  || echo "WARNING: layout check inconclusive -- read the dumps before trusting the table below"

echo
echo "############################################################"
echo "# side by side"
echo "############################################################"
python3 "$HERE/collate.py" "$RESULTS/bench_stock.csv" "$RESULTS/bench_patched.csv"

echo
echo "############################################################"
echo "# register pressure"
echo "############################################################"
OUT="$OUT" python3 "$HERE/spill_stats.py" "$OUT/dump_stock" "$OUT/dump_patched" \
  || echo "(spill stats unavailable -- check the dump dirs)"

echo
echo "results in $RESULTS"

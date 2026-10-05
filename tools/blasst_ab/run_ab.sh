#!/bin/bash
# The PR's headline measurement: AITER's BLASST attention kernel on UNMODIFIED
# upstream Triton vs the same Triton with the chain-dot patch.
#
#   bash tools/blasst_ab/run_ab.sh                       # defaults
#   bash tools/blasst_ab/run_ab.sh --seqlens 8192,16384  # args pass through
#   INERT_CHECK=1 bash tools/blasst_ab/run_ab.sh         # + third arm, see below
#
# Needs an MI300X/MI355X and Docker. See README.md for the full procedure.
#
# ---------------------------------------------------------------------------
# WHY TWO IMAGES RATHER THAN ONE
#
# The patch is opt-in, and returns false before changing anything when the env
# var is unset, so a single patched image could produce both arms. That makes
# the baseline a "patched build with the patch switched off", which a reader has
# to take on trust. Arm A here is a genuinely unpatched image instead -- see
# Dockerfile.stock, which asserts at build time that the patch is absent.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="${OUT:-$REPO/.blasst-out}"
RESULTS="${RESULTS:-$OUT/blasst_ab}"

STOCK="${STOCK:-triton-blasst-ab:stock}"
PATCHED="${PATCHED:-triton-blasst-ab:patched}"

AITER_SHA="${AITER_SHA:-3fc0ebd668030b3d12a94ce034deb87704092899}"
TARBALL="$HERE/aiter-${AITER_SHA:0:9}.tar.gz"
# Point this at a clone pinned to AITER_SHA:
#   AITER_DIR=/path/to/aiter bash run_ab.sh
# A pre-extracted tarball next to this script is used instead when present.
AITER_DIR="${AITER_DIR:-$OUT/aiter-${AITER_SHA:0:9}}"

# The shape the published numbers were measured at. Override via arguments.
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

# ---------------------------------------------------------------------------
# aiter, at a pinned commit, extracted from the tarball.
#
# Use ONE pinned tree for all of it -- kernel, wrapper and benchmark together.
# They evolve as a unit: the skip_counter dtype, for instance, changed across
# all three in the same commit, so mixing a benchmark from one revision with a
# wrapper from another trips an assertion in unified_attention.py.
# ---------------------------------------------------------------------------
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

# Rebuild if an image is MISSING **or STALE**.
#
# Checking only for absence is not enough, and the failure is silent and
# expensive: after rebasing onto a new upstream commit and updating the pin,
# both images still existed, so this loop skipped build.sh entirely and the
# whole run measured the OLD base. The provenance print was the only clue, and
# only because it is printed per arm.
#
# The image records what it was built from in /opt/triton-src-commit.txt;
# compare that against the commit the patch was actually cut from.
PATCH_BASE=$(head -1 "$HERE/patch-base.txt" 2>/dev/null || echo "")
for img in "$STOCK" "$PATCHED"; do
  img_base=$(docker run --rm "$img" head -1 /opt/triton-src-commit.txt 2>/dev/null || echo "")
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
done

# ---------------------------------------------------------------------------
# One arm.
#
# Per-arm TRITON_CACHE_DIR and TRITON_DUMP_DIR. Separate images already make
# cross-contamination impossible, but a shared cache has produced wrong A/B
# results before now, and separating them costs nothing.
#
# aiter is mounted READ-ONLY and used via PYTHONPATH, never `pip install -e`.
# aiter's setup.py replaces the installed Triton unless AITER_USE_SYSTEM_TRITON
# is set (setup.py:100-141) -- in an image whose entire purpose is a specific
# Triton build, that is the one thing that must not happen. The Triton path is
# pure Python, so PYTHONPATH is sufficient.
# ---------------------------------------------------------------------------
run_arm() {
  local name="$1" image="$2" force="$3"
  local cache="$OUT/tcache_$name" dump="$OUT/dump_$name"

  echo
  echo "############################################################"
  echo "# arm: $name"
  echo "#   image: $image"
  echo "#   TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF=${force:-<unset = OFF>}"
  echo "############################################################"

  # Wipe the per-arm dirs IN A CONTAINER, not on the host.
  #
  # Docker runs as root, so everything the previous run wrote into the bind
  # mount is root-owned and a host-side `rm -rf` fails with "Permission denied"
  # on every file. The first run works (nothing to delete) and every rerun dies,
  # which makes this look like a cluster permissions problem rather than a bug
  # here. Deleting as root from inside the same image is reliable and needs no
  # sudo on the host.
  #
  # Wiping at all is not optional: reusing these across shapes left two .amdgcn
  # per kernel in the cache and the analyser silently picked an arbitrary one,
  # mixing results from different shapes into one table.
  docker run --rm -v "$OUT":/out "$image" \
    rm -rf "/out/tcache_$name" "/out/dump_$name"
  mkdir -p "$cache" "$dump"

  local vid ren
  vid=$(getent group video | cut -d: -f3)
  ren=$(getent group render | cut -d: -f3 || echo "$vid")

  # Pass the flag ONLY when it is on. getBoolEnv returns false for an empty
  # string (GetEnv.h:91-99), so -e VAR="" would also work -- but the stock image
  # has no patch and therefore does not list this name in
  # CACHE_INVALIDATING_ENV_VARS. Leaving it unset keeps the baseline's
  # environment identical to a reviewer's plain `docker run`.
  local force_env=()
  [ -n "$force" ] && force_env=(-e "TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF=$force")

  docker run --rm -i \
    --device=/dev/kfd --device=/dev/dri --group-add "$vid" --group-add "$ren" \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --network=host --ipc=host --shm-size=64g \
    -e ROCR_VISIBLE_DEVICES \
    -e PYTHONUNBUFFERED=1 \
    -e AITER_SRC=/aiter \
    -e PYTHONPATH=/aiter \
    -e AITER_TRITON_ONLY=1 \
    "${force_env[@]}" \
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

# ROCR_VISIBLE_DEVICES: Slurm's value is relative to the job's device cgroup,
# but --device=/dev/kfd bypasses that cgroup, so inside the container GPUs are
# numbered globally and forwarding the relative index selects somebody else's
# card. SLURM_STEP_GPUS is already global. Harmless outside Slurm, where none of
# these are set.
if [ -n "${SLURM_STEP_GPUS:-}" ]; then
  ROCR_VISIBLE_DEVICES="$SLURM_STEP_GPUS"
elif [ -n "${SLURM_JOB_GPUS:-}" ]; then
  ROCR_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
fi
export ROCR_VISIBLE_DEVICES
echo "GPU visibility: ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset:all>}"

run_arm stock   "$STOCK"   ""
run_arm patched "$PATCHED" "1"

# Optional third arm: the PATCHED image with the flag unset. It should match
# `stock` to within noise, confirming the patch costs nothing when disabled. If
# it does NOT match, the patch is not inert when off -- a far more serious
# finding than anything about performance.
if [ "${INERT_CHECK:-0}" = "1" ]; then
  run_arm patched_off "$PATCHED" ""
fi

echo
echo "############################################################"
echo "# layout check -- does the recovered warpsPerCTA differ?"
echo "############################################################"
# This gate comes BEFORE the timings on purpose. If the two arms produced the
# same layout, the patch did not take effect and the timing difference, whatever
# it is, is measuring noise.
python3 "$HERE/layout_check.py" "$OUT/dump_stock" "$OUT/dump_patched" \
  || echo "WARNING: layout check inconclusive -- read the dumps before trusting the table below"

echo
echo "############################################################"
echo "# side by side"
echo "############################################################"
python3 "$HERE/collate.py" "$RESULTS/bench_stock.csv" "$RESULTS/bench_patched.csv"

echo
echo "############################################################"
echo "# register pressure (the mechanism, not the symptom)"
echo "############################################################"
# bench_blasst_ua2d.py measures time and elision only. Spills are what convert a
# layout difference into time -- the PR claims the win is confined to FA-shaped
# kernels that would otherwise spill, and this is the direct evidence for it.
OUT="$OUT" python3 "$HERE/spill_stats.py" "$OUT/dump_stock" "$OUT/dump_patched" \
  || echo "(spill stats unavailable -- check the dump dirs)"

echo
echo "results in $RESULTS"

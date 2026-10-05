#!/bin/bash
# Build both arms of the BLASST A/B: stock upstream Triton, and the same build
# plus the chain-dot patch.
#
#   bash tools/blasst_ab/build.sh
#
# Run this ON A COMPUTE NODE (it needs docker, not a GPU -- compiling Triton is
# CPU work -- the benchmark in run_ab.sh is what needs the GPU).
#
# First run is ~10-15 min for the stock image; the patched image derives from it
# and should take ~2. Both are cached afterwards, so a rerun is seconds.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

STOCK="${STOCK:-triton-blasst-ab:stock}"
PATCHED="${PATCHED:-triton-blasst-ab:patched}"
TRITON_COMMIT="${TRITON_COMMIT:-b1c34b00c7f6a6819927577c00a98c0753c1709f}"

command -v docker >/dev/null || {
  echo "ERROR: no docker here. Are you on a login node? This must run on a compute node." >&2
  exit 1
}

# ---------------------------------------------------------------------------
# Verify the patch -- do NOT regenerate it here.
#
# This script runs on a compute node, where there is no git repository: the
# sync deliberately excludes .git/ (~700 MB of an 842 MB checkout, and the
# remote pushes an already-patched tree so it never needs history). An earlier
# version called `git diff` here and died with "not a git repository" the first
# time it ran on the cluster.
#
# Both the patch and patch-base.txt ship alongside this script, so these checks
# normally just pass. They exist so that a patch cut against one base can never
# be built on top of another -- which is exactly what the pin is for.
# ---------------------------------------------------------------------------
PATCH="$HERE/0001-force-chain-dot-across-scf-if.patch"
STAMP="$HERE/patch-base.txt"
echo "=== verifying patch provenance ==="
[ -s "$PATCH" ] || {
  echo "ERROR: $PATCH missing or empty -- it should ship with this script." >&2
  exit 1
}
[ -f "$STAMP" ] || {
  echo "ERROR: $STAMP missing -- cannot tell which commit the patch was cut from." >&2
  exit 1
}
PATCH_BASE=$(head -1 "$STAMP")
if [ "$PATCH_BASE" != "$TRITON_COMMIT" ]; then
  echo "ERROR: patch was cut from $PATCH_BASE but the images pin $TRITON_COMMIT." >&2
  echo "       Build the matching base with TRITON_COMMIT=$PATCH_BASE if that is" >&2
  echo "       intended, or restore a patch cut from $TRITON_COMMIT." >&2
  exit 1
fi
echo "patch: $(wc -l < "$PATCH") lines, cut from $PATCH_BASE -- matches the pinned commit"

# ---------------------------------------------------------------------------
# Build. Context is $HERE so the patch COPY resolves; the Dockerfiles pull
# Triton from GitHub rather than from this tree, so the context stays tiny --
# EXCEPT for the aiter tarball, which is big and not needed at build time.
# ---------------------------------------------------------------------------
echo
echo "=== [1/2] stock: upstream Triton @ ${TRITON_COMMIT:0:9}, no patch ==="
time docker build --network=host \
  -f "$HERE/Dockerfile.stock" \
  --build-arg "TRITON_COMMIT=$TRITON_COMMIT" \
  --tag "$STOCK" \
  "$HERE"

echo
echo "=== [2/2] patched: FROM stock + chain-dot patch (incremental rebuild) ==="
time docker build --network=host \
  -f "$HERE/Dockerfile.patched" \
  --build-arg "STOCK_IMAGE=$STOCK" \
  --tag "$PATCHED" \
  "$HERE"

# ---------------------------------------------------------------------------
# Provenance. Printed every build because the single most expensive mistake
# available here is measuring two arms that are secretly the same build.
# ---------------------------------------------------------------------------
echo
echo "=== provenance ==="
for img in "$STOCK" "$PATCHED"; do
  echo "--- $img"
  docker run --rm "$img" cat /opt/triton-src-commit.txt
done

echo
echo "Built: $STOCK and $PATCHED"
echo "Next:  bash tools/blasst_ab/run_ab.sh"

#!/bin/bash
# Build both arms: stock upstream Triton, and the same build plus the patch.
# Needs Docker, not a GPU. ~15 min for stock, a few minutes for patched.
#
#   bash build.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

STOCK="${STOCK:-triton-blasst-ab:stock}"
PATCHED="${PATCHED:-triton-blasst-ab:patched}"
TRITON_COMMIT="${TRITON_COMMIT:-b1c34b00c7f6a6819927577c00a98c0753c1709f}"

command -v docker >/dev/null || {
  echo "ERROR: no docker here. Are you on a login node? This must run on a compute node." >&2
  exit 1
}

# The patch must have been cut from the commit the images build.
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

# Provenance: base commit, and which patch (if any) each image contains.
echo
echo "=== provenance ==="
for img in "$STOCK" "$PATCHED"; do
  echo "--- $img"
  docker run --rm "$img" cat /opt/triton-src-commit.txt
done

echo
echo "Built: $STOCK and $PATCHED"
echo "Next:  bash run_ab.sh"

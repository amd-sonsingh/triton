# Reproducing the chain-dot-across-`scf.if` measurement

Companion to the PR **"[AMD] Allow chain-dot detection across one level of `scf.if`"**
against `triton-lang/triton`. It lives on its own branch so the PR diff stays at the two
files it actually changes.

**What is being measured.** The AMD backend's chain-dot detection requires both dots to
live in the same MLIR region, so a conditional between a QK dot and a P@V dot defeats it and
`planWarps()` falls back from `warpsPerCTA = [num_warps, 1]` to a generic layout. The patch
lets the detectors see across exactly one level of `scf.if`, behind
`TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF`.

Every step below has a **PASS** line. If one does not hold, stop.

---

## 0. Prerequisites

| | |
|---|---|
| GPU | MI300X or MI355X (`gfx942` / `gfx950`). The block-skip path is not arch-gated, but the numbers below are MI355X. |
| Docker | Required. Runs as a normal user with access to `/dev/kfd`. |
| Disk | ~40 GB for the two images. |
| Time | ~20 min, almost all of it compiling Triton once. |
| Network | The build clones `triton-lang/triton` from GitHub. |

```bash
ls /dev/kfd && docker ps >/dev/null && echo "prereqs OK"
```

**PASS:** prints `prereqs OK`. No `/dev/kfd` means no GPU visible.

All commands below run **from this directory** (`tools/blasst_ab/`).

---

## 1. Get the kernel under test

The benchmark drives AITER's `unified_attention` 2D kernel, whose block-skip path puts the
P@V dot one `scf.if` below the QK dot — that nesting is the entire subject of the patch:

```python
if not (ENABLE_BLOCK_SKIP and all_skip):
    acc = tl.dot(P.to(V.dtype), V, acc=acc)
```

Pinned, because the measurement is only meaningful against a known kernel:

```bash
git clone https://github.com/amd-sonsingh/aiter.git /tmp/aiter
git -C /tmp/aiter checkout 3fc0ebd668030b3d12a94ce034deb87704092899
```

Confirm the conditional is actually there:

```bash
grep -n "ENABLE_BLOCK_SKIP and all_skip" \
  /tmp/aiter/aiter/ops/triton/_triton_kernels/attention/unified_attention.py
```

should return at least one hit.

> Nothing is installed from this tree — it is mounted read-only and used via `PYTHONPATH`.
> Do **not** `pip install -e` it: aiter's `setup.py` replaces the installed Triton unless
> `AITER_USE_SYSTEM_TRITON=1`, which would defeat the purpose of these images entirely.

---

## 2. Build the two arms

```bash
bash build.sh
```

Two images, both from `triton-lang/triton@6c14a3d5e8`:

| image | Triton |
|---|---|
| `triton-blasst-ab:stock` | unmodified upstream — the patch is **absent**, not disabled |
| `triton-blasst-ab:patched` | `FROM` stock, + `0001-force-chain-dot-across-scf-if.patch` |

The patched image derives from the stock one rather than building in parallel, so "the only
difference is this patch" is a property of the image graph rather than a claim. Each asserts
its own state at build time: stock fails if the patch helper is present, patched fails if
the flag string is missing from the built `libtriton.so`.

Expect ~15 min for stock (Triton from source) and ~1–4 min for patched (incremental
relink). Watch for these lines:

```
verified: stock image has no chain-dot patch
Applied patch include/triton/Tools/Sys/GetEnv.h cleanly.
Applied patch third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp cleanly.
rebuild with patch took ~120s (expect ~120s incremental, ~600s full)
verified: patch present in the built binary (/opt/triton-src/python/triton/_C/libtriton.so)
```

**PASS:** `build.sh` exits 0 and prints the provenance block:

```
--- triton-blasst-ab:stock
6c14a3d5e824fcab6e8c4f7ea137ccd839c87140
patched: NONE (stock upstream)
--- triton-blasst-ab:patched
6c14a3d5e824fcab6e8c4f7ea137ccd839c87140
patched: 0001-force-chain-dot-across-scf-if.patch
```

Both must show the **same commit**; exactly one must say `patched:`.

---

## 3. Prove the two arms are genuinely different builds

The single most expensive mistake available here is measuring two arms that are secretly the
same build. `build.sh` already asserts this, but check it independently:

```bash
for img in stock patched; do
  echo -n "$img source:  "
  docker run --rm "triton-blasst-ab:$img" sh -c \
    'grep -c isSameOrAcrossOneIfLevel /opt/triton-src/third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp || true'
  echo -n "$img binary:  "
  docker run --rm "triton-blasst-ab:$img" sh -c \
    'grep -aq TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF /opt/triton-src/python/triton/_C/libtriton.so \
       && echo PRESENT || echo ABSENT'
done
```

**PASS:**

```
stock source:   0
stock binary:   ABSENT
patched source: 3
patched binary: PRESENT
```

Checking the **binary** and not only the source matters: an incremental rebuild that
silently failed to pick up the patch would still show the source change.

---

## 4. Run the A/B

```bash
AITER_DIR=/tmp/aiter bash run_ab.sh
```

Defaults: 32 q-heads / 8 kv-heads, seqlen 16384, head_dim 128, random bf16 inputs. Extra
arguments pass straight through to the benchmark:

```bash
AITER_DIR=/tmp/aiter bash run_ab.sh --shapes 32x8 --seqlens 8192,16384
```

Each arm gets its own `TRITON_CACHE_DIR` and `TRITON_DUMP_DIR`. Separate images already make
cross-contamination impossible; the separation is belt-and-braces because a shared cache has
produced wrong A/B results before now.

> A wall of `nanobind: leaked ... instances/types/functions` at the end of each arm is
> normal Triton interpreter-shutdown noise and does not affect the measurement.

---

## 5. Check the results, in this order

### 5a. Layout — the gate

Printed **before** the timings on purpose. If both arms chose the same layout, the patch did
not take effect and the timings are noise regardless of how good they look.

```
kernel                              stock                patched
kernel_unified_attention_2d.ttgir   [4,1],[2,2],[2,2]    [4,1],[4,1],[4,1]   <-- CHANGED
```

**PASS:** `OK: 1 kernel(s) got a different warp layout under the patch.` Stock must contain
`[2,2]`; patched must be all `[4,1]`.

### 5b. Timings

```
case                          lambda   elide  stock ms   patched  patch worth
32q/8kv seqlen=16384           1e-09    0.0%     3.395     3.096      1.0968x
32q/8kv seqlen=16384         0.02878    0.0%     3.389     3.101      1.0930x
32q/8kv seqlen=16384             0.1    0.0%     3.365     3.042      1.1063x
32q/8kv seqlen=16384             0.3    0.0%     3.362     3.050      1.1026x
32q/8kv seqlen=16384             1.0   21.4%     2.946     2.682      1.0983x
32q/8kv seqlen=16384             1.3   44.3%     2.539     2.329      1.0901x
32q/8kv seqlen=16384             2.0   70.9%     2.095     1.924      1.0887x

  no elision      : patch worth 1.0997x  (4 pts)
  elision active  : patch worth 1.0924x  (3 pts)
```

**PASS:** patch worth roughly **1.08–1.12×** on every row, and **near-constant** across the
elision sweep.

The flatness is the result, not an incidental detail. Warp layout is chosen at compile time,
so the cost is paid because the conditional *exists*, not because it is taken. A figure that
climbed with elision would mean the benchmark was measuring something else, and `collate.py`
prints a warning if the two buckets diverge by more than 0.05×.

### 5c. Numerics

```
  stock drifts from its own dense output where nothing elided;
  patched matches dense EXACTLY on those same rows
```

**PASS:** patched `rel_err` is exactly `0` on every 0%-elision row.

Stock showing ~1e-5 there is expected, and is the layout difference surfacing in the output:
dense has no conditional so it keeps `[4,1]`, unpatched block-skip falls back to `[2,2]`,
and two layouts reduce in different orders. A non-zero `rel_err` on the **patched** arm
would be a real failure and `collate.py` exits non-zero for it.

### 5d. Register pressure

```
arm        vgpr_spill_count  sgpr_spill_count  ScratchSize  vgpr_count  scratch_ops
stock                     0                 2            0         243            0
patched                   0                 2            0         237            0
```

**PASS:** prints without error. Zero spills on both arms is the expected and correct result
here — see the caveats.

---

## 6. Optional: verify the patch is inert when disabled

```bash
INERT_CHECK=1 AITER_DIR=/tmp/aiter bash run_ab.sh
```

Adds a third arm: the **patched image with the flag unset**. It should match `stock` to
within noise, confirming the patch costs nothing when off.

---

## Caveats, stated upfront

**The payoff is narrower than the detection.** `isChainDotHead` / `isChainDotTail` are
purely structural — they ask whether one `tt.dot`'s result feeds another's operand, and
already fire on any two chained dots in the same region. The patch only extends which region
boundaries count. But the layout detection unlocks, `[num_warps, 1]`, is tuned for
flash-attention shapes: it keeps a softmax row inside one warp so the row-reduction needs no
inter-warp communication. **A chain-dot kernel with no row reduction between the dots gains
nothing**, and can lose — because the conditional's cost is paid with nothing to offset it.
That is why the patch is opt-in rather than unconditional.

**The baseline is a real baseline.** Both arms could have come from one image by toggling the
env var — the patch returns false immediately when unset. Two images were built anyway, so
the comparison is against genuinely unmodified upstream rather than "patched build with the
patch switched off".

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `ERROR: ... patch was cut from X but the images pin Y` | `patch-base.txt` disagrees with the pinned commit. Intentional guard — do not override without understanding why they differ. |
| `no .ttgir found` in the layout check | `TRITON_KERNEL_DUMP=1` did not reach the container; the layout gate cannot run and timings are unverifiable. |
| `*** NO LAYOUT CHANGED between the arms` | The flag never reached the compiler, or this shape does not discriminate. **Do not record the timings.** |
| `cmake` build failure in the stock image | `pyproject.toml` at this commit requires `cmake>=3.20,<4.0`; bare `pip install cmake` resolves to 4.x. The Dockerfile pins it. |
| `assert skip_counter.dtype == ...` | aiter is not at `3fc0ebd66`. The counter dtype changed after that commit and the benchmark and wrapper must come from the same tree. |
| `ERROR: no /dev/kfd` | Login node, or a GPU-less allocation. |

---

## Files

| | |
|---|---|
| `Dockerfile.stock` | upstream Triton @ `6c14a3d5e8`, unpatched |
| `Dockerfile.patched` | `FROM` stock + the patch, incremental rebuild |
| `0001-force-chain-dot-across-scf-if.patch` | the change under test, identical to the PR diff |
| `patch-base.txt` | the commit the patch was cut from; `build.sh` refuses to proceed if it disagrees with the pinned commit |
| `build.sh` | builds both images, prints provenance |
| `run_ab.sh` | runs both arms, collates |
| `collate.py` | side-by-side table; splits on *measured* elision, not the λ value |
| `layout_check.py` | the gate in 5a |
| `spill_stats.py` | register pressure from the emitted AMDGCN |

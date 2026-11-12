# Reproducing the chain-dot-across-`scf.if` performance measurement

Companion to the PR **"[AMD] Allow chain-dot detection across one level of `scf.if`"**
against `triton-lang/triton`. This branch is the PR plus this directory. It's kept separate
so the PR diff contains only the files the change touches.

**This covers performance only.** For correctness, use the lit test in the PR: see *How to
verify* in the PR description. That needs no GPU, no Docker and nothing from this
directory.

**What is being measured.** The AMD backend's chain-dot detection requires both dots to
live in the same MLIR region. A conditional between a QK dot and a P@V dot therefore defeats
it, and `planWarps()` falls back from `warpsPerCTA = [num_warps, 1]` to a generic layout. The
patch lets the detectors see across exactly one level of `scf.if`. Here it is measured on a
real attention kernel, against unmodified upstream Triton.

Every step below has a **PASS** line. If one does not hold, stop.

---

## 0. Prerequisites

| | |
|---|---|
| GPU | MI355X (`gfx950`). All numbers below are from MI355X. MI300X (`gfx942`) should run, since nothing here is arch-gated, but hasn't been tested. |
| Docker | Required. Runs as a normal user with access to `/dev/kfd`. |
| Disk | ~40 GB for the two images. |
| Time | ~15 min, almost all of it compiling Triton once. |
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

Pinned, because the measurement is only meaningful against a known kernel. The kernel and
benchmark are on a personal fork of `ROCm/aiter`:

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

Two images, both from `triton-lang/triton@b1c34b00c7`:

| image | Triton |
|---|---|
| `triton-blasst-ab:stock` | unmodified upstream — the patch is **absent**, not disabled |
| `triton-blasst-ab:patched` | `FROM` stock, + `0001-force-chain-dot-across-scf-if.patch` (the PR's full diff) |

The patched image derives from the stock one rather than being built separately. That makes
"the only difference is this patch" a property of how the images are built, not a claim.
Each image checks its own state at build time:

- **stock** fails if the patch's helper function is in the source.
- **patched** compiles the PR's new lit case with its own `triton-opt`, and fails unless the
  branched chain-dot case gets the same layout as the unbranched one. That proves the patch
  reached the built binary, not just the source.

Expect ~10 min for stock (Triton from source) and ~1 min for patched (incremental relink).
Tags default to `triton-blasst-ab:{stock,patched}`; set `STOCK=` and `PATCHED=` for
`build.sh`, step 3 and `run_ab.sh` to use others. Watch for these lines:

```
verified: stock image has no chain-dot patch
Applied patch test/TritonGPU/amd/accelerate-amd-matmul-chain-dot.mlir cleanly.
Applied patch third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp cleanly.
verified: patch present in source
mfma_chain_dot_BM64   (no branch): [4,1] [4,1]
chain_dot_scf_if_BM64 (branched) : [4,1] [4,1]
OK: branched case matches the unbranched one
verified: patch present in the built binary (...)
```

**PASS:** `build.sh` exits 0 and prints the provenance block:

```
--- triton-blasst-ab:stock
b1c34b00c7f6a6819927577c00a98c0753c1709f
patched: NONE (stock upstream)
--- triton-blasst-ab:patched
b1c34b00c7f6a6819927577c00a98c0753c1709f
patched: 0001-force-chain-dot-across-scf-if.patch sha256=<sha256 of the patch file>
```

Both must show the **same commit**, and exactly one must name the patch. Its `sha256` must
match `sha256sum 0001-force-chain-dot-across-scf-if.patch`. `run_ab.sh` checks the same
thing and rebuilds if it doesn't match, so a revised patch can't be measured with an old
image.

---

## 3. Prove the two arms are genuinely different builds

The single most expensive mistake available here is measuring two arms that are secretly the
same build. `build.sh` already asserts this. These checks confirm it independently.

**The patch is the PR.** Run from this directory, on this branch:

```bash
git diff b1c34b00c7 HEAD -- ':(top,exclude)tools' | diff - 0001-force-chain-dot-across-scf-if.patch && echo "patch == PR diff"
```

This branch is the PR plus commits that touch only `tools/`, so everything outside `tools/`
is exactly the PR.

**The binaries differ in behaviour.** Compile the PR's lit test with each image's own
`triton-opt`:

```bash
for img in "${STOCK:-triton-blasst-ab:stock}" "${PATCHED:-triton-blasst-ab:patched}"; do
  echo "--- $img"
  docker run --rm -v "$PWD/../../test:/t:ro" -v "$PWD:/b:ro" "$img" sh -c \
    '$(find /opt/triton-src -name triton-opt -type f -perm -u+x | head -1) \
       /t/TritonGPU/amd/accelerate-amd-matmul-chain-dot.mlir -split-input-file \
       --tritonamdgpu-accelerate-matmul="gfx-arch=gfx942 matrix-instruction-size=16" \
     | python3 /b/check_binary.py'
done
```

**PASS:**

```
--- triton-blasst-ab:stock
mfma_chain_dot_BM64   (no branch): [4,1] [4,1]
chain_dot_scf_if_BM64 (branched) : [4,1] [1,4]
FATAL: the branched case did not get the chain-dot layout, so this triton-opt does not contain the change. ...
--- triton-blasst-ab:patched
mfma_chain_dot_BM64   (no branch): [4,1] [4,1]
chain_dot_scf_if_BM64 (branched) : [4,1] [4,1]
OK: branched case matches the unbranched one
```

Stock **must** fail here and patched **must** pass. This checks the compiled binaries, not
the source, so an incremental rebuild that silently didn't pick up the patch can't get
through.

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

Each arm gets its own `TRITON_CACHE_DIR` and `TRITON_DUMP_DIR`. Results go to
`.blasst-out/` at the repo root; files written by the containers are root-owned.

> Both arms print `chain-dot patch    TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF=0`. That line is
> left over from an earlier, opt-in version of the patch and has no effect: the arms differ
> by image, not by environment.

> A wall of `nanobind: leaked ... instances/types/functions` at the end of each arm is
> normal Triton interpreter-shutdown noise and does not affect the measurement.

---

## 5. Check the results, in this order

### 5a. Layout — the gate

Printed **before** the timings on purpose. If both arms chose the same layout, the patch did
not take effect and the timings are noise regardless of how good they look.

```
kernel                                         stock                  patched
kernel_unified_attention_2d.ttgir              [4,1],[2,2],[2,2]      [4,1],[4,1],[4,1]       <-- CHANGED

OK: 1 kernel(s) got a different warp layout under the patch.
```

**PASS:** `OK: 1 kernel(s) got a different warp layout under the patch.` Stock must contain
`[2,2]`; patched must be all `[4,1]`.

### 5b. Timings

```
case                          lambda   elide  stock ms   patched  patch worth err stock err patch
32q/8kv seqlen=16384         0.02878    0.0%     3.371     3.093      1.0898x  0.000115         0
32q/8kv seqlen=16384             0.1    0.0%     3.360     3.041      1.1048x  6.77e-05         0
32q/8kv seqlen=16384             0.3    0.0%     3.356     3.044      1.1027x  0.000101         0
32q/8kv seqlen=16384             1.0   21.4%     2.943     2.681      1.0975x     0.246     0.245
32q/8kv seqlen=16384             1.3   44.3%     2.538     2.330      1.0889x     0.498     0.498
32q/8kv seqlen=16384           1e-09    0.0%     3.391     3.074      1.1032x  1.28e-05         0
32q/8kv seqlen=16384             2.0   70.9%     2.088     1.920      1.0874x      1.03      1.03

  split by: measured elision (<1% vs >=1% of tiles)
  no elision      : patch worth 1.1001x  (4 pts) -- skip-CHECK overhead only
  elision active  : patch worth 1.0913x  (3 pts) -- the conditional dot is being taken
```

**PASS:** patch worth roughly **1.08–1.11×** on every row, and **near-constant** across the
elision sweep. Expect **~1.10×** with no elision, about ±0.01× run to run; don't read
individual rows to three decimals.

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

Stock showing 1e-5 to 1e-4 there is expected, and is the layout difference surfacing:
dense has no conditional so it keeps `[4,1]`, unpatched block-skip falls back to `[2,2]`,
and two layouts reduce in different orders. A non-zero `rel_err` on the **patched** arm
would be a real failure and `collate.py` exits non-zero for it.

### 5d. Register pressure

```
kernel                                   arm        vgpr_spill_count  sgpr_spill_count       ScratchSize        vgpr_count       scratch_ops
kernel_unified_attention_2d.amdgcn       stock                     0                 2                 0               243                 0
kernel_unified_attention_2d.amdgcn       patched                   0                 2                 0               237                 0
```

**PASS:** prints without error. Zero spills on both arms is the expected and correct result
here — see the caveats.

---

## Caveats, stated upfront

**The payoff is narrower than the detection.** `isChainDotHead` / `isChainDotTail` are
purely structural. They ask whether one `tt.dot`'s result feeds another's operand, and they
already fire on any two chained dots in the same region; the patch only extends which region
boundaries count. The layout that detection unlocks, `[num_warps, 1]`, is tuned for
flash-attention shapes. It keeps a softmax row inside one warp, so the row-reduction needs no
inter-warp communication. **The speedup measured here needs a row-reduction between the
dots.** A chain-dot kernel without one gains nothing from it. With a branch between its
dots, it now gets the same layout `main` already picks for the same kernel without the
branch.

**It is not a register-spilling effect.** Both arms show zero VGPR spills (5d), and the patch
is still worth ~1.10×. Spilling would amplify the effect where it occurs, but it isn't
required.

**The baseline is genuinely unpatched.** The stock image is upstream Triton at the PR's base
commit, with nothing added.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `ERROR: ... patch was cut from X but the images pin Y` | `patch-base.txt` disagrees with the pinned commit. Intentional guard — do not override without understanding why they differ. |
| `no .ttgir found` in the layout check | `TRITON_KERNEL_DUMP=1` did not reach the container; the layout gate cannot run and timings are unverifiable. |
| `*** NO LAYOUT CHANGED between the arms` | The patched image doesn't contain the patch (check step 3), or this shape doesn't discriminate. **Do not record the timings.** |
| `FATAL: the branched case did not get the chain-dot layout` while building `patched` | The patch applied to the source but the incremental rebuild didn't reach `triton-opt`. Rebuild without the Docker cache. |
| `cmake` build failure in the stock image | `pyproject.toml` at this commit requires `cmake>=3.20,<4.0`; bare `pip install cmake` resolves to 4.x. The Dockerfile pins it. |
| `assert skip_counter.dtype == ...` | aiter is not at `3fc0ebd66`. The counter dtype changed after that commit and the benchmark and wrapper must come from the same tree. |
| `ERROR: no /dev/kfd` | Login node, or a GPU-less allocation. |

---

## Files

| | |
|---|---|
| `Dockerfile.stock` | upstream Triton @ `b1c34b00c7`, unpatched |
| `Dockerfile.patched` | `FROM` stock + the patch, incremental rebuild |
| `0001-force-chain-dot-across-scf-if.patch` | the change under test: the PR's full diff (source and lit test), byte for byte |
| `check_binary.py` | checks the built `triton-opt` carries the patch, by compiling the PR's lit case (step 2, step 3) |
| `patch-base.txt` | the commit the patch was cut from; `build.sh` refuses to proceed if it disagrees with the pinned commit |
| `build.sh` | builds both images, prints provenance |
| `run_ab.sh` | runs both arms, collates |
| `collate.py` | side-by-side table; splits on *measured* elision, not the λ value |
| `layout_check.py` | the gate in 5a |
| `spill_stats.py` | register pressure from the emitted AMDGCN |

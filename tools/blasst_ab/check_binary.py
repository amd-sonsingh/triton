#!/usr/bin/env python3
"""Check whether a built triton-opt contains the chain-dot change.

Reads the output of

  triton-opt accelerate-amd-matmul-chain-dot.mlir -split-input-file \
    --tritonamdgpu-accelerate-matmul="gfx-arch=gfx942 matrix-instruction-size=16"

on stdin and compares the dot layouts (warpsPerCTA) of mfma_chain_dot_BM64 (no
branch) and chain_dot_scf_if_BM64 (second dot inside an scf.if). They match
only if the change is in the binary.
"""
import re
import sys

ALIAS = re.compile(r"^(#\w+) = #ttg\.amd_mfma<\{.*?warpsPerCTA = \[(\d+), (\d+)\]")
FUNC = re.compile(r"tt\.func (?:public )?@(\w+)")
DOT_RESULT = re.compile(r"->\s*tensor<[^,>]+,\s*(#\w+)>\s*$")

aliases, funcs, cur = {}, {}, None
for line in sys.stdin:
    if m := ALIAS.match(line):
        aliases[m.group(1)] = f"[{m.group(2)},{m.group(3)}]"
    elif m := FUNC.search(line):
        cur = m.group(1)
        funcs[cur] = []
    elif cur and " tt.dot " in line and (m := DOT_RESULT.search(line)):
        funcs[cur].append(aliases.get(m.group(1), m.group(1)))

ref, new = funcs.get("mfma_chain_dot_BM64"), funcs.get("chain_dot_scf_if_BM64")
if not ref or not new:
    sys.exit("FATAL: test functions not found in triton-opt output -- "
             "did the patch (which adds chain_dot_scf_if_BM64) apply?")
print(f"mfma_chain_dot_BM64   (no branch): {' '.join(ref)}")
print(f"chain_dot_scf_if_BM64 (branched) : {' '.join(new)}")
if new != ref:
    sys.exit("FATAL: the branched case did not get the chain-dot layout, so this triton-opt "
             "does not contain the change. Expected for the stock image. For the patched "
             "image it means the rebuild did not pick up the patch.")
print("OK: branched case matches the unbranched one")

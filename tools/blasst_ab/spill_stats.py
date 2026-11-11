#!/usr/bin/env python3
"""Register pressure per arm, straight from the emitted AMDGCN.

Spilling would amplify a layout difference, but it is not required for the
measured win: both arms show zero VGPR spills.

    python3 spill_stats.py dump_stock dump_patched
"""
import os
import re
import sys

# Read both the .amdhsa_kernel directives and the YAML metadata; which one is
# populated varies across LLVM versions.
FIELDS = {
    "vgpr_spill_count": re.compile(r"\.vgpr_spill_count:\s*(\d+)"),
    "sgpr_spill_count": re.compile(r"\.sgpr_spill_count:\s*(\d+)"),
    "vgpr_count": re.compile(r"\.vgpr_count:\s*(\d+)"),
    "sgpr_count": re.compile(r"\.sgpr_count:\s*(\d+)"),
    "ScratchSize": re.compile(r"\.private_segment_fixed_size:\s*(\d+)"),
}
SCRATCH_OP = re.compile(r"^\s*scratch_(load|store)", re.M)


def stats_for(root):
    out = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.endswith(".amdgcn"):
                continue
            txt = open(os.path.join(dirpath, f), errors="ignore").read()
            row = {}
            for name, rx in FIELDS.items():
                hits = [int(m) for m in rx.findall(txt)]
                # Several kernels per file: report the worst.
                row[name] = max(hits) if hits else None
            row["scratch_ops"] = len(SCRATCH_OP.findall(txt))
            out[f] = row
    return out


def fmt(v):
    return "-" if v is None else str(v)


def main(stock_dir, patched_dir):
    a, b = stats_for(stock_dir), stats_for(patched_dir)
    if not a and not b:
        print(f"no .amdgcn under {stock_dir} or {patched_dir} "
              "(TRITON_KERNEL_DUMP=1 not set?)")
        return 2

    cols = ["vgpr_spill_count", "sgpr_spill_count", "ScratchSize",
            "vgpr_count", "scratch_ops"]
    names = sorted(set(a) | set(b))

    print(f"{'kernel':<40} {'arm':<9} " + " ".join(f"{c:>17}" for c in cols))
    for n in names:
        for arm, src in (("stock", a), ("patched", b)):
            row = src.get(n)
            if row is None:
                continue
            print(f"{n[:40]:<40} {arm:<9} "
                  + " ".join(f"{fmt(row.get(c)):>17}" for c in cols))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))

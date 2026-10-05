#!/usr/bin/env python3
"""Did the patch actually change the warp layout? Gate the timings on this.

The whole experiment is an A/B on one compile-time decision. If both arms chose
the same warpsPerCTA then the patch did not take effect, and whatever the
timings say they are not saying it about this patch. This check exists because a
run where the env var never reached the compiler produced perfectly plausible
numbers that meant nothing.

Reads TTGIR out of the per-arm TRITON_DUMP_DIR.

    python3 layout_check.py dump_stock dump_patched
"""
import collections
import os
import re
import sys

MFMA = re.compile(r"#ttg\.amd_mfma<.*?warpsPerCTA\s*=\s*(\[[0-9,\s]*\])")


def layouts(root):
    """warpsPerCTA of every amd_mfma encoding, per dumped kernel."""
    found = collections.defaultdict(list)
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.endswith(".ttgir"):
                continue
            path = os.path.join(dirpath, f)
            with open(path, errors="ignore") as fh:
                for line in fh:
                    m = MFMA.search(line)
                    if m:
                        # Normalise whitespace so [4, 1] and [4,1] compare equal.
                        found[f].append(re.sub(r"\s+", "", m.group(1)))
    return found


def main(stock_dir, patched_dir):
    a, b = layouts(stock_dir), layouts(patched_dir)

    if not a and not b:
        print(f"no .ttgir found under {stock_dir} or {patched_dir}.")
        print("Was TRITON_KERNEL_DUMP=1 set? Without the dumps this cannot be checked.")
        return 2

    names = sorted(set(a) | set(b))
    print(f"{'kernel':<46} {'stock':<22} {'patched':<22}")
    differing = 0
    for n in names:
        sa = ",".join(a.get(n, [])) or "(none)"
        sb = ",".join(b.get(n, [])) or "(none)"
        mark = "" if sa == sb else "  <-- CHANGED"
        if sa != sb:
            differing += 1
        print(f"{n[:46]:<46} {sa:<22} {sb:<22}{mark}")

    print()
    if differing:
        print(f"OK: {differing} kernel(s) got a different warp layout under the patch.")
        print("The timings below are attributable to that change.")
        return 0

    print("*** NO LAYOUT CHANGED between the arms. ***")
    print("Either the env var never reached the compiler, or this shape does not")
    print("discriminate, or the patch did not apply. The timings mean nothing")
    print("until that is resolved -- do not record them.")
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))

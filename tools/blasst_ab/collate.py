#!/usr/bin/env python3
"""Side-by-side of the two arms: stock upstream Triton vs the patched build.

The important piece is the SPLIT RULE -- see below.

    python3 collate.py bench_stock.csv bench_patched.csv
"""
import csv
import sys


def load(path):
    with open(path) as fh:
        return {(r["case"], r["threshold"]): r for r in csv.DictReader(fh)}


def main(stock_path, patched_path):
    stock, patched = load(stock_path), load(patched_path)
    keys = [k for k in patched if k in stock]
    if not keys:
        print("no overlapping rows -- did the two arms run the same shapes?")
        return 1

    print(f"{'case':<26} {'lambda':>9} {'elide':>7} "
          f"{'stock ms':>9} {'patched':>9} {'patch worth':>12} "
          f"{'err stock':>9} {'err patch':>9}")

    def err_of(row):
        v = row["rel_err"]
        return float(v) if v not in ("", None) else float("nan")

    graded = []          # (threshold, measured elision or None, patch worth)
    bad_patched = []     # patched arm: elision off but output moved -- a real bug
    drift_stock = []     # stock arm ditto; expected, see the note below
    for case, thr in sorted(keys):
        s, p = stock[(case, thr)], patched[(case, thr)]
        worth = float(s["ms"]) / float(p["ms"])
        raw = p["elide"]
        el = float(raw) if raw not in ("", None) else None
        es, ep = err_of(s), err_of(p)
        print(f"{case:<26} {thr:>9} "
              f"{'      -' if el is None else f'{100 * el:6.1f}%'} "
              f"{float(s['ms']):9.3f} {float(p['ms']):9.3f} {worth:11.4f}x "
              f"{es:9.3g} {ep:9.3g}")
        graded.append((float(thr), el, worth))
        # Nothing elided means nothing was skipped, so the output must match
        # this arm's OWN dense reference. Check BOTH arms: an earlier version
        # looked only at `patched` and silently missed that the stock arm drifts.
        if el is not None and el < 1e-12:
            if ep != 0.0:
                bad_patched.append((case, thr, ep))
            if es != 0.0:
                drift_stock.append((case, thr, es))

    # -----------------------------------------------------------------------
    # Split on MEASURED elision, not on the threshold value.
    #
    # A low threshold can elide nothing at all, so bucketing by lambda files
    # genuine overhead-floor points under "skipping active" and OVERSTATES the
    # gap.
    # -----------------------------------------------------------------------
    have_elide = all(e is not None for _, e, _ in graded)
    if have_elide:
        floor = [g for _, e, g in graded if e < 0.01]
        live = [g for _, e, g in graded if e >= 0.01]
        basis = "measured elision (<1% vs >=1% of tiles)"
    else:
        floor = [g for t, _, g in graded if t <= 1e-6]
        live = [g for t, _, g in graded if t > 1e-6]
        basis = ("threshold value -- the elide column is missing, so this split "
                 "is unreliable; a low lambda that elides nothing lands in the "
                 "wrong bucket")

    print()
    print(f"  split by: {basis}")
    if floor:
        print(f"  no elision      : patch worth {sum(floor)/len(floor):.4f}x  "
              f"({len(floor)} pts) -- skip-CHECK overhead only")
    if live:
        print(f"  elision active  : patch worth {sum(live)/len(live):.4f}x  "
              f"({len(live)} pts) -- the conditional dot is being taken")

    # The patch is a compile-time layout decision, so its cost is incurred
    # because the conditional EXISTS, not because it is taken. A large gap
    # between the two buckets is therefore suspicious rather than encouraging.
    if floor and live:
        gap = abs(sum(live)/len(live) - sum(floor)/len(floor))
        if gap > 0.05:
            print(f"\n  NOTE: the two buckets differ by {gap:.3f}x. Warp layout is "
                  "chosen at\n        compile time, so the patch's effect should be "
                  "near-constant\n        regardless of how much actually elides. A "
                  "large gap suggests the\n        benchmark is also measuring "
                  "something else.")

    # -----------------------------------------------------------------------
    # The stock arm drifting from its own dense reference at 0% elision is NOT
    # a bug, and it is not noise either -- it is the layout mismatch showing up
    # in the output. Dense has no conditional, so it keeps warpsPerCTA=[4,1];
    # unpatched block-skip falls back to [2,2]; two layouts means two reduction
    # orders means a different bf16 rounding. Under the patch both are [4,1] and
    # the outputs agree bit for bit.
    #
    # Worth reporting rather than hiding: "the patched kernel is numerically
    # identical to dense when nothing is skipped, and the unpatched one is not"
    # is a cleaner statement of what the patch does than any timing.
    # -----------------------------------------------------------------------
    if drift_stock and not bad_patched:
        print("\n  stock drifts from its own dense output where nothing elided;")
        print("  patched matches dense EXACTLY on those same rows:")
        for case, thr, err in drift_stock:
            print(f"        lambda={thr:<10} stock rel_err={err:.3g}   patched rel_err=0")
        print("  Consistent with the layout difference above, not a correctness fault.")

    if bad_patched:
        print("\n  *** CORRECTNESS: the PATCHED arm's output moved on rows that")
        print("  *** elided nothing. With no tiles skipped it must match dense")
        print("  *** exactly. This is a real failure -- do not record the timings.")
        for case, thr, err in bad_patched:
            print(f"        {case} lambda={thr} rel_err={err:g}")
        return 1

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))

#!/usr/bin/env python
"""From results.txt, copy correctly-classified reals (truth=real, pred=real) whose match_score is
below a cutoff into an output folder. These are 'low-confidence correct reals' (model leaned fake
but stayed under threshold). Prints a per-subset breakdown."""
from __future__ import annotations
import argparse
import collections
import os
import re
import shutil

LINE = re.compile(r"^(OK|XX)\s+truth=(\w+)\s+pred=(\w+)\s+type=\S+\s+fake=([0-9.]+)\s+match=([0-9.]+)\s+(.*)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.txt"))
    ap.add_argument("--out", default="real_low")
    ap.add_argument("--match-max", type=float, default=2.0)
    ap.add_argument("--match-min", type=float, default=-1.0)
    ap.add_argument("--truth", default="real", help="ground-truth class: real / pad / fake (fake matches pad+deepfake)")
    ap.add_argument("--pred", default="real", help="predicted class (real/fake)")
    args = ap.parse_args()

    def truth_ok(t):
        return t == args.truth or (args.truth == "fake" and t in ("pad", "deepfake"))

    os.makedirs(args.out, exist_ok=True)
    picked, missing = [], 0
    by_subset = collections.Counter()
    n_match = 0
    for ln in open(args.results):
        m = LINE.match(ln.rstrip("\n"))
        if not m:
            continue
        truth, pred, fake, match, img = m.group(2), m.group(3), float(m.group(4)), float(m.group(5)), m.group(6)
        if truth_ok(truth) and pred == args.pred:
            n_match += 1
            if args.match_min < match < args.match_max:
                picked.append((img, fake, match))
                sub = os.path.basename(img).split("__")[0] if "__" in os.path.basename(img) else truth
                by_subset[sub] += 1

    copied = 0
    for img, fake, match in picked:
        if not os.path.exists(img):
            missing += 1
            continue
        dst = os.path.join(args.out, os.path.basename(img))
        shutil.copy2(img, dst)
        copied += 1

    lo = f"{args.match_min:g}" if args.match_min > -1 else "-inf"
    hi = f"{args.match_max:g}" if args.match_max < 2 else "+inf"
    print(f"truth={args.truth} & pred={args.pred}                        : {n_match}")
    print(f"  of those, {lo} < match_score < {hi}     : {len(picked)} "
          f"({100*len(picked)/max(n_match,1):.1f}%)")
    print(f"copied -> {args.out}/                            : {copied}" + (f"  (missing src: {missing})" if missing else ""))
    if picked:
        fs = sorted(f for _, f, _ in picked)
        ms = sorted(m for _, _, m in picked)
        print(f"  match_score range: {ms[0]:.3f} .. {ms[-1]:.3f}   (fake_score {fs[0]:.3f} .. {fs[-1]:.3f})")
    print("\nper subset:")
    for s in sorted(by_subset):
        print(f"   {s:34s} {by_subset[s]:5d}")


if __name__ == "__main__":
    main()

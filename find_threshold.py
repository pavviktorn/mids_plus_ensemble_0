#!/usr/bin/env python
"""Find the threshold that MAXIMIZES PAD-recall subject to REAL-recall >= floor, from results.txt.

The 'fake' column in results.txt is the raw ensemble P(fake) (threshold-independent), so the whole
PAD/REAL frontier is recoverable without re-running the models. Prints the frontier + the optimal
threshold, and the per-PAD-type / per-REAL-subset recall at that threshold.
"""
from __future__ import annotations
import argparse
import collections
import os
import re

LINE = re.compile(r"^(OK|XX)\s+truth=(\w+)\s+pred=\w+\s+type=\S+\s+fake=([0-9.]+)\s+match=[0-9.]+\s+(.*)$")


def parse(path):
    rows = []  # (truth, fake, subset)
    for ln in open(path):
        m = LINE.match(ln.rstrip("\n"))
        if not m:
            continue
        truth, fake, img = m.group(2), float(m.group(3)), m.group(4)
        base = os.path.basename(img)
        subset = base.split("__")[0] if "__" in base else truth
        rows.append((truth, fake, subset))
    return rows


def recalls(rows, tau):
    real = [f for t, f, _ in rows if t == "real"]
    pad = [f for t, f, _ in rows if t == "pad"]
    rr = sum(f < tau for f in real) / len(real) if real else float("nan")
    pr = sum(f >= tau for f in pad) / len(pad) if pad else float("nan")
    return rr, pr, len(real), len(pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.txt"))
    ap.add_argument("--floor", type=float, default=0.85)
    args = ap.parse_args()
    rows = parse(args.results)
    real = sorted(f for t, f, _ in rows if t == "real")
    pad = [f for t, f, _ in rows if t == "pad"]
    nR, nP = len(real), len(pad)
    print(f"parsed {len(rows)} images: real={nR} pad={nP}\n")

    # sweep a fine grid of candidate thresholds
    grid = sorted(set([i / 10000 for i in range(0, 10001)] + real))
    best = None
    for tau in grid:
        rr = sum(f < tau for f in real) / nR
        if rr >= args.floor:
            pr = sum(f >= tau for f in pad) / nP
            # smallest tau meeting the floor maximizes pad-recall (both monotonic)
            if best is None or pr > best[2] or (pr == best[2] and tau < best[0]):
                best = (tau, rr, pr)
            break  # grid ascending -> first tau meeting floor is the optimum
    tau_star, rr_star, pr_star = best

    print("frontier (tau : real-recall / pad-recall):")
    for fl in [0.80, 0.85, 0.86, 0.88, 0.90, 0.95]:
        # tau achieving real-recall ~= fl  (the fl-quantile of real scores)
        idx = min(nR - 1, max(0, int(round(fl * nR)) - 1))
        tau = real[idx]
        rr, pr, _, _ = recalls(rows, tau + 1e-9)
        print(f"   real>={fl:.2f}:  tau={tau + 1e-9:.4f}  real={rr*100:6.2f}%  pad={pr*100:6.2f}%")

    print(f"\n*** OPTIMAL: tau = {tau_star:.4f}  ->  real-recall = {rr_star*100:.2f}%  "
          f"pad-recall = {pr_star*100:.2f}%  (max PAD s.t. real >= {args.floor*100:.0f}%) ***")

    # per-subset recall at tau_star
    stot, scor, sg = collections.Counter(), collections.Counter(), {}
    for t, f, s in rows:
        sg[s] = t
        stot[s] += 1
        ok = (f >= tau_star) if t == "pad" else (f < tau_star)
        scor[s] += int(ok)
    print(f"\nper-PAD-type recall @ tau={tau_star:.4f}:")
    for s in sorted(x for x in stot if sg[x] == "pad"):
        print(f"   {s:34s} n={stot[s]:5d}  recall={100*scor[s]/stot[s]:6.2f}%")
    print(f"per-REAL-subset recall @ tau={tau_star:.4f}:")
    for s in sorted(x for x in stot if sg[x] == "real"):
        print(f"   {s:34s} n={stot[s]:5d}  recall={100*scor[s]/stot[s]:6.2f}%")
    print(f"\nTAU_STAR={tau_star:.4f}")


if __name__ == "__main__":
    main()

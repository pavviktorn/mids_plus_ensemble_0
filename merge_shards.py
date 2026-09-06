#!/usr/bin/env python
"""Merge sharded --test outputs into one results.txt + a combined PAD/real accuracy summary."""
from __future__ import annotations
import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, help="per-shard result .txt paths (each has a .tally.json)")
    ap.add_argument("--out", default="results.txt")
    a = ap.parse_args()

    g, gc, st, sc = (collections.Counter() for _ in range(4))
    sgroup, tau = {}, None
    for s in a.shards:
        t = json.load(open(s + ".tally.json"))
        tau = t["threshold"]
        for k, v in t["gtot"].items(): g[k] += v
        for k, v in t["gcor"].items(): gc[k] += v
        for k, v in t["stot"].items(): st[k] += v
        for k, v in t["scor"].items(): sc[k] += v
        sgroup.update(t["sgroup"])

    def pct(c, t):
        return 100.0 * c / t if t else float("nan")

    nr, np_, cr, cp = g["real"], g["pad"], gc["real"], gc["pad"]
    L = ["", "=" * 64, f"SUMMARY (threshold={tau}) — merged {len(a.shards)} shards", "=" * 64,
         f"REAL : n={nr:6d}  correct={cr:6d}  real-recall = {pct(cr,nr):6.2f}%",
         f"PAD  : n={np_:6d}  correct={cp:6d}  pad-recall  = {pct(cp,np_):6.2f}%",
         f"OVERALL accuracy (real+pad) = {pct(cr+cp,nr+np_):6.2f}%   (balanced = {(pct(cr,nr)+pct(cp,np_))/2:6.2f}%)",
         "", "-- per PAD attack type (recall = % flagged fake) --"]
    for s in sorted(x for x in st if sgroup.get(x) == "pad"):
        L.append(f"   {s:34s} n={st[s]:5d}  recall={pct(sc[s],st[s]):6.2f}%")
    L += ["", "-- per REAL subset (recall = % kept real) --"]
    for s in sorted(x for x in st if sgroup.get(x) == "real"):
        L.append(f"   {s:34s} n={st[s]:5d}  recall={pct(sc[s],st[s]):6.2f}%")
    block = "\n".join(L)

    with open(a.out, "w") as out:
        out.write(f"# MIDS++ ensemble full test (merged {len(a.shards)} shards) | threshold={tau}\n")
        out.write("# columns: OK/XX  truth  pred  type  fake_score  match_score  image\n")
        for s in a.shards:
            for line in open(s):
                if line[:2] in ("OK", "XX", "ER"):
                    out.write(line)
        out.write(block + "\n")
    print(block)
    print(f"\n[merged -> {a.out} ; {sum(g.values())} images]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Standalone inference for the MIDS++ A1+A2+A3 (9-class) MLLM-free ensemble.

Per image it reports: real/fake decision, forgery type, match score, processing time.

Single / batch:
    python inference.py path/to/face.jpg
    python inference.py img1.jpg img2.png --json
    python inference.py face.jpg --threshold 0.6257      # max-PAD operating point

Full labelled test over a directory (labels inferred from .../real/ vs .../pad/ paths),
writes every image's result to results.txt and prints PAD/real accuracy:
    python inference.py --test --dir .../heldout3/frames --results results.txt
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import time

from mids_ensemble import MidsEnsemble

_HERE = os.path.dirname(os.path.abspath(__file__))


def gather(paths, directory):
    imgs = list(paths)
    if directory:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.JPG", "*.PNG"):
            imgs += sorted(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return imgs


def group_truth_subset(path):
    """Infer (group, truth, subset) from the frame path. Frames are named '<subset>__<id>.jpg'
    with subset starting 'real'/'pad'; fall back to the parent '/real/' or '/pad/' directory."""
    base = os.path.basename(path)
    subset = base.split("__")[0] if "__" in base else None
    if subset and subset.startswith("pad"):
        return "pad", 1, subset
    if subset and subset.startswith("real"):
        return "real", 0, subset
    sep = os.sep
    if f"{sep}pad{sep}" in path:
        return "pad", 1, (subset or "pad")
    if f"{sep}real{sep}" in path:
        return "real", 0, (subset or "real")
    return "unknown", None, (subset or "unknown")


def run_test(eng, images, results_path, threshold, batch_size, dump_tally_path=None):
    tau = eng.threshold if threshold is None else float(threshold)
    n = len(images)
    # tallies
    gtot = collections.Counter()       # group -> n
    gcor = collections.Counter()       # group -> correct
    stot = collections.Counter()       # subset -> n
    scor = collections.Counter()       # subset -> correct
    sgroup = {}                        # subset -> group
    done = t0 = time.time()
    processed = 0

    with open(results_path, "w") as fh:
        fh.write(f"# MIDS++ ensemble full test | threshold={tau} | {n} images\n")
        fh.write("# columns: OK/XX  truth  pred  type  fake_score  match_score  image\n")
        for i in range(0, n, batch_size):
            chunk = images[i:i + batch_size]
            res = eng.predict_batch(chunk, threshold=tau, batch_size=batch_size)
            # predict_batch may reorder errors to the end; map by path
            by_path = {r["image"]: r for r in res}
            for p in chunk:
                r = by_path.get(p, {"error": "missing"})
                group, truth, subset = group_truth_subset(p)
                if "error" in r:
                    fh.write(f"ERR  truth={group:5} pred=----  err={r['error']}  {p}\n")
                    continue
                dec = r["decision"]
                correct = (dec == "fake") == (truth == 1) if truth is not None else False
                gtot[group] += 1; stot[subset] += 1; sgroup[subset] = group
                if correct:
                    gcor[group] += 1; scor[subset] += 1
                fh.write(f"{'OK' if correct else 'XX'}  truth={group:5} pred={dec:4} "
                         f"type={r['forgery_type']:8} fake={r['forgery_score']:.4f} "
                         f"match={r['match_score']:.4f}  {p}\n")
            processed += len(chunk)
            if time.time() - done > 10:
                done = time.time()
                rate = processed / max(time.time() - t0, 1e-9)
                print(f"  [{processed}/{n}] {rate:.0f} img/s", flush=True)
            fh.flush()

        # ---- summary ----
        def pct(c, t):
            return 100.0 * c / t if t else float("nan")
        nr, np_ = gtot["real"], gtot["pad"]
        cr, cp = gcor["real"], gcor["pad"]
        overall = pct(cr + cp, nr + np_)
        lines = []
        lines.append("\n" + "=" * 64)
        lines.append(f"SUMMARY (threshold={tau})")
        lines.append("=" * 64)
        lines.append(f"REAL : n={nr:6d}  correct={cr:6d}  real-recall = {pct(cr,nr):6.2f}%")
        lines.append(f"PAD  : n={np_:6d}  correct={cp:6d}  pad-recall  = {pct(cp,np_):6.2f}%")
        lines.append(f"OVERALL accuracy (real+pad) = {overall:6.2f}%   (balanced = {(pct(cr,nr)+pct(cp,np_))/2:6.2f}%)")
        lines.append("\n-- per PAD attack type (recall = % flagged fake) --")
        for s in sorted([s for s in stot if sgroup[s] == "pad"]):
            lines.append(f"   {s:34s} n={stot[s]:5d}  recall={pct(scor[s],stot[s]):6.2f}%")
        lines.append("\n-- per REAL subset (recall = % kept real) --")
        for s in sorted([s for s in stot if sgroup[s] == "real"]):
            lines.append(f"   {s:34s} n={stot[s]:5d}  recall={pct(scor[s],stot[s]):6.2f}%")
        block = "\n".join(lines)
        if dump_tally_path is None:
            fh.write(block + "\n")
    tallies = {"gtot": dict(gtot), "gcor": dict(gcor), "stot": dict(stot),
               "scor": dict(scor), "sgroup": sgroup, "threshold": tau}
    if dump_tally_path is not None:
        json.dump(tallies, open(dump_tally_path, "w"))
        print(f"[shard done -> {results_path}]  real {gcor['real']}/{gtot['real']}  pad {gcor['pad']}/{gtot['pad']}")
    else:
        print(block)
        print(f"\n[wrote per-image results + summary -> {results_path}]")
    return tallies


def main():
    ap = argparse.ArgumentParser(description="MIDS++ ensemble face real/fake inference")
    ap.add_argument("images", nargs="*", help="image path(s)")
    ap.add_argument("--dir", default="/datasets/work/vLLM/temp/mids_plus/runs/real/heldout3/frames",
                    help="directory of images (recursive)")
    ap.add_argument("--config", default=os.path.join(_HERE, "config.json"))
    ap.add_argument("--device", default=None, help="cuda | cpu | cuda:0 (default: auto)")
    ap.add_argument("--threshold", type=float, default=None, help="override decision threshold")
    ap.add_argument("--json", action="store_true", help="print one JSON object per image")
    ap.add_argument("--test", action="store_true",
                    help="labelled full test over --dir (real/ vs pad/): write results.txt + accuracy")
    ap.add_argument("--results", default=os.path.join(_HERE, "results.txt"), help="results file for --test")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--shard", default=None, help="i/n : process only shard i of n (1-indexed) for multi-GPU runs")
    args = ap.parse_args()

    if args.test:
        images = gather(args.images, args.dir)
        if not images:
            print(f"No images under {args.dir}", file=sys.stderr); sys.exit(2)
        dump_tally = None
        if args.shard:
            si, sn = (int(x) for x in args.shard.split("/"))
            images = images[si - 1::sn]               # strided -> balanced shards
            dump_tally = args.results + ".tally.json"
        eng = MidsEnsemble(config_path=args.config, device=args.device)
        tau = args.threshold if args.threshold is not None else eng.threshold
        print(f"[full test{(' shard ' + args.shard) if args.shard else ''}] {len(images)} images on "
              f"{eng.device} | threshold={tau} | models loaded in {eng.load_time_sec}s", flush=True)
        run_test(eng, images, args.results, args.threshold, args.batch_size, dump_tally_path=dump_tally)
        return

    images = gather(args.images, args.dir)
    if not images:
        print("No images given. Pass image path(s) or --dir FOLDER.", file=sys.stderr)
        sys.exit(2)
    eng = MidsEnsemble(config_path=args.config, device=args.device)
    if not args.json:
        print(f"[loaded {len(eng.models)} models on {eng.device} in {eng.load_time_sec}s | "
              f"threshold={args.threshold if args.threshold is not None else eng.threshold}]")
    for path in images:
        try:
            r = eng.predict(path, threshold=args.threshold)
        except Exception as e:
            print(json.dumps({"image": path, "error": str(e)}) if args.json else f"\n{path}\n  ERROR: {e}")
            continue
        if args.json:
            print(json.dumps(r))
        else:
            tp = r["type_probs"]
            print(f"\n{path}")
            print(f"  decision        : {r['decision'].upper()}")
            print(f"  forgery_type    : {r['forgery_type']}")
            print(f"  match_score     : {r['match_score']:.4f}   (confidence in the decision)")
            print(f"  forgery_score   : {r['forgery_score']:.4f}   (P(fake); threshold {r['threshold']})")
            print(f"  type_probs      : real {tp['real']:.3f} | pad {tp['pad']:.3f} | deepfake {tp['deepfake']:.3f}")
            print(f"  processing_time : {r['processing_time_sec']:.4f} s")


if __name__ == "__main__":
    main()

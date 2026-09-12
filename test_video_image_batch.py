#!/usr/bin/env python3
"""Batch-test every image AND video under a folder with the standalone MIDS++ ensemble.

Ensemble counterpart of the FFAA `test_video_image_batch.py`: same function (recurse a tree, infer
each still image and each video frame, score real-vs-fake against the path label, copy misses, write
sidecar outputs, print a per-label accuracy summary) but driven by `MidsEnsemble` (MLLM-free,
frame-based) instead of LLaVA+MIDS.

Differences from the FFAA version (kept deliberately):
  * No MLLM generation -> far faster; one ensemble forward batch per chunk.
  * Multi-GPU by FILE SHARDING (one process per device, round-robin over media files) instead of the
    producer/queue/worker/writer pipeline -- simpler, and since shards are disjoint there is no
    output-file contention.
  * Outputs use a distinct `.ensemble` suffix and a separate miss dir, so they sit beside (never
    overwrite) the FFAA `.txt` / `.frames.jsonl` outputs on the same dataset.

Ground truth: a path component named `real` or `fake`. Prediction in {real, fake, ambiguous}
(fake = PAD or deepfake). For accuracy, `ambiguous` is counted as `fake` (so an ambiguous on a
fake item is correct, and an ambiguous on a real item is a miss).

Outputs:
  * still image  -> `<image>.ensemble.txt`               (pretty JSON of the full result)
  * video        -> `<video>.ensemble.frames.jsonl`      (one JSON record per frame)
  * misclassified -> copied (re-encoded JPEG) to `<miss-dir>/<relative-parent>/<name>.jpg`
  * EVERY ambiguous prediction is also copied there with `_ambiguous` before the extension --
    including ambiguous FAKE items, which are counted correct (ambiguous=fake) but still saved
    for review.

Example:
    python test_video_image_batch.py --input-dir /datasets/work/vLLM/data/axonlabs_data --devices all
    python test_video_image_batch.py --device 0 --batch-size 64 --threshold 0.6337 --frame-stride 5
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

# quiet the encoders before transformers loads (via MidsEnsemble)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_INPUT_DIR = "/datasets/work/vLLM/data/axonlabs_data_1"
DEFAULT_MISS_DIR = "/datasets/work/vLLM/data/miss_axonlabs_data_ensemble"
DEFAULT_CONFIG = os.path.join(_HERE, "config.json")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
LABELS = ("real", "fake")
JPEG_QUALITY = 95
OUT_INFIX = ".ensemble"

FORGERY_LABEL = {"real": "None", "pad": "PAD (presentation attack / spoof)", "deepfake": "Deepfake"}


# ------------------------------------------------------------- real face-quality filter
class FaceQualityFilter:
    """Mirror of build_real_filtered.py `_passes`: keep only frames with ONE dominant, frontal,
    proper-size, unoccluded face. insightface buffalo_l (SCRFD detection + 3D-68 landmark/pose) on CPU.
    Used to skip low-quality REAL frames (heavy head pose / wrong face size); never applied to fakes."""

    def __init__(self, score, pose, hmin, hmax, minpx, maxside, det_size=640):
        from insightface.app import FaceAnalysis
        self.score, self.pose = float(score), float(pose)
        self.hmin, self.hmax = float(hmin), float(hmax)
        self.minpx, self.maxside = int(minpx), int(maxside)
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "landmark_3d_68"],
                                providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    def _downscale(self, bgr):
        import cv2
        h, w = bgr.shape[:2]
        sc = self.maxside / max(h, w)
        return cv2.resize(bgr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else bgr

    def passes(self, bgr):
        """Return (keep: bool, reason: str). reason is the first failing check (or 'ok')."""
        import numpy as np
        img = self._downscale(bgr)
        h = img.shape[0]
        faces = self.app.get(img)
        if not faces:
            return False, "noface"
        faces.sort(key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        f = faces[0]
        x1, y1, x2, y2 = f.bbox
        fw, fh = x2 - x1, y2 - y1
        area = fw * fh
        if len(faces) > 1:
            f2 = faces[1]
            if (f2.bbox[2] - f2.bbox[0]) * (f2.bbox[3] - f2.bbox[1]) > 0.5 * area:
                return False, "multiface"
        if f.det_score < self.score:
            return False, "lowdet"
        if not (self.hmin <= fh / h <= self.hmax):
            return False, "size"
        if min(fw, fh) < self.minpx:
            return False, "minpx"
        pose = getattr(f, "pose", None)
        if pose is None or float(np.max(np.abs(pose))) > self.pose:
            return False, "pose"
        return True, "ok"


def subset_for(source: Path) -> str:
    """Per-subset key for the summary: the folder right under the real/fake label dir, else the label."""
    i = get_label_index(source)
    if i is not None and i + 1 < len(source.parts) - 1:
        return source.parts[i + 1]
    return true_label_from_path(source) or "unknown"


# ----------------------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch-test images and videos with the MIDS++ ensemble.")
    p.add_argument("--input-dir", "--input_dir", dest="input_dir", default=DEFAULT_INPUT_DIR,
                   help=f"Folder to scan (recursive). Default: {DEFAULT_INPUT_DIR}")
    p.add_argument("--miss-dir", "--miss_dir", dest="miss_dir", default=DEFAULT_MISS_DIR,
                   help=f"Folder to copy misclassified items into. Default: {DEFAULT_MISS_DIR}")
    p.add_argument("--devices", default="all", help="CUDA devices: 'all' or e.g. '0,1,2,3'. Default: all")
    p.add_argument("--device", type=int, default=None, help="Single-GPU alias; overrides --devices.")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="Ensemble config.json path.")
    p.add_argument("--threshold", type=float, default=None, help="Decision threshold (default: config).")
    p.add_argument("--real-ambiguous-match-min", "--real_ambiguous_match_min",
                   dest="real_ambiguous_match_min", type=float, default=0.9,
                   help="A 'real' with match_score below this becomes 'ambiguous'. 0 = pure binary "
                        "(set 0.9 to mirror the API).")
    p.add_argument("--ambiguous-margin", "--ambiguous_margin", dest="ambiguous_margin",
                   type=float, default=0.0, help="Symmetric |fake-score - tau| band -> 'ambiguous'.")
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64,
                   help="Images/frames per ensemble forward batch. Default: 64")
    p.add_argument("--frame-stride", "--frame_stride", dest="frame_stride", type=int, default=1,
                   help="Process every Nth video frame (1 = every frame, like FFAA). Default: 1")
    p.add_argument("--skip-existing", type=int, default=1,
                   help="1: skip items whose .ensemble output already exists and is non-empty. 0: redo.")
    p.add_argument("--print-ok", action="store_true", help="Print correctly-classified items too.")
    p.add_argument("--progress-interval", type=float, default=10.0,
                   help="Seconds between per-GPU progress lines (0 disables).")
    p.add_argument("--results", default=os.path.join(_HERE, "results_ensemble.txt"),
                   help="Consolidated per-item results file (results.txt style). "
                        "Default: results_ensemble.txt next to this script.")
    # --- real face-quality filter (heavy pose / wrong face size), mirrors build_real_filtered.py ---
    p.add_argument("--filter-real", "--filter_real", dest="filter_real", type=int, default=1,
                   help="1: skip low-quality REAL images/frames (non-frontal / wrong face size / "
                        "occluded / no single dominant face) and EXCLUDE them from accuracy. "
                        "0: keep all. PAD/fake items are never filtered.")
    p.add_argument("--face-score", dest="face_score", type=float, default=0.65,
                   help="Min detector confidence (occlusion/quality proxy).")
    p.add_argument("--face-pose", dest="face_pose", type=float, default=28.0,
                   help="Max |pitch|,|yaw|,|roll| degrees (frontal).")
    p.add_argument("--face-hmin", dest="face_hmin", type=float, default=0.15,
                   help="Min face-height fraction of the frame.")
    p.add_argument("--face-hmax", dest="face_hmax", type=float, default=0.85,
                   help="Max face-height fraction of the frame.")
    p.add_argument("--face-minpx", dest="face_minpx", type=int, default=80,
                   help="Min face side in pixels (on the downscaled frame).")
    p.add_argument("--face-maxside", dest="face_maxside", type=int, default=1280,
                   help="Downscale the long side to this before face detection.")
    return p.parse_args()


def parse_devices(args: argparse.Namespace) -> list[int]:
    import torch
    if args.device is not None:
        return [int(args.device)]
    n = torch.cuda.device_count()
    if args.devices.strip().lower() == "all":
        return list(range(n))
    return [int(x) for x in args.devices.split(",") if x.strip()]


# ----------------------------------------------------------------------------- discovery / paths
def iter_media_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in (IMAGE_EXTENSIONS | VIDEO_EXTENSIONS):
            yield path


def get_label_index(path: Path) -> Optional[int]:
    for i, part in enumerate(path.parts[:-1]):
        if part.lower() in ("real", "fake"):
            return i
    return None


def true_label_from_path(path: Path) -> Optional[str]:
    i = get_label_index(path)
    return path.parts[i].lower() if i is not None else None


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent); return True
    except ValueError:
        return False


def output_path_for(source: Path) -> Path:
    if source.suffix.lower() in IMAGE_EXTENSIONS:
        return source.with_name(source.stem + OUT_INFIX + ".txt")
    return Path(f"{source}{OUT_INFIX}.frames.jsonl")


def has_nonempty_output(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def relative_parent_for(source: Path, input_dir: Path) -> Path:
    try:
        return source.relative_to(input_dir).parent
    except ValueError:
        i = get_label_index(source)
        return Path(*source.parts[i:-1]) if i is not None else source.parent


def infer_source_name(source: Path, input_dir: Path) -> str:
    try:
        parts = source.relative_to(input_dir).parent.parts
    except ValueError:
        parts = source.parent.parts
    for part in reversed(parts):
        if part.lower() not in LABELS:
            return part
    return source.stem


def unique_path(candidate: Path) -> Path:
    if not candidate.exists():
        return candidate
    stem, suffix, i = candidate.stem, candidate.suffix, 1
    while True:
        nxt = candidate.with_name(f"{stem}_{i}{suffix}")
        if not nxt.exists():
            return nxt
        i += 1


# ----------------------------------------------------------------------------- stats helpers
def build_counts() -> dict[str, int]:
    return {k: 0 for k in (
        "total_items", "still_images_processed", "video_frames_extracted",
        "correct", "incorrect", "failed", "miss_saved", "failed_sources", "skipped_sources",
        "real_skipped_quality")}


def build_label_stats() -> dict[str, dict[str, int]]:
    return {lab: {k: 0 for k in ("found", "processed", "evaluated", "correct", "miss", "failed",
                                 "skipped_quality")}
            for lab in LABELS}


def merge_counts(dst: dict[str, int], src: dict[str, int]) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + int(v)


def merge_label_stats(dst: dict, src: dict) -> None:
    for lab, st in src.items():
        for k, v in st.items():
            dst[lab][k] = dst[lab].get(k, 0) + int(v)


def normalize_prediction(decision: str) -> str:
    """Map a raw analysis result to {real, fake, ambiguous}. Both 'ambiguous' and 'likely_fake'
    (borderline labels, as produced by FFAA's finalize_best_answer) are treated as 'ambiguous' ->
    counted as fake for accuracy and copied to the miss dir with '_ambiguous' in the filename."""
    d = (decision or "").strip().lower()
    if d == "real":
        return "real"
    if d in ("ambiguous", "likely_fake"):
        return "ambiguous"
    return "fake"


# ----------------------------------------------------------------------------- result formatting
_REASON = {
    "real": "Consistent facial texture, lighting and natural 3D depth; no manipulation or PAD cues.",
    "pad": "Presentation-attack cues (screen/print recapture, mask/silicone, paper/textile, makeup).",
    "deepfake": "Face-manipulation cues (blending artifacts, inconsistent lighting/integration).",
}


def to_result(raw: dict, tau: float, margin: float, real_amb_min: float) -> tuple[dict, str, str]:
    """Build the full result dict + (decision, match_score_str), applying the ambiguity rules."""
    ff = float(raw["forgery_score"])
    tp = raw["type_probs"]
    decision = "fake" if ff >= tau else "real"
    ftype = "real" if decision == "real" else ("pad" if tp["pad"] >= tp["deepfake"] else "deepfake")
    match = ff if decision == "fake" else 1.0 - ff
    result = decision
    if decision == "real" and match < real_amb_min:
        result = "ambiguous"
    elif margin > 0 and abs(ff - tau) <= margin:
        result = "ambiguous"
    sides = {(float(v) >= tau) for v in raw.get("per_model_fake", {}).values()}
    difficulty = "easy" if len(sides) <= 1 else "hard"

    face_liveness = {
        "Analysis result": result,
        "Forgery type": FORGERY_LABEL[ftype] if result != "real" else "None",
        "Match score": f"{match:.4f}",
        "Forgery score": f"{ff:.4f}",
        "Type probabilities": {k: round(float(v), 4) for k, v in tp.items()},
        "Difficulty": difficulty,
        "Forgery reasoning": _REASON["real"] if result == "real" else _REASON[ftype],
        "Model": "MIDS++ A1+A2+A3 (9-class) MLLM-free ensemble",
        "Threshold": round(tau, 4),
    }
    full = {
        "success": True,
        "decision": result,
        "forgery_type": ftype,
        "match_score": round(match, 4),
        "forgery_score": round(ff, 4),
        "processing_time_sec": raw.get("processing_time_sec"),
        "face_liveness": face_liveness,
        "details": {"type_probs": face_liveness["Type probabilities"],
                    "per_model_fake": raw.get("per_model_fake", {}),
                    "difficulty": difficulty, "threshold": round(tau, 4)},
    }
    return full, result, f"{match:.4f}"


# ----------------------------------------------------------------------------- GPU worker (one per device)
def gpu_worker(device_id: int, media_paths: list[str], args_dict: dict, n_devices: int, result_q) -> None:
    import cv2
    import torch
    from mids_ensemble import MidsEnsemble

    args = argparse.Namespace(**args_dict)
    input_dir = Path(args.input_dir)
    miss_dir = Path(args.miss_dir)
    counts = build_counts()
    label_stats = build_label_stats()
    subtally: dict[str, dict] = {}
    res_path = f"{args.results}.shard{device_id}"
    os.makedirs(os.path.dirname(os.path.abspath(res_path)) or ".", exist_ok=True)
    res_fh = open(res_path, "w", encoding="utf-8")

    def bump(subset, group, evaluated=False, correct=False, skipped=False):
        s = subtally.setdefault(subset, {"group": group, "n": 0, "correct": 0, "skipped": 0})
        if skipped:
            s["skipped"] += 1
        if evaluated:
            s["n"] += 1
        if correct:
            s["correct"] += 1

    def res_pred(tag, lab, pred, ftype, ff, ms, path):    # results.txt-style scored line
        res_fh.write(f"{tag}  truth={lab:5} pred={pred:5} type={ftype:9} "
                     f"fake={ff:.4f} match={ms:.4f}  {path}\n")

    def res_note(tag, lab, pred, info, path):             # SK (skipped) / ER (error) line
        res_fh.write(f"{tag}  truth={lab:5} pred={pred:5} type={info:9} "
                     f"fake=------ match=------  {path}\n")

    # avoid CPU oversubscription across the per-GPU processes during the SVD-reconstruction load
    try:
        torch.set_num_threads(max(2, (os.cpu_count() or 8) // max(1, n_devices)))
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    print(f"[GPU {device_id}] loading ensemble ({len(media_paths)} media files in shard) ...", flush=True)
    eng = MidsEnsemble(config_path=args.config, device=f"cuda:{device_id}")
    tau = eng.threshold if args.threshold is None else float(args.threshold)
    print(f"[GPU {device_id}] ready on {eng.device} in {eng.load_time_sec}s | threshold={tau}", flush=True)

    filt = None
    if args.filter_real:
        filt = FaceQualityFilter(args.face_score, args.face_pose, args.face_hmin,
                                 args.face_hmax, args.face_minpx, args.face_maxside)
        print(f"[GPU {device_id}] real face-quality filter ON "
              f"(score>={args.face_score}, pose<={args.face_pose}, "
              f"hfrac[{args.face_hmin},{args.face_hmax}], minpx={args.face_minpx})", flush=True)

    def encode_jpeg(rgb):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            raise RuntimeError("failed to JPEG-encode")
        return buf.tobytes()

    def save_miss(item, rgb) -> Optional[str]:
        try:
            dst = unique_path(miss_dir / Path(item["relative_parent"]) / item["miss_filename"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(encode_jpeg(rgb))
            return str(dst)
        except Exception as exc:
            print(f"[MISS-SAVE-FAIL] {item['display_path']} error={exc}", file=sys.stderr)
            return None

    buffer: list[dict] = []
    progress_t0 = time.time()
    last_report = time.time()

    def write_skip(item, reason):
        """A low-quality REAL frame: record it, but EXCLUDE it from accuracy (not evaluated)."""
        lab = item["true_label"]
        counts["real_skipped_quality"] += 1
        label_stats[lab]["skipped_quality"] += 1
        bump(item["subset"], lab, skipped=True)
        res_note("SK", lab, "skip", reason, item["display_path"])
        out = Path(item["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        rec = {"success": False, "skipped": "low_quality_real", "reason": reason,
               "source": item["source_path"]}
        if item["kind"] == "image":
            out.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            rec["frame_index"] = item["frame_index"]
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if args.print_ok:
            print(f"[SK] gpu={device_id} {item['display_path']} reason={reason}", flush=True)

    def handle(item, raw):
        nonlocal last_report
        lab = item["true_label"]
        counts["total_items"] += 1
        label_stats[lab]["found"] += 1
        label_stats[lab]["processed"] += 1
        counts["still_images_processed" if item["kind"] == "image" else "video_frames_extracted"] += 1

        if raw is None or "error" in raw:
            counts["failed"] += 1
            label_stats[lab]["failed"] += 1
            err = (raw or {}).get("error", "inference failed")
            write_output(item, {"success": False, "error": err, "source": item["source_path"]},
                         None, "fake", None, None)
            res_note("ER", lab, "----", "error", item["display_path"])
            print(f"[FAIL] gpu={device_id} {item['display_path']} error={err} true={lab}", file=sys.stderr)
            return

        full, decision, match_str = to_result(raw, tau, args.ambiguous_margin, args.real_ambiguous_match_min)
        predicted = normalize_prediction(decision)
        pred_eval = "fake" if predicted in ("fake", "ambiguous") else "real"   # ambiguous counts as FAKE
        is_ambiguous = predicted == "ambiguous"
        label_stats[lab]["evaluated"] += 1
        correct = pred_eval == lab
        if correct:
            counts["correct"] += 1
            label_stats[lab]["correct"] += 1
            tag, state = "OK", "OK"
        else:
            counts["incorrect"] += 1
            label_stats[lab]["miss"] += 1
            tag, state = "XX", "MISS"
        # Copy to the miss dir if misclassified OR ambiguous (ambiguous saved for review even when
        # counted correct, e.g. an ambiguous fake). Ambiguous copies get `_ambiguous` in the filename.
        miss_copy = None
        if (not correct) or is_ambiguous:
            mf = item["miss_filename"]
            if is_ambiguous:
                stem, dot, ext = mf.rpartition(".")
                mf = f"{stem}_ambiguous.{ext}" if dot else f"{mf}_ambiguous"
            miss_copy = save_miss({**item, "miss_filename": mf}, item["rgb"])
            if miss_copy:
                counts["miss_saved"] += 1
        bump(item["subset"], lab, evaluated=True, correct=correct)
        res_pred(tag, lab, predicted, full["forgery_type"],
                 full["forgery_score"], full["match_score"], item["display_path"])
        write_output(item, full, decision, predicted, match_str, miss_copy)
        if state != "OK" or args.print_ok:
            extra = f" miss_copy={miss_copy}" if miss_copy else ""
            print(f"[{state}] gpu={device_id} {item['display_path']} true={lab} "
                  f"pred={predicted} type={full['forgery_type']} match={match_str}{extra}", flush=True)

    def write_output(item, full, decision, predicted, match_str, miss_copy):
        out = Path(item["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        if item["kind"] == "image":
            out.write_text(json.dumps(full, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return
        record = {
            "source_video": item["source_path"], "frame_index": item["frame_index"],
            "frame_name": item["miss_filename"], "true_label": item["true_label"],
            "analysis_result": decision, "predicted_label": predicted,
            "match_score": match_str, "forgery_type": full.get("forgery_type"),
            "forgery_score": full.get("forgery_score"), "miss_copy": miss_copy, "response": full,
        }
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def flush():
        nonlocal last_report
        if not buffer:
            return
        raws = eng.predict_rgb_batch([(i, it["rgb"]) for i, it in enumerate(buffer)],
                                     threshold=tau, batch_size=args.batch_size)
        by_key = {r["image"]: r for r in raws}
        for i, it in enumerate(buffer):
            handle(it, by_key.get(i))
            it["rgb"] = None
        buffer.clear()
        if args.progress_interval > 0 and time.time() - last_report >= args.progress_interval:
            rate = counts["total_items"] / max(time.time() - progress_t0, 1e-9)
            print(f"[GPU {device_id}] {counts['total_items']} items  {rate:.0f}/s  "
                  f"correct={counts['correct']} miss={counts['incorrect']} fail={counts['failed']}",
                  flush=True)
            last_report = time.time()

    try:
        for sp in media_paths:
            source = Path(sp)
            lab = true_label_from_path(source)
            if lab is None:
                counts["skipped_sources"] += 1
                print(f"[SKIP] {source} (no /real/ or /fake/ in path)")
                continue
            out_path = output_path_for(source)
            if args.skip_existing == 1 and has_nonempty_output(out_path):
                counts["skipped_sources"] += 1
                continue

            rel_parent = str(relative_parent_for(source, input_dir))
            src_name = infer_source_name(source, input_dir)
            subset = f"{lab}/{subset_for(source)}"   # label-prefixed so real/ and fake/ never collide

            if source.suffix.lower() in IMAGE_EXTENSIONS:
                bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
                if bgr is None:
                    counts["failed"] += 1
                    counts["total_items"] += 1
                    label_stats[lab]["found"] += 1; label_stats[lab]["processed"] += 1
                    label_stats[lab]["failed"] += 1
                    counts["still_images_processed"] += 1
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(
                        {"success": False, "error": "failed to decode image", "source": str(source)},
                        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    res_note("ER", lab, "----", "decodefail", str(source))
                    print(f"[FAIL] gpu={device_id} {source} error=failed to decode", file=sys.stderr)
                    continue
                item = {
                    "kind": "image", "source_path": str(source), "display_path": str(source),
                    "true_label": lab, "output_path": str(out_path), "relative_parent": rel_parent,
                    "subset": subset, "miss_filename": f"{src_name}_{source.stem}.jpg"}
                if filt is not None and lab == "real":
                    keep, reason = filt.passes(bgr)
                    if not keep:
                        write_skip(item, reason)
                        continue
                item["rgb"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                buffer.append(item)
                if len(buffer) >= args.batch_size:
                    flush()
                continue

            # video: truncate its jsonl, then stream frames
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("", encoding="utf-8")
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                counts["failed_sources"] += 1
                with out_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"source_video": str(source),
                                         "error": "failed to open video", "success": False}) + "\n")
                print(f"[FAIL-VIDEO] {source} error=failed to open", file=sys.stderr)
                continue
            fidx = 0
            video_name = source.stem
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                fidx += 1
                if args.frame_stride > 1 and (fidx - 1) % args.frame_stride != 0:
                    continue
                item = {
                    "kind": "video_frame", "source_path": str(source),
                    "display_path": f"{source}#frame={fidx:06d}", "true_label": lab,
                    "output_path": str(out_path), "frame_index": fidx, "relative_parent": rel_parent,
                    "subset": subset, "miss_filename": f"{video_name}_{fidx:06d}.jpg"}
                if filt is not None and lab == "real":
                    keep, reason = filt.passes(frame)
                    if not keep:
                        write_skip(item, reason)
                        continue
                item["rgb"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                buffer.append(item)
                if len(buffer) >= args.batch_size:
                    flush()
            cap.release()
            if fidx == 0:
                counts["failed_sources"] += 1
                with out_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"source_video": str(source),
                                         "error": "no frames extracted", "success": False}) + "\n")
        flush()
    except Exception as exc:
        print(f"[WORKER-ERROR] gpu={device_id} {exc!r}", file=sys.stderr)
    finally:
        try:
            res_fh.close()
        except Exception:
            pass
        result_q.put({"device_id": device_id, "counts": counts, "label_stats": label_stats,
                      "subtally": subtally, "res_path": res_path})


# ----------------------------------------------------------------------------- summary
def format_summary(args, input_dir, n_images, n_videos, counts, label_stats, subtally) -> str:
    def pct(c, t):
        return 100.0 * c / t if t else float("nan")
    tau_disp = args.threshold if args.threshold is not None else "config-default"
    real, fake = label_stats["real"], label_stats["fake"]
    ne_r, cr = real["evaluated"], real["correct"]
    ne_f, cf = fake["evaluated"], fake["correct"]
    L = ["", "=" * 64,
         f"SUMMARY (threshold={tau_disp}) — MIDS++ ensemble batch test",
         "=" * 64,
         f"REAL : n={ne_r:6d}  correct={cr:6d}  real-recall = {pct(cr, ne_r):6.2f}%   "
         f"(low-quality reals skipped & EXCLUDED: {counts['real_skipped_quality']})",
         f"FAKE : n={ne_f:6d}  correct={cf:6d}  fake-recall = {pct(cf, ne_f):6.2f}%",
         f"OVERALL accuracy (real+fake) = {pct(cr + cf, ne_r + ne_f):6.2f}%   "
         f"(balanced = {(pct(cr, ne_r) + pct(cf, ne_f)) / 2:6.2f}%)",
         "",
         f"Input: {input_dir}",
         f"images={n_images} videos={n_videos} | still={counts['still_images_processed']} "
         f"frames={counts['video_frames_extracted']} failed={counts['failed']} "
         f"failed-videos={counts['failed_sources']} skipped-sources={counts['skipped_sources']} "
         f"miss-saved={counts['miss_saved']}",
         "",
         "-- per REAL subset (recall = % kept real; low-quality excluded) --"]
    for s in sorted(x for x in subtally if subtally[x]["group"] == "real"):
        t = subtally[s]
        L.append(f"   {s:34s} n={t['n']:6d}  recall={pct(t['correct'], t['n']):6.2f}%  "
                 f"skipped={t['skipped']}")
    L.append("")
    L.append("-- per FAKE subset (recall = % flagged fake) --")
    for s in sorted(x for x in subtally if subtally[x]["group"] == "fake"):
        t = subtally[s]
        L.append(f"   {s:34s} n={t['n']:6d}  recall={pct(t['correct'], t['n']):6.2f}%")
    return "\n".join(L)


def main() -> int:
    import torch
    args = parse_args()
    if args.batch_size < 1:
        print("--batch-size must be >= 1", file=sys.stderr); return 1
    if not torch.cuda.is_available():
        print("CUDA is required.", file=sys.stderr); return 1

    devices = parse_devices(args)
    n = torch.cuda.device_count()
    bad = [d for d in devices if d < 0 or d >= n]
    if not devices or bad:
        print(f"Invalid devices {bad or devices}; visible count {n}.", file=sys.stderr); return 1

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        print(f"Input folder does not exist: {input_dir}", file=sys.stderr); return 1
    miss_dir = Path(args.miss_dir).expanduser().resolve()

    media = [p for p in iter_media_files(input_dir) if not is_relative_to(p, miss_dir)]
    if not media:
        print(f"No image/video files under {input_dir}"); return 0
    n_images = sum(1 for p in media if p.suffix.lower() in IMAGE_EXTENSIONS)
    n_videos = sum(1 for p in media if p.suffix.lower() in VIDEO_EXTENSIONS)
    print(f"source_image_count:{n_images}")
    print(f"source_video_count:{n_videos}")
    print(f"[INFO] {len(devices)} GPU worker(s): {','.join(map(str, devices))}")

    shards = [[str(p) for p in media[i::len(devices)]] for i in range(len(devices))]
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    procs = []
    for dev, shard in zip(devices, shards):
        proc = ctx.Process(target=gpu_worker, args=(dev, shard, vars(args), len(devices), result_q))
        proc.start()
        procs.append(proc)

    counts = build_counts()
    label_stats = build_label_stats()
    subtally: dict[str, dict] = {}
    shard_files: list[str] = []
    received = 0
    while received < len(procs):
        msg = result_q.get()
        merge_counts(counts, msg["counts"])
        merge_label_stats(label_stats, msg["label_stats"])
        for k, v in msg.get("subtally", {}).items():
            s = subtally.setdefault(k, {"group": v["group"], "n": 0, "correct": 0, "skipped": 0})
            s["n"] += v["n"]; s["correct"] += v["correct"]; s["skipped"] += v["skipped"]
        if msg.get("res_path"):
            shard_files.append((msg["device_id"], msg["res_path"]))
        received += 1
        print(f"[GPU {msg['device_id']}-DONE] eval={msg['counts']['correct'] + msg['counts']['incorrect']} "
              f"correct={msg['counts']['correct']} miss={msg['counts']['incorrect']} "
              f"skipped-real={msg['counts']['real_skipped_quality']} fail={msg['counts']['failed']}")
    for proc in procs:
        proc.join()

    summary = format_summary(args, input_dir, n_images, n_videos, counts, label_stats, subtally)

    # assemble the consolidated results file (results.txt style): header + per-item lines + summary
    res_out = Path(args.results)
    res_out.parent.mkdir(parents=True, exist_ok=True)
    with open(res_out, "w", encoding="utf-8") as out:
        out.write(f"# MIDS++ ensemble batch test | threshold="
                  f"{args.threshold if args.threshold is not None else 'config-default'} "
                  f"| filter_real={bool(args.filter_real)} | input={input_dir}\n")
        out.write("# columns: OK/XX/SK/ER  truth  pred  type  fake_score  match_score  image"
                  "   (SK = low-quality real, excluded from accuracy)\n")
        for _dev, sf in sorted(shard_files):
            if os.path.exists(sf):
                with open(sf, encoding="utf-8") as fh:
                    out.write(fh.read())
        out.write(summary + "\n")
    for _dev, sf in shard_files:
        try:
            os.remove(sf)
        except OSError:
            pass

    print(summary)
    print(f"\n[wrote consolidated per-item results + summary -> {res_out}]")
    if counts["failed"] or counts["failed_sources"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

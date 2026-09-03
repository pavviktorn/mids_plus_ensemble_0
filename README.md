# mids_plus_ensemble_0

Standalone, self-contained face **real / fake** detector — an ensemble of three MIDS++ **9-class**
models (A1 +SVD, A2 +SVD+GenD, A3 full MIDS++), run **MLLM-free** (fixed candidate answers) and
**frame-based** (no temporal aggregation), on the **whole image** (letterbox, never crops the face).

It needs nothing from the development tree: model code, the three checkpoints, and the base
CLIP + T5 encoders are all bundled here.

## Output
For an input image, `inference.py` prints:
- **decision** — `real` / `fake`
- **forgery_type** — `real` / `pad` / `deepfake` (the predicted attack family when fake; PAD includes makeup)
- **match_score** — confidence in the decision (∈ [0,1])
- **processing_time** — seconds for the forward pass
- (extras) `forgery_score` = P(fake), per-type probabilities, per-model fake scores

## Usage
```bash
# one-time: create an env
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

# single image
python inference.py /path/to/face.jpg

# many images / a folder / machine-readable
python inference.py a.jpg b.png --json
python inference.py --dir /path/to/folder --json

# pick a different operating point (see config.json -> threshold_options)
python inference.py face.jpg --threshold 0.6257     # max-PAD @ real>=85
python inference.py face.jpg --device cpu
```

Programmatic:
```python
from mids_ensemble import MidsEnsemble
eng = MidsEnsemble("config.json")          # device auto (cuda if available)
print(eng.predict("face.jpg"))
```

## How the decision is made
1. The full image (letterbox→336, CLIP-normalized, **no crop**) and 3 fixed candidate answers are fed
   to each model → per-model softmax over 9 classes (`label = 3*true + claim`, with
   `true/claim ∈ {real, pad, deepfake}`).
2. Per-model `P(fake) = mean over the 3 answers of [1 − P(true=real)]`.
3. Ensemble `P(fake) = mean` of the three per-model scores.
4. `fake if P(fake) ≥ threshold else real`; `forgery_type` from the ensemble true-class marginal.

The threshold is the only knob. Default `0.6337` (validated **PAD 98.6 / real 86.2 / deepfake 91.9**).
`config.json → threshold_options` lists the real-recall/PAD trade-off (e.g. `0.6257` → PAD 98.7 @ real 85.3).

## Validation (frame-based, MLLM-free)
Tuned on a **source-disjoint** set; reported on a **30K video/identity-disjoint** held-out eval
(real frames passed through a frontal / proper-size / no-occlusion face-quality filter). Every PAD
attack type ≥ 96% recall at the default threshold. See the parent project's
`FINAL_REPORT_MLLMfree.md` for the full frontier and methodology.

## REST API (FastAPI, port 3001)
A server mirroring the FFAA `app_fastapi_json.py` conventions but backed by this ensemble:
```bash
pip install -r requirements.txt            # adds fastapi / uvicorn / python-multipart
DEVICE=cuda:0 PORT=3001 python app_fastapi_json.py
# docs: http://<host>:3001/docs
```
Endpoints (JSON-object responses, CORS, dynamic GPU request batching):
- `POST /face_liveness` — multipart upload (`face=@img.jpg`); optional `?threshold=`
- `POST /face_liveness_base64` — `{"image_base64": "...", "threshold": 0.6337}`
- `POST /face_liveness_base64_batch` — `{"images_base64": [...]}`
- `GET /health`, `GET /status` (models, device, batching, threshold options)

Each response carries the ensemble's native outputs at the top level
(`decision`, `forgery_type`, `match_score`, `forgery_score`, `processing_time_sec`) plus a
legacy-style `face_liveness` object for drop-in FFAA-client compatibility. The decision threshold
is per-request (`threshold`); `forgery_score` is threshold-independent so dynamic batching stays
threshold-safe. Env knobs: `DEVICE`, `PORT`, `MAX_BATCH_SIZE`, `DYNAMIC_BATCHING`,
`AMBIGUOUS_MARGIN`, `SAVE_REQUESTS`.

## Batch test over a folder (images + videos)
`test_video_image_batch.py` is the ensemble counterpart of FFAA's `test_video_image_batch.py`:
it recurses a tree, scores every still image and every video frame, compares to the ground-truth
label (a `real`/`fake` path component), copies misclassified items to a miss folder, writes sidecar
outputs, and prints a per-label accuracy summary. Multi-GPU by file sharding (one process per device).
```bash
python test_video_image_batch.py --input-dir /datasets/work/vLLM/data/axonlabs_data --devices all
python test_video_image_batch.py --device 0 --batch-size 64 --threshold 0.6337 --frame-stride 5
```
- still image -> `<image>.ensemble.txt` (full result JSON); video -> `<video>.ensemble.frames.jsonl`
  (one record per frame). The `.ensemble` suffix and the separate `--miss-dir` keep these from
  overwriting the FFAA `.txt` / `.frames.jsonl` outputs on the same dataset.
- **Consolidated `--results results_ensemble.txt`** (default) — one file in the `results.txt`
  line format: `OK/XX/SK/ER  truth  pred  type  fake_score  match_score  image` + a per-subset
  accuracy summary. (For a complete file run a full pass, e.g. `--skip-existing 0`.)
- **Real face-quality filter (`--filter-real 1`, default on)** — skips low-quality REAL
  images/frames (heavy head pose / wrong face size / occluded / no single dominant face) using
  insightface buffalo_l, mirroring `build_real_filtered.py` (`--face-score 0.65 --face-pose 28
  --face-hmin 0.15 --face-hmax 0.85 --face-minpx 80`). **Skipped reals are EXCLUDED from accuracy**
  (logged as `SK` lines); PAD/fake items are never filtered.
- `--skip-existing 1` (default) skips items whose `.ensemble` output already exists.
- prediction is `real` / `fake` (= PAD or deepfake) / `ambiguous`; `--real-ambiguous-match-min`
  (default 0.9) mirrors the API's ambiguity rule. **For accuracy, `ambiguous` is counted as `fake`**
  (ambiguous on a fake = correct; ambiguous on a real = miss), and an ambiguous miss gets
  `_ambiguous` inserted into the copied miss filename.

Uses `MidsEnsemble.predict_rgb_batch(...)` to score decoded frames in-memory (no temp files).

## Layout
```
inference.py            CLI
config.json             members, base-model paths, threshold (+ options), fixed answers
mids_ensemble/          self-contained package (model + loader + ensemble)
weights/                A1_svd_9c.pt, A2_svdgend_9c.pt, A3_midspp_9c.pt
base_models/            clip-vit-large-patch14-336/, t5-base/   (bundled encoders)
```

Notes: ~5–6 GB GPU for the 3 models (runs on CPU too, slower). PAD vs deepfake is reported but the
headline decision is binary real/fake. Not for unauthorized surveillance; intended for
anti-spoofing / deepfake-detection use.

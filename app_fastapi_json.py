#!/usr/bin/env python
"""FastAPI server for the MIDS++ A1+A2+A3 (9-class) MLLM-free real/fake ensemble.

This mirrors the FFAA `app_fastapi_json.py` request/response conventions (same endpoint names,
status routes, CORS + no-cache middleware, dynamic GPU request batching, JSON-object responses)
but the model is the standalone `MidsEnsemble` from this project — no MLLM, frame-based, full image.

For each face image it returns the ensemble's native outputs:
    decision (real/fake) + forgery_type (real/pad/deepfake) + match_score + processing_time
plus a legacy-style `face_liveness` object so existing FFAA clients keep working.

Run:
    python app_fastapi_json.py                 # serves on 0.0.0.0:3001
    PORT=3001 DEVICE=cuda:0 python app_fastapi_json.py
Docs:  http://<host>:3001/docs
"""
from __future__ import annotations

import base64
import os
import time
import tempfile
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from datetime import datetime
from io import BytesIO
from queue import Queue, Empty
from threading import Thread
from typing import Any, Dict, List, Optional

# keep encoder/tokenizer loading quiet before transformers is imported (via MidsEnsemble)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from PIL import Image, UnidentifiedImageError

from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from mids_ensemble import MidsEnsemble

# --------------------------------------------------------------------------------------
# configuration (all overridable via environment)
# --------------------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.environ.get("ENSEMBLE_CONFIG", os.path.join(_HERE, "config.json"))
DEVICE = os.environ.get("DEVICE")                       # None -> auto (cuda if available)

MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "16"))
SAVE_REQUESTS = os.environ.get("SAVE_REQUESTS", "0") == "1"   # keep uploaded images per client IP
REQUEST_DIR = os.environ.get("REQUEST_DIR", os.path.join(_HERE, "request_images"))

# ambiguity band: if |fake_score - threshold| <= margin, report "ambiguous" (0 = pure binary).
AMBIGUOUS_MARGIN = float(os.environ.get("AMBIGUOUS_MARGIN", "0.0"))
# a "real" decision whose confidence (match_score) is below this is downgraded to "ambiguous".
REAL_AMBIGUOUS_MATCH_MIN = float(os.environ.get("REAL_AMBIGUOUS_MATCH_MIN", "0.9"))

DYNAMIC_BATCHING = os.environ.get("DYNAMIC_BATCHING", "1") == "1"
DYNAMIC_BATCH_MAX_WAIT_MS = int(os.environ.get("DYNAMIC_BATCH_MAX_WAIT_MS", "20"))
INFERENCE_TIMEOUT_SEC = int(os.environ.get("INFERENCE_TIMEOUT_SEC", "120"))

IMAGE_FORMAT_TO_EXT = {"JPEG": ".jpg", "JPG": ".jpg", "PNG": ".png",
                       "WEBP": ".webp", "BMP": ".bmp"}

FORGERY_LABEL = {
    "real": "None",
    "pad": "PAD (presentation attack / spoof)",
    "deepfake": "Deepfake",
}
_REASON = {
    "real": ("All three MIDS++ models read consistent facial texture, lighting and natural 3D depth "
             "with no visible manipulation or presentation-attack cues."),
    "pad": ("The ensemble detects presentation-attack cues (e.g. screen/print recapture, mask/silicone, "
            "paper/textile, or heavy makeup) rather than a live face."),
    "deepfake": ("The ensemble detects face-manipulation cues (blending artifacts, inconsistent lighting, "
                 "unnatural integration between the face and its surroundings)."),
}

# --------------------------------------------------------------------------------------
# load the ensemble once at startup
# --------------------------------------------------------------------------------------
print(f"[startup] loading MidsEnsemble from {CONFIG_PATH} (device={DEVICE or 'auto'}) ...", flush=True)
engine = MidsEnsemble(config_path=CONFIG_PATH, device=DEVICE)
print(f"[startup] loaded {len(engine.models)} models {engine.names} on {engine.device} "
      f"in {engine.load_time_sec}s | default threshold={engine.threshold}", flush=True)
os.makedirs(REQUEST_DIR, exist_ok=True)


# --------------------------------------------------------------------------------------
# decision helpers — recompute decision from threshold-independent raw scores
# --------------------------------------------------------------------------------------
def _tau(threshold: Optional[float]) -> float:
    return engine.threshold if threshold is None else float(threshold)


def _difficulty(raw: dict, tau: float) -> str:
    """easy if the 3 per-model fake-scores agree on the side of the threshold, else hard."""
    pm = list(raw.get("per_model_fake", {}).values())
    sides = {(float(v) >= tau) for v in pm}
    return "easy" if len(sides) <= 1 else "hard"


def _compose(result: str, ftype: str, ff: float, tau: float, match: float) -> str:
    if result == "ambiguous":
        return (f"Borderline case: the decision is not confident (match score {match:.3f}); "
                f"ensemble fake-probability {ff:.3f} vs threshold {tau:.3f}. " + _REASON.get(ftype, _REASON["real"]))
    base = _REASON["real"] if result == "real" else _REASON[ftype]
    side = "below" if result == "real" else "at/above"
    return base + f" (ensemble fake-probability {ff:.3f} {side} threshold {tau:.3f}.)"


def build_response(raw: dict, threshold: Optional[float], margin: float) -> dict:
    """Turn one raw ensemble dict (forgery_score + type_probs + per_model_fake) into the API response,
    applying this request's threshold. `raw` is threshold-independent, so batching is threshold-safe."""
    tau = _tau(threshold)
    ff = float(raw["forgery_score"])
    tp = raw["type_probs"]
    decision = "fake" if ff >= tau else "real"
    forgery_type = "real" if decision == "real" else ("pad" if tp["pad"] >= tp["deepfake"] else "deepfake")
    match_score = ff if decision == "fake" else 1.0 - ff
    # ambiguity: (a) a low-confidence "real" (match_score below REAL_AMBIGUOUS_MATCH_MIN), or
    #            (b) optional symmetric band around the threshold (AMBIGUOUS_MARGIN).
    result = decision
    if decision == "real" and match_score < REAL_AMBIGUOUS_MATCH_MIN:
        result = "ambiguous"
    elif margin > 0 and abs(ff - tau) <= margin:
        result = "ambiguous"
    difficulty = _difficulty(raw, tau)

    face_liveness = {
        "Analysis result": result,                                  # real / fake / ambiguous
        "Forgery type": FORGERY_LABEL[forgery_type] if result != "real" else "None",
        "Match score": f"{match_score:.4f}",
        "Forgery score": f"{ff:.4f}",
        "Type probabilities": {k: round(float(v), 4) for k, v in tp.items()},
        "Difficulty": difficulty,
        "Forgery reasoning": _compose(result, forgery_type, ff, tau, match_score),
        "Model": "MIDS++ A1+A2+A3 (9-class) MLLM-free ensemble",
        "Threshold": round(tau, 4),
    }
    return {
        "success": True,
        "decision": result,
        "forgery_type": forgery_type,
        "match_score": round(match_score, 4),
        "forgery_score": round(ff, 4),
        "processing_time_sec": raw.get("processing_time_sec"),
        "face_liveness": face_liveness,
        "details": {
            "type_probs": {k: round(float(v), 4) for k, v in tp.items()},
            "per_model_fake": raw.get("per_model_fake", {}),
            "difficulty": difficulty,
            "threshold": round(tau, 4),
        },
    }


# --------------------------------------------------------------------------------------
# dynamic request batcher (computes raw scores once; decision applied per-request afterwards)
# --------------------------------------------------------------------------------------
def _infer_paths(paths: List[str]) -> Dict[str, dict]:
    """Run the ensemble on a list of image paths -> {path: raw_dict}. raw_dict may carry 'error'."""
    res = engine.predict_batch(paths, threshold=engine.threshold, batch_size=MAX_BATCH_SIZE)
    return {r["image"]: r for r in res}


class DynamicInferenceBatcher:
    """Coalesce concurrent requests into one GPU batch for higher throughput under parallel traffic."""

    def __init__(self, max_batch_size=16, max_wait_ms=20, timeout_sec=120):
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_s = max(0.0, float(max_wait_ms) / 1000.0)
        self.timeout_sec = timeout_sec
        self.queue: Queue = Queue()
        self.worker = Thread(target=self._loop, daemon=True, name="ensemble-batcher")
        self.worker.start()

    def submit_many(self, paths: List[str]) -> List[Optional[dict]]:
        fut: Future = Future()
        self.queue.put((list(paths), fut))
        return fut.result(timeout=self.timeout_sec)

    def submit(self, path: str) -> Optional[dict]:
        return self.submit_many([path])[0]

    def _loop(self):
        while True:
            first_paths, first_fut = self.queue.get()
            items = [(first_paths, first_fut)]
            total = len(first_paths)
            deadline = time.time() + self.max_wait_s
            while total < self.max_batch_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    paths, fut = self.queue.get(timeout=remaining)
                except Empty:
                    break
                items.append((paths, fut)); total += len(paths)
            while total < self.max_batch_size:                 # drain late arrivals w/o extra latency
                try:
                    paths, fut = self.queue.get_nowait()
                except Empty:
                    break
                items.append((paths, fut)); total += len(paths)

            flat: List[str] = []
            spans = []
            cur = 0
            for paths, _f in items:
                flat.extend(paths); spans.append((cur, cur + len(paths))); cur += len(paths)
            try:
                by_path = _infer_paths(flat)
            except Exception as exc:                            # fail the whole group
                for _p, fut in items:
                    if not fut.done():
                        fut.set_exception(exc)
                continue
            for (paths, fut), (a, b) in zip(items, spans):
                if not fut.done():
                    fut.set_result([by_path.get(p) for p in paths])


inference_batcher = DynamicInferenceBatcher(
    max_batch_size=MAX_BATCH_SIZE,
    max_wait_ms=DYNAMIC_BATCH_MAX_WAIT_MS,
    timeout_sec=INFERENCE_TIMEOUT_SEC,
) if DYNAMIC_BATCHING else None


def analyze_paths(paths: List[str]) -> List[Optional[dict]]:
    """Raw scores for a list of paths, using the batcher when enabled."""
    if inference_batcher is not None:
        return inference_batcher.submit_many(paths)
    by_path = _infer_paths(paths)
    return [by_path.get(p) for p in paths]


# --------------------------------------------------------------------------------------
# image I/O helpers
# --------------------------------------------------------------------------------------
def validate_suffix(image_bytes: bytes) -> str:
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image is too large")
    try:
        with Image.open(BytesIO(image_bytes)) as im:
            im.load()
            fmt = (im.format or "").upper()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Failed to decode image") from exc
    return IMAGE_FORMAT_TO_EXT.get(fmt, ".png")


def client_ip(request: Optional[Request]) -> str:
    if request is None:
        return "unknown"
    ip = request.headers.get("X-Forwarded-For") or (request.client.host if request.client else None)
    if ip:
        ip = ip.split(",")[0].strip()
    return ip or "unknown"


def write_image(image_bytes: bytes, suffix: str, request: Optional[Request]) -> str:
    """Persist request bytes to a file the ensemble can read. Returns the path."""
    if SAVE_REQUESTS:
        subdir = os.path.join(REQUEST_DIR, client_ip(request))
        os.makedirs(subdir, exist_ok=True)
        path = os.path.join(subdir, datetime.now().strftime("%Y%m%d_%H%M%S_%f") + suffix)
        with open(path, "wb") as fh:
            fh.write(image_bytes)
        return path
    fd, path = tempfile.mkstemp(suffix=suffix, dir=REQUEST_DIR)
    with os.fdopen(fd, "wb") as fh:
        fh.write(image_bytes)
    return path


def cleanup(paths: List[str]) -> None:
    if SAVE_REQUESTS:
        return
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def decode_b64(image_base64: str) -> bytes:
    if "," in image_base64:                                   # strip data URL prefix
        image_base64 = image_base64.split(",", 1)[1]
    try:
        return base64.b64decode(image_base64, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 image") from exc


# --------------------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------------------
tags_metadata = [
    {"name": "status", "description": "Health, runtime configuration, and batching diagnostics."},
    {"name": "liveness", "description": "Face real/fake (PAD + deepfake) analysis via the MIDS++ ensemble."},
]

app = FastAPI(
    title="MIDS++ Ensemble Face Real/Fake API",
    version=os.environ.get("API_VERSION", "1.0-ensemble"),
    description=(
        "Frame-based, MLLM-free face anti-spoofing + deepfake detection using the standalone "
        "MIDS++ A1+A2+A3 (9-class) ensemble. Dynamic GPU request batching; per-request threshold. "
        "Swagger UI at `/docs`, ReDoc at `/redoc`, schema at `/openapi.json`."
    ),
    openapi_tags=tags_metadata,
    swagger_ui_parameters={"displayRequestDuration": True, "filter": True, "tryItOutEnabled": True},
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_headers(request: Request, call_next):
    started = time.time()
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["X-Process-Time-Ms"] = f"{(time.time() - started) * 1000:.2f}"
    return response


# ---- request / response models ----
class Base64ImageRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 image (data URL prefix accepted).",
                              examples=["/9j/4AAQSkZJRgABAQAAAQABAAD..."])
    threshold: Optional[float] = Field(None, description="Override decision threshold (default from config).")


class Base64BatchRequest(BaseModel):
    images_base64: Optional[List[str]] = Field(None, description="List of base64 images. Preferred field.")
    image_base64_list: Optional[List[str]] = Field(None, description="Backward-compatible alias.")
    threshold: Optional[float] = Field(None, description="Override decision threshold for all images.")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def homepage():
    return (
        "<html><body><h3>MIDS++ Ensemble Face Real/Fake API</h3>"
        "<p>FastAPI server is running (port 3001).</p><ul>"
        "<li><a href='/docs'>Swagger UI</a></li>"
        "<li><a href='/redoc'>ReDoc</a></li>"
        "<li><a href='/status'>Status</a></li></ul></body></html>"
    )


@app.get("/health", tags=["status"], summary="Health check")
async def health():
    return {"success": True, "status": "ok"}


@app.get("/status", tags=["status"], summary="Model / batching / runtime settings")
async def status():
    return {
        "success": True,
        "models": engine.names,
        "device": str(engine.device),
        "image_size": engine.size,
        "default_threshold": engine.threshold,
        "ambiguous_margin": AMBIGUOUS_MARGIN,
        "real_ambiguous_match_min": REAL_AMBIGUOUS_MATCH_MIN,
        "dynamic_batching": inference_batcher is not None,
        "max_batch_size": MAX_BATCH_SIZE,
        "max_wait_ms": DYNAMIC_BATCH_MAX_WAIT_MS,
        "queue_size": inference_batcher.queue.qsize() if inference_batcher is not None else 0,
        "inference_timeout_sec": INFERENCE_TIMEOUT_SEC,
        "threshold_options": engine.config.get("threshold_options", {}),
    }


# ---- liveness endpoints ----
@app.post("/face_liveness", tags=["liveness"], summary="Analyze an uploaded face image")
async def face_liveness(
    request: Request,
    face: UploadFile = File(..., description="Face image file"),
    threshold: Optional[float] = Query(None, description="Override decision threshold"),
):
    if not face.filename:
        raise HTTPException(status_code=400, detail="no face image file.")
    image_bytes = await face.read()
    try:
        suffix = validate_suffix(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    path = write_image(image_bytes, suffix, request)
    try:
        raw = await run_in_threadpool(analyze_paths, [path])
    except FutureTimeoutError:
        raise HTTPException(status_code=504, detail="Inference timed out")
    finally:
        cleanup([path])

    r = raw[0]
    if r is None or "error" in r:
        return JSONResponse({"success": False, "error": (r or {}).get("error", "inference failed")})
    return JSONResponse(build_response(r, threshold, AMBIGUOUS_MARGIN))


@app.post("/face_liveness_base64", tags=["liveness"], summary="Analyze a base64 face image")
async def face_liveness_base64(request: Request, payload: Base64ImageRequest = Body(...)):
    try:
        image_bytes = decode_b64(payload.image_base64)
        suffix = validate_suffix(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    path = write_image(image_bytes, suffix, request)
    try:
        raw = await run_in_threadpool(analyze_paths, [path])
    except FutureTimeoutError:
        raise HTTPException(status_code=504, detail="Inference timed out")
    finally:
        cleanup([path])

    r = raw[0]
    if r is None or "error" in r:
        return JSONResponse({"success": False, "error": (r or {}).get("error", "inference failed")})
    return JSONResponse(build_response(r, payload.threshold, AMBIGUOUS_MARGIN))


@app.post("/face_liveness_base64_batch", tags=["liveness"], summary="Analyze a batch of base64 face images")
async def face_liveness_base64_batch(request: Request, payload: Base64BatchRequest = Body(...)):
    images_base64 = payload.images_base64 or payload.image_base64_list
    if not isinstance(images_base64, list) or not images_base64:
        raise HTTPException(status_code=400, detail="images_base64 must be a non-empty list")
    if len(images_base64) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Batch size exceeds limit of {MAX_BATCH_SIZE}")

    results: List[Optional[dict]] = [None] * len(images_base64)
    valid_idx: List[int] = []
    paths: List[str] = []
    for i, b64 in enumerate(images_base64):
        if not isinstance(b64, str) or not b64.strip():
            results[i] = {"success": False, "error": "image_base64 must be a non-empty string"}
            continue
        try:
            image_bytes = decode_b64(b64)
            suffix = validate_suffix(image_bytes)
        except ValueError as exc:
            results[i] = {"success": False, "error": str(exc)}
            continue
        paths.append(write_image(image_bytes, suffix, request))
        valid_idx.append(i)

    if paths:
        try:
            raws = await run_in_threadpool(analyze_paths, paths)
        except FutureTimeoutError:
            raise HTTPException(status_code=504, detail="Inference timed out")
        finally:
            cleanup(paths)
        for local, original in enumerate(valid_idx):
            r = raws[local]
            if r is None or "error" in r:
                results[original] = {"success": False, "error": (r or {}).get("error", "inference failed")}
            else:
                results[original] = build_response(r, payload.threshold, AMBIGUOUS_MARGIN)

    for i, r in enumerate(results):
        if r is None:
            results[i] = {"success": False, "error": "Unknown input error"}
    return JSONResponse({"success": True, "results": results})


if __name__ == "__main__":
    import uvicorn

    # Pass the already-constructed `app` object (NOT the "module:app" import string): with the import
    # string uvicorn re-imports this module in the server process and would load the 3 models a second
    # time. workers=1 + no reload means the in-process app is all we need.
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "3001")),
        workers=1,
    )

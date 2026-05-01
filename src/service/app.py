"""
app.py — FastAPI service layer v3.0

Fixes:
  - CORS: allow_credentials=True is incompatible with allow_origins=["*"].
    Now uses explicit origins for credentialed requests.
  - Static directory is only mounted if it actually exists — prevents
    startup crash when the directory is missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.service.federation_manager import FederationManager

logger = logging.getLogger(__name__)

app = FastAPI(
    title   = "Hospital FL Client — Multi-Federation Service",
    version = "3.0",
)

# CORS fix: allow_origins=["*"] with allow_credentials=True is invalid —
# browsers reject wildcard origins on credentialed requests.
# Use explicit dev origins instead.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

BASE_DIR   = Path(__file__).resolve().parents[2]
templates  = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Only mount /static if the directory exists — prevents startup crash
_static_dir = BASE_DIR / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
else:
    logger.warning(
        "app.py: static/ directory not found at %s — /static not mounted",
        _static_dir,
    )

manager = FederationManager(
    federations_dir = str(BASE_DIR / "conf" / "federations"),
    output_base_dir = str(BASE_DIR / "output"),
)


# ── Request models ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    inputs: List[List[float]]


# ── Health ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True, "version": "3.0"}


# ── Federation listing ───────────────────────────────────────────────────────

@app.get("/federations")
def list_federations():
    return {
        "federations": manager.list_federations(),
        "status":      manager.status_all(),
    }


# ── Per-federation — training control ────────────────────────────────────────

@app.get("/federations/{federation_id}/status")
def get_status(federation_id: str):
    return _run(federation_id, lambda: manager.status(federation_id))


@app.get("/federations/{federation_id}/config")
def get_config(federation_id: str):
    return _run(federation_id, lambda: manager.get_config(federation_id))


@app.post("/federations/{federation_id}/config")
async def save_config(federation_id: str, request: Request):
    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    result = _run(federation_id, lambda: manager.save_config(federation_id, body))
    if not result.get("saved"):
        raise HTTPException(status_code=422, detail=result.get("message"))
    return result


@app.post("/federations/{federation_id}/start")
def start_training(federation_id: str):
    return _run(federation_id, lambda: manager.start(federation_id))


@app.post("/federations/{federation_id}/stop")
def stop_training(federation_id: str):
    return _run(federation_id, lambda: manager.stop(federation_id))


@app.post("/federations/{federation_id}/reload")
def reload_config(federation_id: str):
    return _run(federation_id, lambda: manager.reload(federation_id))


@app.get("/federations/{federation_id}/metrics")
def get_metrics(federation_id: str):
    return _run(federation_id, lambda: {"metrics": manager.metrics(federation_id)})


# ── Per-federation — inference ───────────────────────────────────────────────

@app.get("/federations/{federation_id}/inference/status")
def inference_status(federation_id: str):
    return _run(federation_id, lambda: manager.inference_status(federation_id))


@app.post("/federations/{federation_id}/predict")
def predict(federation_id: str, body: PredictRequest):
    """
    Run inference on the latest saved checkpoint.

    Body:  { "inputs": [[f1, f2, ..., f13]] }   ← EHR: 13 features
           { "inputs": [[t1, t2, ..., t187]] }   ← ECG: 187 features

    Returns predictions with label names and confidence scores.
    HTTP 503 if no checkpoint is available yet.
    """
    result = _run(
        federation_id,
        lambda: manager.predict(federation_id, body.inputs),
    )
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(
            status_code = 503,
            detail      = result.get("message", "Inference not available"),
        )
    return result


# ── Bulk control ─────────────────────────────────────────────────────────────

@app.post("/start-all")
def start_all():
    return manager.start_all()


@app.post("/stop-all")
def stop_all():
    return manager.stop_all()


# ── UI — Jinja2 (temporary, React UI planned) ────────────────────────────────

@app.get("/ui")
@app.get("/")
def ui_index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request":     request,
        "federations": manager.list_federations(),
        "status":      manager.status_all(),
        "title":       "FL Client Dashboard",
    })


@app.get("/ui/federations/{federation_id}/status")
def ui_status(federation_id: str, request: Request):
    return templates.TemplateResponse("status.html", {
        "request":       request,
        "federation_id": federation_id,
        "title":         f"Status — {federation_id}",
    })


@app.get("/ui/federations/{federation_id}/logs")
def ui_logs(federation_id: str, request: Request):
    return templates.TemplateResponse("logs.html", {
        "request":       request,
        "federation_id": federation_id,
        "title":         f"Logs — {federation_id}",
    })


# ── Helper ───────────────────────────────────────────────────────────────────

def _run(federation_id: str, fn):
    try:
        return fn()
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error for federation '%s': %s", federation_id, e)
        raise HTTPException(status_code=500, detail=str(e))
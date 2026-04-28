"""FastAPI demo server: upload a receipt image, see DONUT extract the fields.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: purely illustrative demo for the DONUT end-to-end comparator.
    Run with: ``make serve`` (or ``uvicorn app.server:app --host 0.0.0.0``).
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from app.predict import LoadedModel, load_model, predict
from core.config import load_config
from core.types import ExpConfig

log = logging.getLogger(__name__)

_INDEX_HTML = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")

app = FastAPI(title="kaggle2 DONUT demo")

_state: dict[str, Any] = {"model": None, "config": None}


def _get_model() -> LoadedModel:
    """Lazy-load DONUT on first request so server boot is instant."""
    if _state["model"] is None:
        config: ExpConfig = _state["config"] or load_config("configs/default.json")
        _state["config"] = config
        log.info("Loading DONUT model (first request)…")
        _state["model"] = load_model(config)
        log.info("DONUT loaded from %s (source=%s).",
                 _state["model"].model_path, _state["model"].source)
    model = _state["model"]
    assert isinstance(model, LoadedModel)
    return model


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the single-page demo UI."""
    return _INDEX_HTML


@app.get("/health")
def health() -> JSONResponse:
    """Report whether the fine-tuned or base model is loaded."""
    model = _state["model"]
    if model is None:
        return JSONResponse({"status": "ok", "model_loaded": False})
    return JSONResponse({
        "status": "ok", "model_loaded": True,
        "model_source": model.source, "model_path": model.model_path,
        "device": model.device,
    })


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> JSONResponse:  # noqa: B008
    """Run DONUT on an uploaded image; return the extracted fields."""
    from PIL import Image, UnidentifiedImageError

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    try:
        image = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Not a valid image.") from exc

    try:
        model = _get_model()
        fields = predict(model, image)
    except Exception as exc:  # noqa: BLE001 — surface any error to the UI
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return JSONResponse({
        "fields": fields, "model_source": model.source,
        "filename": file.filename,
    })

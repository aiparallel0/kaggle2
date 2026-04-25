"""Per-stage best-epoch + epochs-run extractor for the training table.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: each of YOLO, TrOCR, and the assigner has its OWN
    early-stopping or best-checkpoint logic — there is no single
    global "best epoch" across the pipeline, but every stage has
    one.  This helper recovers the per-stage values from the
    artefacts each stage already writes, and folds them into the
    paper metrics dict so :func:`report.inject_tables.render_training_table`
    resolves ``{stage}_best_epoch`` and ``{stage}_epochs_run`` to
    real measurements rather than ``\\textit{n/a}``.

Sources:

  * **YOLO** — Ultralytics writes ``yolo/run/results.csv`` with one
    row per epoch.  The ``best_epoch`` is the row whose
    ``metrics/mAP50-95(B)`` (or ``mAP_0.5:0.95``, depending on
    Ultralytics version) is maximised; ``epochs_run`` is the row
    count.  Falls back to ``map_50`` if the 50-95 column is absent.
  * **TrOCR** — HuggingFace Trainer writes ``trocr/trainer_state.json``
    when ``output_dir`` is configured.  ``state.best_metric`` and
    ``state.log_history[*]['epoch']`` recover the best epoch;
    ``state.epoch`` (last logged) is the total epochs run.
  * **Assigner** — ``assigner_metrics.json`` already exposes
    ``best_epoch`` and ``stopped_at`` via
    :func:`report.combine.merge_assigner_metrics`; this helper
    additionally surfaces them as ``assigner_epochs_run``.

Every reader is best-effort: a missing artefact silently leaves the
key absent so the inject layer renders ``\\MissingCell`` (and the
build gate flags it), which is what we want for a partial run.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from core.types import ExpConfig

log = logging.getLogger("kaggle2")


def _yolo_results_csv(run_dir: Path) -> Path | None:
    """Return the first existing Ultralytics ``results.csv`` under ``run_dir``."""
    candidates = (
        run_dir / "yolo" / "run" / "results.csv",
        run_dir / "yolo" / "results.csv",
    )
    for p in candidates:
        if p.is_file():
            return p
    # Fallback: glob — ultralytics has historically used multiple layouts.
    matches = sorted((run_dir / "yolo").glob("**/results.csv")) if (
        run_dir / "yolo"
    ).is_dir() else []
    return matches[0] if matches else None


def _read_yolo_best(run_dir: Path) -> tuple[int | None, int | None]:
    """Return ``(best_epoch, epochs_run)`` from Ultralytics' results.csv."""
    csv_path = _yolo_results_csv(run_dir)
    if csv_path is None:
        return None, None
    try:
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        log.warning("best_epoch: cannot read %s (%s)", csv_path, exc)
        return None, None
    if not rows:
        return None, None
    # Ultralytics emits whitespace-padded headers historically; normalise.
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    epoch_key = next(
        (k for k in rows[0] if k.lower().startswith("epoch")), None,
    )
    metric_key = next(
        (k for k in rows[0]
         if "mAP" in k and ("50-95" in k or "0.5:0.95" in k)),
        None,
    ) or next(
        (k for k in rows[0] if "mAP" in k and "50" in k), None,
    )
    if epoch_key is None or metric_key is None:
        return None, len(rows)

    def _f(s: str) -> float:
        try:
            return float(s)
        except (TypeError, ValueError):
            return float("nan")

    best_idx = max(
        range(len(rows)),
        key=lambda i: (_f(rows[i].get(metric_key, "nan")), -i),
    )
    # Ultralytics' ``results.csv`` is 0-indexed (epoch column starts at
    # 0).  Papers and Ultralytics' own console output / ``best.pt``
    # filename use 1-indexed epoch numbers, so we add 1 here to match
    # the convention reviewers see in the rest of the paper.  Falling
    # back to ``best_idx + 1`` (the row position) covers the rare case
    # where the epoch column is non-numeric / missing.
    try:
        best = int(float(rows[best_idx][epoch_key])) + 1
    except (TypeError, ValueError):
        best = best_idx + 1
    return best, len(rows)


def _read_trocr_best(run_dir: Path) -> tuple[int | None, int | None]:
    """Return ``(best_epoch, epochs_run)`` from HF Trainer state."""
    candidates = (
        run_dir / "trocr" / "trainer_state.json",
        run_dir / "trocr_trainer_state.json",
    )
    state_path = next((p for p in candidates if p.is_file()), None)
    if state_path is None:
        # HF Trainer also drops checkpoint-N folders each containing
        # ``trainer_state.json``; pick the highest-N as the final state.
        ckpts = sorted(
            (run_dir / "trocr").glob("checkpoint-*/trainer_state.json"),
            key=lambda p: int(p.parent.name.split("-")[-1]),
        ) if (run_dir / "trocr").is_dir() else []
        if ckpts:
            state_path = ckpts[-1]
    if state_path is None:
        return None, None
    try:
        state = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("best_epoch: cannot read %s (%s)", state_path, exc)
        return None, None
    epochs_run_raw = state.get("epoch")
    epochs_run = int(round(float(epochs_run_raw))) if isinstance(
        epochs_run_raw, int | float,
    ) else None
    best_metric = state.get("best_metric")
    best_epoch: int | None = None
    if best_metric is not None:
        # Find the eval log entry matching the best metric value.
        log_history = state.get("log_history") or []
        if isinstance(log_history, list):
            for entry in log_history:
                if not isinstance(entry, dict):
                    continue
                # The metric key was set at trainer-construction time as
                # ``metric_for_best_model``; HF logs it under that name.
                for k, v in entry.items():
                    if k.startswith("eval_") and v == best_metric:
                        ep = entry.get("epoch")
                        if isinstance(ep, int | float):
                            best_epoch = int(round(float(ep)))
                            break
                if best_epoch is not None:
                    break
    return best_epoch, epochs_run


def _read_donut_best(run_dir: Path) -> tuple[int | None, int | None]:
    """Return ``(best_epoch, epochs_run)`` from DONUT's training_log.json."""
    log_path = run_dir / "training_log.json"
    if not log_path.is_file():
        return None, None
    try:
        data = json.loads(log_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, None
    epochs = data.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        return None, None
    epochs_run = len(epochs)
    # Best epoch = argmax(eval_f1) when present, else argmin(eval_loss).
    eval_f1 = data.get("eval_f1") or []
    eval_loss = data.get("eval_loss") or []
    best_epoch: int | None = None
    if isinstance(eval_f1, list) and len(eval_f1) == epochs_run:
        try:
            best_idx = max(range(epochs_run), key=lambda i: float(eval_f1[i]))
            best_epoch = int(epochs[best_idx])
        except (TypeError, ValueError):
            best_epoch = None
    if best_epoch is None and isinstance(eval_loss, list) and len(eval_loss) == epochs_run:
        try:
            best_idx = min(range(epochs_run), key=lambda i: float(eval_loss[i]))
            best_epoch = int(epochs[best_idx])
        except (TypeError, ValueError):
            best_epoch = None
    return best_epoch, epochs_run


def merge_best_epochs(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold per-stage best/total epoch into the paper metrics dict.

    Idempotent: ``setdefault`` is used so an explicit producer-written
    value (e.g. a future ``yolo_metrics.json`` that emits ``best_epoch``
    directly) takes precedence over the on-disk extraction.
    """
    run_dir = Path(config.output_dir)
    donut_best, donut_run = _read_donut_best(run_dir)
    if donut_best is not None:
        metrics.setdefault("donut_best_epoch", donut_best)
    if donut_run is not None:
        metrics.setdefault("donut_epochs", donut_run)
    yolo_best, yolo_run = _read_yolo_best(run_dir)
    if yolo_best is not None:
        metrics.setdefault("yolo_best_epoch", yolo_best)
    if yolo_run is not None:
        metrics.setdefault("yolo_epochs_run", yolo_run)
    trocr_best, trocr_run = _read_trocr_best(run_dir)
    if trocr_best is not None:
        metrics.setdefault("trocr_best_epoch", trocr_best)
    if trocr_run is not None:
        metrics.setdefault("trocr_epochs_run", trocr_run)
    # Assigner already exposes ``assigner_best_epoch``; add the
    # ``epochs_run`` counterpart so the training-table column reads
    # uniformly across stages.
    stopped_at = metrics.get("assigner_stopped_at")
    if isinstance(stopped_at, int | float):
        metrics.setdefault("assigner_epochs_run", int(stopped_at))
    # Pipeline best-epoch is genuinely composite (three trained
    # sub-stages, each with its OWN best).  Rather than fabricate a
    # single number, ``inject_tables.render_training_table`` reads
    # ``pipeline_best_epoch_label`` directly and renders it raw (no
    # numeric formatting) so reviewers see one consistent message.
    metrics.setdefault("pipeline_best_epoch_label", "\\textit{see sub-stages}")
    parts: list[int] = []
    for k in ("yolo_epochs_run", "trocr_epochs_run", "assigner_epochs_run"):
        v = metrics.get(k)
        if isinstance(v, int | float):
            parts.append(int(v))
    if parts:
        metrics.setdefault("pipeline_epochs", sum(parts))

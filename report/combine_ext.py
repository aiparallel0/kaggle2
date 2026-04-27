"""Extended merge helpers — fold new diagnostics sidecars into the paper dict.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: :mod:`report.combine` caps at 161 LOC and already hosts the
    original three merge helpers (``merge_assigner_metrics``,
    ``merge_pipeline_diagnostics``, ``merge_cost_json``).  This sibling
    module adds the six new merge helpers required by the Section-B
    metric expansion so the per-file cap stays honoured.  Every helper
    has the same 2-in/1-out signature — ``(config, metrics_dict)`` →
    None, mutating in place — and never raises on missing files: a
    flaky train run simply leaves some ``\\VAR{}`` placeholders
    resolving to the ``---`` backstop.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from core.types import ExpConfig

log = logging.getLogger("kaggle2")


def _load_json(path: str | Path) -> dict[str, object] | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with p.open() as fh:
            obj = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("combine_ext: cannot read %s (%s)", p, exc)
        return None
    return obj if isinstance(obj, dict) else None


def _flatten(prefix: str, data: dict[str, object], target: dict[str, object]) -> None:
    """Fold a flat diagnostics dict into ``target`` with ``<prefix>_<key>``."""
    for k, v in data.items():
        if k == "schema_version":
            continue
        if isinstance(v, dict):
            # Two-level dict (e.g. per_class_ap, cer_per_field) expands
            # as ``prefix_key_subkey`` so the LaTeX injector can address
            # e.g. \VAR{yolo_per_class_ap_total}.
            for sk, sv in v.items():
                target[f"{prefix}_{k}_{sk}"] = sv
        else:
            target[f"{prefix}_{k}"] = v


def merge_yolo_diagnostics(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold ``metrics/yolo_metrics.json`` into the paper metrics dict."""
    path = os.path.join(config.output_dir, "metrics", "yolo_metrics.json")
    data = _load_json(path) or _load_json(os.path.join(config.output_dir, "yolo_metrics.json"))
    if data is not None:
        _flatten("yolo", data, metrics)


def merge_trocr_diagnostics(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold ``metrics/trocr_metrics.json`` into the paper metrics dict."""
    path = os.path.join(config.output_dir, "metrics", "trocr_metrics.json")
    data = _load_json(path) or _load_json(os.path.join(config.output_dir, "trocr_metrics.json"))
    if data is not None:
        _flatten("trocr", data, metrics)


def merge_assigner_diag(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold ``metrics/assigner_diagnostics.json`` into the paper metrics dict."""
    path = os.path.join(config.output_dir, "metrics", "assigner_diagnostics.json")
    data = _load_json(path) or _load_json(
        os.path.join(config.output_dir, "assigner_diagnostics.json"),
    )
    if data is not None:
        _flatten("assigner", data, metrics)


def merge_donut_diag(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold ``metrics/donut_diagnostics.json`` into the paper metrics dict."""
    path = os.path.join(config.output_dir, "metrics", "donut_diagnostics.json")
    data = _load_json(path) or _load_json(
        os.path.join(config.output_dir, "donut_diagnostics.json"),
    )
    if data is not None:
        _flatten("donut", data, metrics)


def merge_latency(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold every ``metrics/latency_<system>.json`` into the paper dict."""
    metrics_dir = Path(config.output_dir) / "metrics"
    roots: list[Path] = []
    if metrics_dir.is_dir():
        roots.extend(metrics_dir.glob("latency_*.json"))
    # Back-compat path (pre-metrics-subdir runs).
    roots.extend(Path(config.output_dir).glob("latency_*.json"))
    for p in roots:
        system = p.stem.removeprefix("latency_")
        data = _load_json(p)
        if data is not None:
            _flatten(f"{system}_latency", data, metrics)


def merge_extended_metrics(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold ``metrics/extended_metrics.json`` (per-field CIs, P/R) into paper."""
    path = os.path.join(config.output_dir, "metrics", "extended_metrics.json")
    data = _load_json(path) or _load_json(
        os.path.join(config.output_dir, "extended_metrics.json"),
    )
    if not data:
        return
    # ``extended_metrics.json`` is already namespaced by system key
    # (``donut_*``, ``pipeline_*``, ``rulebased_*``); no prefix added.
    for k, v in data.items():
        metrics.setdefault(k, v)


def merge_env(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold ``env/hostinfo.json`` into the paper metrics dict."""
    path = os.path.join(config.output_dir, "env", "hostinfo.json")
    data = _load_json(path)
    if data is None:
        return
    for k, v in data.items():
        if k == "host" and isinstance(v, dict):
            for sk, sv in v.items():
                metrics.setdefault(f"host_{sk}", sv)
        elif k != "schema_version":
            metrics.setdefault(f"env_{k}", v)


def merge_ablations(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold optional ``metrics/ablations.json`` into the paper dict.

    Ablation runs write a flat ``{"ablation_<tag>_<key>": value, ...}``
    JSON; the helper just forwards every entry into ``metrics``.
    """
    path = os.path.join(config.output_dir, "metrics", "ablations.json")
    data = _load_json(path)
    if data is None:
        return
    for k, v in data.items():
        metrics.setdefault(k, v)


def merge_assigner_arch(config: ExpConfig, metrics: dict[str, object]) -> None:
    """PR-A / L1 — single-source assigner architecture from the live ckpt.

    Introspects ``runs/<id>/assigner/assigner.pt`` (or the legacy
    ``./results/assigner.pt``) and emits the paper's ``\\VAR{}`` keys
    that historically drifted between the shipped checkpoint and the
    code-default constants:

    * ``assigner_d_model``     — encoder ``d_model``
    * ``assigner_n_layers``    — number of encoder layers
    * ``assigner_n_heads``     — multi-head attention heads
    * ``assigner_ff_mult``     — FFN expansion factor (FFN / d_model)
    * ``assigner_dropout``     — encoder dropout (rendered to 2 dp)
    * ``assigner_n_priors``    — text-prior dim (6/9/14)
    * ``assigner_params_k``    — total trainable params, rounded to K
    * ``assigner_params_m``    — same, rounded to 2 dp in millions
    * ``assigner_params_phrase`` — typeset phrase, e.g. ``≈1.16M``
    * ``assigner_recipe_json`` — compact JSON of the above for debug
    * ``param_ratio_pct``      — pipeline_total / donut_total * 100, 1 dp
    * ``param_ratio_phrase``   — typeset phrase from
                                 :func:`report.combine._ratio_phrase`

    Wired from ``stages/paper.py`` BEFORE :func:`merge_assigner_diag`
    so the live values land first and downstream merges only fill
    gaps.  Never raises: a missing checkpoint or a torch-less CI
    environment leaves the keys unresolved (``\\MissingCell{}`` in
    the rendered paper).
    """
    intro = _introspect_assigner_ckpt(config)
    if intro is None:
        return
    metrics.setdefault("assigner_d_model", intro["d_model"])
    metrics.setdefault("assigner_n_layers", intro["n_layers"])
    metrics.setdefault("assigner_n_heads", intro["n_heads"])
    metrics.setdefault("assigner_ff_mult", intro["ff_mult"])
    # Render dropout / params with display-friendly precision so the
    # LaTeX injector emits human-readable text not raw float repr.
    dropout_v = intro["dropout"]
    if isinstance(dropout_v, int | float):
        metrics.setdefault("assigner_dropout", round(float(dropout_v), 2))
    metrics.setdefault("assigner_n_priors", intro["n_priors"])
    params_v = intro["params"]
    if not isinstance(params_v, int):
        return
    n_params = int(params_v)
    params_k = int(round(n_params / 1000.0))
    params_m = round(n_params / 1_000_000.0, 2)
    metrics.setdefault("assigner_params_k", params_k)
    metrics.setdefault("assigner_params_m", params_m)
    metrics.setdefault(
        "assigner_params_phrase", f"\\approx{params_m:.2f}M",
    )
    metrics.setdefault("assigner_recipe_json", json.dumps(intro))
    # Mirror onto :mod:`report.combine`'s param-ratio surface so the
    # paper's title / abstract / conclusion all read the same numeric.
    donut_m = metrics.get("donut_params_m")
    if isinstance(donut_m, int | float) and float(donut_m) > 0:
        # Pipeline = TrOCR + YOLO + assigner.  We only have ``params_m``
        # for the assigner here; defer the full param-ratio computation
        # to :func:`merge_pipeline_diagnostics` which has every arm.
        metrics.setdefault(
            "assigner_params_pct_of_donut",
            round(float(params_m) / float(donut_m) * 100.0, 1),
        )


def _introspect_assigner_ckpt(
    config: ExpConfig,
) -> dict[str, object] | None:
    """Return architecture + param-count from the saved assigner ckpt.

    Returns ``None`` (silently) when torch is not installed or no
    checkpoint exists at any of the standard locations — keeps the
    paper stage running on torch-less CI boxes without raising.
    """
    try:
        from models.attention_assign import _load_assigner
    except Exception:  # noqa: BLE001 — torch import inside attention_assign
        return None
    candidates = [
        Path(config.output_dir) / "assigner" / "assigner.pt",
        Path(config.output_dir) / "assigner.pt",
        Path("./results") / "assigner.pt",
    ]
    ckpt = next((p for p in candidates if p.is_file()), None)
    if ckpt is None:
        return None
    try:
        m = _load_assigner(str(ckpt))
    except Exception as exc:  # noqa: BLE001
        log.warning("merge_assigner_arch: cannot load %s (%s)", ckpt, exc)
        return None
    n_params = sum(int(p.numel()) for p in m.parameters())
    # FFN dim is the encoder's ``linear1.out_features``; cf.
    # ``nn.TransformerEncoderLayer`` ctor — the dim is ``hidden *
    # ff_mult`` and we recover the multiplier by integer division.
    try:
        ff_dim = int(m.encoder.layers[0].linear1.out_features)
        d_model = int(m.encoder.layers[0].self_attn.embed_dim)
        dropout_p = float(
            getattr(m.encoder.layers[0].self_attn, "dropout", 0.0),
        )
    except (AttributeError, IndexError):
        d_model = int(m.hidden_dim)
        ff_dim = d_model * 2
        dropout_p = 0.0
    n_layers = (
        int(len(m.encoder.layers)) if hasattr(m.encoder, "layers")
        else int(m.n_layers)
    )
    return {
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": int(m.n_heads),
        "ff_mult": max(1, ff_dim // d_model) if d_model else 1,
        "dropout": dropout_p,
        "n_priors": int(m.n_text_priors),
        "params": n_params,
        "ckpt_path": str(ckpt),
    }


def merge_carbon(config: ExpConfig, metrics: dict[str, object]) -> None:
    """PR-D — fold ``runs/<id>/env/env_snapshot.json`` carbon estimate.

    Reads ``gpu_tdp_w`` + ``wallclock_seconds`` (when present) and
    derives ``carbon_kgco2e = tdp_w * wallclock_h * grid_factor``.
    The grid factor defaults to ``ExpConfig.
    carbon_grid_factor_kgco2e_per_kwh`` (override via
    ``ExpConfig.extra["grid_factor"]`` for region-specific reruns).
    Never raises: missing fields leave the paper key unresolved.
    """
    path = os.path.join(config.output_dir, "env", "env_snapshot.json")
    data = _load_json(path)
    if data is None:
        return
    tdp = data.get("gpu_tdp_w")
    wall = data.get("wallclock_seconds")
    if not (isinstance(tdp, int | float) and isinstance(wall, int | float)):
        return
    extra_factor = config.extra.get("grid_factor")
    grid = (
        float(extra_factor)
        if isinstance(extra_factor, int | float)
        else float(config.carbon_grid_factor_kgco2e_per_kwh)
    )
    kwh = float(tdp) * float(wall) / 3600.0 / 1000.0
    metrics.setdefault("carbon_kgco2e", round(kwh * grid, 4))
    metrics.setdefault("carbon_kwh", round(kwh, 4))
    metrics.setdefault("carbon_grid_factor", grid)

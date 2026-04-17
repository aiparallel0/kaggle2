"""kaggle2 orchestrator: --stage train | eval | paper | all."""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from core.config import load_config
from core.errors import EvalError, TrainError
from core.seed import seed_everything
from core.types import AssignerData, ExpConfig, PipelinePaths
from data.sroie import download_sroie, extract_crops, extract_receipt_regions, load_or_create_split
from models.assigner_train import train_assigner
from models.donut_eval import eval_donut
from models.donut_train import train_donut
from models.pipeline_eval import eval_pipeline
from models.trocr_train import train_trocr
from models.yolo_train import train_yolo
from report.inject import inject_results

log = logging.getLogger("kaggle2")


def _validate_f1(global_f1: float, arch: str, config: ExpConfig) -> None:
    """Post-eval F1 guardrails: hard raise below floor; soft WARN below expected.

    F1 is stochastic (GPU, HF weights, SROIE label noise); no specific number
    can be guaranteed. Floors flag *bugs*, not underperformance.
    """
    if arch == "donut" and global_f1 < 0.50:
        raise TrainError(
            f"DONUT F1={global_f1:.4f} < 0.50 — likely lm_head dedup (Bug 1), "
            "wrong decoder_start_token_id (Bug 2), or token2json list (Bug 3).",
        )
    if arch == "pipeline" and global_f1 == 0.0:
        raise TrainError(
            "Pipeline F1=0.0 — YOLO imgsz mismatch (Bug 5) "
            "or TrOCR undertrained (Bug 6).",
        )
    if global_f1 < config.expected_f1_warn:
        log.warning("%s F1=%.4f below expected_f1_warn=%.2f (not an error).",
                    arch, global_f1, config.expected_f1_warn)


def _split_cache(config: ExpConfig) -> Path:
    return Path(config.output_dir) / "split.json"


def _write_pipeline_meta(config: ExpConfig) -> None:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(config.output_dir, "pipeline_meta.json"), "w") as f:
        json.dump({"yolo_img_size": config.yolo_img_size}, f)


def _stage_train(config: ExpConfig) -> None:
    log.info("=== Stage: train ===")
    data_path = download_sroie(config)
    data = load_or_create_split(data_path, config.seed, _split_cache(config))
    log.info("Split: %d train / %d val / %d test",
             len(data.train), len(data.val), len(data.test))
    donut_path = train_donut(config, data)
    log.info("DONUT → %s", donut_path)
    yolo_path = train_yolo(config, data)
    log.info("YOLO  → %s", yolo_path)
    crops = extract_crops(data.train, config.fields)
    regions = extract_receipt_regions(data.train, config.fields)
    log.info("Extracted %d labeled crops / %d receipt region-groups",
             len(crops), len(regions))
    if not crops:
        raise TrainError("No labeled SROIE crops — check box/ annotations.")
    trocr_path = train_trocr(config, crops)
    log.info("TrOCR → %s", trocr_path)
    assigner_data = AssignerData(trocr_path=trocr_path, crops=crops, regions=regions)
    assigner_path = train_assigner(config, assigner_data)
    log.info("Assigner → %s", assigner_path)
    _write_pipeline_meta(config)


def _stage_eval(config: ExpConfig) -> None:
    log.info("=== Stage: eval ===")
    data_path = download_sroie(config)
    data = load_or_create_split(data_path, config.seed, _split_cache(config))
    donut_model = os.path.join(config.output_dir, "donut")
    dm = eval_donut(donut_model, data.test, config)
    _validate_f1(dm.global_f1, "donut", config)
    log.info("DONUT F1=%.4f", dm.global_f1)
    paths = PipelinePaths(
        yolo=os.path.join(config.output_dir, "yolo", "run", "weights", "best.pt"),
        trocr=os.path.join(config.output_dir, "trocr"),
        assigner=os.path.join(config.output_dir, "assigner.pt"),
    )
    pm = eval_pipeline(paths, data.test, config)
    _validate_f1(pm.assigner.global_f1, "pipeline", config)
    log.info("Pipeline (assigner)  F1=%.4f", pm.assigner.global_f1)
    log.info("Pipeline (rulebased) F1=%.4f", pm.rulebased.global_f1)
    combined: dict[str, object] = {
        "donut_f1": dm.global_f1, "donut_ned": dm.global_ned, "donut_em": dm.global_em,
        "pipeline_f1": pm.assigner.global_f1,
        "pipeline_ned": pm.assigner.global_ned,
        "pipeline_em": pm.assigner.global_em,
        "rulebased_f1": pm.rulebased.global_f1,
        "rulebased_ned": pm.rulebased.global_ned,
        "f1_gap": round(dm.global_f1 - pm.assigner.global_f1, 4),
        "assigner_delta": round(pm.assigner.global_f1 - pm.rulebased.global_f1, 4),
        "donut_f1_company": dm.per_field_f1.get("company", 0.0),
        "donut_f1_date": dm.per_field_f1.get("date", 0.0),
        "donut_f1_address": dm.per_field_f1.get("address", 0.0),
        "donut_f1_total": dm.per_field_f1.get("total", 0.0),
        "epochs_donut": config.epochs_donut, "epochs_trocr": config.epochs_trocr,
        "epochs_yolo": config.epochs_yolo, "batch_size": config.batch_size,
        "lr": config.lr, "precision": config.precision,
        "label_smoothing": config.label_smoothing,
        "yolo_img_size": config.yolo_img_size,
        "img_w": config.image_size[0], "img_h": config.image_size[1],
    }
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(config.output_dir, "combined_metrics.json"), "w") as f:
        json.dump(combined, f, indent=2)


def _compile_paper_pdf(tex_path: Path, bib_src: Path) -> Path | None:
    """Compile ``tex_path`` to PDF.

    Prefers ``tectonic`` (self-contained LaTeX engine, handles bibtex+rerun
    automatically in one call) and falls back to ``pdflatex`` + ``bibtex`` +
    ``pdflatex`` x2 when only a traditional TeX Live install is available.

    Returns the resulting PDF path on success, or ``None`` if neither engine
    is installed — we warn instead of failing so ``make paper`` still works
    on machines without a LaTeX toolchain. A non-zero compiler exit is fatal:
    the README advertises ``report/paper_filled.pdf`` as the deliverable, so
    a silent compilation failure would defeat the purpose of running the
    stage.
    """
    work = tex_path.parent
    # bibtex/tectonic reads references.bib next to the .tex file.
    if bib_src.exists() and bib_src.resolve() != (work / bib_src.name).resolve():
        shutil.copy(bib_src, work / bib_src.name)

    if shutil.which("tectonic") is not None:
        # Tectonic: one invocation compiles, resolves citations, and reruns
        # as needed. --keep-intermediates lets us inspect .aux / .log on
        # failure; --chatter minimal keeps the stdout tidy.
        result = subprocess.run(
            [
                "tectonic",
                "--keep-intermediates",
                "--keep-logs",
                "--chatter", "minimal",
                tex_path.name,
            ],
            cwd=work,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise EvalError(
                f"tectonic failed for {tex_path.name}:\n"
                f"stdout: {result.stdout[-2000:]}\n"
                f"stderr: {result.stderr[-2000:]}",
            )
        pdf = work / f"{tex_path.stem}.pdf"
        if not pdf.exists():
            raise EvalError(f"tectonic finished but {pdf} was not produced.")
        return pdf

    if shutil.which("pdflatex") is None:
        log.warning(
            "No LaTeX engine found (tried tectonic, pdflatex) — skipping PDF "
            "compilation. Install tectonic (scripts/vastai_bootstrap.sh does "
            "this) to generate %s.pdf.", tex_path.stem,
        )
        return None

    stem = tex_path.stem
    cmds: list[list[str]] = [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        ["bibtex", stem],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
    ]
    for cmd in cmds:
        # bibtex exits non-zero if there are no citations on the first run;
        # tolerate that specifically, but let genuine pdflatex failures raise.
        result = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
        if result.returncode != 0 and cmd[0] == "pdflatex":
            raise EvalError(
                f"pdflatex failed for {tex_path.name}:\n"
                f"stdout: {result.stdout[-2000:]}\n"
                f"stderr: {result.stderr[-2000:]}",
            )
    pdf = work / f"{stem}.pdf"
    if not pdf.exists():
        raise EvalError(f"pdflatex finished but {pdf} was not produced.")
    return pdf


def _stage_paper(config: ExpConfig) -> None:
    log.info("=== Stage: paper ===")
    metrics_path = os.path.join(config.output_dir, "combined_metrics.json")
    if not Path(metrics_path).exists():
        raise EvalError(f"Run eval stage first — {metrics_path} not found.")
    with open(metrics_path) as f:
        metrics: dict[str, object] = json.load(f)
    with open(config.paper_template) as f:
        template = f.read()
    filled = inject_results(template, metrics)
    tex_out = Path(config.paper_output)
    tex_out.parent.mkdir(parents=True, exist_ok=True)
    with open(tex_out, "w") as f:
        f.write(filled)
    log.info("Paper LaTeX written to %s", tex_out)
    bib_src = Path(config.paper_template).parent / "references.bib"
    pdf = _compile_paper_pdf(tex_out, bib_src)
    if pdf is not None:
        log.info("Paper PDF written to %s", pdf)


def main() -> None:
    parser = argparse.ArgumentParser(description="kaggle2 KIE pipeline")
    parser.add_argument(
        "--stage", choices=["train", "eval", "paper", "all"], default="all",
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    seed_everything(config.seed)
    if args.stage in ("train", "all"):
        _stage_train(config)
    if args.stage in ("eval", "all"):
        _stage_eval(config)
    if args.stage in ("paper", "all"):
        _stage_paper(config)


if __name__ == "__main__":
    main()

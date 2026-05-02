"""Paper 2 — receipt key-information extraction with a non-learned modular pipeline.

This package is the self-contained Paper 2 system: end-to-end DONUT
fine-tuned under the original Kim et al. 2022 recipe versus a
deliberately non-learned modular pipeline (YOLOv8 + TrOCR + regex
field assignment + 3-state header/items/totals zone-prior HMM +
per-field deterministic post-processing) on the canonical SROIE
Task-3 split.

Paper 2 contains no learned cross-attention assigner, no structural
arithmetic verifier, no graph-attention head, no convolutional visual
features, no confidence-gated cascade.  Its contribution is a faithful
replication study with a documented catalogue of 14 silent
F1-destroying implementation pitfalls.

The Paper 2 system writes its run artefacts under ``paper2/runs/<id>/``,
loads its configuration from ``paper2/configs/default.json``, and
renders its manuscript through ``paper2/report/template_paper2.tex``.
None of these artefacts are read or written by the sibling Paper 3
system — the two systems are independent end-to-end.
"""

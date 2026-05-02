"""Paper 3 — Structure-Verified Document KIE (SVKIE) with FOCUS-Σ verification.

This package is the self-contained Paper 3 system: a multi-prior
neural framework for document key-information extraction in which
five orthogonal structural priors (arithmetic, spatial, visual,
lexical, epistemic) are each encoded by a dedicated module and a
unifying verification layer (FOCUS-Σ) enforces consistency across
priors via three arithmetic identities with provable soundness and
completeness.

Paper 3's pipeline replaces rule-based field assignment with a
three-headed neural ensemble (cross-attention assigner + graph-
attention over line-graph + frozen-CNN visual features, fused by a
learned gating MLP), grounds candidate values in FOCUS-Σ subset-sum
verification with Hamming-1/2 OCR-drift recovery, and falls through a
confidence-gated cascade when priors disagree.  The framework is
applied as an architecture-agnostic wrapper around DONUT, LayoutLMv3,
and the in-house FOCUS-T head, evaluated on canonical SROIE Task-3
plus CORD-v2 with multi-seed paired-bootstrap confidence intervals.

The Paper 3 system writes its run artefacts under ``paper3/runs/<id>/``,
loads its configuration from ``paper3/configs/default.json``, and
renders its manuscript through ``paper3/report/template_paper3.tex``.
None of these artefacts are read or written by the sibling Paper 2
system — the two systems are independent end-to-end.
"""

# Scope lock

**Paper:** The Two Questions of Trust (two-axis trust framework for numeric
document extraction).

**Status:** structure only. Experiments NOT run. Internal results-QA NOT done.
(These are the two items deliberately excluded from this build.)

**Venue plan**
- Primary: IJDAR (Int. J. on Document Analysis and Recognition).
- Safety: IEEE Access.
- Conference alternative: DAS (Document Analysis Systems).
- Explicitly NOT targeted: NeurIPS/ICML/ICLR main, TPAMI, JMLR. The paper's
  own "applied formalisation, no methodological-novelty claim" boundary is
  real and would be rejected there. Do not soften that boundary to chase tier.

**Relationship to the two ASYU papers (hard rule)**
- The two ASYU papers are complete, standalone results. This paper is a NEW
  paper for a different venue. It CITES them and uses each as a component.
- It must NOT re-claim their results, must NOT restate their core findings as
  this paper's novelty, and must NOT reduce them to fragments. Novelty here =
  framework + non-redundancy mechanism + integrated pre-registered evaluation
  + broadened evidence. See DELINEATION.md.

**Honesty invariants (carried from the component papers)**
- Axis B is a population-level signal, not a per-receipt detector; no
  per-receipt triage. Carried forward unsoftened.
- Framework is an applied formalisation; no methodological-novelty claim.
- Every reported number must trace to either (a) a cited prior-work fact, or
  (b) a new experiment actually run. Until run, it stays a visible
  `\pending{}` tag in main.tex. No fabricated numbers, ever.

**Build:** `latexmk -pdf main.tex` (skeleton uses `article`; camera-ready
will switch to the venue class).

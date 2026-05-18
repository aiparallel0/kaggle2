# New-vs-prior delineation

Required for a legitimate extended-version submission and to guarantee the
two standalone ASYU papers are not diluted. Every row states what was
established in prior work (cited, not re-claimed) vs. what is new here.

| Element | Prior ASYU work (cited, NOT re-claimed) | New in this paper |
|---|---|---|
| Symbolic gate (Axis A) | Subset-sum precision gate; pooled precision 0.989 (Wilson LB 0.961); orthogonal to softmax; either signal alone < 0.92 | Alternative-verifier bake-off; full precision-coverage-cost frontier; power-resolving per-corpus replication; real end-to-end deployment latency |
| Distributional signal (Axis B) | Beam-margin variance as a location-invariant shift signal; variance-ratio vs. KS on one natural pair; below-baseline per-receipt AUROC (structural, not retracted) | Multiple natural shift pairs; controlled mechanism study with ablations + negative controls; operational monitor numbers (prior F6/F7 were pending) |
| Trust framework | None: neither prior paper proposes a two-axis theory | Formal two-axis theory; composition operator; accept/abstain/flag-shift policy |
| Non-redundancy | Each prior paper only notes its own signal is "complementary" informally | Formal + empirical non-redundancy via error-decorrelation; mechanistic account |
| Integrated system | None: the two signals were never combined or co-evaluated | Shared released benchmark; pre-registered four-way head-to-head at matched false-alarm and matched cost |
| Boundary | Each prior paper states its own limitation | Limitations restated in the framework's language; carried forward unsoftened, not erased |

**Rule:** rows in the middle column are referenced via `\priorfact{}` /
citation only. Rows in the right column are the paper's contribution and are
`\pending{}` until the corresponding experiment is actually run.

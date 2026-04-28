"""PR-A / T-B — V2 → V3 prior-vector migration round-trip.

A legacy assigner checkpoint with a 9-d prior projection must load
into a V3 (14-d) :class:`AttentionAssigner` via
:func:`models.attention_assign.migrate_v2_checkpoint` with bit-exact
copies for every shared parameter.  The five extra prior columns are
zero-initialised so the migrated model is functionally identical to
the V2 model on receipts whose new distractor bits do not fire.

Skipped when torch is not installed.
"""
from __future__ import annotations

import pytest


def test_migrate_v2_to_v3_preserves_shared_weights(tmp_path: object) -> None:
    torch = pytest.importorskip("torch")
    from models.attention_assign import (
        N_TEXT_PRIORS_V2,
        N_TEXT_PRIORS_V3,
        AttentionAssigner,
        migrate_v2_checkpoint,
        save_assigner,
    )
    src = AttentionAssigner(
        hidden_dim=192, n_layers=3, n_heads=8, n_text_priors=N_TEXT_PRIORS_V2,
    )
    ckpt_path = str(tmp_path / "v2.pt")  # type: ignore[operator]
    save_assigner(src, ckpt_path)
    dst = migrate_v2_checkpoint(ckpt_path)
    assert dst.n_text_priors == N_TEXT_PRIORS_V3
    # Shared encoder weights must be bit-equal (not just close).
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    for k in src_sd:
        if k.startswith("prior_proj."):
            continue
        assert k in dst_sd, f"missing key after migrate: {k}"
        assert torch.equal(src_sd[k], dst_sd[k]), f"weight diverged at {k}"
    # New prior columns must be zero so the model is bit-equivalent on
    # receipts where the new distractor bits do not fire.
    sw = src_sd["prior_proj.weight"]
    dw = dst_sd["prior_proj.weight"]
    assert torch.equal(sw, dw[:, : sw.shape[1]])


def test_migrate_v2_rejects_non_v2(tmp_path: object) -> None:
    pytest.importorskip("torch")
    from models.attention_assign import (
        N_TEXT_PRIORS_V3,
        AttentionAssigner,
        migrate_v2_checkpoint,
        save_assigner,
    )
    src = AttentionAssigner(n_text_priors=N_TEXT_PRIORS_V3)
    ckpt = str(tmp_path / "v3.pt")  # type: ignore[operator]
    save_assigner(src, ckpt)
    with pytest.raises(ValueError, match="expected n_text_priors=9"):
        migrate_v2_checkpoint(ckpt)

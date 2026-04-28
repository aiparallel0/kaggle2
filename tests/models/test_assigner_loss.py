"""Unit tests for the composite assigner loss (Bug 18).

Pins the three new contracts:

  * CTKR pushes the BRN-line attention below ``a_min - 0.5*margin``
    after 200 SGD steps on a synthetic 8-line / 3-gold-line / BRN-
    confuser fixture (the parent prompt's plain Σ-neg-mass term fails
    this same assertion — that is the empirical justification for
    upgrading to CTKR).
  * Soft-IoU on the attention row reaches ``> 0.99`` once attention is
    one-hot on the gold mask, and ``0`` when there is zero overlap.
  * The address / total distractor masks fire on the canonical SROIE
    boilerplate lines named in the merge-gate spot-check.
"""
from __future__ import annotations

import pytest

from models.assigner_distractors import (
    address_distractor_mask,
    field_distractor_mask,
    total_distractor_mask,
)


def test_address_distractor_regex_fires_on_spot_check_set() -> None:
    """The merge-gate names {INV NO, INVOICE NO, CASHIER, BRN, GST NO,
    TAX INVOICE, TEL, FAX, ROC NO, TABLE} as the canonical address-
    field boilerplate; the distractor regex must flag every one."""
    lines = [
        "INV NO 12345",
        "INVOICE NO: 999",
        "CASHIER: ALICE",
        "BRN 0123456",
        "GST NO 0011223",
        "TAX INVOICE",
        "TEL: 03-1234567",
        "FAX 03-9999",
        "ROC NO 12345",
        "TABLE NO 7",
        "TABLE 7",
        # And an actual address line should NOT fire.
        "NO 12 JALAN MAJU 5",
    ]
    mask = address_distractor_mask(lines)
    assert mask[:-1] == [True] * 11
    assert mask[-1] is False


def test_total_distractor_regex_fires_on_spot_check_set() -> None:
    """SUBTOTAL / TAX / CHANGE / ROUNDING / CASH-PAID must all flag."""
    lines = [
        "SUBTOTAL RM 50.00",
        "SUB TOTAL 50.00",
        "TAX 3.00",
        "GST 6%",
        "CHANGE 2.00",
        "KEMBALIAN 2.00",
        "ROUNDING -0.01",
        "CASH TENDERED 100",
        "TENDERED 100",
        "PAID",
        # A grand-total-style line should NOT fire.
        "TOTAL RM 53.00",
    ]
    mask = total_distractor_mask(lines)
    assert mask[:-1] == [True] * 10
    assert mask[-1] is False


def test_field_distractor_mask_dispatch_no_op_for_unannotated_fields() -> None:
    """Company / date have no distractor regex — mask must be all False."""
    lines = ["foo", "bar", "baz"]
    assert field_distractor_mask("company", lines) == [False, False, False]
    assert field_distractor_mask("date", lines) == [False, False, False]


def test_soft_iou_attention_one_hot_match_is_unit() -> None:
    """``soft_iou_attention`` ≈ 0 when Â = m exactly."""
    torch = pytest.importorskip("torch")
    from models.assigner_loss import soft_iou_attention

    n = 6
    probs = torch.zeros(n)
    probs[2] = 1.0
    probs[3] = 1.0
    probs[4] = 1.0
    l_iou = soft_iou_attention(probs, [2, 3, 4])
    assert float(l_iou.item()) == pytest.approx(0.0, abs=1e-6)


def test_soft_iou_attention_zero_overlap_is_one() -> None:
    """``soft_iou_attention`` = 1 when Â and m have disjoint supports."""
    torch = pytest.importorskip("torch")
    from models.assigner_loss import soft_iou_attention

    n = 6
    probs = torch.zeros(n)
    probs[0] = 1.0  # row max on a non-gold line
    l_iou = soft_iou_attention(probs, [3, 4])
    # min(1.0, 0) + min(0,1)+min(0,1) = 0; max sum = 1+1+1 = 3 → 1 - 0/3 = 1
    assert float(l_iou.item()) == pytest.approx(1.0, abs=1e-6)


def test_ctkr_drives_brn_line_below_weakest_gold() -> None:
    """The CTKR convergence guarantee — 8-line / 3-gold / BRN fixture.

    Synthetic receipt: 8 lines, gold ``T_f = {3, 4, 5}``, BRN line at
    index 6.  Initialise ``A_f`` ≈ uniform.  After 200 SGD steps with
    ``L_pos + λ_ctkr * L_ctkr + λ_iou * L_iou_attn``, the BRN-line
    attention must sit at least ``0.5 * margin`` below the weakest
    gold attention.  The plain Σ-neg-mass term proposed in the parent
    prompt's step C does NOT satisfy this on this fixture (gradient is
    diluted across 5 negatives instead of concentrated on the BRN
    line); CTKR's referenced-to-a_min margin fixes that.
    """
    torch = pytest.importorskip("torch")
    from models.assigner_loss import composite_field_loss

    torch.manual_seed(0)
    n = 8
    pos_idxs = [3, 4, 5]
    # Distractor mask — BRN line at 6 is flagged; other negatives are
    # plain neutral lines (e.g. an item line at 0/1/2/7).
    distractor_mask = [False] * n
    distractor_mask[6] = True
    margin = 0.05

    # Free-parameter logits, softmax to produce ``probs``.
    logits = torch.zeros(n, requires_grad=True)
    opt = torch.optim.SGD([logits], lr=0.5)

    for _ in range(200):
        opt.zero_grad()
        probs = torch.softmax(logits, dim=0)
        loss, _, _ = composite_field_loss(
            probs, pos_idxs, distractor_mask,
            ctkr_k=4, ctkr_margin=margin,
            ctkr_weight=1.0, iou_weight=1.0,
        )
        loss.backward()
        opt.step()

    with torch.no_grad():
        final = torch.softmax(logits, dim=0)
        a_min = float(final[pos_idxs].min().item())
        a_brn = float(final[6].item())

    # Spec: A_{f, BRN} < a_min - 0.5 * margin.
    assert a_brn < a_min - 0.5 * margin, (
        f"CTKR did not separate BRN from weakest gold: "
        f"a_brn={a_brn:.4f}, a_min={a_min:.4f}, margin={margin}"
    )


def test_composite_field_loss_returns_diagnostic_floats() -> None:
    """Composite loss returns ``(Tensor, ctkr_active∈{0,1}, iou∈[0,1])``."""
    torch = pytest.importorskip("torch")
    from models.assigner_loss import composite_field_loss

    n = 5
    probs = torch.softmax(torch.zeros(n), dim=0)  # uniform 0.2
    loss, ctkr_active, iou_v = composite_field_loss(
        probs, pos_idxs=[1, 2], distractor_mask=[False] * n,
        ctkr_k=2, ctkr_margin=0.05, ctkr_weight=1.0, iou_weight=1.0,
    )
    assert isinstance(loss, torch.Tensor)
    assert ctkr_active in (0.0, 1.0)
    # IoU on uniform Â vs 2-of-5 mask: Â=1 everywhere after row-max
    # normalisation → inter = 2, union = 5 → IoU = 0.4.
    assert 0.0 <= iou_v <= 1.0

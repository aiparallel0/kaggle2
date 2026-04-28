"""PR-A / T-E — mini_assigner namespace + smoke contract.

The PR-A spec collapses the legacy ``gat_assigner`` graph-attention
opt-in into a "mini" assigner backbone (option-b in the plan).  This
test pins the new namespace + the fully-dense attention smoke contract
on five regions so any rename / API drift is caught in CI.

The mini path is config-gated; the smoke test exercises the module
import + tensor-typed forward only when torch is installed.
"""
from __future__ import annotations

import pytest


def test_mini_assigner_module_importable() -> None:
    """The mini-assigner namespace must be importable without torch.

    The implementation is a torch-only opt-in but the module shell
    must be reachable so callers can guard on ``hasattr(mini_assigner,
    'mini_assign')`` without crashing on a torch-less CI box.
    """
    import importlib

    mod = importlib.import_module("models.focus_mini")
    assert hasattr(mod, "mini_assign"), "mini_assigner missing public entry"
    assert hasattr(mod, "AssignerInput")
    assert hasattr(mod, "FieldAssignment")


def test_mini_assigner_dense_smoke_5_regions() -> None:
    """Fully-dense attention on 5 regions must run end-to-end."""
    torch = pytest.importorskip("torch")
    from models.focus_mini import AssignerInput, mini_assign

    n = 5
    inp = AssignerInput(
        texts=[f"line {i}" for i in range(n)],
        text_feats=torch.zeros(n, 768),
        bboxes=torch.tensor(
            [[0.0, i * 0.2, 1.0, (i + 1) * 0.2] for i in range(n)],
        ),
        priors=torch.zeros(n, 6),
        fields=["company", "address", "date", "total"],
    )

    class _Cfg:
        gat_enabled = True

    out = mini_assign(inp, _Cfg())
    assert isinstance(out.values, dict)
    assert set(out.values.keys()) == {"company", "address", "date", "total"}

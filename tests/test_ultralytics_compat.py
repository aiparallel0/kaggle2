"""Guard the ultralytics / PyTorch version compatibility.

PyTorch 2.6 changed the default for ``weights_only`` in ``torch.load`` from
``False`` to ``True``.  Ultralytics < 8.3.0 calls ``torch.load`` without
specifying ``weights_only``, which causes an ``UnpicklingError`` when loading
``yolov8n.pt`` because ``ultralytics.nn.tasks.DetectionModel`` is not an
allowed global under the new default.  Ultralytics 8.3.0 added the
``weights_only=False`` argument and ``torch.serialization.add_safe_globals``
registration to fix this.

This test enforces that the installed ultralytics satisfies the minimum
version so the crash is caught in CI rather than at training time on vast.ai.
"""
from __future__ import annotations

import importlib.metadata

import pytest


def test_ultralytics_version_at_least_8_3_0() -> None:
    """ultralytics must be >= 8.3.0 to work with PyTorch >= 2.6."""
    try:
        version_str = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("ultralytics not installed in this environment")

    from packaging.version import Version

    version = Version(version_str)
    assert version >= Version("8.3.0"), (
        f"ultralytics=={version_str} is incompatible with PyTorch>=2.6. "
        "torch.load now defaults to weights_only=True, which rejects "
        "ultralytics.nn.tasks.DetectionModel as an un-whitelisted global. "
        "Upgrade to ultralytics>=8.3.0."
    )

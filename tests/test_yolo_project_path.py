"""train_yolo must pass an absolute project path to ultralytics.

Bug 8: ultralytics >=8.3 resolves a *relative* ``project=`` against its
internal ``settings.runs_dir`` (defaults to ``runs/detect/``), not against
the current working directory.  Passing ``project="./results/yolo"``
therefore writes weights to ``./runs/detect/results/yolo/run/weights/best.pt``
while ``train_yolo`` (and downstream ``eval_pipeline``) look for them at
``./results/yolo/run/weights/best.pt`` — the eval stage then crashes with
``YOLO training finished but best.pt not found at ...``.

Resolving the path to absolute before passing to ``model.train`` makes the
two locations agree byte-for-byte.
"""
from __future__ import annotations

import inspect

from models import yolo_train


def test_train_yolo_uses_absolute_project_path() -> None:
    src = inspect.getsource(yolo_train.train_yolo)
    assert ".resolve()" in src, (
        "train_yolo must convert the project directory to an absolute path "
        "(Path(...).resolve()) before passing it to model.train.  Otherwise "
        "ultralytics >=8.3 prefixes 'runs/detect/' to it and the saved "
        "best.pt ends up in a directory the rest of the pipeline doesn't "
        "look in (Bug 8)."
    )

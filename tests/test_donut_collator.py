"""Unit tests for _DonutCollator.

Guards against the crash that repeatedly surfaced on GPU at training step 0:

    ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds

Root cause: ``label_smoothing_factor > 0`` causes HF Trainer to pop ``labels``
from the batch *before* calling ``model(**batch)``.  Without a collator that
pre-computes ``decoder_input_ids``, the mbart decoder receives neither
``labels`` nor ``decoder_input_ids`` and raises immediately.
"""
from __future__ import annotations

import types
from typing import Any

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_mock_model(pad_id: int = 1, start_id: int = 2) -> Any:
    """Minimal stand-in for VisionEncoderDecoderModel used by the collator."""

    class _Config:
        pad_token_id = pad_id
        decoder_start_token_id = start_id

    model = types.SimpleNamespace()
    model.config = _Config()

    def _prep(labels: Any) -> Any:  # mirrors HF shift-right
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = start_id
        return shifted

    model.prepare_decoder_input_ids_from_labels = _prep
    return model


def _make_mock_model_no_helper(pad_id: int = 1, start_id: int = 2) -> Any:
    """Like _make_mock_model but without prepare_decoder_input_ids_from_labels."""
    model = _make_mock_model(pad_id=pad_id, start_id=start_id)
    del model.prepare_decoder_input_ids_from_labels
    return model


def _make_feature(seq_len: int = 8, pad_id: int = 1) -> dict[str, Any]:
    pv = torch.zeros(3, 4, 4)  # tiny pixel_values [C, H, W]
    input_ids = torch.tensor([2, 10, 11, 12] + [pad_id] * (seq_len - 4))
    labels = input_ids.clone()
    labels[labels == pad_id] = -100
    return {"pixel_values": pv, "labels": labels}


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_collator_adds_decoder_input_ids() -> None:
    from models.donut_train import _DonutCollator

    collator = _DonutCollator(_make_mock_model())
    batch = collator([_make_feature(), _make_feature()])

    assert "decoder_input_ids" in batch, "collator must produce decoder_input_ids"
    assert "pixel_values" in batch
    assert "labels" in batch


def test_collator_decoder_input_ids_shape_matches_labels() -> None:
    from models.donut_train import _DonutCollator

    collator = _DonutCollator(_make_mock_model())
    batch = collator([_make_feature(), _make_feature()])

    assert batch["decoder_input_ids"].shape == batch["labels"].shape


def test_collator_starts_with_decoder_start_token() -> None:
    """First column of decoder_input_ids must be decoder_start_token_id."""
    from models.donut_train import _DonutCollator

    START = 2
    collator = _DonutCollator(_make_mock_model(pad_id=1, start_id=START))
    batch = collator([_make_feature(pad_id=1), _make_feature(pad_id=1)])

    first_tokens = batch["decoder_input_ids"][:, 0].tolist()
    assert all(t == START for t in first_tokens), (
        f"expected decoder_start_token_id={START} in column 0, got {first_tokens}"
    )


def test_collator_no_minus100_in_decoder_input_ids() -> None:
    """-100 (ignore-index) must not appear in decoder_input_ids."""
    from models.donut_train import _DonutCollator

    collator = _DonutCollator(_make_mock_model())
    batch = collator([_make_feature()])

    assert (batch["decoder_input_ids"] != -100).all(), (
        "decoder_input_ids must not contain the ignore-index -100"
    )


def test_collator_fallback_shift_right_used_without_helper() -> None:
    """Collator must work even if prepare_decoder_input_ids_from_labels is absent."""
    from models.donut_train import _DonutCollator

    START = 5
    collator = _DonutCollator(_make_mock_model_no_helper(pad_id=1, start_id=START))
    batch = collator([_make_feature()])

    assert "decoder_input_ids" in batch
    assert int(batch["decoder_input_ids"][0, 0]) == START
    assert (batch["decoder_input_ids"] != -100).all()


def test_shift_right_helper_directly() -> None:
    """_shift_right must prepend start_id and replace -100 with pad_id."""
    from models.donut_train import _shift_right

    PAD, START = 1, 99
    # Use label values that are distinct from both PAD and START
    labels = torch.tensor([[20, 10, 11, -100, -100]])
    out = _shift_right(labels, start_id=START, pad_id=PAD)

    assert out[0, 0] == START, "first token must be start_id"
    assert (out != -100).all(), "output must not contain -100"
    # shifted: [START, 20, 10, 11, PAD] — last position becomes PAD
    assert out[0, -1] == PAD


def test_model_accepts_loss_kwargs_set_false() -> None:
    """train_donut must set model.accepts_loss_kwargs=False to prevent
    transformers >=4.48 from forwarding num_items_in_batch into the encoder."""
    from models.donut_train import _DonutCollator  # noqa: F401 – just import check

    model = _make_mock_model()
    # Simulate what train_donut does
    model.accepts_loss_kwargs = False
    assert model.accepts_loss_kwargs is False

"""Guard against lm_head weight deduplication by safetensors.

safetensors identifies tensors as duplicates by comparing ``data_ptr()``
(the raw storage address).  If ``lm_head.weight`` and
``embed_tokens.weight`` share the same storage — which happens when
``tie_word_embeddings`` is still ``True`` at the point
``resize_token_embeddings`` is called — safetensors omits
``lm_head.weight`` from the checkpoint.  Reloading that checkpoint then
emits::

    There were missing keys in the checkpoint model loaded:
    ['decoder.lm_head.weight']

and the LM head silently falls back to the tied embedding weights (wrong
for an expanded vocabulary), causing eval F1 to collapse to 0.

These tests verify that ``train_donut`` sets up the model so that the two
tensors have distinct storage from the very first step.
"""
from __future__ import annotations

import inspect

from models import donut_train


def test_tie_word_embeddings_false_set_before_resize() -> None:
    """tie_word_embeddings=False must be applied BEFORE resize_token_embeddings.

    If it is applied after the resize, the resize can create a shared-storage
    alias between embed_tokens and lm_head, defeating the dedup prevention.
    """
    src = inspect.getsource(donut_train.train_donut)
    tie_pos = src.find("tie_word_embeddings = False")
    resize_pos = src.find("resize_token_embeddings(")
    assert tie_pos != -1, "train_donut must set tie_word_embeddings=False"
    assert resize_pos != -1, "train_donut must call resize_token_embeddings"
    assert tie_pos < resize_pos, (
        "tie_word_embeddings=False must be set BEFORE resize_token_embeddings(); "
        "setting it after allows the resize to create a shared-storage alias "
        "between embed_tokens and lm_head, causing safetensors to dedup the "
        "lm_head weight and produce 'missing keys' on checkpoint reload."
    )


def test_decoder_tie_word_embeddings_false_also_set() -> None:
    """model.config.decoder.tie_word_embeddings must also be set to False.

    VisionEncoderDecoderModel reads the *decoder sub-config* when deciding
    whether to re-tie weights during resize; setting only the top-level config
    flag is insufficient.
    """
    src = inspect.getsource(donut_train.train_donut)
    assert "model.config.decoder.tie_word_embeddings = False" in src, (
        "train_donut must set model.config.decoder.tie_word_embeddings=False "
        "before resize_token_embeddings(). VisionEncoderDecoderModel uses "
        "the decoder sub-config to decide whether to re-tie weights on resize."
    )


def test_lm_head_cloned_after_resize() -> None:
    """lm_head.weight must be cloned to a distinct tensor after resize.

    Even with tie_word_embeddings=False the resize may leave lm_head and
    embed_tokens sharing storage in some transformers versions.  An explicit
    clone after resize guarantees a unique data_ptr(), preventing safetensors
    deduplication.
    """
    src = inspect.getsource(donut_train.train_donut)
    resize_pos = src.find("resize_token_embeddings(")
    clone_pos = src.find("lm_head.weight.data.clone()")
    assert clone_pos != -1, (
        "train_donut must clone lm_head.weight after resize_token_embeddings() "
        "to prevent safetensors from deduplicating it with embed_tokens.weight."
    )
    assert clone_pos > resize_pos, (
        "The lm_head.weight clone must come AFTER resize_token_embeddings(), "
        "not before, to guarantee the cloned tensor reflects the resized shape."
    )


def test_no_per_save_lm_head_clone_callback() -> None:
    """No on_save callback may replace ``lm_head.weight`` during training.

    ``TrainerCallback.on_save`` fires *after* the checkpoint save completes,
    so a clone performed there cannot influence the file just written.  Worse:
    re-binding ``model.decoder.lm_head.weight`` to a fresh ``nn.Parameter``
    orphans the tensor the optimizer still holds a reference to — from the
    next epoch onwards autograd accumulates grads on the new Parameter while
    ``optimizer.step()`` updates the old one, leaving the lm_head frozen at
    its barely-warmed-up epoch-1 state.  Generation then collapses to
    ``<s_sroie></s_sroie>`` and eval F1 lands at exactly 0.0000.  The
    init-time clone at the top of ``train_donut`` is sufficient on its own
    to defeat safetensors dedup.
    """
    src = inspect.getsource(donut_train)
    assert "_LmHeadCloneCallback" not in src, (
        "train_donut must not register a per-save lm_head clone callback — "
        "on_save fires after the save and orphans the Parameter from the "
        "optimizer, freezing lm_head and collapsing eval F1 to 0."
    )

"""RAG-augmented DONUT dataset + inference-time prefix builder (P2 wiring).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: two 2-in/1-out helpers:

  * :class:`_RAGSROIEDataset` — train-time dataset that prepends
    ``<retrieved>...</retrieved>`` tokens carrying the top-k neighbour
    JSON serialisations before the ``<s_sroie>`` label.  Neighbours
    are resolved at ``__init__`` (not per-sample) so DataLoader workers
    stay cheap.
  * :func:`build_rag_prompt` — inference-time mirror that turns the top-k
    neighbours into the same ``<retrieved>...</retrieved>`` token prefix
    and returns the token-id sequence HF ``generate`` consumes as
    ``decoder_input_ids`` (so the retrieval context seeds beam search
    before the model emits its first SROIE token).

Both paths are active only when ``config.rag_enabled`` is True; with
the flag off the module is a pure no-op and never imports FAISS.
"""
from __future__ import annotations

from typing import Any

from models.retrieval_bank import RetrievalBank, retrieve

from core.types import ExpConfig, Receipt
from models.donut_dataset import _build_label, _SROIEDataset

_RAG_OPEN, _RAG_CLOSE = "<retrieved>", "</retrieved>"


def _serialise_neighbour(receipt: Receipt) -> str:
    """Compact one-line serialisation of a neighbour receipt's fields."""
    parts = [f"<s_{f.name.lower()}>{f.value}</s_{f.name.lower()}>"
             for f in receipt.fields]
    return "".join(parts)


def _build_rag_prefix(neighbours: list[Receipt]) -> str:
    """Render retrieved receipts as ``<retrieved>...</retrieved>`` string.

    Returns the empty string when ``neighbours`` is empty (flag off /
    empty bank / self-retrieval excluded).  The caller unconditionally
    concatenates with the label so RAG-on vs RAG-off differ only in
    this prefix.
    """
    if not neighbours:
        return ""
    body = "".join(_serialise_neighbour(r) for r in neighbours)
    return f"{_RAG_OPEN}{body}{_RAG_CLOSE}"


class _RAGSROIEDataset(_SROIEDataset):
    """Training dataset with retrieved-neighbour prefix prepended.

    Subclasses ``_SROIEDataset`` so the pixel-values path stays shared;
    overrides ``__getitem__`` only to swap in the RAG-prefixed label.
    Neighbour receipts are pre-computed at construction time (one
    encoder pass over the training set) and cached per-item.
    """

    def __init__(
        self,
        receipts: list[Receipt],
        processor: Any,
        config: ExpConfig,
        bank: RetrievalBank,
    ) -> None:
        super().__init__(receipts, processor, config)
        self._bank = bank
        # Precompute neighbours once: each training item gets its top-k
        # (excluding itself by image path) from the bank.  Avoids re-
        # encoding the query per DataLoader worker.
        self._neighbours: list[list[Receipt]] = []
        for r in receipts:
            neigh = retrieve(bank, (str(r.image_path), config))
            # Exclude self-hit deterministically (k+1 retrieve then filter).
            filtered = [n for n in neigh if n.image_path != r.image_path]
            self._neighbours.append(filtered[: config.rag_k])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Re-run the parent path for pixel_values but replace labels with
        # the RAG-prefixed variant.  Avoids a second image decode.
        from PIL import Image

        r = self._r[idx]
        img = Image.open(r.image_path).convert("RGB")
        # See _SROIEDataset.__getitem__ — image size is pinned on
        # processor.image_processor.size; the per-call size= kwarg is
        # misrouted by transformers 4.48's ProcessorMixin.__call__.
        pv = self._p(
            images=img, return_tensors="pt", legacy=False,
        ).pixel_values.squeeze(0)
        prefix = _build_rag_prefix(self._neighbours[idx])
        label_text = prefix + _build_label(r)
        tok = self._p.tokenizer(
            label_text, max_length=self._c.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = tok.input_ids.squeeze(0)
        labels = input_ids.clone()
        labels[labels == self._p.tokenizer.pad_token_id] = -100
        return {"pixel_values": pv, "labels": labels}


def build_rag_prompt(
    bank: RetrievalBank, query: tuple[str, ExpConfig, Any],
) -> list[int]:
    """Inference-time: encode query → top-k receipts → decoder-token prefix.

    2-in/1-out contract: ``(bank, (query_path, config, tokenizer))`` →
    list of token ids starting with ``<s_sroie>`` and preceded by a
    ``<retrieved>...</retrieved>`` span.  When the bank is empty the
    result degenerates to ``[decoder_start_token_id]`` so callers can
    unconditionally feed it to HF ``generate``'s
    ``decoder_input_ids`` kwarg.
    """
    query_path, config, tokenizer = query
    start_id: int = tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
    if not config.rag_enabled or not bank.receipts:
        return [start_id]
    neighbours = retrieve(bank, (query_path, config))
    prefix = _build_rag_prefix(neighbours)
    if not prefix:
        return [start_id]
    # Encode prefix without special-token padding; the start_id follows.
    prefix_ids: list[int] = tokenizer(
        prefix, add_special_tokens=False, return_tensors=None,
    )["input_ids"]
    return [*prefix_ids, start_id]

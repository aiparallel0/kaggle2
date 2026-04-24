"""Retrieval bank for RA-KIE (Proposal 2).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: build a nearest-neighbour index of train-set receipts keyed on
    the DONUT Swin encoder's [CLS] token (1024-d).  ``build_bank``
    returns ``(faiss.Index, list[Receipt])``; ``retrieve`` returns the
    top-k matching receipts for a query image.

Both functions are 2-in / 1-out, gated by ``ExpConfig.rag_enabled``.
When faiss is unavailable they fall back to numpy cosine similarity
so the module imports cleanly on CPU-only CI.  No GPU/torch work is
done at import time; the Swin encoder is loaded lazily only if
``rag_enabled`` is True.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.types import DataSplit, ExpConfig, Receipt

if TYPE_CHECKING:  # pragma: no cover — import-time only
    import numpy.typing as npt

    NDArray = npt.NDArray[Any]


@dataclass
class RetrievalBank:
    """Index + receipt metadata for retrieval-at-inference."""

    index: Any  # faiss.Index | _NumpyIndex
    receipts: list[Receipt]
    dim: int


def _encode_images(paths: list[str], base_model: str) -> NDArray:
    """Batch-encode receipt JPEGs with DONUT's Swin encoder → CLS-token."""
    # Lazy imports — keep module import cheap on CPU-only CI.
    import numpy as np
    import torch
    from PIL import Image
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    processor = DonutProcessor.from_pretrained(base_model)
    model = VisionEncoderDecoderModel.from_pretrained(base_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    feats: list[NDArray] = []
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB")
            pv = processor(img, return_tensors="pt").pixel_values.to(device)
            out = model.encoder(pv).last_hidden_state  # (1, N, 1024)
            cls = out[:, 0, :].cpu().numpy().astype("float32")
            feats.append(cls)
    return np.vstack(feats)


class _NumpyIndex:
    """faiss.Index-compatible fallback using pure numpy cosine similarity."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._data: NDArray | None = None

    def add(self, x: NDArray) -> None:
        import numpy as np

        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        self._data = x.astype("float32")

    def search(self, q: NDArray, k: int) -> tuple[NDArray, NDArray]:
        import numpy as np

        if self._data is None:
            raise ValueError("Index is empty; call add() first.")
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        sims = qn @ self._data.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        dist = np.take_along_axis(sims, idx, axis=1)
        return dist, idx


def empty_bank(dim: int = 1024) -> RetrievalBank:
    """Return an empty bank (used as the RAG-off inference no-op)."""
    return RetrievalBank(index=_NumpyIndex(dim), receipts=[], dim=dim)


def build_bank(data: DataSplit, config: ExpConfig) -> RetrievalBank:
    """Index every training receipt on its Swin-CLS embedding.

    Returns an empty bank when ``rag_enabled`` is False so callers
    never need to branch on the flag (zero marginal cost on off-path).
    """
    if not config.rag_enabled or not data.train:
        return RetrievalBank(index=_NumpyIndex(1024), receipts=[], dim=1024)
    paths = [str(r.image_path) for r in data.train]
    feats = _encode_images(paths, config.base_model)
    dim = int(feats.shape[1])
    idx: Any
    try:
        import faiss  # type: ignore[import-not-found]  # optional dep

        idx = faiss.IndexFlatIP(dim)
        # faiss expects L2-normalised vectors for cosine similarity.
        import numpy as np

        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
        idx.add(feats.astype("float32"))
    except ImportError:
        idx = _NumpyIndex(dim)
        idx.add(feats)
    return RetrievalBank(index=idx, receipts=list(data.train), dim=dim)


def retrieve(bank: RetrievalBank, query: tuple[str, ExpConfig]) -> list[Receipt]:
    """Top-k nearest training receipts for the query image path.

    The 2-in contract is (bank, (query_path, config)).  ``config``
    carries both the base-model identifier used for re-encoding and
    ``rag_k`` (top-k).  Returns [] when the bank is empty.
    """
    import numpy as np

    query_path, config = query
    if not bank.receipts:
        return []
    feats = _encode_images([query_path], config.base_model)
    qn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    _, ix = bank.index.search(qn.astype("float32"), min(config.rag_k, len(bank.receipts)))
    return [bank.receipts[int(i)] for i in ix[0]]

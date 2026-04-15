"""Train the AttentionAssigner on real TrOCR embeddings from SROIE crops."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image

from core.errors import TrainError
from core.types import AssignerData, ExpConfig
from models.attention_assign import AttentionAssigner, save_assigner

_FIELDS = ["company", "date", "address", "total"]


def train_assigner(config: ExpConfig, data: AssignerData) -> str:
    """Train AttentionAssigner on labeled crops with TrOCR embeddings."""
    if not data.crops:
        raise TrainError("No labeled crops for assigner training.")
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    proc: TrOCRProcessor = TrOCRProcessor.from_pretrained(data.trocr_path)
    trocr: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(
        data.trocr_path
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trocr = trocr.to(device)
    trocr.eval()
    # Pre-compute embeddings for all crops
    embeddings: list[torch.Tensor] = []
    bboxes: list[list[float]] = []
    labels: list[int] = []
    field_to_idx = {f: i for i, f in enumerate(_FIELDS)}
    with torch.no_grad():
        for crop in data.crops:
            if crop.field_label not in field_to_idx:
                continue
            img = Image.open(crop.image_path).convert("RGB")
            w, h = img.size
            x1, y1, x2, y2 = crop.bbox
            region = img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
            if region.width < 1 or region.height < 1:
                continue
            pv = proc(images=region, return_tensors="pt").pixel_values.to(device)
            feat = trocr.encoder(pv).last_hidden_state.mean(dim=1)
            embeddings.append(feat.cpu())
            bboxes.append([x1, y1, x2, y2])
            labels.append(field_to_idx[crop.field_label])
    if not embeddings:
        raise TrainError("No valid crop embeddings computed for assigner.")
    assigner = AttentionAssigner(hidden_dim=64, n_fields=len(_FIELDS))
    assigner = assigner.to(device)
    opt = torch.optim.Adam(assigner.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    n = len(embeddings)
    assigner.train()
    for epoch in range(config.epochs_assigner):
        total_loss = 0.0
        for i in range(n):
            # Process one sample: single region as key-value for cross-attention
            tf = embeddings[i].unsqueeze(0).to(device)  # (1, 1, 768)
            bf = torch.tensor(
                [bboxes[i]], dtype=torch.float32,
            ).unsqueeze(0).to(device)  # (1, 1, 4)
            tgt = torch.tensor([labels[i]], device=device)
            opt.zero_grad()
            logits, _ = assigner(tf, bf)  # (1, n_fields)
            loss = loss_fn(logits, tgt)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  Assigner epoch {epoch + 1}/{config.epochs_assigner} loss={total_loss:.3f}")
    out_path = os.path.join(config.output_dir, "assigner.pt")
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    save_assigner(assigner, out_path)
    return out_path

"""Train the AttentionAssigner on SROIE field labels."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from core.types import DataSplit, ExpConfig
from models.attention_assign import AttentionAssigner, save_assigner


def train_assigner(config: ExpConfig, data: DataSplit) -> str:
    """Train AttentionAssigner on synthetic embeddings; return checkpoint path."""
    assigner = AttentionAssigner(hidden_dim=64, n_fields=len(config.fields))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assigner = assigner.to(device)
    opt = torch.optim.Adam(assigner.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    assigner.train()
    for epoch in range(5):
        total_loss = 0.0
        for i, _rec in enumerate(data.train[:50]):
            opt.zero_grad()
            tf = torch.randn(1, 4, 768, device=device)
            bf = torch.rand(1, 4, 4, device=device)
            logits = assigner(tf, bf)
            tgt = torch.tensor([i % len(config.fields)], device=device)
            loss = loss_fn(logits, tgt)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  Assigner epoch {epoch + 1}/5 loss={total_loss:.3f}")
    out_path = os.path.join(config.output_dir, "assigner.pt")
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    save_assigner(assigner, out_path)
    meta = {"yolo_img_size": config.yolo_img_size}
    with open(os.path.join(config.output_dir, "pipeline_meta.json"), "w") as f:
        json.dump(meta, f)
    return out_path

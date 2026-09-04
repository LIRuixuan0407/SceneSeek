from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from sceneseek.training.data import TemporalFeatureDataset, collate_temporal_batch
from sceneseek.training.evaluate import evaluate_temporal_checkpoint
from sceneseek.training.losses import multi_positive_contrastive_loss
from sceneseek.training.temporal_adapter import TemporalAdapterConfig, create_temporal_adapter


def train_temporal_adapter(
    feature_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    temperature: float = 0.07,
    num_layers: int = 2,
    num_heads: int = 8,
    dropout: float = 0.1,
    max_frames: int = 32,
    device: str = "auto",
    seed: int = 7,
    num_workers: int = 0,
) -> dict[str, object]:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise RuntimeError("请先安装 `pip install -e '.[ml]'`") from error

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法访问 CUDA")

    train_data = TemporalFeatureDataset(feature_dir, split="train")
    config = TemporalAdapterConfig(
        dimension=train_data.dimension,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_frames=max_frames,
    )
    model = create_temporal_adapter(config).to(device)
    loader = DataLoader(
        train_data,
        batch_size=max(1, batch_size),
        shuffle=True,
        num_workers=max(0, num_workers),
        collate_fn=collate_temporal_batch,
        pin_memory=device.startswith("cuda"),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "temporal-adapter.pt"
    history: list[dict[str, float | int]] = []
    best_r10 = float("-inf")
    best_epoch = 0

    for epoch in range(1, max(1, epochs) + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            texts = batch["texts"].to(device, non_blocking=True)
            if frames.shape[1] > config.max_frames:
                frames = frames[:, : config.max_frames]
                mask = mask[:, : config.max_frames]
            video_embeddings = model(frames, mask)
            loss = multi_positive_contrastive_loss(
                video_embeddings,
                texts,
                batch["video_ids"],
                temperature=temperature,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1

        mean_loss = total_loss / max(1, batches)
        torch.save(
            {
                "config": config.to_dict(),
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "train_loss": mean_loss,
                "feature_dir": str(feature_dir.resolve()),
            },
            checkpoint,
        )
        try:
            metrics = evaluate_temporal_checkpoint(
                feature_dir,
                checkpoint,
                split="val",
                device=device,
                batch_size=batch_size,
            )
        except ValueError:
            metrics = {"r@10": float("nan")}

        record: dict[str, float | int] = {"epoch": epoch, "train_loss": mean_loss}
        for key, value in metrics.items():
            if key not in {"queries", "videos"}:
                record[f"val_{key}"] = value
        history.append(record)
        r10 = float(metrics.get("r@10", float("nan")))
        if np.isfinite(r10) and r10 > best_r10:
            best_r10 = r10
            best_epoch = epoch
            torch.save(
                {
                    "config": config.to_dict(),
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "train_loss": mean_loss,
                    "val_metrics": metrics,
                    "feature_dir": str(feature_dir.resolve()),
                },
                output_dir / "best.pt",
            )

    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "device": device,
        "epochs": max(1, epochs),
        "best_epoch": best_epoch,
        "best_val_r@10": best_r10 if np.isfinite(best_r10) else None,
        "checkpoint": str(checkpoint.resolve()),
        "best_checkpoint": str((output_dir / "best.pt").resolve()) if best_epoch else None,
        "history": str((output_dir / "history.json").resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary

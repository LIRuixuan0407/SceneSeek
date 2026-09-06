from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from sceneseek.training.data import TemporalFeatureDataset, collate_temporal_batch
from sceneseek.training.evaluate import evaluate_temporal_checkpoint
from sceneseek.training.hard_negatives import HardNegativeBatchSampler, HardNegativeLookup
from sceneseek.training.losses import (
    hard_negative_margin_loss,
    multi_positive_contrastive_loss,
)
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
    selection_metric: str = "mrr",
    early_stopping_patience: int = 3,
    min_delta: float = 0.0,
    hard_negative_index: Path | None = None,
    hard_negative_pool: int = 16,
    hard_negative_mode: str = "sampler",
    hard_negative_loss_weight: float = 0.25,
    hard_negative_margin: float = 0.05,
) -> dict[str, object]:
    try:
        import torch
        import torch.nn.functional as F
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

    if selection_metric not in {"mrr", "r@1", "r@5", "r@10"}:
        raise ValueError("selection_metric 必须是 mrr/r@1/r@5/r@10")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience 不能小于 0")
    if min_delta < 0:
        raise ValueError("min_delta 不能小于 0")
    if hard_negative_mode not in {"sampler", "ranking"}:
        raise ValueError("hard_negative_mode 必须是 sampler/ranking")
    if hard_negative_loss_weight < 0:
        raise ValueError("hard_negative_loss_weight 不能小于 0")
    if hard_negative_margin < 0:
        raise ValueError("hard_negative_margin 不能小于 0")
    if hard_negative_index is None and hard_negative_mode == "ranking":
        raise ValueError("ranking 模式需要 --hard-negatives")

    train_data = TemporalFeatureDataset(feature_dir, split="train")
    config = TemporalAdapterConfig(
        dimension=train_data.dimension,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_frames=max_frames,
    )
    model = create_temporal_adapter(config).to(device)
    loader_kwargs = {
        "num_workers": max(0, num_workers),
        "collate_fn": collate_temporal_batch,
        "pin_memory": device.startswith("cuda"),
    }
    ranking_lookup = None
    if hard_negative_index is not None and hard_negative_mode == "sampler":
        batch_sampler = HardNegativeBatchSampler(
            train_data,
            hard_negative_index,
            batch_size=max(2, batch_size),
            candidate_pool=hard_negative_pool,
            seed=seed,
        )
        loader = DataLoader(train_data, batch_sampler=batch_sampler, **loader_kwargs)
    else:
        loader = DataLoader(
            train_data,
            batch_size=max(1, batch_size),
            shuffle=True,
            **loader_kwargs,
        )
        if hard_negative_index is not None and hard_negative_mode == "ranking":
            ranking_lookup = HardNegativeLookup(
                train_data,
                hard_negative_index,
                candidate_pool=hard_negative_pool,
                seed=seed,
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "temporal-adapter.pt"
    history: list[dict[str, float | int]] = []
    best_score = float("-inf")
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    stale_epochs = 0
    stopped_early = False

    for epoch in range(1, max(1, epochs) + 1):
        model.train()
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_ranking_loss = 0.0
        total_active_fraction = 0.0
        total_score_gap = 0.0
        ranking_batches = 0
        batches = 0

        for step, batch in enumerate(loader):
            frames = batch["frames"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            texts = batch["texts"].to(device, non_blocking=True)
            if frames.shape[1] > config.max_frames:
                frames = frames[:, : config.max_frames]
                mask = mask[:, : config.max_frames]

            video_embeddings = model(frames, mask)
            contrastive_loss = multi_positive_contrastive_loss(
                video_embeddings,
                texts,
                batch["video_ids"],
                temperature=temperature,
            )
            loss = contrastive_loss

            if ranking_lookup is not None:
                negative_indices = ranking_lookup.sample_dataset_indices(
                    batch["sample_ids"],
                    epoch=epoch,
                    step=step,
                )
                negative_items = [train_data[index] for index in negative_indices]
                negative_batch = collate_temporal_batch(negative_items)
                negative_frames = negative_batch["frames"].to(device, non_blocking=True)
                negative_mask = negative_batch["mask"].to(device, non_blocking=True)
                if negative_frames.shape[1] > config.max_frames:
                    negative_frames = negative_frames[:, : config.max_frames]
                    negative_mask = negative_mask[:, : config.max_frames]

                negative_embeddings = model(negative_frames, negative_mask)
                ranking_loss = hard_negative_margin_loss(
                    texts,
                    video_embeddings,
                    negative_embeddings,
                    margin=hard_negative_margin,
                )
                loss = loss + hard_negative_loss_weight * ranking_loss

                with torch.no_grad():
                    normalized_texts = F.normalize(texts, dim=-1)
                    normalized_positives = F.normalize(video_embeddings, dim=-1)
                    normalized_negatives = F.normalize(negative_embeddings, dim=-1)
                    positive_scores = (normalized_texts * normalized_positives).sum(dim=-1)
                    negative_scores = (normalized_texts * normalized_negatives).sum(dim=-1)
                    violations = hard_negative_margin - positive_scores + negative_scores
                    active_fraction = (violations > 0).float().mean()
                    score_gap = (positive_scores - negative_scores).mean()
                total_ranking_loss += float(ranking_loss.detach().cpu())
                total_active_fraction += float(active_fraction.detach().cpu())
                total_score_gap += float(score_gap.detach().cpu())
                ranking_batches += 1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_contrastive_loss += float(contrastive_loss.detach().cpu())
            batches += 1

        mean_loss = total_loss / max(1, batches)
        mean_contrastive_loss = total_contrastive_loss / max(1, batches)
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

        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "train_contrastive_loss": mean_contrastive_loss,
        }
        if ranking_batches:
            record.update(
                {
                    "train_hard_negative_loss": total_ranking_loss / ranking_batches,
                    "train_hard_negative_active_fraction": (
                        total_active_fraction / ranking_batches
                    ),
                    "train_hard_negative_score_gap": total_score_gap / ranking_batches,
                }
            )
        for key, value in metrics.items():
            if key not in {"queries", "videos"}:
                record[f"val_{key}"] = value
        history.append(record)

        score = float(metrics.get(selection_metric, float("nan")))
        improved = np.isfinite(score) and score > best_score + min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            best_metrics = {key: float(value) for key, value in metrics.items()}
            stale_epochs = 0
            torch.save(
                {
                    "config": config.to_dict(),
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "train_loss": mean_loss,
                    "val_metrics": metrics,
                    "selection_metric": selection_metric,
                    "hard_negative_mode": (
                        hard_negative_mode if hard_negative_index is not None else None
                    ),
                    "hard_negative_loss_weight": (
                        hard_negative_loss_weight if ranking_lookup is not None else None
                    ),
                    "hard_negative_margin": (
                        hard_negative_margin if ranking_lookup is not None else None
                    ),
                    "feature_dir": str(feature_dir.resolve()),
                },
                output_dir / "best.pt",
            )
        elif np.isfinite(score):
            stale_epochs += 1

        if (
            early_stopping_patience > 0
            and best_epoch > 0
            and stale_epochs >= early_stopping_patience
        ):
            stopped_early = True
            break

    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "device": device,
        "epochs": max(1, epochs),
        "epochs_requested": max(1, epochs),
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "selection_metric": selection_metric,
        "best_epoch": best_epoch,
        f"best_val_{selection_metric}": best_score if np.isfinite(best_score) else None,
        "best_val_metrics": best_metrics,
        "hard_negative_index": (
            str(hard_negative_index.resolve()) if hard_negative_index is not None else None
        ),
        "hard_negative_mode": hard_negative_mode if hard_negative_index is not None else None,
        "hard_negative_pool": hard_negative_pool if hard_negative_index is not None else None,
        "hard_negative_loss_weight": (
            hard_negative_loss_weight if ranking_lookup is not None else None
        ),
        "hard_negative_margin": hard_negative_margin if ranking_lookup is not None else None,
        "checkpoint": str(checkpoint.resolve()),
        "best_checkpoint": str((output_dir / "best.pt").resolve()) if best_epoch else None,
        "history": str((output_dir / "history.json").resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from sceneseek.training.reranker import RerankerConfig, create_query_conditioned_reranker
from sceneseek.training.reranker_data import RerankerCandidateStore


def _collate_pairs(store: RerankerCandidateStore, query_indices: list[int], device: str):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("请先安装 `pip install -e '.[ml]'`") from error

    pair_items: list[dict[str, object]] = []
    pair_texts: list[np.ndarray] = []
    coarse_scores: list[float] = []
    target_positions: list[int] = []
    for query_index in query_indices:
        target_position = int(store.target_positions[query_index])
        if target_position < 0:
            raise ValueError("训练 candidates 必须包含 ground-truth video")
        target_positions.append(target_position)
        text = store.text(query_index)
        items = store.candidate_video_items(query_index)
        pair_items.extend(items)
        pair_texts.extend([text] * len(items))
        coarse_scores.extend(store.scores[query_index].tolist())

    max_steps = max(np.asarray(item["frames"]).shape[0] for item in pair_items)
    max_steps = min(max_steps, 10_000)
    dimension = store.dimension
    frames = torch.zeros(
        (len(pair_items), max_steps, dimension), dtype=torch.float32, device=device
    )
    mask = torch.zeros((len(pair_items), max_steps), dtype=torch.bool, device=device)
    for row, item in enumerate(pair_items):
        array = np.asarray(item["frames"], dtype=np.float32)
        steps = array.shape[0]
        frames[row, :steps] = torch.from_numpy(array).to(device)
        mask[row, :steps] = True
    texts = torch.from_numpy(np.stack(pair_texts).astype(np.float32, copy=False)).to(device)
    coarse = torch.tensor(coarse_scores, dtype=torch.float32, device=device)
    targets = torch.tensor(target_positions, dtype=torch.long, device=device)
    return texts, frames, mask, coarse, targets


def _metrics_from_ranks(ranks: np.ndarray, videos: int) -> dict[str, float]:
    values = np.asarray(ranks, dtype=np.int64)
    return {
        "queries": float(len(values)),
        "videos": float(videos),
        "r@1": float(np.mean(values <= 1)),
        "r@5": float(np.mean(values <= 5)),
        "r@10": float(np.mean(values <= 10)),
        "mrr": float(np.mean(1.0 / values)),
        "median_rank": float(np.median(values)),
        "mean_rank": float(np.mean(values)),
    }


def load_reranker_checkpoint(path: Path, device: str = "cpu"):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("请先安装 `pip install -e '.[ml]'`") from error

    payload = torch.load(path, map_location=device, weights_only=False)
    format_version = int(payload.get("format_version", 1))
    if format_version < 2:
        raise ValueError(
            "该 checkpoint 来自未约束 residual 的 reranker v1；请用 v2 重新训练后再评测"
        )
    config = RerankerConfig(**payload["config"])
    model = create_query_conditioned_reranker(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


def evaluate_reranker(
    feature_dir: Path,
    candidates: Path,
    checkpoint: Path,
    *,
    split: str,
    device: str = "auto",
    batch_size: int = 16,
) -> dict[str, float]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("请先安装 `pip install -e '.[ml]'`") from error

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, payload = load_reranker_checkpoint(checkpoint, device)
    store = RerankerCandidateStore(feature_dir, candidates, split=split)
    config = RerankerConfig(**payload["config"])

    reranked_ranks = store.target_ranks.copy()
    stage1_ranks = store.target_ranks.copy()
    present = np.flatnonzero(store.target_positions >= 0)
    oracle_ranks = stage1_ranks.copy()
    oracle_ranks[present] = 1
    absolute_delta_sum = 0.0
    absolute_delta_count = 0
    maximum_absolute_delta = 0.0

    for start in range(0, len(present), max(1, batch_size)):
        query_indices = present[start : start + batch_size].tolist()
        texts, frames, mask, coarse, _ = _collate_pairs(store, query_indices, device)
        if frames.shape[1] > config.max_frames:
            frames = frames[:, : config.max_frames]
            mask = mask[:, : config.max_frames]
        with torch.inference_mode():
            flat_scores = model(texts, frames, mask, coarse)
        absolute_delta = (flat_scores - coarse).abs()
        absolute_delta_sum += float(absolute_delta.sum().cpu())
        absolute_delta_count += int(absolute_delta.numel())
        if absolute_delta.numel():
            maximum_absolute_delta = max(
                maximum_absolute_delta, float(absolute_delta.max().cpu())
            )
        scores = flat_scores.view(len(query_indices), store.top_k).cpu().numpy()
        for row, query_index in enumerate(query_indices):
            target_position = int(store.target_positions[query_index])
            target_score = float(scores[row, target_position])
            reranked_ranks[query_index] = 1 + int(np.sum(scores[row] > target_score))

    result = _metrics_from_ranks(reranked_ranks, len(store.video_ids))
    stage1 = _metrics_from_ranks(stage1_ranks, len(store.video_ids))
    oracle = _metrics_from_ranks(oracle_ranks, len(store.video_ids))
    result.update(
        {
            f"stage1_{key}": value
            for key, value in stage1.items()
            if key not in {"queries", "videos"}
        }
    )
    result.update(
        {
            f"oracle_{key}": value
            for key, value in oracle.items()
            if key not in {"queries", "videos"}
        }
    )
    for key in ("r@1", "r@5", "r@10", "mrr"):
        result[f"delta_{key}"] = result[key] - stage1[key]
    result["candidate_recall"] = float(np.mean(store.target_positions >= 0))
    result["top_k"] = float(store.top_k)
    result["mean_abs_delta"] = absolute_delta_sum / max(1, absolute_delta_count)
    result["max_abs_delta"] = maximum_absolute_delta
    return result


def train_reranker(
    feature_dir: Path,
    train_candidates: Path,
    val_candidates: Path,
    output_dir: Path,
    *,
    epochs: int = 8,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    num_heads: int = 8,
    hidden_dimension: int = 512,
    dropout: float = 0.1,
    max_frames: int = 16,
    device: str = "auto",
    seed: int = 7,
    selection_metric: str = "mrr",
    early_stopping_patience: int = 3,
    min_delta: float = 0.0,
    temperature: float = 0.05,
    residual_scale: float = 0.05,
) -> dict[str, object]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as error:
        raise RuntimeError("请先安装 `pip install -e '.[ml]'`") from error

    if selection_metric not in {"mrr", "r@1", "r@5", "r@10"}:
        raise ValueError("selection_metric 必须是 mrr/r@1/r@5/r@10")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience 不能小于 0")
    if min_delta < 0:
        raise ValueError("min_delta 不能小于 0")
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    if residual_scale <= 0:
        raise ValueError("residual_scale 必须大于 0")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法访问 CUDA")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    candidate_metadata_path = train_candidates.with_suffix(train_candidates.suffix + ".json")
    if candidate_metadata_path.is_file():
        candidate_metadata = json.loads(candidate_metadata_path.read_text(encoding="utf-8"))
        if bool(candidate_metadata.get("ensure_positive", False)):
            raise ValueError(
                "train candidates 由 --ensure-positive 构建；"
                "请重新构建自然 Stage-1 Top-K candidates 后训练 reranker"
            )

    train_store = RerankerCandidateStore(feature_dir, train_candidates, split="train")
    injected_positive = (train_store.target_positions >= 0) & (
        train_store.target_ranks > train_store.top_k
    )
    if np.any(injected_positive):
        raise ValueError(
            "train candidates 包含注入的 ground-truth；"
            "请重新构建自然 Stage-1 Top-K candidates 后训练 reranker"
        )
    config = RerankerConfig(
        dimension=train_store.dimension,
        num_heads=num_heads,
        hidden_dimension=hidden_dimension,
        dropout=dropout,
        max_frames=max_frames,
        residual_scale=residual_scale,
    )
    model = create_query_conditioned_reranker(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "reranker.pt"
    best_checkpoint = output_dir / "best.pt"
    history: list[dict[str, float | int]] = []
    stale_epochs = 0
    stopped_early = False

    train_indices = [
        index for index in range(len(train_store)) if train_store.target_positions[index] >= 0
    ]
    if not train_indices:
        raise ValueError("训练 candidates 中没有包含 positive 的 query")

    identity_payload = {
        "format_version": 2,
        "config": config.to_dict(),
        "state_dict": model.state_dict(),
        "epoch": 0,
        "train_loss": None,
        "temperature": temperature,
        "train_candidates": str(train_candidates.resolve()),
        "val_candidates": str(val_candidates.resolve()),
    }
    torch.save(identity_payload, best_checkpoint)
    baseline_metrics = evaluate_reranker(
        feature_dir,
        val_candidates,
        best_checkpoint,
        split="val",
        device=device,
        batch_size=batch_size,
    )
    best_score = float(baseline_metrics[selection_metric])
    best_epoch = 0
    best_metrics: dict[str, float] = {
        key: float(value) for key, value in baseline_metrics.items()
    }

    for epoch in range(1, max(1, epochs) + 1):
        model.train()
        rng = random.Random(seed + epoch)
        rng.shuffle(train_indices)
        total_loss = 0.0
        total_absolute_delta = 0.0
        total_delta_count = 0
        maximum_absolute_delta = 0.0
        batches = 0
        for start in range(0, len(train_indices), max(1, batch_size)):
            query_indices = train_indices[start : start + batch_size]
            texts, frames, mask, coarse, targets = _collate_pairs(
                train_store, query_indices, device
            )
            if frames.shape[1] > config.max_frames:
                frames = frames[:, : config.max_frames]
                mask = mask[:, : config.max_frames]
            flat_scores = model(texts, frames, mask, coarse)
            absolute_delta = (flat_scores - coarse).detach().abs()
            total_absolute_delta += float(absolute_delta.sum().cpu())
            total_delta_count += int(absolute_delta.numel())
            if absolute_delta.numel():
                maximum_absolute_delta = max(
                    maximum_absolute_delta, float(absolute_delta.max().cpu())
                )
            scores = flat_scores.view(len(query_indices), train_store.top_k)
            loss = F.cross_entropy(scores / temperature, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1

        mean_loss = total_loss / max(1, batches)
        mean_absolute_delta = total_absolute_delta / max(1, total_delta_count)
        torch.save(
            {
                "format_version": 2,
                "config": config.to_dict(),
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "train_loss": mean_loss,
                "temperature": temperature,
                "train_candidates": str(train_candidates.resolve()),
                "val_candidates": str(val_candidates.resolve()),
            },
            checkpoint,
        )
        metrics = evaluate_reranker(
            feature_dir,
            val_candidates,
            checkpoint,
            split="val",
            device=device,
            batch_size=batch_size,
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "train_mean_abs_delta": mean_absolute_delta,
            "train_max_abs_delta": maximum_absolute_delta,
        }
        for key, value in metrics.items():
            if key not in {"queries", "videos"}:
                record[f"val_{key}"] = value
        history.append(record)

        score = float(metrics[selection_metric])
        improved = np.isfinite(score) and score > best_score + min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            best_metrics = {key: float(value) for key, value in metrics.items()}
            stale_epochs = 0
            torch.save(
                {
                    "format_version": 2,
                    "config": config.to_dict(),
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "train_loss": mean_loss,
                    "temperature": temperature,
                    "val_metrics": metrics,
                    "selection_metric": selection_metric,
                    "train_candidates": str(train_candidates.resolve()),
                    "val_candidates": str(val_candidates.resolve()),
                },
                best_checkpoint,
            )
        else:
            stale_epochs += 1

        if early_stopping_patience > 0 and stale_epochs >= early_stopping_patience:
            stopped_early = True
            break

    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "device": device,
        "epochs_requested": max(1, epochs),
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "selection_metric": selection_metric,
        "temperature": temperature,
        "residual_scale": residual_scale,
        "best_epoch": best_epoch,
        f"best_val_{selection_metric}": best_score if np.isfinite(best_score) else None,
        "baseline_val_metrics": baseline_metrics,
        "best_val_metrics": best_metrics,
        "train_queries": len(train_indices),
        "train_candidate_recall": float(len(train_indices) / len(train_store)),
        "top_k": train_store.top_k,
        "checkpoint": str(checkpoint.resolve()),
        "best_checkpoint": str(best_checkpoint.resolve()),
        "history": str((output_dir / "history.json").resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary

from __future__ import annotations

from collections.abc import Sequence


def multi_positive_contrastive_loss(
    video_embeddings,
    text_embeddings,
    group_ids: Sequence[str],
    temperature: float = 0.07,
):
    """Symmetric InfoNCE where captions from the same video are not negatives."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as error:
        raise RuntimeError("请先安装 `pip install -e '.[ml]'` 以训练模型") from error

    if video_embeddings.shape != text_embeddings.shape:
        raise ValueError("video_embeddings 与 text_embeddings shape 必须一致")
    if video_embeddings.ndim != 2:
        raise ValueError("embedding 必须是 [batch, dim]")
    if len(group_ids) != video_embeddings.shape[0]:
        raise ValueError("group_ids 数量必须与 batch size 一致")
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")

    videos = F.normalize(video_embeddings, dim=-1)
    texts = F.normalize(text_embeddings, dim=-1)
    logits = texts @ videos.T / temperature
    positive_mask = torch.tensor(
        [[left == right for right in group_ids] for left in group_ids],
        dtype=torch.bool,
        device=logits.device,
    )

    def directional_loss(matrix, mask):
        numerator = torch.logsumexp(matrix.masked_fill(~mask, float("-inf")), dim=1)
        denominator = torch.logsumexp(matrix, dim=1)
        return -(numerator - denominator).mean()

    text_to_video = directional_loss(logits, positive_mask)
    video_to_text = directional_loss(logits.T, positive_mask.T)
    return 0.5 * (text_to_video + video_to_text)


def hard_negative_margin_loss(
    text_embeddings,
    positive_video_embeddings,
    negative_video_embeddings,
    *,
    margin: float = 0.05,
):
    """Margin ranking loss for text -> positive video vs mined hard negative video."""
    try:
        import torch.nn.functional as F
    except ImportError as error:
        raise RuntimeError("请先安装 `pip install -e '.[ml]'` 以训练模型") from error

    if margin < 0:
        raise ValueError("margin 不能小于 0")
    if text_embeddings.shape != positive_video_embeddings.shape:
        raise ValueError("text 与 positive video embedding shape 必须一致")
    if text_embeddings.shape != negative_video_embeddings.shape:
        raise ValueError("text 与 negative video embedding shape 必须一致")
    if text_embeddings.ndim != 2:
        raise ValueError("embedding 必须是 [batch, dim]")

    texts = F.normalize(text_embeddings, dim=-1)
    positives = F.normalize(positive_video_embeddings, dim=-1)
    negatives = F.normalize(negative_video_embeddings, dim=-1)
    positive_scores = (texts * positives).sum(dim=-1)
    negative_scores = (texts * negatives).sum(dim=-1)
    violations = margin - positive_scores + negative_scores
    return F.relu(violations).mean()

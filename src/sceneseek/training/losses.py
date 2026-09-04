from __future__ import annotations

from collections.abc import Sequence


def multi_positive_contrastive_loss(video_embeddings, text_embeddings, group_ids: Sequence[str], temperature: float = 0.07):
    """Symmetric InfoNCE where captions from the same video are not treated as negatives."""
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

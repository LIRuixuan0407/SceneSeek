from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class RerankerConfig:
    dimension: int
    num_heads: int = 8
    hidden_dimension: int = 512
    dropout: float = 0.1
    max_frames: int = 16

    def validate(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension 必须为正整数")
        if self.num_heads <= 0 or self.dimension % self.num_heads != 0:
            raise ValueError("num_heads 必须为正数且能整除 dimension")
        if self.hidden_dimension <= 0:
            raise ValueError("hidden_dimension 必须为正整数")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1)")
        if self.max_frames <= 0:
            raise ValueError("max_frames 必须为正整数")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def create_query_conditioned_reranker(config: RerankerConfig):
    """Create a light query-to-frame reranker for Stage-1 top-k candidates."""
    config.validate()
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("请先安装 `pip install -e '.[ml]'` 以训练 reranker") from error

    class QueryConditionedReranker(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.head_dimension = config.dimension // config.num_heads
            self.query_norm = nn.LayerNorm(config.dimension)
            self.frame_norm = nn.LayerNorm(config.dimension)
            self.query_projection = nn.Linear(config.dimension, config.dimension)
            self.key_projection = nn.Linear(config.dimension, config.dimension)
            self.value_projection = nn.Linear(config.dimension, config.dimension)
            self.context_projection = nn.Linear(config.dimension, config.dimension)
            fusion_dimension = config.dimension * 4 + 1
            self.scorer = nn.Sequential(
                nn.LayerNorm(fusion_dimension),
                nn.Linear(fusion_dimension, config.hidden_dimension),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dimension, 1),
            )
            self.coarse_scale = nn.Parameter(torch.tensor(1.0))
            # Start as an identity reranker: before learning, preserve Stage-1 ordering.
            nn.init.zeros_(self.scorer[-1].weight)
            nn.init.zeros_(self.scorer[-1].bias)

        def forward(self, texts, frames, valid_mask, coarse_scores):
            if texts.ndim != 2:
                raise ValueError("texts 必须是 [batch, dim]")
            if frames.ndim != 3:
                raise ValueError("frames 必须是 [batch, time, dim]")
            if valid_mask.shape != frames.shape[:2]:
                raise ValueError("valid_mask 必须与 frames 的前两维一致")
            if frames.shape[0] != texts.shape[0] or frames.shape[2] != texts.shape[1]:
                raise ValueError("texts 与 frames shape 不兼容")
            if coarse_scores.ndim != 1 or coarse_scores.shape[0] != texts.shape[0]:
                raise ValueError("coarse_scores 必须是 [batch]")
            if frames.shape[1] > config.max_frames:
                raise ValueError(
                    f"输入帧数 {frames.shape[1]} 超过 reranker max_frames={config.max_frames}"
                )

            batch, steps, dimension = frames.shape
            heads = config.num_heads
            head_dim = self.head_dimension

            normalized_texts = self.query_norm(texts)
            normalized_frames = self.frame_norm(frames)
            queries = self.query_projection(normalized_texts).view(batch, heads, head_dim)
            keys = self.key_projection(normalized_frames).view(batch, steps, heads, head_dim)
            values = self.value_projection(normalized_frames).view(batch, steps, heads, head_dim)

            attention = torch.einsum("bhd,bthd->bht", queries, keys)
            attention = attention / (head_dim**0.5)
            attention = attention.masked_fill(~valid_mask[:, None, :].bool(), float("-inf"))
            weights = torch.softmax(attention, dim=-1)
            context = torch.einsum("bht,bthd->bhd", weights, values).reshape(batch, dimension)
            context = self.context_projection(context)

            text_unit = torch.nn.functional.normalize(texts, dim=-1)
            context_unit = torch.nn.functional.normalize(context, dim=-1)
            fusion = torch.cat(
                (
                    text_unit,
                    context_unit,
                    text_unit * context_unit,
                    torch.abs(text_unit - context_unit),
                    coarse_scores[:, None],
                ),
                dim=-1,
            )
            delta = self.scorer(fusion).squeeze(-1)
            return self.coarse_scale * coarse_scores + delta

    return QueryConditionedReranker()

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class TemporalAdapterConfig:
    dimension: int
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    max_frames: int = 32

    def validate(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension 必须为正整数")
        if self.num_layers <= 0:
            raise ValueError("num_layers 必须为正整数")
        if self.num_heads <= 0 or self.dimension % self.num_heads != 0:
            raise ValueError("num_heads 必须为正数且能整除 dimension")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1)")
        if self.max_frames <= 0:
            raise ValueError("max_frames 必须为正整数")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def create_temporal_adapter(config: TemporalAdapterConfig):
    """Create the torch module lazily so the base SceneSeek install stays CPU-light."""
    config.validate()
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("请先安装 `pip install -e '.[ml]'` 以训练 Temporal Adapter") from error

    class TemporalAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.input_norm = nn.LayerNorm(config.dimension)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dimension))
            self.position = nn.Parameter(
                torch.empty(1, config.max_frames + 1, config.dimension)
            )
            layer = nn.TransformerEncoderLayer(
                d_model=config.dimension,
                nhead=config.num_heads,
                dim_feedforward=config.dimension * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=config.num_layers, enable_nested_tensor=False
            )
            self.output_norm = nn.LayerNorm(config.dimension)
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.position, std=0.01)

        def forward(self, frames, valid_mask=None):
            if frames.ndim != 3:
                raise ValueError("frames 必须是 [batch, time, dim]")
            batch, steps, dim = frames.shape
            if dim != config.dimension:
                raise ValueError(
                    f"输入 embedding 维度为 {dim}，checkpoint 期望 {config.dimension}"
                )
            if steps > config.max_frames:
                raise ValueError(
                    f"输入帧数 {steps} 超过 Temporal Adapter max_frames={config.max_frames}"
                )
            if valid_mask is None:
                valid_mask = torch.ones((batch, steps), dtype=torch.bool, device=frames.device)
            if valid_mask.shape != (batch, steps):
                raise ValueError("valid_mask 必须与 frames 的前两维一致")

            normalized = self.input_norm(frames)
            cls = self.cls_token.expand(batch, -1, -1)
            tokens = torch.cat((cls, normalized), dim=1)
            tokens = tokens + self.position[:, : steps + 1]

            cls_mask = torch.ones((batch, 1), dtype=torch.bool, device=frames.device)
            token_mask = torch.cat((cls_mask, valid_mask.bool()), dim=1)
            encoded = self.encoder(tokens, src_key_padding_mask=~token_mask)
            pooled = self.output_norm(encoded[:, 0])
            return torch.nn.functional.normalize(pooled, dim=-1)

    return TemporalAdapter()

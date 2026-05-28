from torch import Tensor, nn

from lib.reflection import build_object_from_class_name
from lib.config.schema import ModelConfig


class ForcedAlignmentModel(nn.Module):
    """Backbone holder for forced alignment.

    Embeds phoneme tokens and delegates to a dynamically-configured two-stream
    backbone that owns the spectrogram input projection.

    The backbone must follow the protocol:
        __init__(x_in_dim, token_in_dim, x_out_dim, token_out_dim, **kwargs)
        forward(x, token, t_mask, n_mask) -> (x_features, token_features)

    where
        x:             [B, T, x_in_dim]      raw spectrogram frames
        token:         [B, N, token_in_dim]  pre-embedded phoneme tokens
        t_mask:        [B, T] bool, True = valid frame
        n_mask:        [B, N] bool, True = valid token
        x_features:    [B, T, x_out_dim]
        token_features:[B, N, token_out_dim]

    token_out_dim is out_dim + 1: the last channel is the authenticity logit,
    sliced off by the holder.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config.max_vocab_size, config.embedding_dim, padding_idx=0,
        )
        self.backbone = build_object_from_class_name(
            config.backbone.cls, nn.Module,
            config.in_dim,          # x_in_dim
            config.embedding_dim,   # token_in_dim
            config.out_dim,         # x_out_dim
            config.out_dim + 1,     # token_out_dim (features + logit)
            **config.backbone.kwargs,
        )

    def forward(
        self,
        spectrogram: Tensor,
        tokens: Tensor,
        t_mask: Tensor,
        n_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        token = self.token_embedding(tokens)
        x_features, token_out = self.backbone(spectrogram, token, t_mask, n_mask)
        token_features = token_out[..., :-1]               # [B, N, out_dim]
        token_logits = token_out[..., -1]                  # [B, N]
        return x_features, token_features, token_logits


class ForcedAlignmentSSLModel(nn.Module):
    """SSL backbone holder.

    Forward returns (x_features, token_features, attn) where attn is a list
    of [B, H, T, N] cross-attention weight tensors.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config.max_vocab_size, config.embedding_dim, padding_idx=0,
        )
        self.backbone = build_object_from_class_name(
            config.backbone.cls, nn.Module,
            config.in_dim,          # x_in_dim
            config.embedding_dim,   # token_in_dim
            config.out_dim,         # x_out_dim
            config.out_dim,         # token_out_dim (no extra logit)
            **config.backbone.kwargs,
        )

    def forward(
        self,
        spectrogram: Tensor,
        tokens: Tensor,
        t_mask: Tensor,
        n_mask: Tensor,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        token = self.token_embedding(tokens)
        x_features, token_features, attn = self.backbone(spectrogram, token, t_mask, n_mask)
        return x_features, token_features, attn

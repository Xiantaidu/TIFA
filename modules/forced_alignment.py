from torch import Tensor, nn

from lib.reflection import build_object_from_class_name
from lib.config.schema import ModelConfig


class ForcedAlignmentModel(nn.Module):
    """Backbone holder for forced alignment.

    Embeds phoneme tokens and delegates to a dynamically-configured two-stream
    backbone that owns the spectrogram input projection.

    The backbone must follow the protocol:
        __init__(audio_in_dim, text_in_dim, out_dim, **kwargs)
        forward(x, tok, t_mask, n_mask) -> (out_x, out_tok, attn)

    where
        x:       [B, T, audio_in_dim] raw spectrogram frames
        tok:     [B, N, text_in_dim]  pre-embedded phoneme tokens
        t_mask:  [B, T] bool, True = valid frame
        n_mask:  [B, N] bool, True = valid token
        out_x:   [B, T, out_dim]
        out_tok: [B, N, out_dim]
        attn:    list of [B, H, T, N] cross-attention weights (one per CA layer)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config.max_vocab_size, config.embedding_dim, padding_idx=0,
        )
        self.token_head = nn.Linear(config.embedding_dim, 1)
        self.backbone = build_object_from_class_name(
            config.backbone.cls, nn.Module,
            config.in_dim,         # audio_in_dim (raw spectrogram bins)
            config.embedding_dim,  # text_in_dim  (token embedding dim)
            config.embedding_dim,  # out_dim
            **config.backbone.kwargs,
        )

    def forward(
        self,
        spectrogram: Tensor,
        tokens: Tensor,
        t_mask: Tensor,
        n_mask: Tensor,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        tok = self.token_embedding(tokens)              # [B, N, embedding_dim]
        out_x, out_tok, attn = self.backbone(spectrogram, tok, t_mask, n_mask)
        out_tok = self.token_head(out_tok).squeeze(-1)  # [B, N]
        return out_x, out_tok, attn

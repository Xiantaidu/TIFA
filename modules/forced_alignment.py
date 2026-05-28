from torch import Tensor, nn

from lib.reflection import build_object_from_class_name
from lib.config.schema import ModelConfig


class ForcedAlignmentModel(nn.Module):
    """Backbone holder for forced alignment.

    Embeds both spectrogram frames and phoneme tokens, then delegates to a
    dynamically-configured two-stream joint-attention backbone.

    The backbone must follow the protocol:
        __init__(in_dim, out_dim, **kwargs)
        forward(x, tok, t_mask, n_mask) -> (out_x, out_tok)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.spectrogram_proj = nn.Linear(config.in_dim, config.embedding_dim)
        self.token_embedding = nn.Embedding(
            config.max_vocab_size, config.embedding_dim, padding_idx=0,
        )
        self.token_head = nn.Linear(config.embedding_dim, 1)
        self.backbone = build_object_from_class_name(
            config.backbone.cls, nn.Module,
            config.embedding_dim,  # in_dim
            config.embedding_dim,  # out_dim
            **config.backbone.kwargs,
        )

    def forward(
        self,
        spectrogram: Tensor,
        tokens: Tensor,
        t_mask: Tensor,
        n_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        x = self.spectrogram_proj(spectrogram)       # [B, T, embedding_dim]
        tok = self.token_embedding(tokens)             # [B, N, embedding_dim]
        out_x, out_tok = self.backbone(x, tok, t_mask, n_mask)
        token_logits = self.token_head(out_tok).squeeze(-1)  # [B, N]
        return out_x, out_tok, token_logits

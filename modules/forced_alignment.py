from torch import Tensor, nn

from lib.reflection import build_object_from_class_name
from lib.config.schema import ModelConfig


ARCHITECTURE_ERROR_MSG = (
    "Unable to initialize '{expected}' architecture with "
    "configuration for '{actual}'. "
    "Be sure to call the correct entry point."
)


class ForcedAlignmentModel(nn.Module):
    """Backbone holder for forced alignment.

    Embeds phoneme tokens and delegates to a dynamically-configured two-stream
    backbone that owns the spectrogram input projection.

    The backbone must follow the protocol:
        __init__(x_in_dim, token_in_dim, x_out_dim, token_out_dim, **kwargs)
        forward(x, token, t_mask, n_mask) -> (x_out, token_out)

    where
        x:             [B, T, x_in_dim]             raw spectrogram frames
        token:         [B, N, token_in_dim]         pre-embedded phoneme tokens
        t_mask:        [B, T] bool, True = valid frame
        n_mask:        [B, N] bool, True = valid token
        x_out:         [B, T, x_out_dim]
        token_out:     [B, N, token_out_dim]

    The holder forward returns (x_features, frame_logits, token_features, token_logits):

        frame_features:  [B, T, out_dim]          backbone frame features
        frame_logits:    [B, T, vocab_size]       per-frame phoneme logits
        token_features:  [B, N, out_dim]          backbone token features
        token_logits:    [B, N, vocab_size]       per-token logits
    """

    def __init__(self, config: ModelConfig, vocab_size: int):
        super().__init__()
        expected = type(self).__name__
        if config.arch != expected:
            raise ValueError(ARCHITECTURE_ERROR_MSG.format(expected=expected, actual=config.arch))
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(
            vocab_size, config.embedding_dim, padding_idx=0,
        )
        self.backbone = build_object_from_class_name(
            config.backbone.cls, nn.Module,
            config.in_dim,              # x_in_dim
            config.embedding_dim,       # token_in_dim
            config.out_dim + vocab_size,         # x_out_dim (features + logit)
            config.out_dim + vocab_size,         # token_out_dim (features + logit)
            **config.backbone.kwargs,
        )

    def forward(
        self,
        spectrogram: Tensor,
        tokens: Tensor,
        t_mask: Tensor,
        n_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        token = self.token_embedding(tokens)
        x_out, token_out = self.backbone(spectrogram, token, t_mask, n_mask)
        frame_features = x_out[..., :-self.vocab_size]          # [B, T, out_dim]
        frame_logits = x_out[..., -self.vocab_size:]        # [B, T, vocab_size]
        token_features = token_out[..., :-self.vocab_size]  # [B, N, out_dim]
        token_logits = token_out[..., -self.vocab_size:]    # [B, N, vocab_size]
        return frame_features, frame_logits, token_features, token_logits


class ForcedAlignmentSSLModel(nn.Module):
    """Backbone holder for SSL forced alignment.

    Encodes tokens and spectrogram via a two-stream EBFEncoderBackbone with
    cross-attention, then optionally injects f0 and decodes to a reconstructed
    spectrogram via a single-stream EBFDecoderBackbone.

    Forward signature:
        forward(spectrogram, tokens, t_mask, n_mask, f0=None, reconstruct=True)
            -> (x_features, token_features, activations, x_recon)

    where
        spectrogram:    [B, T, in_dim]          raw spectrogram frames
        tokens:         [B, N] int64            phoneme token IDs
        t_mask:         [B, T] bool             True = valid frame
        n_mask:         [B, N] bool             True = valid token
        f0:             [B, T] | None           pitch in Hz
        reconstruct:    bool                    enable decoder + f0 injection
        x_features:     [B, T, embedding_dim]   frame features
        token_features: [B, N, embedding_dim]   token features
        activations:           list of [B, H, T, N]    CA weights (one per layer)
        x_recon:        [B, T, in_dim] | None   reconstructed spectrogram

    Set ``reconstruct=False`` for inference (no f0, no decoder).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        expected = type(self).__name__
        if config.arch != expected:
            raise ValueError(ARCHITECTURE_ERROR_MSG.format(expected=expected, actual=config.arch))
        if config.reconstructor is None:
            raise ValueError(
                f"'{expected}' requires 'reconstructor' in the model configuration."
            )
        self.token_embedding = nn.Embedding(
            config.max_vocab_size, config.embedding_dim, padding_idx=0,
        )
        self.backbone = build_object_from_class_name(
            config.backbone.cls, nn.Module,
            config.in_dim,              # x_in_dim
            config.embedding_dim,       # token_in_dim
            config.embedding_dim,       # x_out_dim
            config.embedding_dim,       # token_out_dim
            **config.backbone.kwargs,
        )
        self.pitch_embedding = nn.Linear(1, config.embedding_dim)
        self.reconstructor = build_object_from_class_name(
            config.reconstructor.cls, nn.Module,
            config.embedding_dim,       # in_dim
            config.in_dim,              # out_dim (reconstruct spectrogram)
            **config.reconstructor.kwargs,
        )

    def forward(
        self,
        spectrogram: Tensor,
        tokens: Tensor,
        t_mask: Tensor,
        n_mask: Tensor,
        f0: Tensor | None = None,
        reconstruct: bool = True,
    ) -> tuple[Tensor, Tensor, list[Tensor], Tensor | None]:
        if reconstruct and f0 is None:
            raise ValueError("f0 is required when reconstruct=True")
        token_emb = self.token_embedding(tokens)
        x_features, token_features, activations = self.backbone(
            spectrogram, token_emb, t_mask, n_mask,
        )
        if reconstruct:
            f0_mel = (1 + f0 / 700).log()
            x_recon = self.reconstructor(
                x_features + self.pitch_embedding(f0_mel[..., None]),
                mask=t_mask,
            )
        else:
            x_recon = None
        return x_features, token_features, activations, x_recon

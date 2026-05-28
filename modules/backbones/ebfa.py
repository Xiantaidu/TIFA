from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from modules.backbones.layers import LayerScale, RMSNorm, GLUFFN, FFN, CgMLP
from modules.backbones.rope import SingleRoPosEmb


class AttnWROPEX(nn.Module):
    def __init__(
            self, dim, num_heads, head_dim,
            use_rope=True, rope_cache=True,
            dropout_attn: float = 0.0,
            out_drop: float = 0.0
    ):
        super().__init__()

        self.num_heads = num_heads
        attn_dim = head_dim * num_heads
        self.q_linear = nn.Linear(dim, out_features=attn_dim, bias=True)
        self.kv_linear = nn.Linear(dim, out_features=attn_dim * 2, bias=True)

        self.out_linear = nn.Linear(attn_dim, dim, bias=True)
        self.dropout_attn = dropout_attn
        self.out_drop = nn.Dropout(out_drop) if out_drop > 0. else nn.Identity()
        if use_rope:
            self.rope = SingleRoPosEmb(head_dim, use_cache=rope_cache)
        else:
            self.rope = None

    def forward(self, x):

        q = self.q_linear(x)

        k, v = self.kv_linear(x).chunk(2, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b t (h c) -> b h t c", h=self.num_heads), (q, k, v)
        )
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        with torch.backends.cuda.sdp_kernel():
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout_attn,
            )

        out = rearrange(out, "b h t c -> b t (h c) ", h=self.num_heads, )
        out = self.out_linear(out)
        out = self.out_drop(out)
        return out




# class CrossAttention(nn.Module):
#     def __init__(
#             self, dim,cross_dim, num_heads, head_dim,
#
#             dropout_attn: float = 0.0,
#             out_drop: float = 0.0
#     ):
#         super().__init__()
#
#         self.num_heads = num_heads
#         attn_dim = head_dim * num_heads
#         self.q_linear = nn.Linear(dim, out_features=attn_dim, bias=True)
#         self.kv_linear = nn.Linear(cross_dim, out_features=attn_dim * 2, bias=True)
#
#         self.out_linear = nn.Linear(attn_dim, dim, bias=True)
#         self.dropout_attn = dropout_attn
#         self.out_drop = nn.Dropout(out_drop) if out_drop > 0. else nn.Identity()
#
#
#     def forward(self, x,y):
#
#         q = self.q_linear(x)
#
#         k, v = self.kv_linear(y).chunk(2, dim=-1)
#
#         q, k, v = map(
#             lambda t: rearrange(t, "b t (h c) -> b h t c", h=self.num_heads), (q, k, v)
#         )
#
#         with torch.backends.cuda.sdp_kernel():
#             out = F.scaled_dot_product_attention(
#                 q, k, v,
#                 dropout_p=self.dropout_attn,
#             )
#
#         out = rearrange(out, "b h t c -> b t (h c) ", h=self.num_heads, )
#         out = self.out_linear(out)
#         out = self.out_drop(out)
#         return out





class CrossAttention(nn.Module):
    def __init__(
            self, dim, cross_dim, num_heads, head_dim,
            dropout_attn: float = 0.0,
            out_drop: float = 0.0
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        attn_dim = head_dim * num_heads

        self.q_linear = nn.Linear(dim, attn_dim, bias=True)
        self.kv_linear = nn.Linear(cross_dim, attn_dim * 2, bias=True)
        self.out_linear = nn.Linear(attn_dim, dim, bias=True)

        self.dropout_attn = nn.Dropout(dropout_attn) if dropout_attn > 0. else nn.Identity()
        self.out_drop = nn.Dropout(out_drop) if out_drop > 0. else nn.Identity()

    def forward(self, x, y, mask=None, return_attn=False):
        """
        x: [B, T, dim]       (audio, query)
        y: [B, S, cross_dim] (text, key/value)
        mask: [B, S] bool, True = padding (optional)
        """
        q = self.q_linear(x)
        k, v = self.kv_linear(y).chunk(2, dim=-1)

        # reshape to multi-head
        q = rearrange(q, "b t (h c) -> b h t c", h=self.num_heads)
        k = rearrange(k, "b s (h c) -> b h s c", h=self.num_heads)
        v = rearrange(v, "b s (h c) -> b h s c", h=self.num_heads)

        # attention scores
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, S]

        # mask padding positions
        if mask is not None:
            # mask: [B, S] -> [B, 1, 1, S]
            attn_logits = attn_logits.masked_fill(mask[:, None, None, :], -1e9)

        # softmax
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, H, T, S]
        attn_weights = self.dropout_attn(attn_weights)

        # weighted sum
        out = torch.matmul(attn_weights, v)  # [B, H, T, C]

        # merge heads
        out = rearrange(out, "b h t c -> b t (h c)")
        out = self.out_linear(out)
        out = self.out_drop(out)

        if return_attn:
            return out, attn_weights  # attn_weights: [B, H, T, S]
        return out



class PAC(nn.Module):
    def __init__(
            self, dim, num_heads, head_dim,
            c_kernel_size=31, m_kernel_size=31, use_rope=True, rope_cache=True,
            dropout_attn: float = 0.0, out_drop: float = 0.0, c_out_drop=0.1,
            c_latent_drop=0.0,
    ):
        super().__init__()
        self.attn = AttnWROPEX(dim, num_heads, head_dim, use_rope, rope_cache, dropout_attn, out_drop)
        self.c = CgMLP(
            dim, kernel_size=c_kernel_size,
            latent_drop=c_latent_drop, out_drop=c_out_drop
        )

        self.a_norm = RMSNorm(dim)
        self.c_norm = RMSNorm(dim)

        self.merge_linear = nn.Linear(dim * 2, dim)
        self.merge_dw_conv = (
            nn.Conv1d(
                dim * 2, dim * 2, kernel_size=m_kernel_size, stride=1,
                padding=m_kernel_size // 2,
                groups=dim * 2
            )
            if m_kernel_size != 0 else
            None
        )

    def forward(self, x):
        a_o = self.attn(self.a_norm(x))
        c_o = self.c(self.c_norm(x))
        m_o = torch.cat([a_o, c_o], dim=-1)

        if self.merge_dw_conv is not None:
            m_o = self.merge_dw_conv(m_o.transpose(1, 2)).transpose(1, 2) + m_o
        m_o = self.merge_linear(m_o)
        return m_o


class EBF(nn.Module):
    def __init__(
            self, dim, num_heads, head_dim,
            c_kernel_size=31, m_kernel_size=31, use_rope=True, rope_cache=True,
            dropout_attn: float = 0.0, out_drop: float = 0.0, c_out_drop=0.1,
            c_latent_drop=0.0, use_ls=True, ffn_type='glu', ffn_latent_drop=0.1, ffn_out_drop=0.1,
            skip_first_ffn=False, skip_out_ffn=False,
    ):
        super().__init__()
        self.skip_first_ffn = skip_first_ffn
        self.skip_out_ffn = skip_out_ffn

        if ffn_type == 'glu':
            if not skip_first_ffn:
                self.ffn1 = GLUFFN(
                    dim, latent_dim=dim * 4, dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop
                )
            if not skip_out_ffn:
                self.ffn2 = GLUFFN(
                    dim, latent_dim=dim * 4, dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop
                )
        elif ffn_type == 'ffn':
            if not skip_first_ffn:
                self.ffn1 = FFN(
                    dim, latent_dim=dim * 4,
                    dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop
                )
            if not skip_out_ffn:
                self.ffn2 = FFN(
                    dim, latent_dim=dim * 4,
                    dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop
                )
        elif ffn_type == 'cgmlp':
            if not skip_first_ffn:
                self.ffn1 = CgMLP(
                    dim, latent_dim=int(dim * 2.5), latent_drop=ffn_latent_drop,
                    out_drop=ffn_out_drop, kernel_size=21
                )
            if not skip_out_ffn:
                self.ffn2 = CgMLP(
                    dim, latent_dim=int(dim * 2.5), latent_drop=ffn_latent_drop,
                    out_drop=ffn_out_drop, kernel_size=7
                )


        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

        self.attn = PAC(
            dim, num_heads, head_dim, c_kernel_size, m_kernel_size, use_rope, rope_cache, dropout_attn,
            out_drop, c_out_drop, c_latent_drop
        )
        if not skip_first_ffn:
            self.norm1 = RMSNorm(dim)
        if not skip_out_ffn:
            self.norm2 = RMSNorm(dim)

        if use_ls:
            if not skip_first_ffn:
                self.lay_scale1 = LayerScale(dim)
            self.lay_scale2 = LayerScale(dim)
            if not skip_out_ffn:
                self.lay_scale3 = LayerScale(dim)
        else:
            if not skip_first_ffn:
                self.lay_scale1 = nn.Identity()
            self.lay_scale2 = nn.Identity()
            if not skip_out_ffn:
                self.lay_scale3 = nn.Identity()

    def forward(self, x, mask=None):
        if not self.skip_first_ffn:
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), 0)
            x = self.lay_scale1(self.ffn1(self.norm1(x))) * 0.5 + x
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        x = self.lay_scale2(self.attn(x)) + x
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        if not self.skip_out_ffn:
            x = self.lay_scale3(self.ffn2(self.norm2(x))) * 0.5 + x
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), 0)
        return x


class EBFBackbone(nn.Module):
    def __init__(
            self, in_dim: int, out_dim: int, return_latent: bool,
            dim: int = 256,
            num_layers: int = 8,
            latent_layer_idx: int = None,
            latent_out_dim: int = 16,
            num_heads: int = 8,
            head_dim: int = 64,
            c_kernel_size: int = 31,
            m_kernel_size: int = 31,
            use_rope: bool = True,
            rope_cache: bool = True,
            dropout_attn: float = 0.0,
            out_drop: float = 0.0,
            c_out_drop: float = 0.1,
            c_latent_drop: float = 0.0,
            use_ls: bool = True,
            ffn_type: str = 'glu',
            ffn_latent_drop: float = 0.1,
            ffn_out_drop: float = 0.1,
            use_out_norm: bool = True,
            skip_first_ffn=True,
            skip_out_ffn=False,
    ):
        super().__init__()


        self.use_out_norm = use_out_norm
        self.return_latent = return_latent
        if return_latent:
            assert latent_layer_idx <= num_layers
        self.latent_layer_idx = latent_layer_idx

        self.input_proj = nn.Linear(in_dim, dim)

        self.layers = nn.ModuleList([
            EBF(dim=dim, num_heads=num_heads, head_dim=head_dim,
                c_kernel_size=c_kernel_size, m_kernel_size=m_kernel_size,
                use_rope=use_rope, rope_cache=rope_cache,
                dropout_attn=dropout_attn, out_drop=out_drop,
                c_out_drop=c_out_drop, c_latent_drop=c_latent_drop,
                use_ls=use_ls, ffn_type=ffn_type,
                ffn_latent_drop=ffn_latent_drop, ffn_out_drop=ffn_out_drop,
                skip_first_ffn=skip_first_ffn, skip_out_ffn=skip_out_ffn,
                )
            for _ in range(num_layers)
        ])

        if self.return_latent:
            self.latent_norm = RMSNorm(dim)
            self.latent_proj = nn.Linear(dim, latent_out_dim)  # -> [B, T, C_latent]
        if self.use_out_norm:
            self.output_norm = RMSNorm(dim)
        self.output_proj = nn.Linear(dim, out_dim)  # -> [B, T, C_out]

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, T, in_dim] input tensor
            mask: [B, T] valid mask
        Returns:
            latent: [B, T, C_latent] intermediate latent tensor for self cosine similarity
            out: [B, T, C_out] output tensor
        """
        x = self.input_proj(x)

        latent = None
        for i, layer in enumerate(self.layers):
            x = layer(x, mask=mask)
            if self.return_latent and i == self.latent_layer_idx - 1:
                latent = self.latent_norm(x)
                latent = self.latent_proj(latent)  # [B, T, C_latent]

        if self.use_out_norm:
            x = self.output_norm(x)
        out = self.output_proj(x)  # [B, T, C_out]

        if self.return_latent:
            return out, latent
        else:
            return out



class EBFAlignmentBackbone(nn.Module):
    """Two-stream encoder-CA-decoder backbone for forced alignment.

    Frame stream (`x`, raw spectrogram at `audio_in_dim`) and token stream
    (`token`, pre-embedded at `token_in_dim`) are projected to `dim` by separate
    input projections. The token stream is encoded with kernel-7 EBF layers;
    the frame stream is encoded with kernel-31 EBF layers, fused with the
    token stream via stacked CrossAttention layers (x attends to token), then
    decoded with kernel-31 EBF layers. Both streams are projected to `out_dim`.

    Constructor signature: (x_in_dim, token_in_dim, x_out_dim, token_out_dim, **kwargs)
    Forward signature:     forward(x, token, t_mask, n_mask)
                           -> (x_features, token_features, attn)
    Matches ForcedAlignmentModel's backbone protocol; `attn` is a list of
    cross-attention weights [B, H, T, N], one per CA layer.
    """

    def __init__(
            self,
            x_in_dim: int,
            token_in_dim: int,
            x_out_dim: int,
            token_out_dim: int,
            # shared params
            dim: int = 256,
            num_heads: int = 8,
            head_dim: int = 64,
            # token encoder
            text_num_layers: int = 4,
            text_c_kernel_size: int = 7,
            text_m_kernel_size: int = 7,
            # frame encoder
            audio_enc_num_layers: int = 4,
            audio_enc_c_kernel_size: int = 31,
            audio_enc_m_kernel_size: int = 31,
            # cross-attention
            num_ca_layers: int = 1,
            ca_dropout_attn: float = 0.0,
            ca_out_drop: float = 0.0,
            # frame decoder
            audio_dec_num_layers: int = 4,
            audio_dec_c_kernel_size: int = 31,
            audio_dec_m_kernel_size: int = 31,
            # general EBF params
            use_rope: bool = True,
            rope_cache: bool = True,
            dropout_attn: float = 0.0,
            out_drop: float = 0.0,
            c_out_drop: float = 0.1,
            c_latent_drop: float = 0.0,
            use_ls: bool = True,
            ffn_type: str = 'glu',
            ffn_latent_drop: float = 0.1,
            ffn_out_drop: float = 0.1,
            skip_first_ffn: bool = True,
            skip_out_ffn: bool = False,
            use_out_norm: bool = True,
    ):
        super().__init__()

        self.use_out_norm = use_out_norm
        self.num_ca_layers = num_ca_layers
        self.dim = dim
        self.x_in_dim = x_in_dim
        self.token_in_dim = token_in_dim
        self.x_out_dim = x_out_dim
        self.token_out_dim = token_out_dim

        # ============ Input Projections (frame & token) ============
        self.audio_input_proj = nn.Linear(x_in_dim, dim)
        self.text_input_proj = nn.Linear(token_in_dim, dim)

        # ============ Token Encoder (kernel=7) ============
        self.text_encoder = nn.ModuleList([
            EBF(dim=dim, num_heads=num_heads, head_dim=head_dim,
                c_kernel_size=text_c_kernel_size, m_kernel_size=text_m_kernel_size,
                use_rope=use_rope, rope_cache=rope_cache,
                dropout_attn=dropout_attn, out_drop=out_drop,
                c_out_drop=c_out_drop, c_latent_drop=c_latent_drop,
                use_ls=use_ls, ffn_type=ffn_type,
                ffn_latent_drop=ffn_latent_drop, ffn_out_drop=ffn_out_drop,
                skip_first_ffn=skip_first_ffn, skip_out_ffn=skip_out_ffn)
            for _ in range(text_num_layers)
        ])
        self.text_enc_norm = RMSNorm(dim)

        # ============ Frame Encoder (kernel=31) ============
        self.audio_encoder = nn.ModuleList([
            EBF(dim=dim, num_heads=num_heads, head_dim=head_dim,
                c_kernel_size=audio_enc_c_kernel_size, m_kernel_size=audio_enc_m_kernel_size,
                use_rope=use_rope, rope_cache=rope_cache,
                dropout_attn=dropout_attn, out_drop=out_drop,
                c_out_drop=c_out_drop, c_latent_drop=c_latent_drop,
                use_ls=use_ls, ffn_type=ffn_type,
                ffn_latent_drop=ffn_latent_drop, ffn_out_drop=ffn_out_drop,
                skip_first_ffn=skip_first_ffn, skip_out_ffn=skip_out_ffn)
            for _ in range(audio_enc_num_layers)
        ])
        self.audio_enc_norm = RMSNorm(dim)

        # ============ Cross-Attention Layers (x <- token) ============
        self.cross_attn_layers = nn.ModuleList()
        self.ca_pre_norms = nn.ModuleList()
        self.ca_lay_scales = nn.ModuleList()

        for _ in range(num_ca_layers):
            self.cross_attn_layers.append(
                CrossAttention(
                    dim=dim, cross_dim=dim,
                    num_heads=num_heads, head_dim=head_dim,
                    dropout_attn=ca_dropout_attn, out_drop=ca_out_drop
                )
            )
            self.ca_pre_norms.append(RMSNorm(dim))
            self.ca_lay_scales.append(LayerScale(dim) if use_ls else nn.Identity())

        # ============ Frame Decoder (kernel=31) ============
        self.audio_decoder = nn.ModuleList([
            EBF(dim=dim, num_heads=num_heads, head_dim=head_dim,
                c_kernel_size=audio_dec_c_kernel_size, m_kernel_size=audio_dec_m_kernel_size,
                use_rope=use_rope, rope_cache=rope_cache,
                dropout_attn=dropout_attn, out_drop=out_drop,
                c_out_drop=c_out_drop, c_latent_drop=c_latent_drop,
                use_ls=use_ls, ffn_type=ffn_type,
                ffn_latent_drop=ffn_latent_drop, ffn_out_drop=ffn_out_drop,
                skip_first_ffn=skip_first_ffn, skip_out_ffn=skip_out_ffn)
            for _ in range(audio_dec_num_layers)
        ])

        # ============ Output Heads ============
        if self.use_out_norm:
            self.output_norm_x = RMSNorm(dim)
            self.output_norm_token = RMSNorm(dim)
        self.output_proj_x = nn.Linear(dim, x_out_dim)
        self.output_proj_token = nn.Linear(dim, token_out_dim)

    def forward(self, x, token, t_mask, n_mask):
        """
        Args:
            x:      [B, T, x_in_dim]      raw spectrogram frames
            token:  [B, N, token_in_dim]  pre-embedded tokens
            t_mask: [B, T] bool, True = valid frame
            n_mask: [B, N] bool, True = valid token
        Returns:
            x_features:     [B, T, x_out_dim]
            token_features: [B, N, token_out_dim]
            attn:           list of [B, H, T, N], length = num_ca_layers
        """
        # ============ Input Projections ============
        x = self.audio_input_proj(x)
        token = self.text_input_proj(token)

        # ============ Token Encoder ============
        for layer in self.text_encoder:
            token = layer(token, mask=n_mask)
        token = self.text_enc_norm(token)

        # ============ Frame Encoder ============
        for layer in self.audio_encoder:
            x = layer(x, mask=t_mask)
        x = self.audio_enc_norm(x)

        # ============ Cross-Attention (x attends to token) ============
        # CrossAttention mask semantic: True = padding, so invert n_mask
        text_padding_mask = ~n_mask if n_mask is not None else None

        attn = []
        for i in range(self.num_ca_layers):
            residual = x
            x_normed = self.ca_pre_norms[i](x)

            ca_out, attn_w = self.cross_attn_layers[i](
                x_normed, token, mask=text_padding_mask, return_attn=True
            )
            attn.append(attn_w)

            x = residual + self.ca_lay_scales[i](ca_out)

            if t_mask is not None:
                x = x.masked_fill(~t_mask.unsqueeze(-1), 0)

        # ============ Frame Decoder ============
        for layer in self.audio_decoder:
            x = layer(x, mask=t_mask)

        # ============ Output ============
        if self.use_out_norm:
            x = self.output_norm_x(x)
            token = self.output_norm_token(token)
        x_features = self.output_proj_x(x)
        token_features = self.output_proj_token(token)

        return x_features, token_features, attn

if __name__ == '__main__':
    # protocol: (x_in_dim, token_in_dim, x_out_dim, token_out_dim, **kwargs)
    #           forward(x, token, t_mask, n_mask) -> (x_features, token_features, attn)
    X_IN_DIM = 80      # raw mel bins
    TOKEN_IN_DIM = 256  # token embedding dim
    X_OUT_DIM = 256
    TOKEN_OUT_DIM = 256

    model = EBFAlignmentBackbone(
        x_in_dim=X_IN_DIM,
        token_in_dim=TOKEN_IN_DIM,
        x_out_dim=X_OUT_DIM,
        token_out_dim=TOKEN_OUT_DIM,
        dim=256,
        num_heads=8,
        head_dim=64,
        text_num_layers=4,
        text_c_kernel_size=7,
        text_m_kernel_size=7,
        audio_enc_num_layers=4,
        audio_dec_num_layers=4,
        num_ca_layers=2,
    )

    torch.manual_seed(0)
    B, T, N = 2, 100, 30
    # x: raw mel; token: output of token_embedding in ForcedAlignmentModel
    x = torch.randn(B, T, X_IN_DIM)
    token = torch.randn(B, N, TOKEN_IN_DIM)

    t_lens = torch.randint(T // 2, T + 1, (B,))
    n_lens = torch.randint(N // 2, N + 1, (B,))
    t_mask = torch.arange(T)[None, :] < t_lens[:, None]  # [B, T]
    n_mask = torch.arange(N)[None, :] < n_lens[:, None]  # [B, N]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params / 1e6:.2f}M")

    model.eval()
    with torch.no_grad():
        x_features, token_features, attn = model(x, token, t_mask, n_mask)
        print(f"x_features shape:     {tuple(x_features.shape)}")    # (B, T, X_OUT_DIM)
        print(f"token_features shape: {tuple(token_features.shape)}")  # (B, N, TOKEN_OUT_DIM)
        for i, aw in enumerate(attn):
            print(f"  attn[{i}] shape: {tuple(aw.shape)}")  # (B, H, T, N)

        assert x_features.shape == (B, T, X_OUT_DIM)
        assert token_features.shape == (B, N, TOKEN_OUT_DIM)
        assert len(attn) == 2 and attn[0].shape == (B, 8, T, N)
        assert not torch.isnan(x_features).any() and not torch.isnan(token_features).any()
        print("OK")

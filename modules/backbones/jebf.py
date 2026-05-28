import torch
import torch.nn as nn

from modules.backbones.eglu import HalfCacheGLUFFN
from modules.backbones.joint_attn import JointAttention, SplitJointAttention
from modules.backbones.layers import LayerScale, RMSNorm, GLUFFN, FFN, CgMLP


class PJAC(nn.Module):
    """Parallel Joint Attention + Convolution block for the two-stream setting."""

    def __init__(
            self, dim,
            num_heads,
            head_dim,
            c_kernel_size_tok=7,
            m_kernel_size_tok=5,
            c_kernel_size_x=31,
            m_kernel_size_x=31,

            c_out_drop_x=0.1,
            c_latent_drop_x=0.0,
            c_out_drop_tok=0.1,
            c_latent_drop_tok=0.0,

            qk_norm=True,
            use_rope=True,
            theta=10000.0,
            dropout_attn: float = 0.0,
            attn_out_drop_x: float = 0.0,
            attn_out_drop_tok: float = 0.0,
            attn_type: str = 'joint',
    ):
        super().__init__()
        if attn_type == 'joint':
            self.jattn = JointAttention(
                dim=dim, num_heads=num_heads,
                qk_norm=qk_norm,
                use_rope=use_rope, theta=theta,
                dropout_attn=dropout_attn, out_drop_x=attn_out_drop_x,
                out_drop_tok=attn_out_drop_tok, head_dim=head_dim,
            )
        elif attn_type == 'split':
            self.jattn = SplitJointAttention(
                dim=dim, num_heads=num_heads,
                qk_norm=qk_norm,
                use_rope=use_rope, theta=theta,
                dropout_attn=dropout_attn, out_drop_x=attn_out_drop_x,
                out_drop_tok=attn_out_drop_tok, head_dim=head_dim,
            )
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

        self.c_x = CgMLP(
            dim, kernel_size=c_kernel_size_x,
            latent_drop=c_latent_drop_x, out_drop=c_out_drop_x,
        )
        self.c_tok = CgMLP(
            dim, kernel_size=c_kernel_size_tok,
            latent_drop=c_latent_drop_tok, out_drop=c_out_drop_tok,
        )

        self.c_norm_x = RMSNorm(dim)
        self.c_norm_tok = RMSNorm(dim)

        self.merge_linear_x = nn.Linear(dim * 2, dim)
        self.merge_dw_conv_x = (
            nn.Conv1d(
                dim * 2, dim * 2, kernel_size=m_kernel_size_x, stride=1,
                padding=m_kernel_size_x // 2, groups=dim * 2,
            )
            if m_kernel_size_x != 0 else None
        )
        self.merge_linear_tok = nn.Linear(dim * 2, dim)
        self.merge_dw_conv_tok = (
            nn.Conv1d(
                dim * 2, dim * 2, kernel_size=m_kernel_size_tok, stride=1,
                padding=m_kernel_size_tok // 2, groups=dim * 2,
            )
            if m_kernel_size_tok != 0 else None
        )

    def forward(self, tok, x, t_mask, n_mask):
        a_tok, a_x = self.jattn(tok, x, t_mask, n_mask)
        c_tok, c_x = self.c_tok(self.c_norm_tok(tok)), self.c_x(self.c_norm_x(x))
        m_tok, m_x = torch.cat([a_tok, c_tok], dim=-1), torch.cat([a_x, c_x], dim=-1)
        if self.merge_dw_conv_tok is not None:
            m_tok = self.merge_dw_conv_tok(m_tok.transpose(1, 2)).transpose(1, 2) + m_tok
        m_tok = self.merge_linear_tok(m_tok)
        if self.merge_dw_conv_x is not None:
            m_x = self.merge_dw_conv_x(m_x.transpose(1, 2)).transpose(1, 2) + m_x
        m_x = self.merge_linear_x(m_x)
        return m_tok, m_x


class JEBF(nn.Module):
    """One JEBF layer: FFN_tok + FFN_x -> PJAC -> FFN_tok + FFN_x.

    Padding is handled via masked_fill (EBF style), no attn_mask.
    """

    def __init__(
            self, dim,
            num_heads,
            head_dim,
            c_kernel_size_tok=7,
            m_kernel_size_tok=5,
            c_kernel_size_x=31,
            m_kernel_size_x=31,

            c_out_drop_x=0.1,
            c_latent_drop_x=0.0,
            c_out_drop_tok=0.1,
            c_latent_drop_tok=0.0,

            qk_norm=True,
            use_rope=True,
            theta=10000.0,
            dropout_attn: float = 0.0,
            attn_out_drop_x: float = 0.0,
            attn_out_drop_tok: float = 0.0,
            skip_first_ffn=False, skip_out_ffn=False,
            use_ls=True, ffn_type='glu', ffn_latent_drop=0.1, ffn_out_drop=0.1,
            attn_type: str = 'joint',
    ):
        super().__init__()
        self.skip_first_ffn = skip_first_ffn
        self.skip_out_ffn = skip_out_ffn

        if ffn_type == 'glu':
            if not skip_first_ffn:
                self.ffn1_x = GLUFFN(
                    dim, latent_dim=dim * 4, dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop,
                )
                self.ffn1_tok = GLUFFN(
                    dim, latent_dim=dim * 4, dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop,
                )
            if not skip_out_ffn:
                self.ffn2_x = GLUFFN(
                    dim, latent_dim=dim * 4, dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop,
                )
                self.ffn2_tok = GLUFFN(
                    dim, latent_dim=dim * 4, dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop,
                )
        elif ffn_type == 'ffn':
            if not skip_first_ffn:
                self.ffn1_x = FFN(
                    dim, latent_dim=dim * 4,
                    dropout_latent=ffn_latent_drop, dropout_output=ffn_out_drop,
                )
                self.ffn1_tok = FFN(
                    dim, latent_dim=dim * 4,
                    dropout_latent=ffn_latent_drop, dropout_output=ffn_out_drop,
                )
            if not skip_out_ffn:
                self.ffn2_x = FFN(
                    dim, latent_dim=dim * 4,
                    dropout_latent=ffn_latent_drop, dropout_output=ffn_out_drop,
                )
                self.ffn2_tok = FFN(
                    dim, latent_dim=dim * 4,
                    dropout_latent=ffn_latent_drop, dropout_output=ffn_out_drop,
                )
        elif ffn_type == 'cgmlp':
            if not skip_first_ffn:
                self.ffn1_x = CgMLP(
                    dim, latent_dim=int(dim * 2.5), latent_drop=ffn_latent_drop,
                    out_drop=ffn_out_drop, kernel_size=21,
                )
                self.ffn1_tok = CgMLP(
                    dim, latent_dim=int(dim * 2.5), latent_drop=ffn_latent_drop,
                    out_drop=ffn_out_drop, kernel_size=21,
                )
            if not skip_out_ffn:
                self.ffn2_x = CgMLP(
                    dim, latent_dim=int(dim * 2.5), latent_drop=ffn_latent_drop,
                    out_drop=ffn_out_drop, kernel_size=7,
                )
                self.ffn2_tok = CgMLP(
                    dim, latent_dim=int(dim * 2.5), latent_drop=ffn_latent_drop,
                    out_drop=ffn_out_drop, kernel_size=7,
                )
        elif ffn_type == 'eglu':
            if not skip_first_ffn:
                self.ffn1_x = HalfCacheGLUFFN(d_model=dim, d_ff=dim * 4, gate_type='silu', quant_bits=0, bias=True)
                self.ffn1_tok = HalfCacheGLUFFN(d_model=dim, d_ff=dim * 4, gate_type='silu', quant_bits=0, bias=True)
            if not skip_out_ffn:
                self.ffn2_x = HalfCacheGLUFFN(d_model=dim, d_ff=dim * 4, gate_type='silu', quant_bits=0, bias=True)
                self.ffn2_tok = HalfCacheGLUFFN(d_model=dim, d_ff=dim * 4, gate_type='silu', quant_bits=0, bias=True)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

        self.attn = PJAC(
            dim=dim, num_heads=num_heads, head_dim=head_dim,
            c_kernel_size_tok=c_kernel_size_tok, m_kernel_size_tok=m_kernel_size_tok,
            c_kernel_size_x=c_kernel_size_x, m_kernel_size_x=m_kernel_size_x,
            c_out_drop_x=c_out_drop_x, c_latent_drop_x=c_latent_drop_x,
            c_out_drop_tok=c_out_drop_tok, c_latent_drop_tok=c_latent_drop_tok,
            qk_norm=qk_norm, use_rope=use_rope, theta=theta,
            dropout_attn=dropout_attn,
            attn_out_drop_x=attn_out_drop_x, attn_out_drop_tok=attn_out_drop_tok,
            attn_type=attn_type,
        )
        if not skip_first_ffn:
            self.norm_ffn1_x = RMSNorm(dim)
            self.norm_ffn1_tok = RMSNorm(dim)
        if not skip_out_ffn:
            self.norm_ffn2_x = RMSNorm(dim)
            self.norm_ffn2_tok = RMSNorm(dim)

        if use_ls:
            if not skip_out_ffn:
                self.layer_scale_ffn2_x = LayerScale(dim)
                self.layer_scale_ffn2_tok = LayerScale(dim)
            if not skip_first_ffn:
                self.layer_scale_ffn1_x = LayerScale(dim)
                self.layer_scale_ffn1_tok = LayerScale(dim)
            self.layer_scale_jpac_x = LayerScale(dim)
            self.layer_scale_jpac_tok = LayerScale(dim)
        else:
            if not skip_out_ffn:
                self.layer_scale_ffn2_x = nn.Identity()
                self.layer_scale_ffn2_tok = nn.Identity()
            if not skip_first_ffn:
                self.layer_scale_ffn1_x = nn.Identity()
                self.layer_scale_ffn1_tok = nn.Identity()
            self.layer_scale_jpac_x = nn.Identity()
            self.layer_scale_jpac_tok = nn.Identity()

    def forward(self, tok, x, t_mask, n_mask):
        if t_mask is not None:
            x = x.masked_fill(~t_mask.unsqueeze(-1), 0)
        if n_mask is not None:
            tok = tok.masked_fill(~n_mask.unsqueeze(-1), 0)

        if not self.skip_first_ffn:
            x = self.layer_scale_ffn1_x(self.ffn1_x(self.norm_ffn1_x(x))) + x
            tok = self.layer_scale_ffn1_tok(self.ffn1_tok(self.norm_ffn1_tok(tok))) + tok

        if t_mask is not None:
            x = x.masked_fill(~t_mask.unsqueeze(-1), 0)
        if n_mask is not None:
            tok = tok.masked_fill(~n_mask.unsqueeze(-1), 0)

        t_o, x_o = self.attn(tok, x, t_mask, n_mask)
        x = self.layer_scale_jpac_x(x_o) + x
        tok = self.layer_scale_jpac_tok(t_o) + tok

        if t_mask is not None:
            x = x.masked_fill(~t_mask.unsqueeze(-1), 0)
        if n_mask is not None:
            tok = tok.masked_fill(~n_mask.unsqueeze(-1), 0)

        if not self.skip_out_ffn:
            x = self.layer_scale_ffn2_x(self.ffn2_x(self.norm_ffn2_x(x))) + x
            tok = self.layer_scale_ffn2_tok(self.ffn2_tok(self.norm_ffn2_tok(tok))) + tok

        return x, tok


class JEBFBackbone(nn.Module):
    """Two-stream joint-attention backbone.

    Receives pre-embedded token and frame features, processes them through
    stacked JEBF layers with joint attention. No region-based masking --
    padding handled via masked_fill (EBF style).

    Constructor signature: (in_dim, out_dim, **kwargs)
    This matches the backbone protocol expected by ForcedAlignmentModel.
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            dim: int = 256,
            num_layers: int = 8,
            num_heads: int = 8,
            head_dim: int = 64,

            c_kernel_size_tok: int = 7,
            m_kernel_size_tok: int = 5,
            c_kernel_size_x: int = 31,
            m_kernel_size_x: int = 31,
            c_out_drop_x: float = 0.1,
            c_latent_drop_x: float = 0.0,
            c_out_drop_tok: float = 0.1,
            c_latent_drop_tok: float = 0.0,
            qk_norm: bool = True,
            use_rope: bool = True,
            theta: float = 10000.0,
            dropout_attn: float = 0.0,
            attn_out_drop_x: float = 0.0,
            attn_out_drop_tok: float = 0.0,
            use_ls: bool = True,
            ffn_type: str = 'glu',
            ffn_latent_drop: float = 0.1,
            ffn_out_drop: float = 0.1,

            use_out_norm: bool = True,
            skip_first_ffn=False,
            skip_out_ffn=False,
            attn_type: str = 'joint',
    ):
        super().__init__()
        self.use_out_norm = use_out_norm
        self.attn_type = attn_type
        self.dim = dim
        self.out_dim = out_dim

        self.input_proj = nn.Linear(in_dim, dim)

        self.layers = nn.ModuleList([
            JEBF(
                dim=dim, num_heads=num_heads, head_dim=head_dim,
                c_kernel_size_tok=c_kernel_size_tok, m_kernel_size_tok=m_kernel_size_tok,
                c_kernel_size_x=c_kernel_size_x, m_kernel_size_x=m_kernel_size_x,
                c_out_drop_x=c_out_drop_x, c_latent_drop_x=c_latent_drop_x,
                c_out_drop_tok=c_out_drop_tok, c_latent_drop_tok=c_latent_drop_tok,
                qk_norm=qk_norm, use_rope=use_rope, theta=theta,
                dropout_attn=dropout_attn,
                attn_out_drop_x=attn_out_drop_x, attn_out_drop_tok=attn_out_drop_tok,
                use_ls=use_ls, ffn_type=ffn_type,
                ffn_latent_drop=ffn_latent_drop, ffn_out_drop=ffn_out_drop,
                skip_first_ffn=skip_first_ffn, skip_out_ffn=skip_out_ffn,
                attn_type=attn_type,
            )
            for _ in range(num_layers)
        ])

        if self.use_out_norm:
            self.output_norm_x = RMSNorm(dim)
            self.output_norm_tok = RMSNorm(dim)
        self.output_proj_x = nn.Linear(dim, out_dim)
        self.output_proj_tok = nn.Linear(dim, out_dim)

    def forward(self, x, tok, t_mask, n_mask):
        """
        Args:
            x:      [B, T, in_dim] frame features
            tok:    [B, N, in_dim] token features (pre-embedded)
            t_mask: [B, T]         valid mask for frames
            n_mask: [B, N]         valid mask for tokens
        Returns:
            out_x:   [B, T, out_dim]
            out_tok: [B, N, out_dim]
        """
        x = self.input_proj(x)

        for layer in self.layers:
            x, tok = layer(tok, x, t_mask, n_mask)

        if self.use_out_norm:
            x = self.output_norm_x(x)
            tok = self.output_norm_tok(tok)

        out_x = self.output_proj_x(x)
        out_tok = self.output_proj_tok(tok)

        return out_x, out_tok

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def compute_inv_freq(dim: int, theta: float = 10000.0):
    """pre-compute inv_freq, dim is fixed"""
    return 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))


def compute_freqs_cis_dynamic(x: torch.Tensor, inv_freq: torch.Tensor):
    """ONNX兼容：动态计算cos/sin，序列长度从xa tensor shape获取"""
    # 用tensor操作获取seq_len，让ONNX能动态trace
    seq_len = x.shape[-2]
    # 生成位置索引 [0, 1, 2, ..., seq_len-1]
    t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
    # outer product: [seq_len, dim//2]
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def single_apply_rotary_emb(
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
):
    """ONNX兼容：手动实现复数乘法"""
    x_ = x.float().reshape(*x.shape[:-1], -1, 2).contiguous()

    # 分离实部和虚部
    x_r, x_i = x_[..., 0], x_[..., 1]

    # 复数乘法: (x_r + x_i*j) * (cos + sin*j) = (x_r*cos - x_i*sin) + (x_r*sin + x_i*cos)*j
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    # 合并实部和虚部
    x_out = torch.stack([x_out_r, x_out_i], dim=-1).flatten(-2)

    return x_out.type_as(x)


class SingleRoPosEmb(nn.Module):
    def __init__(self, dim: int, max_len=5000, theta=10000.0, use_cache=True):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.use_cache = use_cache
        # inv_freq是固定的，可以预计算
        self.register_buffer('inv_freq', compute_inv_freq(dim, theta), persistent=False)
        # 缓存模式下预计算pe
        if use_cache:
            pe_cos, pe_sin = compute_freqs_cis_dynamic(
                torch.zeros(1, max_len, dim), self.inv_freq)
            self.register_buffer('pe_cos', pe_cos[None, :, :], persistent=False)
            self.register_buffer('pe_sin', pe_sin[None, :, :], persistent=False)

    def extend_pe(self, x):
        """Reset the positional encodings (only for use_cache=True mode)."""
        if self.pe_cos.size(1) >= x.size(-2):
            return
        pe_cos, pe_sin = compute_freqs_cis_dynamic(x, self.inv_freq)
        self.pe_cos = pe_cos[None, :, :].to(device=x.device)
        self.pe_sin = pe_sin[None, :, :].to(device=x.device)

    def get_pe_dynamic(self, x):
        """ONNX模式：完全动态计算"""
        ndim = x.ndim
        seq_len = x.shape[-2]
        pe_cos, pe_sin = compute_freqs_cis_dynamic(x, self.inv_freq)
        pe_cos = pe_cos.view(*((1,) * (ndim - 2)), seq_len, self.dim // 2)
        pe_sin = pe_sin.view(*((1,) * (ndim - 2)), seq_len, self.dim // 2)
        return pe_cos, pe_sin

    def get_pe_cached(self, x):
        """Cache模式：从缓存切片"""
        ndim = x.ndim
        seq_len = x.size(-2)
        pe_cos = self.pe_cos[:, :seq_len]
        pe_sin = self.pe_sin[:, :seq_len]
        pe_cos = pe_cos.view(*((1,) * (ndim - 2)), seq_len, self.dim // 2)
        pe_sin = pe_sin.view(*((1,) * (ndim - 2)), seq_len, self.dim // 2)
        return pe_cos, pe_sin

    def forward(self, x):
        if self.use_cache:
            self.extend_pe(x)
            pe_cos, pe_sin = self.get_pe_cached(x)
        else:
            # ONNX模式：完全动态计算
            pe_cos, pe_sin = self.get_pe_dynamic(x)
        return single_apply_rotary_emb(x, pe_cos, pe_sin)



class LayScale(nn.Module):
    def __init__(self, dim, lay_scale_init_value=1e-6):
        super().__init__()
        sp = torch.ones(dim) * lay_scale_init_value

        self.scale = nn.Parameter(sp)

        self.dim = dim

    def unc(self, res):
        n_dim = res.ndim
        if n_dim == 1:
            return self.scale
        else:
            return self.scale.view(*((1,) * (n_dim - 1)), self.dim)

    def forward(self, x):
        return x * self.unc(x)


class RMSnorm(torch.nn.Module):
    def __init__(self, dim: int, init_num=1, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim) * init_num)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x)
        return output * self.weight


class GLUFFN(nn.Module):
    def __init__(self, dim, latent_dim=None, dropout_latent: float = 0.1, dropout_output: float = 0.1):
        super().__init__()
        if latent_dim is None:
            latent_dim = dim * 4
        self.ln1 = nn.Linear(dim, latent_dim * 2)

        self.ln2 = nn.Linear(latent_dim, dim)
        self.dropout_latent = nn.Dropout(dropout_latent) if dropout_latent > 0. else nn.Identity()
        self.dropout_output = nn.Dropout(dropout_output) if dropout_output > 0. else nn.Identity()

    def forward(self, x):
        x1, x2 = self.ln1(x).chunk(2, dim=-1)
        x = F.gelu(x1) * x2
        x = self.dropout_latent(x)
        x = self.ln2(x)
        return self.dropout_output(x)


class FFN(nn.Module):
    def __init__(self, dim, latent_dim=None, dropout_latent: float = 0.1, dropout_output: float = 0.1):
        super().__init__()
        if latent_dim is None:
            latent_dim = dim * 4
        self.ln1 = nn.Linear(dim, latent_dim)
        self.ln2 = nn.Linear(latent_dim, dim)
        self.dropout_latent = nn.Dropout(dropout_latent) if dropout_latent > 0. else nn.Identity()
        self.dropout_output = nn.Dropout(dropout_output) if dropout_output > 0. else nn.Identity()

    def forward(self, x):
        x = self.ln1(x)
        x = F.gelu(x)
        x = self.dropout_latent(x)
        x = self.ln2(x)
        return self.dropout_output(x)


class CgMLP(nn.Module):
    def __init__(
            self, dim: int,
            kernel_size: int = 31,
            out_drop=0.1,
            latent_drop=0.0,
            bias: bool = True,
            use_dw_act=True,
            latent_dim: Optional[int] = None
    ):
        super().__init__()
        if latent_dim is None:
            latent_dim = dim
        self.pw1 = nn.Conv1d(
            dim,
            latent_dim * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        self.use_dw_act = use_dw_act
        self.norm = RMSnorm(latent_dim)
        padding = (kernel_size - 1) // 2
        self.dw = nn.Conv1d(
            latent_dim, latent_dim, kernel_size,
            stride=1,
            padding=padding,
            groups=latent_dim,
            bias=bias
        )
        self.pw2 = nn.Conv1d(
            latent_dim,
            dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        self.out_drop = nn.Dropout(out_drop) if out_drop > 0. else nn.Identity()
        self.latent_drop = nn.Dropout(latent_drop) if latent_drop > 0. else nn.Identity()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pw1(x)
        x = F.gelu(x)
        x1, x2 = x.chunk(2, dim=1)
        x2 = self.norm(x2.transpose(1, 2)).transpose(1, 2)
        x2 = self.dw(x2)
        if self.use_dw_act:
            x2 = F.gelu(x2)
        x = x1 * x2
        x = self.latent_drop(x)
        x = self.pw2(x)
        return self.out_drop(x).transpose(1, 2)


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
        mask: [B, S] bool, True = padding (可选)
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

        self.a_norm = RMSnorm(dim)
        self.c_norm = RMSnorm(dim)

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
            self.norm1 = RMSnorm(dim)
        if not skip_out_ffn:
            self.norm2 = RMSnorm(dim)

        if use_ls:
            if not skip_first_ffn:
                self.lay_scale1 = LayScale(dim)
            self.lay_scale2 = LayScale(dim)
            if not skip_out_ffn:
                self.lay_scale3 = LayScale(dim)
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
            self.latent_norm = RMSnorm(dim)
            self.latent_proj = nn.Linear(dim, latent_out_dim)  # -> [B, T, C_latent]
        if self.use_out_norm:
            self.output_norm = RMSnorm(dim)
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
    (`tok`, pre-embedded at `text_in_dim`) are projected to `dim` by separate
    input projections. The token stream is encoded with kernel-7 EBF layers;
    the frame stream is encoded with kernel-31 EBF layers, fused with the
    token stream via stacked CrossAttention layers (x attends to tok), then
    decoded with kernel-31 EBF layers. Both streams are projected to `out_dim`.

    Constructor signature: (audio_in_dim, text_in_dim, out_dim, **kwargs)
    Forward signature:     forward(x, tok, t_mask, n_mask)
                           -> (out_x, out_tok, attn)
    Matches ForcedAlignmentModel's backbone protocol; `attn` is a list of
    cross-attention weights [B, H, T, N], one per CA layer.
    """

    def __init__(
            self,
            audio_in_dim: int,
            text_in_dim: int,
            out_dim: int,
            # 共享参数
            dim: int = 256,
            num_heads: int = 8,
            head_dim: int = 64,
            # Token (text) encoder
            text_num_layers: int = 4,
            text_c_kernel_size: int = 7,
            text_m_kernel_size: int = 7,
            # Frame (audio) encoder
            audio_enc_num_layers: int = 4,
            audio_enc_c_kernel_size: int = 31,
            audio_enc_m_kernel_size: int = 31,
            # Cross-Attention
            num_ca_layers: int = 1,
            ca_dropout_attn: float = 0.0,
            ca_out_drop: float = 0.0,
            # Frame (audio) decoder
            audio_dec_num_layers: int = 4,
            audio_dec_c_kernel_size: int = 31,
            audio_dec_m_kernel_size: int = 31,
            # 通用 EBF 参数
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
        self.audio_in_dim = audio_in_dim
        self.text_in_dim = text_in_dim
        self.out_dim = out_dim

        # ============ Input Projections (frame & token) ============
        self.audio_input_proj = nn.Linear(audio_in_dim, dim)
        self.text_input_proj = nn.Linear(text_in_dim, dim)

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
        self.text_enc_norm = RMSnorm(dim)

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
        self.audio_enc_norm = RMSnorm(dim)

        # ============ Cross-Attention Layers (x <- tok) ============
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
            self.ca_pre_norms.append(RMSnorm(dim))
            self.ca_lay_scales.append(LayScale(dim) if use_ls else nn.Identity())

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
            self.output_norm_x = RMSnorm(dim)
            self.output_norm_tok = RMSnorm(dim)
        self.output_proj_x = nn.Linear(dim, out_dim)
        self.output_proj_tok = nn.Linear(dim, out_dim)

    def forward(self, x, tok, t_mask, n_mask):
        """
        Args:
            x:      [B, T, audio_in_dim] 帧特征 (raw spectrogram)
            tok:    [B, N, text_in_dim]  token 特征 (pre-embedded)
            t_mask: [B, T] bool, True = 有效帧
            n_mask: [B, N] bool, True = 有效 token
        Returns:
            out_x:   [B, T, out_dim]
            out_tok: [B, N, out_dim]
            attn:    list of [B, H, T, N], 长度 = num_ca_layers
        """
        # ============ Input Projections ============
        x = self.audio_input_proj(x)
        tok = self.text_input_proj(tok)

        # ============ Token Encoder ============
        for layer in self.text_encoder:
            tok = layer(tok, mask=n_mask)
        tok = self.text_enc_norm(tok)

        # ============ Frame Encoder ============
        for layer in self.audio_encoder:
            x = layer(x, mask=t_mask)
        x = self.audio_enc_norm(x)

        # ============ Cross-Attention (x attends to tok) ============
        # CrossAttention 的 mask 语义: True = padding，需要从 n_mask 反转
        text_padding_mask = ~n_mask if n_mask is not None else None

        attn = []
        for i in range(self.num_ca_layers):
            residual = x
            x_normed = self.ca_pre_norms[i](x)

            ca_out, attn_w = self.cross_attn_layers[i](
                x_normed, tok, mask=text_padding_mask, return_attn=True
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
            tok = self.output_norm_tok(tok)
        out_x = self.output_proj_x(x)
        out_tok = self.output_proj_tok(tok)

        return out_x, out_tok, attn

if __name__ == '__main__':
    # 协议: (audio_in_dim, text_in_dim, out_dim, **kwargs)
    #       forward(x, tok, t_mask, n_mask) -> (out_x, out_tok, attn)
    AUDIO_IN_DIM = 80    # raw mel
    TEXT_IN_DIM = 256    # token embedding dim
    OUT_DIM = 256

    model = EBFAlignmentBackbone(
        audio_in_dim=AUDIO_IN_DIM,
        text_in_dim=TEXT_IN_DIM,
        out_dim=OUT_DIM,
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
    # x: 原始 mel；tok: ForcedAlignmentModel 里 token_embedding 之后的输出
    x = torch.randn(B, T, AUDIO_IN_DIM)
    tok = torch.randn(B, N, TEXT_IN_DIM)

    t_lens = torch.randint(T // 2, T + 1, (B,))
    n_lens = torch.randint(N // 2, N + 1, (B,))
    t_mask = torch.arange(T)[None, :] < t_lens[:, None]  # [B, T]
    n_mask = torch.arange(N)[None, :] < n_lens[:, None]  # [B, N]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params / 1e6:.2f}M")

    model.eval()
    with torch.no_grad():
        out_x, out_tok, attn = model(x, tok, t_mask, n_mask)
        print(f"out_x shape:   {tuple(out_x.shape)}")    # (B, T, OUT_DIM)
        print(f"out_tok shape: {tuple(out_tok.shape)}")  # (B, N, OUT_DIM)
        for i, aw in enumerate(attn):
            print(f"  attn[{i}] shape: {tuple(aw.shape)}")  # (B, H, T, N)

        assert out_x.shape == (B, T, OUT_DIM)
        assert out_tok.shape == (B, N, OUT_DIM)
        assert len(attn) == 2 and attn[0].shape == (B, 8, T, N)
        assert not torch.isnan(out_x).any() and not torch.isnan(out_tok).any()
        print("OK")

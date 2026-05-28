from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F

from modules.backbones.layers import RMSNorm
from modules.backbones.rope import SingleRoPosEmb


class JointAttention(nn.Module):
    """Joint attention between token stream and frame stream.

    Q/K/V are projected separately per stream, RoPE is applied per stream,
    then Q/K/V are concatenated for a single SDPA call. No attn_mask --
    padding is handled upstream via masked_fill (EBF style).
    """

    def __init__(
            self, dim,
            num_heads,
            head_dim,
            qk_norm=True,
            use_rope=True,
            theta=10000.0,
            dropout_attn: float = 0.0,
            out_drop_x: float = 0.0,
            out_drop_tok: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.attn_dim = num_heads * head_dim
        self.out_drop_x = nn.Dropout(out_drop_x) if out_drop_x > 0. else nn.Identity()
        self.out_drop_tok = nn.Dropout(out_drop_tok) if out_drop_tok > 0. else nn.Identity()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.use_rope = use_rope
        self.dropout_attn = dropout_attn

        self.tok_qkv = nn.Linear(dim, self.attn_dim * 3, bias=True)
        self.x_qkv = nn.Linear(dim, self.attn_dim * 3, bias=True)

        self.qk_norm = qk_norm
        if qk_norm:
            self.tok_q_norm = RMSNorm(head_dim)
            self.tok_k_norm = RMSNorm(head_dim)
            self.x_q_norm = RMSNorm(head_dim)
            self.x_k_norm = RMSNorm(head_dim)

        self.tok_norm = RMSNorm(dim)
        self.x_norm = RMSNorm(dim)

        if use_rope:
            self.tok_rope = SingleRoPosEmb(head_dim, theta=theta)
            self.x_rope = SingleRoPosEmb(head_dim, theta=theta)

        self.tok_out = nn.Linear(self.attn_dim, self.dim, bias=True)
        self.x_out = nn.Linear(self.attn_dim, self.dim, bias=True)

    def _to_heads(self, x):
        return rearrange(x, 'b t (h d) -> b h t d', h=self.num_heads)

    def _flatten_heads(self, x):
        return rearrange(x, 'b h t d -> b t (h d)')

    def _project_qkv(self, tok, x):
        tok_q, tok_k, tok_v = self.tok_qkv(self.tok_norm(tok)).chunk(3, dim=-1)
        x_q, x_k, x_v = self.x_qkv(self.x_norm(x)).chunk(3, dim=-1)

        tok_q, tok_k, tok_v = map(self._to_heads, (tok_q, tok_k, tok_v))
        x_q, x_k, x_v = map(self._to_heads, (x_q, x_k, x_v))

        if self.qk_norm:
            tok_q = self.tok_q_norm(tok_q)
            tok_k = self.tok_k_norm(tok_k)
            x_q = self.x_q_norm(x_q)
            x_k = self.x_k_norm(x_k)

        return tok_q, tok_k, tok_v, x_q, x_k, x_v

    def _apply_rope(self, tok_q, tok_k, x_q, x_k):
        if not self.use_rope:
            return tok_q, tok_k, x_q, x_k
        tok_q = self.tok_rope(tok_q)
        tok_k = self.tok_rope(tok_k)
        x_q = self.x_rope(x_q)
        x_k = self.x_rope(x_k)
        return tok_q, tok_k, x_q, x_k

    def _mask_outputs(self, tok, x, t_mask, n_mask):
        tok = tok * n_mask.unsqueeze(-1).float()
        x = x * t_mask.unsqueeze(-1).float()
        return tok, x

    def forward(self, tok, x, t_mask, n_mask):
        N = tok.shape[1]
        tok_q, tok_k, tok_v, x_q, x_k, x_v = self._project_qkv(tok, x)
        tok_q, tok_k, x_q, x_k = self._apply_rope(tok_q, tok_k, x_q, x_k)

        q = torch.cat([tok_q, x_q], dim=2)
        k = torch.cat([tok_k, x_k], dim=2)
        v = torch.cat([tok_v, x_v], dim=2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_attn if self.training else 0.0,
        )

        tok_attn = self._flatten_heads(out[:, :, :N, :])
        x_attn = self._flatten_heads(out[:, :, N:, :])

        tok_attn = self.tok_out(tok_attn)
        x_attn = self.x_out(x_attn)
        tok = self.out_drop_tok(tok_attn)
        x = self.out_drop_x(x_attn)
        tok, x = self._mask_outputs(tok, x, t_mask, n_mask)

        return tok, x


class SplitJointAttention(nn.Module):
    """4-way split joint attention.

    Separates same-stream (tok->tok, x->x) and cross-stream (tok->x, x->tok)
    attention, then merges each stream's same+cross outputs via learned linear.
    No attn_mask -- padding handled via masked_fill (EBF style).
    """

    def __init__(
            self, dim,
            num_heads,
            head_dim,
            qk_norm=True,
            use_rope=True,
            theta=10000.0,
            dropout_attn: float = 0.0,
            out_drop_x: float = 0.0,
            out_drop_tok: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.attn_dim = num_heads * head_dim
        self.out_drop_x = nn.Dropout(out_drop_x) if out_drop_x > 0. else nn.Identity()
        self.out_drop_tok = nn.Dropout(out_drop_tok) if out_drop_tok > 0. else nn.Identity()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.use_rope = use_rope
        self.dropout_attn = dropout_attn

        self.tok_qkv = nn.Linear(dim, self.attn_dim * 3, bias=True)
        self.x_qkv = nn.Linear(dim, self.attn_dim * 3, bias=True)

        self.qk_norm = qk_norm
        if qk_norm:
            self.tok_q_norm = RMSNorm(head_dim)
            self.tok_k_norm = RMSNorm(head_dim)
            self.x_q_norm = RMSNorm(head_dim)
            self.x_k_norm = RMSNorm(head_dim)

        self.tok_norm = RMSNorm(dim)
        self.x_norm = RMSNorm(dim)

        if use_rope:
            self.tok_rope = SingleRoPosEmb(head_dim, theta=theta)
            self.x_rope = SingleRoPosEmb(head_dim, theta=theta)

        self.tok_merge = nn.Linear(self.attn_dim * 2, dim, bias=True)
        self.x_merge = nn.Linear(self.attn_dim * 2, dim, bias=True)

    def _to_heads(self, x):
        return rearrange(x, 'b t (h d) -> b h t d', h=self.num_heads)

    def _flatten_heads(self, x):
        return rearrange(x, 'b h t d -> b t (h d)')

    def _project_qkv(self, tok, x):
        tok_q, tok_k, tok_v = self.tok_qkv(self.tok_norm(tok)).chunk(3, dim=-1)
        x_q, x_k, x_v = self.x_qkv(self.x_norm(x)).chunk(3, dim=-1)

        tok_q, tok_k, tok_v = map(self._to_heads, (tok_q, tok_k, tok_v))
        x_q, x_k, x_v = map(self._to_heads, (x_q, x_k, x_v))

        if self.qk_norm:
            tok_q = self.tok_q_norm(tok_q)
            tok_k = self.tok_k_norm(tok_k)
            x_q = self.x_q_norm(x_q)
            x_k = self.x_k_norm(x_k)

        return tok_q, tok_k, tok_v, x_q, x_k, x_v

    def _apply_rope(self, tok_q, tok_k, x_q, x_k):
        if not self.use_rope:
            return tok_q, tok_k, x_q, x_k
        tok_q = self.tok_rope(tok_q)
        tok_k = self.tok_rope(tok_k)
        x_q = self.x_rope(x_q)
        x_k = self.x_rope(x_k)
        return tok_q, tok_k, x_q, x_k

    def _attention(self, q, k, v):
        return F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_attn if self.training else 0.0,
        )

    def _mask_outputs(self, tok, x, t_mask, n_mask):
        tok = tok * n_mask.unsqueeze(-1).float()
        x = x * t_mask.unsqueeze(-1).float()
        return tok, x

    def forward(self, tok, x, t_mask, n_mask):
        tok_q, tok_k, tok_v, x_q, x_k, x_v = self._project_qkv(tok, x)
        tok_q_r, tok_k_r, x_q_r, x_k_r = self._apply_rope(
            tok_q, tok_k, x_q, x_k
        )

        # 1. tok -> tok (same-stream)
        tt_out = self._attention(tok_q_r, tok_k_r, tok_v)
        # 2. x -> x (same-stream)
        xx_out = self._attention(x_q_r, x_k_r, x_v)
        # 3. tok -> x (cross-stream)
        tx_out = self._attention(tok_q_r, x_k_r, x_v)
        # 4. x -> tok (cross-stream)
        xt_out = self._attention(x_q_r, tok_k_r, tok_v)

        tt_flat = self._flatten_heads(tt_out)
        tx_flat = self._flatten_heads(tx_out)
        xx_flat = self._flatten_heads(xx_out)
        xt_flat = self._flatten_heads(xt_out)

        tok_attn = self.tok_merge(torch.cat([tt_flat, tx_flat], dim=-1))
        x_attn = self.x_merge(torch.cat([xx_flat, xt_flat], dim=-1))

        tok = self.out_drop_tok(tok_attn)
        x = self.out_drop_x(x_attn)
        tok, x = self._mask_outputs(tok, x, t_mask, n_mask)

        return tok, x

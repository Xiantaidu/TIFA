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
            out_drop_token: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.attn_dim = num_heads * head_dim
        self.out_drop_x = nn.Dropout(out_drop_x) if out_drop_x > 0. else nn.Identity()
        self.out_drop_token = nn.Dropout(out_drop_token) if out_drop_token > 0. else nn.Identity()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.use_rope = use_rope
        self.dropout_attn = dropout_attn

        self.token_qkv = nn.Linear(dim, self.attn_dim * 3, bias=True)
        self.x_qkv = nn.Linear(dim, self.attn_dim * 3, bias=True)

        self.qk_norm = qk_norm
        if qk_norm:
            self.token_q_norm = RMSNorm(head_dim)
            self.token_k_norm = RMSNorm(head_dim)
            self.x_q_norm = RMSNorm(head_dim)
            self.x_k_norm = RMSNorm(head_dim)

        self.token_norm = RMSNorm(dim)
        self.x_norm = RMSNorm(dim)

        if use_rope:
            self.token_rope = SingleRoPosEmb(head_dim, theta=theta)
            self.x_rope = SingleRoPosEmb(head_dim, theta=theta)

        self.token_out = nn.Linear(self.attn_dim, self.dim, bias=True)
        self.x_out = nn.Linear(self.attn_dim, self.dim, bias=True)

    def _to_heads(self, x):
        return rearrange(x, 'b t (h d) -> b h t d', h=self.num_heads)

    def _flatten_heads(self, x):
        return rearrange(x, 'b h t d -> b t (h d)')

    def _project_qkv(self, token, x):
        token_q, token_k, token_v = self.token_qkv(self.token_norm(token)).chunk(3, dim=-1)
        x_q, x_k, x_v = self.x_qkv(self.x_norm(x)).chunk(3, dim=-1)

        token_q, token_k, token_v = map(self._to_heads, (token_q, token_k, token_v))
        x_q, x_k, x_v = map(self._to_heads, (x_q, x_k, x_v))

        if self.qk_norm:
            token_q = self.token_q_norm(token_q)
            token_k = self.token_k_norm(token_k)
            x_q = self.x_q_norm(x_q)
            x_k = self.x_k_norm(x_k)

        return token_q, token_k, token_v, x_q, x_k, x_v

    def _apply_rope(self, token_q, token_k, x_q, x_k):
        if not self.use_rope:
            return token_q, token_k, x_q, x_k
        token_q = self.token_rope(token_q)
        token_k = self.token_rope(token_k)
        x_q = self.x_rope(x_q)
        x_k = self.x_rope(x_k)
        return token_q, token_k, x_q, x_k

    def _mask_outputs(self, token, x, t_mask, n_mask):
        token = token * n_mask.unsqueeze(-1).float()
        x = x * t_mask.unsqueeze(-1).float()
        return token, x

    def forward(self, token, x, t_mask, n_mask):
        N = token.shape[1]
        token_q, token_k, token_v, x_q, x_k, x_v = self._project_qkv(token, x)
        token_q, token_k, x_q, x_k = self._apply_rope(token_q, token_k, x_q, x_k)

        q = torch.cat([token_q, x_q], dim=2)
        k = torch.cat([token_k, x_k], dim=2)
        v = torch.cat([token_v, x_v], dim=2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_attn if self.training else 0.0,
        )

        token_attn = self._flatten_heads(out[:, :, :N, :])
        x_attn = self._flatten_heads(out[:, :, N:, :])

        token_attn = self.token_out(token_attn)
        x_attn = self.x_out(x_attn)
        token = self.out_drop_token(token_attn)
        x = self.out_drop_x(x_attn)
        token, x = self._mask_outputs(token, x, t_mask, n_mask)

        return token, x


class SplitJointAttention(nn.Module):
    """4-way split joint attention.

    Separates same-stream (token->token, x->x) and cross-stream (token->x,
    x->token) attention, then merges each stream's same+cross outputs via
    learned linear. No attn_mask -- padding handled via masked_fill (EBF style).
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
            out_drop_token: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.attn_dim = num_heads * head_dim
        self.out_drop_x = nn.Dropout(out_drop_x) if out_drop_x > 0. else nn.Identity()
        self.out_drop_token = nn.Dropout(out_drop_token) if out_drop_token > 0. else nn.Identity()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.use_rope = use_rope
        self.dropout_attn = dropout_attn

        self.token_qkv = nn.Linear(dim, self.attn_dim * 3, bias=True)
        self.x_qkv = nn.Linear(dim, self.attn_dim * 3, bias=True)

        self.qk_norm = qk_norm
        if qk_norm:
            self.token_q_norm = RMSNorm(head_dim)
            self.token_k_norm = RMSNorm(head_dim)
            self.x_q_norm = RMSNorm(head_dim)
            self.x_k_norm = RMSNorm(head_dim)

        self.token_norm = RMSNorm(dim)
        self.x_norm = RMSNorm(dim)

        if use_rope:
            self.token_rope = SingleRoPosEmb(head_dim, theta=theta)
            self.x_rope = SingleRoPosEmb(head_dim, theta=theta)

        self.token_merge = nn.Linear(self.attn_dim * 2, dim, bias=True)
        self.x_merge = nn.Linear(self.attn_dim * 2, dim, bias=True)

    def _to_heads(self, x):
        return rearrange(x, 'b t (h d) -> b h t d', h=self.num_heads)

    def _flatten_heads(self, x):
        return rearrange(x, 'b h t d -> b t (h d)')

    def _project_qkv(self, token, x):
        token_q, token_k, token_v = self.token_qkv(self.token_norm(token)).chunk(3, dim=-1)
        x_q, x_k, x_v = self.x_qkv(self.x_norm(x)).chunk(3, dim=-1)

        token_q, token_k, token_v = map(self._to_heads, (token_q, token_k, token_v))
        x_q, x_k, x_v = map(self._to_heads, (x_q, x_k, x_v))

        if self.qk_norm:
            token_q = self.token_q_norm(token_q)
            token_k = self.token_k_norm(token_k)
            x_q = self.x_q_norm(x_q)
            x_k = self.x_k_norm(x_k)

        return token_q, token_k, token_v, x_q, x_k, x_v

    def _apply_rope(self, token_q, token_k, x_q, x_k):
        if not self.use_rope:
            return token_q, token_k, x_q, x_k
        token_q = self.token_rope(token_q)
        token_k = self.token_rope(token_k)
        x_q = self.x_rope(x_q)
        x_k = self.x_rope(x_k)
        return token_q, token_k, x_q, x_k

    def _attention(self, q, k, v):
        return F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_attn if self.training else 0.0,
        )

    def _mask_outputs(self, token, x, t_mask, n_mask):
        token = token * n_mask.unsqueeze(-1).float()
        x = x * t_mask.unsqueeze(-1).float()
        return token, x

    def forward(self, token, x, t_mask, n_mask):
        token_q, token_k, token_v, x_q, x_k, x_v = self._project_qkv(token, x)
        token_q_r, token_k_r, x_q_r, x_k_r = self._apply_rope(
            token_q, token_k, x_q, x_k
        )

        # 1. token -> token (same-stream)
        tt_out = self._attention(token_q_r, token_k_r, token_v)
        # 2. x -> x (same-stream)
        xx_out = self._attention(x_q_r, x_k_r, x_v)
        # 3. token -> x (cross-stream)
        tx_out = self._attention(token_q_r, x_k_r, x_v)
        # 4. x -> token (cross-stream)
        xt_out = self._attention(x_q_r, token_k_r, token_v)

        tt_flat = self._flatten_heads(tt_out)
        tx_flat = self._flatten_heads(tx_out)
        xx_flat = self._flatten_heads(xx_out)
        xt_flat = self._flatten_heads(xt_out)

        token_attn = self.token_merge(torch.cat([tt_flat, tx_flat], dim=-1))
        x_attn = self.x_merge(torch.cat([xx_flat, xt_flat], dim=-1))

        token = self.out_drop_token(token_attn)
        x = self.out_drop_x(x_attn)
        token, x = self._mask_outputs(token, x, t_mask, n_mask)

        return token, x

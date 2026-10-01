"""Warehouse demand model (Phase 0 section 1).

    E = X W_e + b_e + P            W_e in R^{5x16}, P in R^{24x16} learned
    Q = E W_Q, K = E W_K, V = E W_V        d_model = d_k = d_v = 16
    S = QK^T/sqrt(d_k) (B,24,24) -> A = softmax(S) -> Y = AV (B,24,16)
    h = concat(Y[:,-1,:], E[:,-1,:]) (B,32) -> MLP(32->32->1, ReLU) -> z-score

Uses ScaledDotProductAttention from Problem 1 (cross-problem integration, PRD s21).
The no-attention control replaces A by uniform 1/24 (mean pooling of V).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.attention import ScaledDotProductAttention

D_IN = 5
D_MODEL = 16
WINDOW = 24


class WarehouseAttentionModel(nn.Module):
    def __init__(self, d_model: int = D_MODEL, hidden: int = 32):
        super().__init__()
        self.embed = nn.Linear(D_IN, d_model)
        self.pos = nn.Parameter(torch.zeros(WINDOW, d_model))
        self.attn = ScaledDotProductAttention(d_model, d_model, d_model, use_scale=True)
        self.head = nn.Sequential(nn.Linear(2 * d_model, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, X: torch.Tensor, return_weights: bool = False):
        E = self.embed(X) + self.pos[None, :, :]       # (B,24,16)
        Y, aux = self.attn(E, return_weights=True)     # (B,24,16)
        h = torch.cat([Y[:, -1, :], E[:, -1, :]], dim=-1)
        out = self.head(h).squeeze(-1)                 # (B,)
        if return_weights:
            return out, aux
        return out


class NoAttentionControl(nn.Module):
    """Identical except A = uniform 1/24 (mean pooling of V). Isolates attention's value (H3)."""

    def __init__(self, d_model: int = D_MODEL, hidden: int = 32):
        super().__init__()
        self.embed = nn.Linear(D_IN, d_model)
        self.pos = nn.Parameter(torch.zeros(WINDOW, d_model))
        self.attn = ScaledDotProductAttention(d_model, d_model, d_model, use_scale=True)
        self.head = nn.Sequential(nn.Linear(2 * d_model, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, X: torch.Tensor, return_weights: bool = False):
        E = self.embed(X) + self.pos[None, :, :]
        Q = E @ self.attn.W_Q
        K = E @ self.attn.W_K
        V = E @ self.attn.W_V
        A = torch.full((E.shape[0], WINDOW, WINDOW), 1.0 / WINDOW,
                       device=E.device, dtype=E.dtype)
        Y = A @ V
        h = torch.cat([Y[:, -1, :], E[:, -1, :]], dim=-1)
        out = self.head(h).squeeze(-1)
        if return_weights:
            return out, {"Q": Q, "K": K, "V": V,
                         "S": Q @ K.transpose(-2, -1) * self.attn.scale, "A": A}
        return out

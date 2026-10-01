"""First-principles scaled dot-product self-attention.

Implements Phase 0 section 1 and PRD section 7 using raw tensor ops only.
No nn.MultiheadAttention / nn.Transformer is used anywhere.

Math:
    Q = X W_Q,  K = X W_K,  V = X W_V
    S = Q K^T / sqrt(d_k)          (or unscaled S = Q K^T for the ablation)
    A = softmax(S)  (row-wise, max-subtracted for stability)
    Y = A V
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """softmax(x) = exp(x - max(x)) / sum(exp(x - max(x))).

    Subtracting the row max is mathematically identical (softmax(x) = softmax(x - c))
    but avoids exp() overflow: naive exp overflows to inf for logits > ~89 in
    float32 (~710 in float64), giving inf/inf = nan.
    """
    m = logits.max(dim=dim, keepdim=True).values
    e = torch.exp(logits - m)
    return e / e.sum(dim=dim, keepdim=True)


def naive_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Unstable reference: exp(x)/sum(exp(x)). Overflows to nan for large logits."""
    e = torch.exp(logits)
    return e / e.sum(dim=dim, keepdim=True)


def numpy_reference_attention(
    X: "object", W_Q: "object", W_K: "object", W_V: "object", scale: float | None = None
):
    """Plain-NumPy reference implementation (no torch), for independent testing."""
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    W_Q = np.asarray(W_Q, dtype=np.float64)
    W_K = np.asarray(W_K, dtype=np.float64)
    W_V = np.asarray(W_V, dtype=np.float64)
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    dk = W_Q.shape[1]
    s = 1.0 / math.sqrt(dk) if scale is None else scale
    S = (Q @ K.swapaxes(-1, -2)) * s
    m = S.max(axis=-1, keepdims=True)
    E = np.exp(S - m)
    A = E / E.sum(axis=-1, keepdims=True)
    Y = A @ V
    return {"Q": Q, "K": K, "V": V, "S": S, "A": A, "Y": Y}


class ScaledDotProductAttention(nn.Module):
    """Single-head, single-layer self-attention from raw ops.

    Args:
        d_in: input feature dim (X last dim).
        d_k: query/key dim. d_v: value dim.
        use_scale: if True divide by sqrt(d_k); if False use raw QK^T (ablation).
        bias: whether QKV projections use a bias term (default False: pure Q=XW_Q).
    """

    def __init__(self, d_in: int, d_k: int, d_v: int | None = None, use_scale: bool = True,
                 bias: bool = False):
        super().__init__()
        d_v = d_k if d_v is None else d_v
        self.d_in, self.d_k, self.d_v = d_in, d_k, d_v
        self.use_scale = use_scale
        std = 1.0 / math.sqrt(d_in)
        self.W_Q = nn.Parameter(torch.empty(d_in, d_k).uniform_(-std, std))
        self.W_K = nn.Parameter(torch.empty(d_in, d_k).uniform_(-std, std))
        self.W_V = nn.Parameter(torch.empty(d_in, d_v).uniform_(-std, std))

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(self.d_k) if self.use_scale else 1.0

    def forward(self, X: torch.Tensor, return_weights: bool = False):
        # Q = X W_Q, K = X W_K, V = X W_V
        Q = X @ self.W_Q          # (..., T, d_k)
        K = X @ self.W_K          # (..., T, d_k)
        V = X @ self.W_V          # (..., T, d_v)
        # S = Q K^T / sqrt(d_k)
        S = (Q @ K.transpose(-2, -1)) * self.scale   # (..., T, T)
        # A = softmax(S) row-wise
        A = stable_softmax(S, dim=-1)
        # Y = A V
        Y = A @ V                 # (..., T, d_v)
        if return_weights:
            return Y, {"Q": Q, "K": K, "V": V, "S": S, "A": A}
        return Y

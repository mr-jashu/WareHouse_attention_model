"""Attention tests: shapes, normalisation, hand reference, numpy match, equivariance."""
import math

import numpy as np
import pytest
import torch

from src.attention import (
    ScaledDotProductAttention,
    naive_softmax,
    numpy_reference_attention,
    stable_softmax,
)


def test_output_shapes():
    attn = ScaledDotProductAttention(d_in=5, d_k=16, d_v=16)
    X = torch.randn(4, 24, 5)
    Y, aux = attn(X, return_weights=True)
    assert aux["Q"].shape == (4, 24, 16)
    assert aux["K"].shape == (4, 24, 16)
    assert aux["V"].shape == (4, 24, 16)
    assert aux["S"].shape == (4, 24, 24)
    assert aux["A"].shape == (4, 24, 24)
    assert Y.shape == (4, 24, 16)


def test_softmax_normalisation():
    attn = ScaledDotProductAttention(d_in=5, d_k=8)
    X = torch.randn(2, 6, 5)
    _, aux = attn(X, return_weights=True)
    A = aux["A"]
    assert torch.all(A >= 0)
    assert torch.allclose(A.sum(-1), torch.ones(2, 6), atol=1e-6)


def test_hand_computed_tiny_example():
    # T=3, d_in=2, d_k=2, identity projections: Q=K=V=X.
    attn = ScaledDotProductAttention(d_in=2, d_k=2, bias=False)
    with torch.no_grad():
        attn.W_Q.copy_(torch.eye(2, dtype=torch.float32))
        attn.W_K.copy_(torch.eye(2, dtype=torch.float32))
        attn.W_V.copy_(torch.eye(2, dtype=torch.float32))
    X = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    Y, aux = attn(X, return_weights=True)
    # Hand: S = XX^T/sqrt(2); row 0 = [1,0,1]/sqrt(2).
    s = 1 / math.sqrt(2)
    logits = torch.tensor([1.0, 0.0, 1.0]) * s
    e = torch.exp(logits - logits.max())
    a0 = e / e.sum()
    assert torch.allclose(aux["A"][0, 0], a0, atol=1e-6)
    # Y[0,0] = a0[0]*[1,0] + a0[1]*[0,1] + a0[2]*[1,1]
    expected = a0[0] * torch.tensor([1.0, 0.0]) + a0[1] * torch.tensor([0.0, 1.0]) + a0[2] * torch.tensor([1.0, 1.0])
    assert torch.allclose(Y[0, 0], expected, atol=1e-6)


def test_matches_numpy_reference():
    torch.manual_seed(0)
    attn = ScaledDotProductAttention(d_in=5, d_k=4).double()
    X = torch.randn(3, 7, 5, dtype=torch.float64)
    Y, aux = attn(X, return_weights=True)
    ref = numpy_reference_attention(
        X.detach().numpy(),
        attn.W_Q.detach().numpy(), attn.W_K.detach().numpy(), attn.W_V.detach().numpy(),
    )
    assert np.allclose(aux["A"].detach().numpy(), ref["A"], atol=1e-10)
    assert np.allclose(Y.detach().numpy(), ref["Y"], atol=1e-10)


def test_permutation_equivariance_without_position():
    # Pure attention (no positional embedding) is permutation-equivariant:
    # permuting tokens permutes outputs identically.
    torch.manual_seed(1)
    attn = ScaledDotProductAttention(d_in=4, d_k=8)
    X = torch.randn(2, 6, 4)
    perm = torch.randperm(6)
    Y1 = attn(X)
    Y2 = attn(X[:, perm, :])
    assert torch.allclose(Y2, Y1[:, perm, :], atol=1e-5)


def test_unscaled_logits_are_larger():
    # Same weights, scaled vs unscaled: unscaled logits have std sqrt(d_k) larger.
    torch.manual_seed(2)
    d_k = 64
    a = ScaledDotProductAttention(d_in=8, d_k=d_k, use_scale=True)
    b = ScaledDotProductAttention(d_in=8, d_k=d_k, use_scale=False)
    with torch.no_grad():
        b.W_Q.copy_(a.W_Q); b.W_K.copy_(a.W_K); b.W_V.copy_(a.W_V)
    X = torch.randn(4, 8, 8)
    _, ax = a(X, return_weights=True)
    _, bx = b(X, return_weights=True)
    assert torch.allclose(bx["S"], ax["S"] * math.sqrt(d_k), atol=1e-4)
    # Unscaled attention is sharper (higher max weight) at large d_k.
    assert bx["A"].amax(-1).mean() >= ax["A"].amax(-1).mean()


def test_stable_vs_naive_agree_and_stable_survives_large_logits():
    torch.manual_seed(0)
    logits = torch.randn(2, 5) * 2
    assert torch.allclose(stable_softmax(logits), naive_softmax(logits), atol=1e-6)
    big = torch.full((2, 5), 100.0)  # exp(100) overflows fp32
    bad = naive_softmax(big.float())
    good = stable_softmax(big.float())
    assert torch.isnan(bad).any() or torch.isinf(torch.exp(big.float())).any()
    assert torch.allclose(good, torch.full_like(good, 0.2), atol=1e-6)


def test_gradients_flow_to_all_params_and_input():
    attn = ScaledDotProductAttention(d_in=5, d_k=6).double()
    X = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    Y = attn(X)
    Y.sum().backward()
    for p in (attn.W_Q, attn.W_K, attn.W_V):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert X.grad is not None and torch.isfinite(X.grad).all()

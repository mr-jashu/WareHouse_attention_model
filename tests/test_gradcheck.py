"""Gradient-verification + stability tests (fast, tiny dims; full sweep in experiments/gradcheck.py)."""
import torch

from src.attention import ScaledDotProductAttention, naive_softmax, stable_softmax


def _central_grads(attn, X, seed_out, eps=1e-5):
    num = {}
    with torch.no_grad():
        for name in ("W_Q", "W_K", "W_V"):
            p = getattr(attn, name)
            g = torch.zeros_like(p)
            it = __import__("itertools").product(*[range(s) for s in p.shape])
            for idx in it:
                p[idx] += eps
                f1 = (attn(X) * seed_out).sum().item()
                p[idx] -= 2 * eps
                f2 = (attn(X) * seed_out).sum().item()
                p[idx] += eps
                g[idx] = (f1 - f2) / (2 * eps)
            num[name] = g
    return num


def test_gradcheck_float64_tiny():
    torch.manual_seed(0)
    attn = ScaledDotProductAttention(2, 2, 2).double()
    X = torch.randn(1, 2, 2, dtype=torch.float64)
    seed_out = torch.randn(1, 2, 2, dtype=torch.float64)
    loss = (attn(X) * seed_out).sum()
    ana = torch.autograd.grad(loss, [attn.W_Q, attn.W_K, attn.W_V])
    num = _central_grads(attn, X, seed_out)
    for a, (k, n) in zip(ana, num.items()):
        rel = ((a - n).abs() / torch.maximum(a.abs(), n.abs()).clamp_min(1e-12)).max().item()
        assert rel < 1e-6, (k, rel)


def test_softmax_stability_threshold():
    big = torch.full((1, 4), 100.0)
    assert torch.isnan(naive_softmax(big)).any()  # naive breaks
    good = stable_softmax(big)
    assert torch.allclose(good, torch.full_like(good, 0.25), atol=1e-6)
    torch.manual_seed(0)
    logits = torch.randn(3, 5) * 2  # moderate: both must agree
    assert torch.allclose(stable_softmax(logits), naive_softmax(logits), atol=1e-6)

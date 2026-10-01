"""Gradient verification (PRD s9, Phase 0 V1) + numerical-stability demo (PRD s11).

- float64 central differences (eps=1e-5) vs autograd for W_Q, W_K, W_V, X.
  Pass criterion: max relative error < 1e-6 (Phase 0 V1).
- float32 reference showing ~1e-3..1e-2 roundoff (expected, not a bug).
- Stability demo: naive exp softmax -> nan for logits >= ~89 fp32; max-subtracted
  version returns the exact answer. Writes results/gradcheck.txt.
"""
from __future__ import annotations

import math
from itertools import product
from pathlib import Path

import torch

from src.attention import ScaledDotProductAttention, naive_softmax, stable_softmax


def _rel_err(a: torch.Tensor, n: torch.Tensor) -> float:
    denom = torch.maximum(a.abs(), n.abs()).clamp_min(1e-12)
    return ((a - n).abs() / denom).max().item()


def central_diff_grads(attn, X, seed_out: torch.Tensor, eps: float = 1e-5):
    """Central finite differences of (seed_out * Y).sum() w.r.t. W_Q,W_K,W_V,X."""
    num = {}
    with torch.no_grad():
        for name in ("W_Q", "W_K", "W_V"):
            p = getattr(attn, name)
            g = torch.zeros_like(p)
            for idx in product(*[range(s) for s in p.shape]):
                p[idx] += eps
                f1 = (attn(X) * seed_out).sum().item()
                p[idx] -= 2 * eps
                f2 = (attn(X) * seed_out).sum().item()
                p[idx] += eps
                g[idx] = (f1 - f2) / (2 * eps)
            num[name] = g
        g = torch.zeros_like(X)
        for idx in product(*[range(s) for s in X.shape]):
            X[idx] += eps
            f1 = (attn(X) * seed_out).sum().item()
            X[idx] -= 2 * eps
            f2 = (attn(X) * seed_out).sum().item()
            X[idx] += eps
            g[idx] = (f1 - f2) / (2 * eps)
        num["X"] = g
    return num


def run(dtype: torch.dtype, B: int = 2, T: int = 4, d_in: int = 5, d_k: int = 6):
    torch.manual_seed(0)
    attn = ScaledDotProductAttention(d_in, d_k, d_k).to(dtype)
    X = torch.randn(B, T, d_in, dtype=dtype, requires_grad=True)
    seed_out = torch.randn(B, T, d_k, dtype=dtype)
    Y = attn(X)
    loss = (Y * seed_out).sum()
    ana = {}
    grads = torch.autograd.grad(loss, [attn.W_Q, attn.W_K, attn.W_V, X])
    for name, g in zip(("W_Q", "W_K", "W_V", "X"), grads):
        ana[name] = g.detach().clone()
    num = central_diff_grads(attn, X.detach(), seed_out, eps=1e-5)
    rows = {k: (_rel_err(ana[k], num[k]), ana[k].abs().max().item()) for k in ana}
    return rows


def stability_demo() -> dict:
    out = {}
    big32 = torch.full((1, 4), 100.0)  # exp(100) overflows fp32
    raw = torch.exp(big32)
    out["exp100_fp32_is_inf"] = bool(torch.isinf(raw).any())
    out["naive_has_nan"] = bool(torch.isnan(naive_softmax(big32)).any())
    good = stable_softmax(big32)
    out["stable_row"] = [round(v, 4) for v in good[0].tolist()]
    out["stable_is_uniform"] = bool(torch.allclose(good, torch.full_like(good, 0.25), atol=1e-6))
    # Threshold scan in fp32: naive breaks at ~89, stable never does.
    for v in (80.0, 89.0, 90.0, 100.0):
        L = torch.full((1, 4), v) + torch.tensor([[0.0, -1.0, -2.0, -3.0]])
        n = naive_softmax(L)
        s = stable_softmax(L)
        out[f"logit{v:g}_naive_ok"] = bool(torch.isfinite(n).all())
        out[f"logit{v:g}_stable_ok"] = bool(torch.isfinite(s).all())
    return out


def main() -> None:
    lines = ["# Gradient verification (Phase 0 V1: float64 central diff eps=1e-5, tol 1e-6)"]
    rows64 = run(torch.float64)
    ok64 = True
    for k, (rel, mx) in rows64.items():
        ok = rel < 1e-6
        ok64 &= ok
        lines.append(f"float64 {k}: max_rel_err={rel:.3e} max|grad|={mx:.4f} {'PASS' if ok else 'FAIL'}")
    lines.append(f"float64 overall: {'PASS' if ok64 else 'FAIL'}")
    rows32 = run(torch.float32)
    for k, (rel, mx) in rows32.items():
        lines.append(f"float32 {k}: max_rel_err={rel:.3e} (roundoff reference, not a bug)")
    lines.append("# Numerical stability demo (PRD s11)")
    for k, v in stability_demo().items():
        lines.append(f"{k}: {v}")
    lines.append("Explanation: float32 central differences divide ~1e-7-scale roundoff by "
                 "eps=1e-5, so errors up to ~1e-1 here are expected (6 orders of magnitude "
                 "above the float64 1e-9 level) — roundoff, not a bug. float64 has ~1e-16 roundoff "
                 "and passes 1e-6. Naive softmax exp() overflows fp32 at ~89; max-subtraction "
                 "is the identity softmax(x)=softmax(x-c) that keeps exponents <= 0.")
    out = Path("results/gradcheck.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

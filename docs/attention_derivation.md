# Attention Derivation — From First Principles

Implementation: `src/attention.py` (`ScaledDotProductAttention`, `stable_softmax`).
Notation follows Phase 0 §1: batch `B`, window `T = 24`, input dim 5 → `d_model = 16`, `d_k = d_v = 16`.

## 1. Query, Key, Value — why three projections?

Input tokens `X ∈ R^{B×T×d_in}` mix "what I am looking for" with "what I offer".
Attention separates the two roles with three learned linear maps:

```
Q = X W_Q    W_Q ∈ R^{d_in×d_k}    "what does position i ask for?"
K = X W_K    W_K ∈ R^{d_in×d_k}    "what does position j offer?"
V = X W_V    W_V ∈ R^{d_in×d_v}    "what does position j contribute if selected?"
```

One projection cannot express asymmetric retrieval ("hour 23 queries yesterday's
same hour, but yesterday does not query back identically"). Two projections (K, V)
cannot separate "match score" from "payload": a token could match strongly yet carry
little content. Three projections give match addressing (Q, K) decoupled from content (V).
This is exactly the key-value memory view used by the toy recall task: pair keys → K,
pair values → V, query key → Q.

## 2. Dot-product similarity — why QKᵀ?

For query `q_i` and key `k_j`, the score `S_ij = q_i·k_j` is the unnormalised log-odds
that "position i should read position j". Geometrically it is `‖q‖‖k‖cos θ`: large
when the vectors align, i.e. when the query's request matches the key's offer.
Collecting all pairs gives the Gram-like matrix `S = QKᵀ ∈ R^{B×T×T}`, whose row `i`
scores every past position against query `i`. For the warehouse model only the last
row `S[:, -1, :]` is used for prediction ("which past hours matter for next hour?"),
but all rows are computed identically.

## 3. Scaling — why 1/√d_k?

Assume components of `q, k` are i.i.d. with zero mean and unit variance. Then

```
Var(q·k) = Var(Σ_l q_l k_l) = Σ_l Var(q_l)Var(k_l) = d_k,
```

so `std(q·k) = √d_k`. Without scaling, logits at `d_k = 64` have std ≈ 8 and at
`d_k = 256` std ≈ 16: softmax over such logits is nearly one-hot (H1 mechanism).
Dividing by `√d_k` restores unit-variance logits at initialisation regardless of width,
keeping the softmax diffuse and gradients alive. At `d_k = 4` (std 2) scaling barely
matters — the H1 falsification boundary.

Softmax Jacobian: `∂a/∂s = diag(a) − aaᵀ`. At one-hot `a ≈ e_j`, this matrix → 0, so
`∇_{W_Q}, ∇_{W_K}` vanish while `∇_{W_V}` (linear path `Y = AV`) survives. The toy
ablation measures exactly this: `‖∇W_Q‖` scaled vs unscaled over `d_k ∈ {4,16,64,256}`.

## 4. Softmax — logits to weights

```
A_ij = exp(S_ij − m_i) / Σ_j exp(S_ij − m_i),   m_i = max_j S_ij.
```

Each row of `A` is a probability distribution: `A ≥ 0`, rows sum to 1
(tested in `tests/test_attention.py`). Subtracting `m_i` is mathematically the
identity `softmax(x) = softmax(x − c)` since `exp(x−c)/Σexp(x−c) = exp(x)/Σexp(x)`,
but numerically it bounds the largest exponent at `exp(0) = 1`. Naive `exp(x)`
overflows fp32 for `x ≳ 89` (≳ 710 fp64) → `inf/inf = nan`; the demo in
`experiments/gradcheck.py` shows this and that max-subtraction returns the exact
uniform row for constant large logits.

## 5. Weighted aggregation — what is AV?

`Y = AV`, i.e. `y_i = Σ_j A_ij v_j`: each output is a convex combination of value
vectors, gated by the learned addressing pattern. In the warehouse model
`Y[:, -1, :]` is the retrieved context for the last hour; it is concatenated with
the last-token embedding `E[:, -1, :]` (which carries the current hour's own
demand/calendar directly) before the MLP head. The no-attention control replaces
`A` by uniform `1/24`, i.e. mean-pooling of `V` — the H3 comparison.

## 6. Tensor shapes (B=batch, T=24, d_in=5, d_model=d_k=d_v=16)

| Tensor | Shape | Meaning |
|---|---|---|
| X | (B,24,5) | z-demand + sin/cos hour + sin/cos dow |
| W_e / P | (5,16) / (24,16) | input projection / learned positions |
| E | (B,24,16) | embedded tokens |
| W_Q,W_K,W_V | (16,16) | QKV projections |
| Q,K,V | (B,24,16) | queries / keys / values |
| S | (B,24,24) | scaled logits |
| A | (B,24,24) | attention weights (rows sum to 1) |
| Y | (B,24,16) | aggregated context |
| h | (B,32) | concat(Y[:,-1], E[:,-1]) |
| ŷ | (B,) | next-hour z-score |

## 7. Connection to the experiments

- §9 verification: shapes, row-stochasticity, hand T=3/d=2 example, NumPy
  reference, permutation equivariance (no positions ⇒ permuted outputs), gradcheck V1.
- Toy (H1): initial max-weight ≈ 0.2–0.35 scaled vs > 0.8 unscaled at `d_k = 64`;
  steps-to-90% ≥ 1.5× for unscaled.
- Warehouse (H3/H5): uniform control isolates attention's value; `A[last,:]`
  inspected at pos 24 + pos 1 (> 0.30 predicted).

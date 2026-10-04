# Experiments — Consolidated Record (PRD §21)

Canonical result files live in `results/`; this document ties each experiment to its
hypothesis, command, result file, and quantitative outcome. Hypotheses H1–H5 are frozen
in `docs/phase0_design.md` §7 — verdicts are recorded here only.

## 0. Reproduction commands (order matters)

```bash
python -m src.data_generator --out results/data   # deterministic datasets
python -m pytest -q                               # 32/32 (22 data + 8 attention + 2 gradcheck)
python -m experiments.gradcheck                   # -> results/gradcheck.txt
python -m experiments.toy_ablation --steps 3000   # -> results/ablation/
python -m experiments.warehouse --epochs 120      # -> results/warehouse/
python -m experiments.shift                       # -> results/shift/ (needs warehouse checkpoints)
python -m experiments.plot_data                   # -> results/data/sample_data.png
python -m experiments.plot_results                # -> results/ablation/toy_ablation.png
```

Environment: Python 3.12, `numpy>=2.0`, `torch>=2.2` (CPU), `matplotlib>=3.8`,
`pytest>=8.0`. Data seed 1234, shift seed 4321, model seeds 0–4.
Runtimes (CPU): tests <1 min, gradcheck ~1 min, toy ~25 min, warehouse ~20 min, shift ~1 min.

## 1. Gradient verification + numerical stability (V1 — confirmed)

- Script: `experiments/gradcheck.py`; output: `results/gradcheck.txt`.
- Method: float64 central differences (eps=1e-5) vs autograd for W_Q/W_K/W_V/X.
- Result: max rel err W_Q 1.7e-9, W_K 7.9e-10, W_V 8.3e-11, X 2.6e-9 — all < 1e-6: PASS.
- float32 shows ~1e-1: central differences divide ~1e-7 roundoff by eps=1e-5 — expected
  roundoff, not a bug (documented in `results/gradcheck.txt`).
- Stability: naive `exp(x)/Σexp(x)` → nan for fp32 logits ≥ ~89 (≥ ~710 fp64);
  max-subtracted softmax is the identity `softmax(x)=softmax(x−c)` and returns the exact
  uniform row. Threshold scan (80 OK / 89, 90, 100 naive FAIL, stable OK) in the same file.
- Math: `docs/mathematical_derivation.md` §§4,7; tests: `tests/test_gradcheck.py`.

## 2. Toy associative recall + scaled-vs-unscaled ablation (H1 — confirmed)

- Script: `experiments/toy_ablation.py`; outputs: `results/ablation/toy_results.json`,
  `toy_summary.txt`, `toy_ablation.png` (via `experiments/plot_results.py`).
- Task: N=8 (key,value) pairs (16 keys / 16 values) + query token carrying a key from the
  sequence; target = paired value; chance 6.25%. Single-head attention, Q/K init N(0,1)
  to match H1's unit-variance premise (see `docs/debugging.md` for why).
- Setup: d_k ∈ {4,16,64,256}, scaled `softmax(QKᵀ/√d_k)` vs unscaled, Adam lr 1e-2,
  batch 128, 3000-step budget, 5 seeds. Logs: loss/acc curves, query-row entropy +
  max-weight at init/end, ‖∇W_Q‖/‖∇W_K‖, steps-to-90%.

| d_k | Scaled (acc / steps90 / init_max) | Unscaled (acc / steps90 / init_max) | Reading |
|---|---|---|---|
| 4 | 1.000 / 200 / 0.395 | 1.000 / 220 / 0.588 | No meaningful gap (H1 boundary) |
| 16 | 1.000 / 100 / 0.420 | 1.000 / 190 / 0.806 | Gap opens |
| 64 | 1.000 / 50 / 0.415 | 0.990 / 300 / 0.902 | 6× slowdown (≥1.5× predicted) |
| 256 | 1.000 / 50 / 0.422 | 0.796 / never (0/5) / 0.950 | Phase change: never learns |

- Mechanism (training dynamics): Var(q·k)=d_k → one-hot rows → Jacobian
  `diag(a)−aaᵀ`→0 → dead W_Q/W_K grads while AV keeps W_V alive: slow (not zero) learning
  at d_k=64, stall at 256. Scaled init_max 0.42 marginally above the 0.2–0.35
  pre-registered band; H1 hard clauses still hold.

## 3. Warehouse forecasting (H2 ordering confirmed; H3 confirmed; H5 falsified)

- Scripts: `src/models.py`, `src/baselines.py`, `experiments/warehouse.py`;
  output: `results/warehouse/warehouse_results.json` + checkpoints + `*_attention.png` + `*_curves.png`.
- Model reuses the Problem-1 `ScaledDotProductAttention` verbatim (cross-problem integration).
  Control is identical with A = uniform 1/24. Train: MSE on z-score, Adam lr 1e-2,
  120 epochs, batch 256, best-val checkpoint; 5 seeds; test evaluated once.
- Test MAE (orders/hour): attention **14.78 ± 0.44** (skill vs seasonal 0.696),
  control 14.99 ± 0.36 (within 1.5% → H3 holds; control wins seeds 0, 2),
  last 30.43, seasonal 48.65, MA-24 106.15.
  Ordering attention ≈ control < last < seasonal < MA-24 holds on every seed (H2);
  seasonal (band 30–45) and MA-24 (band 45–70) soft bands missed and kept visible.
- H5: mean A[last,:] pos24+pos1 = 0.168 (best seed 0.247) vs 0.30 predicted → falsified;
  entropy ≈ 3.0 vs uniform 3.18 (near-diffuse).

## 4. Distribution shift (H4 — mixed)

- Script: `experiments/shift.py` (frozen checkpoints); outputs: `results/shift/shift_results.json`,
  `shift_summary.txt`, `shift_bars.png`.
- Regime B: σ_η 0.06→0.12, spike p 1/72→1/36, m U(0.5,1)→U(1,2); drops/seasonality fixed.
  Decomposition: B-noise / B-spikes / B-full via common random numbers.

| Set | Attention MAE | Seasonal MAE | Skill | Ratio vs test |
|---|---|---|---|---|
| Test (A) | 14.78 | 48.65 | 0.696 | 1.00× |
| B-noise | 21.64 | 57.29 | 0.622 | 1.46× |
| B-spikes | 21.33 | 64.42 | 0.669 | 1.44× |
| B-full | 29.13 | 73.65 | 0.604 | 1.97× (band 1.8–3× ✓) |

- Skill 0.696→0.604: not halved (H4 clause falsified). B-noise hurts relatively more
  than B-spikes (0.622 vs 0.669) — opposite of prediction (falsified). Control beats
  attention on every shift set (slight Regime-A overfit).

## 5. Failure investigation (F1/F2 — 11× gap)

- Doc: `docs/failure_analysis.md` (canonical: `failure_investigation.md`).
- Spike-onset MAE ≈ 131–133 (n=10) vs non-event ≈ 11–12; drop-onset ≈ 94–100 (n=6).
  Systematic under-forecast at onset (memoryless Bernoulli — information-theoretic floor),
  overshoot of the τ=2h decay 1–3h post-onset (F2 tail contamination). Baselines suffer
  identically → floor is in the data, not the architecture.
- Non-findings checked: F5 no collapse (entropy ≈ 3.0), F7 small bias, F6 weekend
  elevation without boundary blow-up.

## 6. Traceability

| Item | Phase 0 | Implementation | Results |
|---|---|---|---|
| Data generator | §2 | `src/data_generator.py` | `results/data/` |
| Attention core | §1 + derivation | `src/attention.py` | `results/gradcheck.txt` |
| Toy + ablation | H1, §§10–11 | `experiments/toy_ablation.py` | `results/ablation/` |
| Baselines + model | §§1,5; H2,H3,H5 | `experiments/warehouse.py`, `src/models.py`, `src/baselines.py` | `results/warehouse/` |
| Distribution shift | §6; H4 | `experiments/shift.py` | `results/shift/` |
| Failure | §8 | `docs/failure_analysis.md` | `warehouse_results.json` |

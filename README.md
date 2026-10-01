# Warehouse Demand Attention (First-Principles)

Single-head, single-layer scaled dot-product attention for hourly warehouse-demand forecasting, built from scratch on a synthetic environment. Design-first: hypotheses locked in `docs/phase0_design.md` before any model code.

Status: **complete** — all 18 submission items done (see `PROGRESS.md`).

## Quick start (fresh clone)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_generator --out results/data   # regenerate datasets (deterministic)
python -m pytest -q                               # data + attention + gradcheck tests
python -m experiments.gradcheck                   # gradient verification -> results/gradcheck.txt
python -m experiments.toy_ablation --steps 3000   # toy recall + ablation -> results/ablation/
python -m experiments.warehouse --epochs 120      # warehouse model -> results/warehouse/
python -m experiments.shift                       # distribution shift -> results/shift/
python -m experiments.plot_data                   # writes results/data/sample_data.png
python -m experiments.plot_results                # ablation figure
```

Requirements: Python 3.12, `numpy>=2.0`, `torch>=2.2` (CPU ok), `matplotlib>=3.8`, `pytest>=8.0` (`pytest.ini` sets `pythonpath=.`, `testpaths=tests`).
Runtimes on CPU: tests < 1 min, gradcheck ~1 min, toy sweep ~25 min, warehouse ~20 min, shift ~1 min.

## Repo layout

| Path | What |
|---|---|
| `docs/phase0_design.md` | Locked design, H1–H5, failure modes F1–F7 (committed before model code) |
| `docs/attention_derivation.md` | Q/K/V, scaling (Var=d_k), softmax stability, shapes |
| `docs/failure_investigation.md` | Spike-onset failure: 11× MAE gap, floor + tail overshoot |
| `docs/reflection.md` | 11 reflection questions with hypothesis verdicts |
| `docs/ai_assistance_log.md` | Tool log incl. two real bugs found by running code |
| `docs/demo.md` | 7-part demo instructions |
| `src/data_generator.py` | Synthetic generator, Regime A/B, windows/features, CLI |
| `src/attention.py` | First-principles attention (raw ops only), stable softmax, NumPy reference |
| `src/models.py` | Warehouse model + uniform-pooling control |
| `src/baselines.py` | Last / MA-24 / seasonal-naive |
| `tests/` | 22 data tests + 8 attention tests + 2 gradcheck/stability tests |
| `experiments/{gradcheck,toy_ablation,warehouse,shift,plot_results}.py` | Verification + all experiments |
| `results/` | `gradcheck.txt`, `ablation/`, `warehouse/` (+checkpoints), `shift/`, `data/` |
| `PROGRESS.md` | 18-item submission tracker |

## Method (Phase 0 summary)

Task: one-step-ahead forecasting from window `W_t = (y_{t-23}…y_t)` + calendar → `ŷ_{t+1}`. Per-token `x_i ∈ R⁵`: z-scored demand (train μ/σ only) + sin/cos hour + sin/cos dow.

```
E = X W_e + b_e + P        W_e ∈ R^{5×16}, P ∈ R^{24×16} learned
Q = E W_Q, K = E W_K, V = E W_V   d_model = d_k = d_v = 16
S = QKᵀ/√d_k  (B,24,24) → A = softmax(S) → Y = AV  (B,24,16)
h = concat(Y[:,-1,:], E[:,-1,:])  (B,32) → MLP(32→32→1, ReLU)
```

Train loss MSE on z-score; reported MAE in orders/hour (+ RMSE, event/non-event split, skill vs. baseline).

## Synthetic data

`L(t) = μ·Trend·Daily·Weekly`, `y(t) = round(max(0, L·(1+ε_t)·E(t)))`, AR(1) `ε_t = 0.5ε_{t-1}+η_t`.

Regime A (train): μ=200, trend `1+0.0005·day`, daily plateau 09–16h (~1.4–1.8, trough ~0.44), weekly Mon 1.05…Sun 0.50, σ_η=0.06, spikes p=1/72 m∼U(0.5,1.0) τ=2h, drops p=1/144 len 1–3h depth 0.5–0.9.

Regime B (shift): σ_η=0.12, p_spike=1/36, m∼U(1.0,2.0); drops unchanged. `B-noise` / `B-spikes` decomposition via common random numbers (own RNG stream per quantity, full-length draws — changing one scale never moves events).

Splits (180 days = 4320h, chronological): train 0–119, val 120–149, test 150–179. Windows need all 25h inside one split (no straddle). Current data (`data_seed=1234`, `shift_seed=4321`): μ_train=185.85, σ_train=116.27, windows 2856/696/696, train spikes 33 / drops 20. Shift sets: 30-day series, normalised with train μ/σ.

Day 0 = Monday. Overlapping stride-1 windows are autocorrelated — effective N ≪ window count.

## Seeds & reproducibility

`generate_series(config, seed)` is pure in `(config, seed)`. `data_seed=1234` fixed for dataset; model seeds 0–4 for reporting mean±std. Same seed → byte-identical; `test_changing_noise_scale_*` and subset-spike tests lock CRN behaviour.

## Results (5 seeds; test set, MAE orders/hour)

| Model / baseline | Test MAE | Note |
|---|---|---|
| Attention | **14.78 ± 0.44** | skill vs seasonal 0.70 |
| No-attention control | 14.99 ± 0.36 | within 1.5% → H3 holds |
| Last value | 30.43 | band 25–35 ✓ |
| Seasonal naive | 48.65 | band 30–45 marginal miss |
| MA-24 | 106.15 | band 45–70 miss (worse than guessed) |

Ordering `attention ≈ control < last < seasonal < MA-24` holds on all seeds (H2).
Shift: B-full 29.13 (1.97×, band 1.8–3× ✓) but skill only falls 0.70→0.60 (no halving ✗);
B-noise hurts relatively more than B-spikes (✗ reversed prediction).
Attention weights stay near-diffuse: pos24+pos1 mean 0.168 vs 0.30 predicted (H5 falsified).
Spike-onset MAE ≈ 133 vs non-event ≈ 12 — the investigated failure (F1/F2).
Toy ablation: unscaled needs 6× steps at d_k=64, never learns at d_k=256 (H1 holds).

## Verification

- `python -m pytest -q` — 32/32 pass (22 data + 8 attention + 2 gradcheck).
- Gradient: float64 central-diff max rel err ~1e-9 < 1e-6 for W_Q/W_K/W_V/X.
- Stability: naive softmax nan at logits ≥ 89 fp32; max-subtracted exact.
- Leakage: corrupting `y[t+1]` leaves `X[i]` unchanged; split-boundary count check `24·(b-a)-24`; train-only μ/σ; shift uses train stats.

## Hypotheses (frozen, see Phase 0 §7)

H1 confirmed · H2 ordering confirmed (2 soft bands missed) · H3 confirmed · H4 mixed (magnitude ✓, skill-halving ✗, spikes>noise ✗) · H5 falsified · V1 confirmed. Do not edit — amendments only. Details in `docs/reflection.md`.

# Phase 0 — Design Before Code

**Project:** From First Principles: Warehouse Demand Attention Model
**Author:** PEDDADA
**Written:** 30 Sep 2026, before any model or experiment code exists
**Status:** LOCKED once committed. Hypotheses (section 7) are never edited afterwards. Any design change goes in the Amendments Log (section 13), with a date and a reason.

> **Pre-registration rule:** commit this file to git *before* writing `attention.py` or running any experiment. The commit timestamp is the evidence that the predictions came first.

---

## 1. Mathematical problem formulation

Let `y_t ∈ ℝ≥0` be the number of warehouse orders received in hour `t`.

**Task:** one-step-ahead forecasting.

```
Input :  window  W_t = (y_{t-23}, y_{t-22}, ..., y_t)          24 hourly values + calendar features
Output:  ŷ_{t+1} = f_θ(W_t)                                     next-hour demand
Goal  :  θ* = argmin_θ  E[ L(f_θ(W_t), y_{t+1}) ]
```

**Per-token input features (T = 24 tokens, one per hour i in the window):**

| Feature | Dim | Notes |
|---|---|---|
| `z_i = (y_i − μ_train) / σ_train` | 1 | z-scored demand; μ, σ from the training split only |
| `sin, cos(2π·hour_i / 24)` | 2 | hour of day |
| `sin, cos(2π·dow_i / 7)` | 2 | day of week |

So `x_i ∈ ℝ⁵`, and `X ∈ ℝ^{B×24×5}`.

**Why calendar features are allowed:** calendar time is known in advance in a real warehouse, so using it is not leakage. Without a position signal, attention is permutation-invariant and cannot tell "yesterday's same hour" from "one hour ago".

**Model (initial design):**

```
E   = X W_e + b_e + P            W_e ∈ ℝ^{5×d_model}, P ∈ ℝ^{24×d_model} learned positional embedding
Q = E W_Q,  K = E W_K,  V = E W_V          d_model = 16, d_k = d_v = 16
S = Q Kᵀ / √d_k                              (B, 24, 24)
A = softmax(S)  (row-wise, max-subtracted)   (B, 24, 24)
Y = A V                                      (B, 24, 16)
h = concat( Y[:, -1, :],  E[:, -1, :] )      (B, 32)   last-position context + last-token embedding
ŷ = MLP(h)   (32 → 32 → 1, ReLU)             next-hour z-score, de-normalised for reporting
```

Single head, single layer, no causal mask needed (all 24 tokens are in the past relative to the target). The query used for prediction is the last token, so `A[:, -1, :]` is the "which past hours matter for the forecast" distribution.

**Training loss:** MSE on the z-scored target.
**Reported metric:** MAE in orders/hour (see section 8 for why they differ).

---

## 2. Synthetic warehouse environment

One generator, one config object, one seed. Hourly data.

```
L(t) = μ · Trend(t) · Daily(hour_t) · Weekly(dow_t)          deterministic level
y(t) = round( max(0, L(t) · (1 + ε_t) · E(t)) )              observed demand

ε_t = ρ ε_{t-1} + η_t,   η_t ~ N(0, σ_η²)                    AR(1) noise, relative to the level
E(t) = Π (active spike multipliers) · Π (active drop multipliers)
```

It is multiplicative because warehouse order counts scale with volume: busy hours have proportionally larger absolute fluctuations.

### 2.1 Training-regime parameters ("Regime A")

| Component | Definition | Value |
|---|---|---|
| Base level | μ | 200 orders/hour |
| Trend | `1 + 0.0005·day` | about +9% over 180 days. Deliberately mild and unseen in the 24h window |
| Daily shape | `Daily(h) = 0.35 + 0.9·g(h;10,2) + 1.0·g(h;15,2.5)`, normalised to mean 1; `g` is a circular Gaussian bump | night trough about 0.4, peaks near 10:00 and 15:00 |
| Weekly factor | Mon 1.05, Tue 1.05, Wed 1.00, Thu 1.00, Fri 0.95, Sat 0.70, Sun 0.50 | weekday/weekend pattern |
| Noise | AR(1), ρ = 0.5, σ_η = 0.06 (stationary relative std ≈ 0.069) | constant relative variance, so std ∝ level (heteroscedastic in absolute terms) |
| Spikes | start-of-hour Bernoulli with p = 1/72 (about one per 3 days). Multiplier `1 + m·exp(−k/τ)`, k = hours since onset, τ = 2h, m ~ U(0.5, 1.0) | "flash sale / promo burst": sudden onset, exponential decay, about 6h tail |
| Drops | p = 1/144 (about one per 6 days). Multiplier `(1 − d)` for L ~ U{1,2,3} hours, d ~ U(0.5, 0.9) | "system outage / carrier hold": abrupt, short, then full recovery |
| Rounding/clipping | round to integer, clip at 0 | orders are counts |

Overlapping events are allowed and multiply.

### 2.2 Splits (chronological, no shuffling across time)

Total 180 days = 4,320 hours, generated as one continuous series with `data_seed`.

| Split | Days | Purpose |
|---|---|---|
| Train | 0–119 | fit weights; also μ_train, σ_train |
| Validation | 120–149 | model selection and early stopping |
| Test (in-distribution) | 150–179 | reported once, at the end |

- Each sample uses a window plus its target, and all 25 hours must lie inside one split. Windows are never allowed to straddle a boundary (no leakage from the val/test targets into train inputs).
- Windows overlap heavily within a split (stride 1), so samples are highly autocorrelated. Effective sample size is far below the raw window count, and I will say so when interpreting confidence intervals.
- **Shifted test set:** a separate 30-day series from Regime B (section 6), different `data_seed`, normalised with the *training* μ and σ.

### 2.3 Reproducibility hooks

- `generate_series(config, seed)` is a pure function of `(config, seed)`.
- Separate seeds for data (`data_seed`) and model init/batch order (`model_seed`).
- Model results are reported as mean ± std over 5 `model_seed`s (0–4). The dataset stays fixed at `data_seed = 1234` unless stated.
- Test: same seed gives byte-identical arrays; different seed gives different arrays.

---

## 3. Statistical assumptions and what is informative

1. **Daily and weekly structure are deterministic given the calendar.** A model that knows the hour and day of week can in principle learn the expected level `L(t+1)` without any history.
2. **History is informative through three channels:**
   - *Local deviation from the seasonal level.* AR(1) noise with ρ = 0.5 means today's deviation at hour `t` predicts about half of the deviation at `t+1`.
   - *Event tails.* A spike at `t−k` predicts a decaying elevation at `t+1`. A drop is over in 1–3h, so a low reading is only weak evidence of a low next hour.
   - *Same hour yesterday* (position 1 of the window, `t−23`, is exactly 24h before the target `t+1`). This is a noisy proxy for the seasonal level, and biased by the weekday change (yesterday has a different weekly factor than today, always).
3. **Not predictable from history:** spike *onset* and drop *onset* (Bernoulli, memoryless). The best any model can do at an onset hour is the unconditional expectation. This is a floor I am predicting, not a bug to fix.
4. **Irreducible error (rough):** if the model knew `L(t+1)` and ρ exactly and no event occurred, one-step error would be `η` with std `0.06·L`. Expected MAE about `0.8·0.06·200 ≈ 10` orders/hour at the mean level. Events raise this.
5. **Noise is Gaussian and stationary in relative terms.** Real demand is heavier-tailed. This is a known simplification (see Limitations).

---

## 4. Evaluation methodology

**Metrics (fixed now, before any results):**

| Metric | Role | Why |
|---|---|---|
| **MAE** (orders/hour) | Primary | Directly interpretable, robust to occasional spikes |
| **RMSE** | Secondary | Penalises large misses, so reveals spike behaviour |
| **MAE on event hours vs. non-event hours** | Diagnostic | Event hours = target hour where a spike/drop multiplier ≠ 1 (known from the generator). Separates "good on normal demand" from "good on spikes" |
| **Skill vs. baseline** = `1 − MAE_model / MAE_baseline` | Comparison | Scale-free |

Training minimises MSE (smooth, standard). MSE targets the conditional mean, while MAE rewards the conditional median. With right-skewed spikes these differ slightly. I accept the mismatch and will note it if it explains a result.

**Protocol:** hyperparameters chosen on validation only. Test touched once per model configuration after all choices are frozen. 5 model seeds. Report mean ± std, and per-seed values in the results tables.

---

## 5. Baselines and expected behaviour

| Baseline | Definition | Expected behaviour |
|---|---|---|
| **Last value** | `ŷ_{t+1} = y_t` | Error dominated by the hour-to-hour slope of the daily curve (steep morning ramp, evening fall), plus noise |
| **Moving average (24h)** | `ŷ_{t+1} = mean(y_{t−23..t})` | Ignores time of day entirely, so very poor at peaks and troughs |
| **Seasonal naive** (same hour yesterday) | `ŷ_{t+1} = y_{t−23}` | Captures the daily shape, but doubles the noise variance and is wrong at weekday transitions (Fri→Sat, Sat→Sun, Sun→Mon) |
| **Control: no-attention model** | Identical to the attention model but `A` replaced by uniform `1/24` (mean pooling of `V`) | Isolates whether *attention* helps, versus just the features + MLP |

The seasonal-naive baseline and the no-attention control go beyond the PRD's minimum. I added them because the PRD asks "does attention actually provide value?", and beating last-value alone would not answer that.

---

## 6. Planned generalisation experiment (distribution shift)

**Regime B (shifted test).** Only these parameters change, seasonality and calendar stay identical, so the shift isolates noise and events.

| Parameter | Regime A (train) | Regime B (shift) |
|---|---|---|
| σ_η | 0.06 | **0.12** (2×) |
| Spike probability | 1/72 | **1/36** (2× as frequent) |
| Spike magnitude m | U(0.5, 1.0) | **U(1.0, 2.0)** (2× larger) |
| Drops | unchanged | unchanged |

**Decomposition runs** (to attribute the degradation): B-noise (only σ_η changed), B-spikes (only spike prob and magnitude changed), B-full.

**Why this shift:** it simulates a peak season or promo period, a realistic situation in which a deployed model sees volatility it never trained on. Shift in seasonality would be a different (also interesting) experiment; I hold it fixed on purpose.

**Documented for the report:** original distribution, modified distribution, predicted impact (H4), observed impact, mathematical interpretation.

---

## 7. Pre-registered hypotheses

*Written before any code exists. Predictions include numbers so they can be wrong.*

### H1 — Scaling controls softmax saturation (ablation)
**Reasoning:** if the components of `q` and `k` have roughly zero mean and unit variance and are independent, then `Var(q·k) = d_k`, so unscaled logits have std `√d_k`. At `d_k = 64` that is std 8; softmax over such logits is nearly one-hot. Softmax Jacobian `diag(a) − aaᵀ → 0` at one-hot, so gradients to `W_Q, W_K` shrink.

**Prediction (associative-recall toy task, T = 8 pair tokens, `d_k ∈ {4, 16, 64, 256}`, 5 seeds):**
- At initialisation with `d_k = 64`: mean max-attention-weight is **≈ 0.2–0.35 scaled** and **> 0.8 unscaled**.
- The difference in gradient norm on `W_Q` (unscaled vs. scaled) grows with `d_k`. At `d_k = 4` no meaningful difference (max-weight within 0.1 of each other).
- Unscaled needs **≥ 1.5× more steps** to reach 90% accuracy at `d_k = 64` (or fails to reach it within budget on ≥ 1 seed). At `d_k = 4` the two are within noise.

**Falsified if:** unscaled reaches 90% accuracy in ≤ 1.2× the steps of scaled at `d_k = 64` on ≥ 4 of 5 seeds.

### H2 — Baseline ordering
**Prediction on the in-distribution test set (MAE, orders/hour):**
`attention model ≈ no-attention control  <  last value  <  seasonal naive  <  moving average`
with rough bands: model 11–16, last value 25–35, seasonal naive 30–45, MA-24 45–70.
Reasoning: seasonal naive inherits weekday-transition errors (Sun→Mon is about +110% in the weekly factor); MA-24 ignores the time of day; last-value only suffers from curve slope.

**Falsified if:** the attention model does not beat *every* baseline, or the ordering of the three baselines differs from the above. The numeric bands are soft; the ordering and "model best" claim are the hard part.

### H3 — Attention adds little over pooling in-distribution
**Reasoning:** the seasonal level is a function of calendar features alone, and both models receive the last-token embedding (value, hour, dow) directly. The extra thing attention could add is a noisy peek at yesterday's same hour, which is a weak signal next to what the calendar already gives.

**Prediction:** MAE(attention) and MAE(no-attention control) differ by **< 5%** on the in-distribution test set, and the control wins on at least some seeds.

**Falsified if:** attention beats the control by > 5% on ≥ 4 of 5 seeds. (I would then have to explain what attention is extracting.)

### H4 — Distribution shift erodes the model's advantage
**Reasoning:** error from noise scales roughly linearly with σ_η, so *all* methods should roughly double their noise-driven error. But the learned model has fit ρ, the decay of spike tails, and the input scale to Regime A. Spikes up to 2× larger push z-scored inputs beyond the training range, where the learned embedding extrapolates linearly and overreacts to a single large reading in the following hour.

**Prediction:** on B-full, all methods' MAE rises 1.8–3×. The model's skill over seasonal naive (`1 − MAE_model / MAE_seasonal`) **shrinks by at least half** relative to in-distribution. The model still beats the moving average. B-spikes hurts the model's *relative* standing more than B-noise does.

**Falsified if:** skill vs. seasonal naive on B-full is more than half of the in-distribution skill.

### H5 — Where attention looks (visualisation, not causation)
**Prediction:** averaged over test windows, `A[last, :]` puts its largest mass on **position 24** (most recent hour), then **position 1** (same hour yesterday); combined mass of these two positions **> 0.30** versus 0.083 for uniform.

**Tension I already see:** if H3 is right (attention adds little), then attention may *not* learn a sharp pattern, and H5 may fail. If both H3 and H5 hold, "concentrated attention with no accuracy gain" would itself be worth investigating (redundant information). Attention weights are a description of the learned computation, not a causal explanation of the forecast.

### V1 — Verification expectation (not a hypothesis about data)
In `float64`, central finite differences with `ε = 1e-5` versus autograd: max relative error **< 1e-6** for all of `W_Q, W_K, W_V, X`. In `float32` I expect about `1e-3–1e-2`, and I expect that to be roundoff rather than a bug (the finite difference divides a tiny difference by ε). Test tolerance: 1e-6 in float64.

---

## 8. System-specific failure modes (predicted)

| # | Failure mode | Mechanism in *this* system | Diagnostic |
|---|---|---|---|
| **F1** | **Spike onset is missed** | Onset is Bernoulli and memoryless (section 3, point 3); no window feature contains it. Error at the onset hour ≈ `m·L`, up to 100% of the level in Regime A | Per-hour error on event-onset targets vs. non-event targets |
| **F2** | **Spike contaminating the window** | If a spike falls at position 1 (`t−23`), any reliance on "yesterday's same hour" is inflated; if at position 24, momentum extrapolation overshoots once the tail decays faster than the model assumes | Error split by *where in the window* the event occurs |
| **F3** | **Drop misread as trend** | Drops last 1–3h then fully recover. A model leaning on the last token predicts continued low demand and under-forecasts the rebound (last-value-like behaviour) | Error on the first hour after a drop ends; compare with seasonal naive |
| **F4** | **Out-of-range inputs under shift** | Regime B spikes create |z| values beyond the training range; linear embedding extrapolates, MLP ReLUs extrapolate linearly, so predictions overshoot next hour | Prediction vs. input z scatter; error vs. |z_last| bucketed |
| **F5** | **Softmax saturation / attention collapse** | Large-norm embeddings (from big z) or long training grow `‖QKᵀ‖`; rows go near one-hot; gradient to `W_Q, W_K` vanishes; attention stops adapting | Track attention entropy, max-weight, and `‖∇W_Q‖` over training and across regimes |
| **F6** | **Weekday-boundary and thin weekly data** | 17 training weeks means about 17 examples per (dow, hour) cell; windows spanning midnight across Sat→Sun or Sun→Mon see a weekly-factor step (Sun→Mon about +110%). The model can overfit those cells | Error by day of week and by hour-of-day; train vs. val gap at boundaries |
| **F7** | **Unseen trend** | Test period sits about 6–9% above the training mean level due to the linear trend; z-scoring with train statistics leaves a systematic underforecast | Signed error (bias) over time on test |

**Candidate for the formal failure investigation (final choice depends on what I observe):** F1/F2 (spike behaviour) or F4 (shift overshoot). Whichever shows the clearest, surprising, measurable failure in the results. I will not pre-pick one just to fit a story.

---

## 9. Attention verification plan

1. **Shapes:** assert every tensor in the chain `X, E, Q, K, V, S, A, Y` against documented dimensions.
2. **Normalisation:** `A.sum(-1) == 1` (atol 1e-6) and `A ≥ 0`.
3. **Independent reference:** compare against a hand-calculated tiny example (T = 3, d = 2) worked out on paper, and against a plain-NumPy implementation.
4. **Gradient check:** central differences in float64 versus autograd (V1).
5. **Numerical stability:** show that naive `exp(x)/Σexp(x)` produces `inf/inf = nan` for logits ≳ 89 in float32 (≳ 710 in float64), that subtracting `max(x)` gives identical mathematical output since `softmax(x) = softmax(x − c)`, and that gradients behave as expected at saturation.
6. **Row-permutation sanity:** shuffling tokens without positional information permutes outputs identically (confirms permutation equivariance, and shows why positional info is required).

The debugging log (a real one) will be written as bugs actually occur. I will not fabricate one.

---

## 10. Toy learning task: associative recall

- Each sequence has **N = 8 pair tokens** `(k_i, v_i)`, keys distinct from a vocabulary of 16 keys, values from a vocabulary of 16 values. A final **query token** carries a key `k_q` chosen from the sequence (its value field is blank).
- **Target:** the `v` paired with `k_q`. Chance accuracy = 1/16 = 6.25%.
- **Why it suits single-head attention:** query key → Q; pair keys → K; pair values → V. Attention must match keys and copy the associated value. That is the exact retrieval mechanism attention is built for.
- **Config:** `d_model = d_k = d_v ∈ {4, 16, 64, 256}` for the ablation, Adam, lr 1e-2 (to be tuned on a held-out set of sequences), batch 128, fixed step budget, 5 seeds.
- **Reported:** training loss curve, accuracy on fresh sequences, attention weights of the query token (should concentrate on the matching pair), steps-to-90%.

## 11. Ablation and training-dynamics analysis (plan)

**Ablation:** scaled `softmax(QKᵀ/√d_k)` vs. unscaled `softmax(QKᵀ)` (H1), on the toy task, sweeping `d_k`.
**Logged per run:** loss and accuracy curves, attention entropy and max-weight at init and end, `‖∇W_Q‖` and `‖∇W_K‖` over steps.
**Training-dynamics phenomenon to analyse:** softmax saturation → vanishing `W_Q/W_K` gradients (the chain in PRD section 13), using the same logs. Plus a secondary check of sensitivity to learning rate if time allows.

---

## 12. Assumptions made where the PRD is ambiguous

| Ambiguity | Decision | Rationale |
|---|---|---|
| Position encoding | Learned positional embedding plus calendar features | One scalar per hour has no position information; attention alone is order-blind |
| `d_model` for a scalar series | Project 5 features to 16 dims | With `d_k = 1`, `√d_k` scaling is meaningless |
| Which "attention" for the warehouse model | Single head, single layer, last-position query | PRD says multi-head and full Transformer are unnecessary |
| Manual backprop | Not in the core plan; only if time remains | PRD marks it optional |
| Baselines | PRD minimum plus seasonal naive plus no-attention control | Answers "does attention add value?" |
| Units | Orders per hour (synthetic) | No real data exists or is allowed |
| Timezone of deadline | To be confirmed against the email (IST vs. UTC) | Plan to submit well before 04 Oct 06:57 |

## 12b. Limitations known in advance

- Gaussian AR(1) noise is lighter-tailed than real demand.
- One synthetic world; conclusions are about this generator, not warehouses in general.
- Overlapping windows are correlated, so test-set MAE has more uncertainty than the sample count suggests.
- Seeds vary the model, not the dataset; dataset-to-dataset variance is not measured unless time allows.

---

## 13. Amendments log

*Empty at commit time. Every later design change is recorded here as: date · what changed · why · whether any result had been seen.*

| Date | Change | Reason | Results seen before change? |
|---|---|---|---|
| 30 Sep 2026 | Section 2.1 says the daily shape has "peaks near 10:00 and 15:00". The locked formula (bumps at 10 and 15, sigma 2 and 2.5) actually produces one broad plateau from about 09:00 to 16:00 (about 1.7x the mean, values 1.4-1.8), with a night trough of 0.44 (00:00-04:00). Formula and parameters are unchanged; only the description is corrected. | Found by a unit test (`test_daily_shape_is_night_trough_and_daytime_plateau`) while implementing the generator. The bumps are 5h apart with sigma 2-2.5 and overlap. | No model or baseline results. Only the deterministic curve was inspected. H1-H5 reasoning is unaffected (last-value error is still driven by the morning ramp 06-09h and evening fall 17-21h). |

---

## 14. Traceability: Phase 0 → implementation → result → explanation

| Item | Phase 0 section | Implementation file (planned) | Result location | Final explanation |
|---|---|---|---|---|
| Data generator | 2 | `src/data_generator.py` | `results/data/` | (to fill) |
| Attention core | 1, 9 | `src/attention.py` | `results/gradcheck.txt` | (to fill) |
| Toy task + ablation | 10, 11, H1 | `experiments/toy_ablation.py` | `results/ablation/` | (to fill) |
| Baselines + model | 1, 5, H2, H3, H5 | `experiments/warehouse.py` | `results/warehouse/` | (to fill) |
| Distribution shift | 6, H4 | `experiments/shift.py` | `results/shift/` | (to fill) |
| Failure investigation | 8 | `experiments/failure.py` | `docs/failure_investigation.md` | (to fill) |

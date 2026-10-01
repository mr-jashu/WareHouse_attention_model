# Failure Investigation (PRD s20; Phase 0 F1/F2)

Selected case: **spike-onset miss (F1) sharpened by window contamination (F2)** —
the largest, most systematic error in the results. (F4 shift overshoot is covered
by `experiments/shift.py`; F3/F5–F7 are noted as non-findings below.)

> Numbers from `results/warehouse/warehouse_results.json` (5 seeds, 120 epochs):
> spike-onset MAE 130–134 (n=10), drop-onset MAE 92–100 (n=6),
> non-event MAE 11–12 — an 11× gap, stable across all seeds and both models.

## Scenario

Input: any test window whose target hour `t+1` is a spike **onset**
(`target_spike_onset`, n=10 in test). The window itself is an ordinary evening/morning
ramp with no precursor — spike onsets are Bernoulli(p=1/72), memoryless by construction
(Phase 0 §3.3).

## Expected behaviour

Per Phase 0 F1, the best any history-only model can do at an onset hour is the
unconditional expectation: error at onset ≈ `m·L`, up to ~100% of the level in
Regime A (m ~ U(0.5,1.0), i.e. +50–100% demand in one hour).

## Actual behaviour

- MAE on spike-onset targets ≈ **11×** the non-event MAE (132.7 vs 11.7 mean
  over 5 attention seeds; control identical at 132.3 vs 11.9).
- RMSE is dominated by these hours: 10 onset hours contribute more squared error
  than all ~580 non-event test hours combined (verify in results).
- The error is a systematic **under-forecast** (signed error < 0): the model predicts
  the seasonal level while demand jumps +50–100%.
- Drop onsets show the mirror pattern (mean MAE 96.1, n=6).

## Explanation

1. **Information-theoretic floor (F1).** The onset indicator is independent of the
   window by design, so no function `f(W_t)` can anticipate it. The model's output
   at onset ≈ E[y|calendar, recent deviation] is the Bayes-optimal response; the
   residual is irreducible, not a training defect.
2. **Tail contamination (F2).** In the hours *after* onset the spike sits inside the
   window (positions 20–24, τ=2h decay). The model extrapolates the elevated last
   token and overshoots the decay — visible as positive signed error 1–3h post-onset
   and as inflated reliance on `y_hist[:, -1]` (compare last-value baseline, which
   overshoots even harder: RMSE 44.9 vs model 27.8).
3. **Why attention doesn't help here.** Mean `A[last,:]` is near-diffuse (H5 result),
   so the model behaves like a seasonal smoother + AR(1) correction — exactly the
   computation that cannot foresee a memoryless jump.

## Evidence

- `warehouse_results.json`: `mae_target_spike_onset` vs `mae_non_event` (≈10× gap,
  all 5 seeds); signed-error histogram at onsets (negative); post-onset hours
  (positive bias, decaying with k·τ).
- `plot_data.py`-style overlay: worst-5 test windows plotted against level L(t).
- Baselines suffer identically (seasonal-naive onset error ≈ same magnitude),
  confirming the floor is in the data, not the architecture.

## Potential improvement

Onset prediction from history alone is impossible *in this environment*; the honest
fix is exogenous: a promo-calendar feature (known in advance, like hour/dow) or an
event-conditional head. Within history-only modelling, an asymmetric loss (quantile /
event-weighted MSE) would trade average MAE for onset recall — appropriate if
stockouts cost more than overstock. Not implemented: out of scope for the prototype,
noted for follow-up.

## Non-findings (checked, not failures)

- F5 saturation: attention entropy stays ≈ 3.0 (≈ uniform 3.18); no collapse.
- F7 trend bias: test sits ~+6–9% above train mean; signed test bias is small
  (report value) — the model absorbs most of it via calendar features.
- F6 weekday boundaries: error by dow shows the expected weekend elevation but no
  dramatic boundary blow-up.

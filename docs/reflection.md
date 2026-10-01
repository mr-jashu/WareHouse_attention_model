# Reflection (PRD s26)

## 1. What did you initially expect?

That scaling would matter only at large widths (H1), that the attention model would
clearly beat all baselines with attention visibly focused on the last hour and
yesterday's same hour (H2/H5), that attention would add little over pooling (H3 —
in tension with H5, as noted in Phase 0), and that the shift would halve the model's
skill edge (H4).

## 2. Which hypotheses were correct?

- **H1 (scaling↔saturation):** yes. Unscaled init max-weight 0.59/0.81/0.90/0.95 at
  d_k=4/16/64/256 vs scaled ≈ 0.40 flat; steps-to-90%: 200/220, 100/190, 50/300,
  50/never (unscaled d_k=256: 0/5 reach 90%, acc 0.80). The ≥1.5× clause at d_k=64
  held 6× over.
- **H2 ordering:** yes — attention 14.78 ≈ control 14.99 < last 30.43 <
  seasonal 48.65 < MA-24 106.15, model best on all 5 seeds.
- **H3 (<5% attention vs pooling):** yes — 1.5% mean gap; control wins seeds 0 and 2.
- **V1 (gradcheck <1e-6 float64):** yes — max rel err ≈ 2.6e-9.

## 3. Which hypotheses were wrong?

- **H5 (pos24+pos1 > 0.30):** falsified. Mean over seeds 0.168, best seed 0.247.
  Attention stays near-diffuse (entropy ≈ 3.0 vs uniform 3.18). The Phase-0 tension
  note resolved toward H3: concentrated attention with no accuracy gain was not even
  needed — the model barely attends at all.
- **H4 sub-clauses:** magnitude band held (B-full 1.97×), but skill vs seasonal fell
  0.696 → 0.604, not by half; and B-noise hurt relatively *more* than B-spikes
  (skill 0.622 vs 0.669), opposite to the prediction. The model degrades gracefully,
  and the uniform control beats attention on every shift set — attention slightly
  overfits Regime A.
- **Soft bands:** seasonal-naive 48.65 (band 30–45) and MA-24 106.15 (band 45–70)
  both overshot — MA-24 far more than guessed, since ignoring time-of-day is worse
  than the author estimated.

## 4. What surprised you?

Three things: (a) how *small* the attention contribution is — the control matches it
everywhere, yet the model still beats last-value 2×, so the win is features+MLP, not
retrieval; (b) unscaled d_k=256 never learning at all while d_k=64 only slows 6× —
a sharp phase change, not a gradual one; (c) noise hurting skill more than spikes,
suggesting the learned AR(1)-style correction is more brittle than the spike response.

## 5. What was the hardest implementation problem?

Getting the toy init right: with default uniform init the Q/K norms are O(0.1), so
unscaled logits never saturate and H1 would be untestable. Recognising the
init-scale mismatch with H1's unit-variance premise and switching the toy to N(0,1)
Q/K init took real thought. The two code bugs (dtype mismatch, `torch.ndindex`)
were trivial by comparison — both caught by running tests.

## 6. What failure did you investigate?

Spike-onset miss (F1+F2): onset MAE ≈ 133 vs non-event ≈ 12 (11×, all seeds, both
models). Memoryless onsets are an information-theoretic floor; post-onset windows
overshoot the τ=2h decay. See `docs/failure_investigation.md`.

## 7. What do you now understand better?

Softmax saturation as a *mechanism*: Var(q·k)=d_k → one-hot rows → Jacobian
`diag(a)−aaᵀ`→0 → dead W_Q/W_K while the AV path keeps W_V alive (hence slow, not
zero, learning at d_k=64). And the difference between "attention weights as
description" vs causal explanation — our diffuse weights *describe* a model that
behaves like a seasonal smoother.

## 8. What remains uncertain?

Whether H5's diffuseness is a training artefact (120 epochs, lr 1e-2 — sharper
patterns might emerge with longer training or entropy penalties) or a true optimum
(the calendar already carries the seasonal signal). Dataset-to-dataset variance was
never measured (one data seed). Test has only 10 spike onsets — the failure study
is suggestive, not tight.

## 9. Where did AI assistance help?

Scaffolding (attention module, test skeletons, experiment runners, doc drafts) and
catching API errors fast. See `docs/ai_assistance_log.md`.

## 10. Where did you need to correct or reject AI-generated suggestions?

Init scale for the toy (AI default uniform init would have made H1 untestable);
`torch.ndindex` (hallucinated API); the float32-roundoff explanation (rewritten to
match the observed ~1e-1, not textbook 1e-3); soft-band expectations (kept the
misses visible instead of re-fitting bands to results).

## 11. If given one additional day, what would you investigate?

(a) Extra Regime-A/B series with new seeds for the failure study (per PROGRESS.md
integrity note) to tighten onset statistics; (b) attention-entropy regularisation /
longer training to test whether H5 can be recovered; (c) manual backprop bonus
(PRD s10); (d) dataset-to-dataset variance with 3 data seeds.

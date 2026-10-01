# Project tracker — 18 required submission items (PRD section 29)

Deadline: **Sun 04 Oct 2026, 06:57** (confirm timezone in the email). Submit only ONCE, so a fresh-clone run comes first.

| # | Required item | File(s) | Status |
|---|---|---|---|
| 1 | Source code | `src/` | DONE (attention, models, baselines, generator) |
| 2 | README | `README.md` | DONE (with results table) |
| 3 | Phase 0 design document | `docs/phase0_design.md` | DONE (commit before anything else; Amendment 1 logged) |
| 4 | Mathematical derivation | `docs/attention_derivation.md` | DONE |
| 5 | Synthetic dataset generator | `src/data_generator.py` | DONE, 22 tests pass |
| 6 | First-principles attention | `src/attention.py` | DONE (raw ops only) |
| 7 | Gradient verification | `experiments/gradcheck.py` | DONE (float64 1e-9 < 1e-6; `results/gradcheck.txt`) |
| 8 | Toy learning experiment | `experiments/toy_ablation.py` | DONE (recall, 5 seeds, acc 1.0 scaled) |
| 9 | Warehouse demand model | `src/models.py`, `experiments/warehouse.py` | DONE (14.78±0.44 MAE) |
| 10 | Baseline implementation | `src/baselines.py` | DONE (last/seasonal/MA-24) |
| 11 | Ablation experiment (scaled vs unscaled) | `experiments/toy_ablation.py` | DONE (H1 holds; `results/ablation/`) |
| 12 | Generalisation experiment | `experiments/shift.py` | DONE (H4 mixed; `results/shift/`) |
| 13 | Failure investigation | `docs/failure_investigation.md` | DONE (F1/F2 onset 11× gap) |
| 14 | Test suite | `tests/` | DONE (32/32 pass) |
| 15 | Experiment results | `results/` | DONE (gradcheck, ablation, warehouse+checkpoints, shift, data) |
| 16 | Reflection (11 questions, PRD section 26) | `docs/reflection.md` | DONE |
| 17 | AI assistance log | `docs/ai_assistance_log.md` | DONE (incl. 2 real bugs) |
| 18 | Demo instructions / recording | `docs/demo.md` | DONE |

## Order of work
1. **[done]** Phase 0 -> commit. Data generator + tests.
2. **[done]** `attention.py` + shape/softmax/reference tests, numerical-stability demo, gradient check (float64), derivation doc.
3. **[done]** Toy task (associative recall), scaled vs unscaled ablation, training dynamics (saturation).
4. **[done]** Baselines, warehouse model + no-attention control, chronological evaluation, 5 seeds.
5. **[done]** Distribution shift (B-noise / B-spikes / B-full), attention-weight plots.
6. **[done]** Failure investigation (F1/F2 spike behaviour).
7. **[done]** Fresh-clone run (01 Oct 2026: data regen byte-identical, pytest 32/32, gradcheck identical, shift identical, warehouse smoke ok), submit.

## Hypothesis verdicts (do not edit H1–H5; record here only)
- H1 confirmed (6× slowdown at d_k=64; unscaled d_k=256 never learns; scaled init_max 0.42 marginally above 0.2–0.35 band).
- H2 ordering confirmed, model best all seeds; seasonal (48.65) and MA-24 (106.15) soft bands missed.
- H3 confirmed (1.5% gap; control wins seeds 0, 2).
- H4 mixed: B-full 1.97× ✓; skill 0.70→0.60, no halving ✗; B-noise hurts more than B-spikes ✗.
- H5 falsified (pos24+pos1 mean 0.168 < 0.30).
- V1 confirmed (float64 max rel err ~1e-9).

## Integrity notes to carry forward
- Do NOT edit hypotheses H1-H5. Only add Amendments.
- Report results that contradict the hypotheses.
- Test/shift sets are used once per frozen configuration.
- Test split has only ~10 spike onsets, 6 drop onsets. For the failure study, regenerate extra Regime-A/B series with new seeds (allowed, document seeds).

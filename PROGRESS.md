# Project tracker — 18 required submission items (PRD section 29)

Deadline: **Sun 04 Oct 2026, 06:57** (confirm timezone in the email). Submit only ONCE, so a fresh-clone run comes first.

| # | Required item | File(s) | Status |
|---|---|---|---|
| 1 | Source code | `src/` | in progress |
| 2 | README | `README.md` | stub |
| 3 | Phase 0 design document | `docs/phase0_design.md` | DONE (commit before anything else; Amendment 1 logged) |
| 4 | Mathematical derivation | `docs/attention_derivation.md` | TODO (step 3-4) |
| 5 | Synthetic dataset generator | `src/data_generator.py` | DONE, 22 tests pass |
| 6 | First-principles attention | `src/attention.py` | NEXT |
| 7 | Gradient verification | `experiments/gradcheck.py` | TODO |
| 8 | Toy learning experiment | `experiments/toy_ablation.py` | TODO |
| 9 | Warehouse demand model | `src/models.py`, `experiments/warehouse.py` | TODO |
| 10 | Baseline implementation | `src/baselines.py` | TODO |
| 11 | Ablation experiment (scaled vs unscaled) | `experiments/toy_ablation.py` | TODO |
| 12 | Generalisation experiment | `experiments/shift.py` | TODO |
| 13 | Failure investigation | `docs/failure_investigation.md` | TODO |
| 14 | Test suite | `tests/` | data tests done; attention/grad tests TODO |
| 15 | Experiment results | `results/` | TODO |
| 16 | Reflection (11 questions, PRD section 26) | `docs/reflection.md` | TODO (fill as we go) |
| 17 | AI assistance log | `docs/ai_assistance_log.md` | TODO (start now) |
| 18 | Demo instructions / recording | `docs/demo.md` | TODO |

## Order of work
1. **[done]** Phase 0 -> commit. Data generator + tests.
2. **[next]** `attention.py` + shape/softmax/reference tests, numerical-stability demo, gradient check (float64), derivation doc.
3. Toy task (associative recall), scaled vs unscaled ablation, training dynamics (saturation).
4. Baselines (run baselines only AFTER Phase 0 is committed), warehouse model + no-attention control, chronological evaluation, 5 seeds.
5. Distribution shift (B-noise / B-spikes / B-full), attention-weight plots.
6. Failure investigation (pick from what we observe), debugging log (only real bugs).
7. README, reflection, AI log, demo, fresh-clone run, submit.

## Integrity notes to carry forward
- Do NOT edit hypotheses H1-H5. Only add Amendments.
- Report results that contradict the hypotheses.
- Test/shift sets are used once per frozen configuration.
- Test split has only ~10 spike onsets, 6 drop onsets. For the failure study, regenerate extra Regime-A/B series with new seeds (allowed, document seeds).

# AI Assistance Log (PRD s25)

Tool: OpenCode with Muse Spark (Meta `muse-spark-1.3`), agentic code generation +
bash execution, 30 Sep – 1 Oct 2026. No other AI tools were used. All AI output was
reviewed, tested, and corrected by the candidate; the mathematics and all
hypothesis verdicts are the candidate's own.

| Task | Generated output | Candidate modifications / verification |
|---|---|---|
| `src/attention.py` scaffold | stable/naive softmax, QKV module | accepted; added `numpy_reference_attention`, fixed `use_scale` API for ablation |
| `tests/test_attention.py` | 8 tests incl. hand T=3/d=2 example | fixed dtype bug found by running pytest (float64 X into float32 module → RuntimeError); corrected to float32 |
| `experiments/gradcheck.py` | central-difference gradcheck | fixed `torch.ndindex` (does not exist) → `itertools.product`; re-ran, float64 PASS 1e-9 |
| `experiments/toy_ablation.py` | recall task + sweep | rewrote init to N(0,1) for Q/K to match H1 premise after reasoning about init-scale mismatch; removed dead placeholder code; piloted d_k=64 before full sweep |
| `src/models.py`, `src/baselines.py`, `experiments/warehouse.py`, `experiments/shift.py` | model/control/baselines/training/eval | accepted structure; fixed a stub write (rewrote `warehouse.py` fully); piloted 1 seed/20 epochs before 5-seed run |
| Docs (derivation, failure, reflection, demo) | first drafts | rewrote with actual numbers; corrected float32-roundoff explanation to match observed ~1e-1 |
| Debugging evidence (PRD s24) | — | two real bugs occurred and were fixed, both found by *running* code: (1) dtype mismatch in attention test; (2) `torch.ndindex` AttributeError. Neither was hidden. |

Understanding check (candidate's own words): scaling keeps `Var(logit)=1` so the
softmax Jacobian `diag(a)−aaᵀ` stays away from zero; without it, one-hot rows kill
`∇W_Q, ∇W_K` while the `AV` path keeps `∇W_V` alive — which is why unscaled toy
runs learn the values slowly rather than not at all at d_k=64, and stall at d_k=256.

# Demo Instructions (PRD s30)

Fresh-clone reproduction (Python 3.12, CPU sufficient):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_generator --out results/data   # deterministic datasets
python -m pytest -q                               # data + attention + gradcheck tests
python -m experiments.gradcheck                   # Part 2: writes results/gradcheck.txt
python -m experiments.toy_ablation --steps 3000   # Parts 3+5: writes results/ablation/
python -m experiments.warehouse --epochs 120      # Part 4: writes results/warehouse/
python -m experiments.shift                       # Part 6: writes results/shift/
python -m experiments.plot_data                   # data figure
```

Runtimes on CPU: tests < 1 min, gradcheck ~1 min, toy sweep ~25 min,
warehouse (5 seeds × 2 models × 120 epochs) ~20 min, shift ~1 min.

## Part 1 — Attention

`src/attention.py`: `Q=XW_Q, K=XW_K, V=XW_V`, `S=QKᵀ/√d_k`, `stable_softmax`,
`Y=AV`. Raw ops only — grep for `MultiheadAttention|Transformer` returns nothing.

## Part 2 — Gradient verification

`cat results/gradcheck.txt`: float64 max rel err ~1e-9 < 1e-6 PASS for
W_Q/W_K/W_V/X; float32 ~1e-1 roundoff reference; naive softmax nan at logits ≥ 89,
max-subtracted exact.

## Part 3 — Training (toy)

`cat results/ablation/toy_summary.txt`: scaled reaches 90% in 50–200 steps at all
d_k; unscaled 220 (d_k=4) → 300 (d_k=64) → never (d_k=256, acc 0.80).

## Part 4 — Warehouse prediction

`cat results/warehouse/warehouse_results.json`: attention vs control vs
last/MA-24/seasonal MAE table; `*_attention.png` shows diffuse weights;
`*_curves.png` training curves.

## Part 5 — Ablation

Same toy summary + `docs/attention_derivation.md` §3 (Var=d_k mechanism).

## Part 6 — Generalisation

`cat results/shift/shift_summary.txt` + `shift_bars.png`.

## Part 7 — Failure

`docs/failure_investigation.md`: spike-onset 10× MAE gap, under-forecast,
irreducible floor + tail overshoot, evidence locations.

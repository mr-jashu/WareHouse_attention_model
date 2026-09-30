# Warehouse Demand Attention (first-principles)

Status: work in progress. Full README (setup, seeds, commands, results) is written at the end.

## Quick start
    pip install -r requirements.txt
    python -m src.data_generator --out results/data     # regenerate datasets (deterministic)
    python -m pytest -q                                   # run tests
    python -m experiments.plot_data                       # sample plots

See `docs/phase0_design.md` for the design and pre-registered hypotheses, `PROGRESS.md` for status.

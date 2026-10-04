# Debugging Evidence (PRD §17)

Two real bugs occurred during the build. Both were found by *running code*, not by
inspection. Neither was hidden. Hypotheses H1–H5 were never edited as a result.

## Bug 1 — dtype mismatch in attention test

- **Initial implementation:** `tests/test_attention.py` built fixtures with
  `torch.randn(...)` (default float32 module, float64 fixtures from NumPy defaults
  in one test path).
- **Unexpected behaviour:** `pytest` → `RuntimeError: expected ... float32 ... got float64`
  when a float64 input hit the default-float32 `ScaledDotProductAttention` module.
- **Hypothesis:** default torch module parameters are float32 while the test fed float64.
- **Diagnostic experiment:** re-ran the single failing test; confirmed dtype of
  `attn.W_Q` (float32) vs `X` (float64).
- **Correction:** cast fixtures to float32 in the affected tests; kept an explicit
  `.double()` path only in the float64 gradcheck/reference tests where it belongs.
- **Verification:** `python -m pytest -q` → attention tests 8/8 pass; full suite 32/32.

## Bug 2 — hallucinated `torch.ndindex` API in gradcheck

- **Initial implementation:** AI-scaffolded `experiments/gradcheck.py` used
  `torch.ndindex(...)` to iterate parameter indices for central differences.
- **Unexpected behaviour:** `python -m experiments.gradcheck` →
  `AttributeError: module 'torch' has no attribute 'ndindex'`.
- **Hypothesis:** the API was invented by the generator (torch has no `ndindex`;
  NumPy-style `ndenumerate` thinking leaked into torch).
- **Diagnostic experiment:** `python -c "import torch; print(hasattr(torch,'ndindex'))"` → False.
- **Correction:** replaced with `itertools.product(*[range(s) for s in p.shape])`.
- **Verification:** re-ran `python -m experiments.gradcheck` → float64 PASS
  (max rel err ~1e-9 < 1e-6); output written to `results/gradcheck.txt`.

## Non-bug that took real thought (documented, not hidden)

Toy Q/K initialisation: default uniform init gives Q/K norms O(0.1), so unscaled logits
never saturate and H1 would be untestable. Recognising the init-scale mismatch with H1's
unit-variance premise and switching the toy to N(0,1) Q/K init (warehouse model keeps
default uniform init) made the effect measurable. See `docs/reflection.md` Q5 and
`docs/ai_assistance_log.md`.

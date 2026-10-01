"""Toy associative recall + scaled-vs-unscaled ablation (PRD s12-14, Phase 0 H1/s10-11).

Task: N=8 (key,value) pairs, keys distinct from 16, values from 16, plus a query
token carrying a key from the sequence (blank value). Target: paired value.
Chance = 1/16. Single-head attention must match query-key -> pair-key and copy value.

Model: X in R^{T=9 x 32} (one-hot key + one-hot value; query value = 0) ->
  ScaledDotProductAttention(32 -> d_k) -> Linear(d_v -> 16) on last position.

Init note (matches H1 premise): W_Q, W_K ~ N(0,1) so q,k components are O(1) and
unscaled logits have std ~ sqrt(d_k). Warehouse model keeps default uniform init.

Logs per run: loss/acc curves, query-row entropy + max-weight at init/end,
||grad W_Q|| / ||grad W_K|| early, steps-to-90%. Saves results/ablation/.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

from src.attention import ScaledDotProductAttention

N_PAIRS = 8
N_KEYS = 16
N_VALS = 16
T = N_PAIRS + 1
D_IN = N_KEYS + N_VALS


class RecallModel(nn.Module):
    def __init__(self, d_k: int, use_scale: bool):
        super().__init__()
        self.attn = ScaledDotProductAttention(D_IN, d_k, d_k, use_scale=use_scale)
        with torch.no_grad():  # H1 unit-variance premise for Q/K addressing
            self.attn.W_Q.normal_(0.0, 1.0)
            self.attn.W_K.normal_(0.0, 1.0)
        self.head = nn.Linear(d_k, N_VALS)

    def forward(self, X):
        Y, aux = self.attn(X, return_weights=True)
        return self.head(Y[:, -1, :]), aux


def fresh_batch(B: int, rng: torch.Generator):
    keys = torch.stack([torch.randperm(N_KEYS, generator=rng)[:N_PAIRS] for _ in range(B)])
    vals = torch.randint(0, N_VALS, (B, N_PAIRS), generator=rng)
    qpos = torch.randint(0, N_PAIRS, (B,), generator=rng)
    X = torch.zeros(B, T, D_IN)
    X[torch.arange(B)[:, None], torch.arange(N_PAIRS)[None, :], keys] = 1.0
    X[torch.arange(B)[:, None], torch.arange(N_PAIRS)[None, :], N_KEYS + vals] = 1.0
    qkey = keys[torch.arange(B), qpos]
    X[torch.arange(B), N_PAIRS, qkey] = 1.0
    target = vals[torch.arange(B), qpos]
    return X, target


def run_once(d_k: int, use_scale: bool, seed: int, steps: int = 3000, B: int = 128, lr: float = 1e-2):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(10_000 + seed)
    eval_rng = torch.Generator().manual_seed(999_000 + seed)
    model = RecallModel(d_k, use_scale)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        X0, _ = fresh_batch(256, eval_rng)
        _, aux0 = model(X0)
        A0 = aux0["A"][:, -1, :]
        init_max = A0.amax(-1).mean().item()
        init_ent = -(A0 * (A0 + 1e-12).log()).sum(-1).mean().item()
    hist = {"loss": [], "acc": [], "gQ": [], "gK": []}
    steps_to_90 = None
    for s in range(steps):
        X, tgt = fresh_batch(B, rng)
        logits, aux = model(X)
        loss = ce(logits, tgt)
        opt.zero_grad()
        loss.backward()
        gQ = model.attn.W_Q.grad.norm().item()
        gK = model.attn.W_K.grad.norm().item()
        opt.step()
        if s % 50 == 0 or s == steps - 1:
            with torch.no_grad():
                Xe, te = fresh_batch(512, eval_rng)
                le = model(Xe)[0]
                acc = (le.argmax(-1) == te).float().mean().item()
            hist["loss"].append(round(loss.item(), 4))
            hist["acc"].append(round(acc, 4))
            hist["gQ"].append(round(gQ, 5))
            hist["gK"].append(round(gK, 5))
            if acc >= 0.90 and steps_to_90 is None:
                steps_to_90 = s
    with torch.no_grad():
        X1, t1 = fresh_batch(512, eval_rng)
        l1, aux1 = model(X1)
        A1 = aux1["A"][:, -1, :]
        acc1 = (l1.argmax(-1) == t1).float().mean().item()
        end_max = A1.amax(-1).mean().item()
        end_ent = -(A1 * (A1 + 1e-12).log()).sum(-1).mean().item()
    return {
        "d_k": d_k, "scaled": use_scale, "seed": seed,
        "final_acc": round(acc1, 4),
        "steps_to_90": steps_to_90 if steps_to_90 is not None else steps,
        "reached_90": steps_to_90 is not None,
        "init_max": round(init_max, 4), "init_ent": round(init_ent, 4),
        "end_max": round(end_max, 4), "end_ent": round(end_ent, 4),
        "hist": hist,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dks", default="4,16,64,256")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--steps", type=int, default=3000)
    a = p.parse_args()
    dks = [int(s) for s in a.dks.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    out = Path("results/ablation")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for dk in dks:
        for scaled in (True, False):
            for sd in seeds:
                r = run_once(dk, scaled, sd, steps=a.steps)
                rows.append(r)
                print(f"d_k={dk} {'scaled' if scaled else 'unscaled'} seed={sd}: "
                      f"acc={r['final_acc']} steps90={r['steps_to_90']} "
                      f"init_max={r['init_max']} end_max={r['end_max']} "
                      f"gQ0={r['hist']['gQ'][0]}", flush=True)
    (out / "toy_results.json").write_text(json.dumps(rows, indent=2))
    # Summary table
    lines = ["# Toy ablation summary (associative recall, 5 seeds, steps budget "
             + str(a.steps) + ")",
             "H1: init max-weight ~0.2-0.35 scaled vs >0.8 unscaled at d_k=64; "
             "unscaled needs >=1.5x steps to 90% at d_k=64; no difference at d_k=4.",
             ""]
    for dk in dks:
        for scaled in (True, False):
            sub = [r for r in rows if r["d_k"] == dk and r["scaled"] == scaled]
            acc = sum(r["final_acc"] for r in sub) / len(sub)
            st = sum(r["steps_to_90"] for r in sub) / len(sub)
            im = sum(r["init_max"] for r in sub) / len(sub)
            ok = sum(r["reached_90"] for r in sub)
            lines.append(f"d_k={dk} {'scaled' if scaled else 'unscaled'}: "
                         f"acc={acc:.3f} steps90={st:.0f} init_max={im:.3f} reached={ok}/5")
    (out / "toy_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Distribution-shift experiment (PRD s19, Phase 0 H4/s6).

Loads best-val warehouse checkpoints (frozen) and evaluates once on:
test (in-dist), shift_noise (B-noise), shift_spikes (B-spikes), shift_full (B-full),
plus baselines on each. Computes H4: MAE rise 1.8-3x on B-full, skill vs seasonal
halving, B-spikes hurting relative standing more than B-noise. Saves results/shift/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.baselines import BASELINES, mae, rmse
from src.data_generator import build_all
from src.models import NoAttentionControl, WarehouseAttentionModel
from experiments.warehouse import predict_raw

SETS = ["test", "shift_noise", "shift_spikes", "shift_full"]


def main() -> None:
    out = Path("results/shift")
    out.mkdir(parents=True, exist_ok=True)
    d = build_all(1234, 4321)
    mu, sigma = d["mu"], d["sigma"]
    ck = Path("results/warehouse/checkpoints")
    seeds = sorted(int(p.name.split("seed")[1].split(".")[0])
                   for p in ck.glob("attention_seed*.pt"))
    res = {}
    for s in SETS:
        w = d[s]
        X = torch.from_numpy(w["X"])
        entry = {"n": len(w["y_next"]), "baselines": {}, "models": {}}
        for k, f in BASELINES.items():
            pv = f(w["y_hist"])
            entry["baselines"][k] = {"mae": round(mae(pv, w["y_next"]), 3),
                                     "rmse": round(rmse(pv, w["y_next"]), 3)}
        for name, cls in (("attention", WarehouseAttentionModel), ("control", NoAttentionControl)):
            maes = []
            for sd in seeds:
                m = cls()
                m.load_state_dict(torch.load(ck / f"{name}_seed{sd}.pt", map_location="cpu"))
                pv = predict_raw(m, X, mu, sigma)
                maes.append(mae(pv, w["y_next"]))
            entry["models"][name] = {"mae_mean": round(float(np.mean(maes)), 3),
                                     "mae_std": round(float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0, 3)}
            s_mae = entry["baselines"]["seasonal"]["mae"]
            entry["models"][name]["skill_vs_seasonal"] = round(1 - entry["models"][name]["mae_mean"] / s_mae, 4)
        res[s] = entry
        print(f"{s}: " + json.dumps(entry), flush=True)

    # H4 summary
    base = res["test"]["models"]["attention"]["mae_mean"]
    full = res["shift_full"]["models"]["attention"]["mae_mean"]
    lines = ["# Shift summary (H4: B-full MAE 1.8-3x; skill vs seasonal halves; "
             "B-spikes hurts relatively more than B-noise)"]
    lines.append(f"attention test MAE={base}, B-full MAE={full}, ratio={full / base:.2f}x")
    for s in SETS:
        e = res[s]
        lines.append(f"{s}: attention={e['models']['attention']} seasonal={e['baselines']['seasonal']}")
    (out / "shift_results.json").write_text(json.dumps(res, indent=2))
    (out / "shift_summary.txt").write_text("\n".join(lines) + "\n")

    # Bar plot: MAE by set for attention vs baselines
    labels = SETS
    x = np.arange(len(labels))
    attn = [res[s]["models"]["attention"]["mae_mean"] for s in labels]
    last = [res[s]["baselines"]["last"]["mae"] for s in labels]
    seas = [res[s]["baselines"]["seasonal"]["mae"] for s in labels]
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.22
    ax.bar(x - w, attn, w, label="attention")
    ax.bar(x, last, w, label="last")
    ax.bar(x + w, seas, w, label="seasonal")
    ax.set_xticks(x)
    ax.set_xticklabels(["test (A)", "B-noise", "B-spikes", "B-full"])
    ax.set_ylabel("MAE (orders/hour)")
    ax.set_title("Distribution shift: MAE by test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "shift_bars.png", dpi=110)
    plt.close(fig)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

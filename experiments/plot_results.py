"""Plots for the toy ablation from results/ablation/toy_results.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    rows = json.loads(Path("results/ablation/toy_results.json").read_text())
    dks = sorted({r["d_k"] for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, metric in enumerate(("steps_to_90", "init_max")):
        s = [np.mean([r[metric] for r in rows if r["d_k"] == dk and r["scaled"]]) for dk in dks]
        u = [np.mean([r[metric] for r in rows if r["d_k"] == dk and not r["scaled"]]) for dk in dks]
        a = ax[i]
        a.plot(dks, s, marker="o", label="scaled")
        a.plot(dks, u, marker="s", label="unscaled")
        a.set_xscale("log", base=4)
        a.set_xticks(dks)
        a.set_xticklabels([str(d) for d in dks])
        a.set_xlabel("d_k")
        a.legend()
    ax[0].set_ylabel("mean steps to 90% (3000 = never reached)")
    ax[0].set_title("Ablation: convergence vs width")
    ax[1].set_ylabel("mean init max-attention-weight")
    ax[1].set_title("Ablation: saturation at init vs width")
    fig.suptitle("Scaled vs unscaled softmax (associative recall, 5 seeds)")
    fig.tight_layout()
    out = Path("results/ablation/toy_ablation.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

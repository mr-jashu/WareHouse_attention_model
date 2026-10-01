"""Warehouse forecasting experiment (PRD s15-18, Phase 0 H2/H3/H5).

Trains WarehouseAttentionModel + NoAttentionControl (5 model seeds 0-4, fixed
dataset seed 1234), picks best-val checkpoints, evaluates once on test:
MAE/RMSE (orders/hour), event/non-event split, skill vs baselines, and
mean A[last,:] for H5. Saves results/warehouse/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.baselines import BASELINES, mae, rmse
from src.data_generator import build_all
from src.models import NoAttentionControl, WarehouseAttentionModel


def tensors(w, mu, sigma):
    X = torch.from_numpy(w["X"])
    z = (w["y_next"].astype(np.float64) - mu) / sigma
    return X, torch.from_numpy(z).float()


@torch.no_grad()
def predict_raw(model, X, mu, sigma, batch: int = 512):
    model.eval()
    outs = []
    for i in range(0, len(X), batch):
        outs.append(model(X[i:i + batch]).cpu())
    z = torch.cat(outs).numpy().astype(np.float64)
    return z * sigma + mu


@torch.no_grad()
def mean_last_attention(model, X, batch: int = 512):
    model.eval()
    acc = None
    n = 0
    for i in range(0, len(X), batch):
        _, aux = model(X[i:i + batch], return_weights=True)
        A = aux["A"][:, -1, :].cpu().numpy().astype(np.float64)
        acc = A.sum(0) if acc is None else acc + A.sum(0)
        n += len(A)
    return acc / n  # (24,)


def train_one(model, trX, trz, vaX, vaz, mu, sigma, seed: int, epochs: int, lr: float):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    tr_ds = torch.utils.data.TensorDataset(trX, trz)
    loader = torch.utils.data.DataLoader(tr_ds, batch_size=256, shuffle=True,
                                         generator=torch.Generator().manual_seed(seed))
    best_va, best_state, curve_tr, curve_va = float("inf"), None, [], []
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, zb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), zb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            va_pred = predict_raw(model, vaX, mu, sigma)
        # val MAE in orders/hour needs raw targets; compute from z here via stored arrays
        curve_tr.append(round(tot / len(trX), 4))
        # val MSE on z for model selection
        with torch.no_grad():
            va_mse = loss_fn(model(vaX), vaz).item()
        curve_va.append(round(va_mse, 4))
        if va_mse < best_va:
            best_va = va_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return curve_tr, curve_va, best_va


def split_metrics(pred, w):
    out = {"mae": round(mae(pred, w["y_next"]), 3),
           "rmse": round(rmse(pred, w["y_next"]), 3)}
    for key in ("target_spike", "target_drop", "target_spike_onset", "target_drop_onset"):
        m = w[key].astype(bool)
        if m.sum() > 0:
            out[f"mae_{key}"] = round(mae(pred[m], w["y_next"][m]), 3)
            out[f"n_{key}"] = int(m.sum())
    non_event = ~(w["target_spike"].astype(bool) | w["target_drop"].astype(bool))
    out["mae_non_event"] = round(mae(pred[non_event], w["y_next"][non_event]), 3)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-2)
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    out = Path("results/warehouse")
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    d = build_all(1234, 4321)
    mu, sigma = d["mu"], d["sigma"]
    trX, trz = tensors(d["train"], mu, sigma)
    vaX, vaz = tensors(d["val"], mu, sigma)
    teX, _ = tensors(d["test"], mu, sigma)

    base_preds = {k: f(d["test"]["y_hist"]) for k, f in BASELINES.items()}
    base_m = {k: {"mae": round(mae(v, d["test"]["y_next"]), 3),
                  "rmse": round(rmse(v, d["test"]["y_next"]), 3)} for k, v in base_preds.items()}

    results = {"mu": mu, "sigma": sigma, "baselines": base_m, "models": {}}
    for name, cls in (("attention", WarehouseAttentionModel), ("control", NoAttentionControl)):
        per_seed = []
        for sd in seeds:
            model = cls()
            ctr, cva, bva = train_one(model, trX, trz, vaX, vaz, mu, sigma, sd, a.epochs, a.lr)
            ck = out / "checkpoints" / f"{name}_seed{sd}.pt"
            torch.save(model.state_dict(), ck)
            pred = predict_raw(model, teX, mu, sigma)
            m = split_metrics(pred, d["test"])
            m["val_mse_z_best"] = round(bva, 4)
            m["seed"] = sd
            Amean = mean_last_attention(model, teX)
            m["attn_pos24"] = round(float(Amean[-1]), 4)
            m["attn_pos1"] = round(float(Amean[0]), 4)
            m["attn_pos24_plus_pos1"] = round(float(Amean[-1] + Amean[0]), 4)
            m["attn_entropy"] = round(float(-(Amean * np.log(Amean + 1e-12)).sum()), 4)
            if sd == seeds[0]:
                np.save(out / f"{name}_Amean_test.npy", Amean)
                fig, ax = plt.subplots(figsize=(9, 3))
                ax.bar(np.arange(1, 25), Amean)
                ax.set_xlabel("window position (1=oldest=y(t-23), 24=most recent=y(t))")
                ax.set_ylabel("mean A[last,:]")
                ax.set_title(f"{name}: mean last-row attention on test")
                fig.tight_layout()
                fig.savefig(out / f"{name}_attention.png", dpi=110)
                plt.close(fig)
                fig, ax = plt.subplots()
                ax.plot(ctr, label="train MSE (z)")
                ax.plot(cva, label="val MSE (z)")
                ax.legend()
                ax.set_xlabel("epoch")
                ax.set_title(f"{name} seed {sd}: training curves")
                fig.tight_layout()
                fig.savefig(out / f"{name}_curves.png", dpi=110)
                plt.close(fig)
            per_seed.append(m)
            print(f"{name} seed {sd}: MAE={m['mae']} RMSE={m['rmse']} "
                  f"pos24+pos1={m['attn_pos24_plus_pos1']}", flush=True)
        maes = [m["mae"] for m in per_seed]
        results["models"][name] = {
            "per_seed": per_seed,
            "mae_mean": round(float(np.mean(maes)), 3),
            "mae_std": round(float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0, 3),
        }
    # Skills vs seasonal naive
    s_mae = base_m["seasonal"]["mae"]
    for name in results["models"]:
        mm = results["models"][name]["mae_mean"]
        results["models"][name]["skill_vs_seasonal"] = round(1 - mm / s_mae, 4)
    (out / "warehouse_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: {n: v for n, v in m.items() if n != "per_seed"}
                      for k, m in results["models"].items()}, indent=2))
    print("baselines:", json.dumps(base_m, indent=2))


if __name__ == "__main__":
    main()

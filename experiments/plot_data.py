"""Sample plots of the generated data (results/data/*.png)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.data_generator import REGIME_A, REGIME_B, daily_shape, generate_series

fig, ax = plt.subplots(3, 1, figsize=(11, 9))
ax[0].plot(daily_shape(), marker="o"); ax[0].set_title("Daily(h): night trough, broad 09-16h plateau (mean = 1)")
ax[0].set_xlabel("hour of day")
for a_, (name, cfg) in zip([ax[1], ax[2]], [("Regime A (train)", REGIME_A), ("Regime B (shift)", REGIME_B)]):
    s = generate_series(cfg, 14, 1234); t = np.arange(len(s["y"]))
    a_.plot(t, s["level"], color="gray", lw=1, label="deterministic level L(t)")
    a_.plot(t, s["y"], lw=0.8, label="demand y(t)")
    a_.scatter(t[s["spike_onset"]], s["y"][s["spike_onset"]], c="r", s=25, label="spike onset", zorder=3)
    a_.scatter(t[s["drop_onset"]], s["y"][s["drop_onset"]], c="k", marker="v", s=25, label="drop onset", zorder=3)
    a_.set_title(f"{name}: 14 days (same data seed)"); a_.set_ylabel("orders/hour"); a_.legend(loc="upper right", fontsize=7)
ax[2].set_xlabel("hour since start (day 0 = Monday)")
plt.tight_layout(); plt.savefig("results/data/sample_data.png", dpi=110)

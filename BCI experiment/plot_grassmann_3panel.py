"""
3-panel figure (a, c, d) for Grassmann:
  (a) F (upper-level, validation)
  (c) Riemannian gradient norm  -- consistent across algorithms (true_grad_norm)
  (d) v-solver rel-error on range(H)  -- apples-to-apples
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

RESULTS = Path("results"); FIGS = Path("figures"); FIGS.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

ALGO_ORDER = ["R-HJFBiO", "RHGD-HINV", "RHGD-CG", "RHGD-NS", "RHGD-AD"]
COLORS = {"R-HJFBiO": "#1f77b4", "RHGD-HINV": "#2ca02c",
          "RHGD-CG":  "#d62728", "RHGD-NS":   "#9467bd", "RHGD-AD": "#ff7f0e"}
STYLES = {"R-HJFBiO":  {"linestyle": "-",  "linewidth": 2.0, "zorder": 5},
          "RHGD-HINV": {"linestyle": "--", "linewidth": 1.5},
          "RHGD-CG":   {"linestyle": "--", "linewidth": 1.5},
          "RHGD-NS":   {"linestyle": "--", "linewidth": 1.5},
          "RHGD-AD":   {"linestyle": "--", "linewidth": 1.5}}


def stack(runs, algo, key):
    arrs = [np.asarray(r[key], dtype=float) for r in runs if r["algo"] == algo and key in r and len(r[key]) > 0]
    if not arrs: return np.zeros((0, 0))
    T = max(len(a) for a in arrs)
    X = np.full((len(arrs), T), np.nan)
    for i, a in enumerate(arrs):
        X[i, :len(a)] = a
        if len(a) < T: X[i, len(a):] = a[-1]
    X = np.where(np.isfinite(X), X, 1e15)
    return np.clip(X, -1e15, 1e15)


def plot_panel(ax, runs, key, ylabel, title, log_y=True, floor=None):
    for algo in ALGO_ORDER:
        Mat = stack(runs, algo, key)
        if Mat.size == 0: continue
        if floor is not None:
            Mat = np.where(Mat == 0, floor, Mat)
        x = np.arange(Mat.shape[1])
        med = np.nanmedian(Mat, axis=0)
        q25 = np.nanpercentile(Mat, 25, axis=0)
        q75 = np.nanpercentile(Mat, 75, axis=0)
        ax.plot(x, med, color=COLORS[algo], label=algo, **STYLES[algo])
        ax.fill_between(x, q25, q75, color=COLORS[algo], alpha=0.10, linewidth=0)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


runs = [json.loads(l) for l in open(RESULTS / "runs_grassmann_diag.jsonl")]
print(f"Loaded {len(runs)} runs")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))

plot_panel(axes[0], runs, "F",
           r"$F(W)$  (validation upper-level loss)",
           "(a) Upper-level objective")
plot_panel(axes[1], runs, "true_grad_norm",
           r"$\|\mathrm{grad}_W F\|$",
           "(c) Riemannian gradient norm")
plot_panel(axes[2], runs, "v_err_range",
           r"$\|P_{\mathrm{rg}}(v_m - v^\ast)\| / \|P_{\mathrm{rg}}v^\ast\|$",
           r"(d) $v$-solver rel-error on $\mathrm{range}(H)$",
           floor=1e-16)
axes[2].set_ylim(1e-17, 1e30)
axes[0].legend(loc="upper right", fontsize=8)

fig.suptitle("BCI IV-2a, Grassmann manifold Gr(22, 8), PL regime (T = 100, 9 subjects × 3 seeds)",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(FIGS / "grassmann_3panel.png", dpi=160, bbox_inches="tight")
fig.savefig(FIGS / "grassmann_3panel.pdf", bbox_inches="tight")
plt.close(fig)
print(f"  -> {FIGS}/grassmann_3panel.{{png,pdf}}")

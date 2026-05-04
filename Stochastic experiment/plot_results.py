"""
Generate figures from saved trajectories.

Two figures:
  1. trajectories.png -- 3 rows (manifolds) x 3 cols (F traj, |grad F|, init spectrum)
     Each F/grad panel overlays mean +/- range bands for B in {1, 4, 16, 30}.
  2. variance_vs_B.png -- final-iterate std of F as a function of mini-batch
     size B, across all three manifolds. Theorem 15 predicts std ~ 1/sqrt(B);
     a 1/sqrt(B) reference line is overlaid.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "results")
TRAJ = os.path.join(OUT, "trajectories.npz")

MANIFOLDS = ["stiefel", "grassmann", "hyperbolic"]
B_LIST    = [1, 4, 16, 40]
N_SEEDS   = 8

# Each B gets a colour from a sequential map (light -> dark)
B_COLORS = {1:  "#fde0a6", 4:  "#f8a652", 16: "#d75f00", 40: "#7f3a00"}
B_LINES  = {1:  "-",       4:  "-",       16: "-",       40: "-"}

DISPLAY = {
    "stiefel":    "Stiefel  St(81, 70)",
    "grassmann":  "Grassmann  Gr(81, 35)",
    "hyperbolic": "Poincare  B$^8$",
}


def collect(data, mname, B, key):
    """Stack the per-seed arrays (T,) -> (S, T)."""
    arrs = []
    for s in range(N_SEEDS):
        arrs.append(data[f"{mname}__B{B}__s{s}__{key}"])
    return np.stack(arrs, axis=0)


def main():
    data = np.load(TRAJ)

    # ----------------------------------------------------------------
    # Figure 1: three rows (manifolds), three columns (F, grad norm, spectrum)
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))

    for row, mname in enumerate(MANIFOLDS):
        ax_F   = axes[row, 0]
        ax_g   = axes[row, 1]
        ax_sp  = axes[row, 2]

        # --- F trajectories --------------------------------------------
        for B in B_LIST:
            F = collect(data, mname, B, "F")               # (S, T)
            it = data[f"{mname}__B{B}__s0__iters"]
            mu = F.mean(0)
            lo, hi = F.min(0), F.max(0)
            ax_F.plot(it, mu, color=B_COLORS[B], lw=1.8,
                      label=f"B = {B}")
            ax_F.fill_between(it, lo, hi, color=B_COLORS[B], alpha=0.18)

        ax_F.set_title(f"{DISPLAY[mname]}: upper-level loss $F(x_t)$")
        ax_F.set_xlabel("outer iteration $t$")
        ax_F.set_ylabel("$F(x_t)$")
        ax_F.legend(loc="best", fontsize=9, frameon=False)
        ax_F.grid(alpha=0.25)

        # --- Riemannian gradient norm ---------------------------------
        for B in B_LIST:
            g = collect(data, mname, B, "gnorm")           # (S, T-1) -- starts at iter 1
            it = data[f"{mname}__B{B}__s0__iters"][1:]
            # Some entries can be huge transiently; clip for plotting only
            mu = np.median(g, axis=0)
            lo = np.quantile(g, 0.10, axis=0)
            hi = np.quantile(g, 0.90, axis=0)
            ax_g.plot(it, mu, color=B_COLORS[B], lw=1.6, label=f"B = {B}")
            ax_g.fill_between(it, lo, hi, color=B_COLORS[B], alpha=0.18)

        ax_g.set_yscale("log")
        ax_g.set_title(r"Riemannian gradient norm  $\|w_{\mathrm{proj}}\|_x$")
        ax_g.set_xlabel("outer iteration $t$")
        ax_g.set_ylabel(r"$\|\mathrm{grad}\, F\|_x$  (median, 10-90 quantile band)")
        ax_g.grid(alpha=0.25, which="both")

        # --- Initial-iterate Hessian spectrum -------------------------
        spec = data[f"{mname}__B{B_LIST[0]}__s0__spec_init"]
        spec = np.sort(spec)[::-1]
        idx  = np.arange(1, len(spec) + 1)
        rank = data[f"{mname}__B{B_LIST[0]}__s0__rank_init"].item()
        mu_pl = data[f"{mname}__B{B_LIST[0]}__s0__mu_pl_init"].item()

        ax_sp.semilogy(idx, np.maximum(spec, 1e-18), "o", ms=4,
                       color="#444", label="eigenvalue")
        ax_sp.axhline(1e-2, color="#c41e3a", lw=1, ls="--",
                      label=r"$\mu_{\mathrm{clip}} = 10^{-2}$")
        d_tot = len(spec)
        ax_sp.set_title(f"Hess$_y g$ spectrum at init  "
                        f"(rank {rank} / {d_tot}, dim ker = {d_tot - rank})")
        ax_sp.set_xlabel("eigenvalue rank")
        ax_sp.set_ylabel("eigenvalue")
        ax_sp.set_ylim(1e-18, 2.0)
        ax_sp.legend(loc="lower left", fontsize=9, frameon=False)
        ax_sp.grid(alpha=0.25, which="both")

    fig.suptitle(
        "SR-HJFBiO on UCI Superconductivity  "
        "(d = 70, |D$_{tr}$| = 40, |D$_{val}$| = 20000;  "
        "PL but not strongly convex)",
        y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    save1 = os.path.join(OUT, "trajectories.png")
    fig.savefig(save1, dpi=140, bbox_inches="tight")
    print(f"Saved {save1}")

    # ----------------------------------------------------------------
    # Figure 2: Theorem 15 diagnostics.
    # Theorem 15 bounds (1/T) sum E||G_gamma(x_t, w_hat_t)||^2 by C_Psi/T
    # + C_var/B + delta_eps^2 terms. So at fixed T (and small delta_eps),
    # the predicted dominant decay is *in B*: the time-averaged squared
    # gradient norm should drop with B. Left panel plots this directly.
    # Right panel shows the per-iteration band width on Stiefel for visual
    # confirmation that smaller B widens the seed envelope.
    fig2, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: time-averaged ||grad F||^2 vs B (the theorem-bounded quantity)
    for mname in MANIFOLDS:
        avg_g2 = []
        for B in B_LIST:
            g = collect(data, mname, B, "gnorm")           # (S, T-1)
            # post-burn-in mean of g^2, then averaged across seeds
            burn = g.shape[1] // 2
            g2 = (g[:, burn:] ** 2).mean()
            avg_g2.append(g2)
        ax_left.plot(B_LIST, avg_g2, "o-", lw=2, ms=8, label=DISPLAY[mname])

    # 1/B reference (Theorem 15 dominant variance term)
    anchors = []
    for mname in MANIFOLDS:
        g = collect(data, mname, B_LIST[0], "gnorm")
        burn = g.shape[1] // 2
        anchors.append((g[:, burn:] ** 2).mean())
    anchor = max(anchors)
    Bs_arr = np.array(B_LIST, dtype=float)
    ref = anchor * (B_LIST[0] / Bs_arr)
    ax_left.plot(B_LIST, ref, "k--", lw=1.2, alpha=0.6,
                 label=r"$\propto 1/B$ reference")

    ax_left.set_xscale("log")
    ax_left.set_yscale("log")
    ax_left.set_xlabel("mini-batch size B")
    ax_left.set_ylabel(r"time-avg $\|\mathrm{grad}\,F(x_t)\|_x^2$  (post burn-in)")
    ax_left.set_title("Theorem 15: time-avg squared gradient $\\propto C/T + C_{\\mathrm{var}}/B$")
    ax_left.set_xticks(B_LIST)
    ax_left.set_xticklabels([str(b) for b in B_LIST])
    ax_left.grid(alpha=0.3, which="both")
    ax_left.legend(loc="best", fontsize=10, frameon=False)

    # --- Right: per-iteration band width vs t, Grassmann (clearest case)
    mname = "grassmann"
    for B in B_LIST:
        F = collect(data, mname, B, "F")                   # (S, T)
        it = data[f"{mname}__B{B}__s0__iters"]
        band = F.max(0) - F.min(0)
        ax_right.plot(it, band, color=B_COLORS[B], lw=1.8,
                      label=f"B = {B}")
    ax_right.set_xlabel("outer iteration $t$")
    ax_right.set_ylabel(r"seed-band width  $\max_s F(x_t^s) - \min_s F(x_t^s)$")
    ax_right.set_title(f"Per-iteration band width on {DISPLAY[mname]}")
    ax_right.legend(loc="best", fontsize=10, frameon=False)
    ax_right.grid(alpha=0.3)

    fig2.tight_layout()
    save2 = os.path.join(OUT, "variance_vs_B.png")
    fig2.savefig(save2, dpi=140, bbox_inches="tight")
    print(f"Saved {save2}")


if __name__ == "__main__":
    main()

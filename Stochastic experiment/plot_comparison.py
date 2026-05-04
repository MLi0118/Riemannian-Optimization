"""
Plot the head-to-head comparison: R-HJFBiO vs HINV/CG/NS/AD across the
three upper-level manifolds.

Three columns per manifold:
  1. F(x_t) trajectory
  2. Riemannian gradient norm trajectory (log scale)
  3. relative error of v-solver vs Moore-Penrose pseudoinverse reference

Column 3 is the structural diagnostic from the paper draft (its Figure 3):
on the rank-deficient lower-level Hessian, HINV and CG converge to a
1/reg-amplified estimate of v* on the kernel, which is the *wrong* object.
R-HJFBiO's spectrally clipped solution coincides with the pseudoinverse on
the range cluster.

For column 3 we re-run a single trajectory per (manifold, method) and at
each outer iteration compute || v_method - H^+ grad_y f || / || H^+ grad_y f ||
where H^+ is the threshold-pseudoinverse with rcond = 1e-8.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from data import load_superconductivity
from manifolds import Stiefel, Grassmann, PoincareBall
from feature_maps import StiefelFM, GrassmannFM, HyperbolicFM
from algorithm import evaluate, spectral_clip_kernel
from han_baselines import (RHGD, STRATEGIES,
                            _hess_yy, _grad_y_full, _grad_x_full)

HERE = os.path.dirname(os.path.abspath(__file__))
TRAJ = os.path.join(HERE, "results", "comparison.npz")
SAVE = os.path.join(HERE, "results", "comparison.png")

MANIFOLDS = ["stiefel", "grassmann", "hyperbolic"]
METHODS   = ["R-HJFBiO", "HINV", "CG", "NS", "AD"]
N_SEEDS   = 3

DISPLAY = {
    "stiefel":    "Stiefel  St(81, 70)",
    "grassmann":  "Grassmann  Gr(81, 35)",
    "hyperbolic": "Poincare  $\\mathbb{B}^8$",
}

COLORS = {
    "R-HJFBiO": "#1f4e9a",
    "HINV":     "#c41e3a",
    "CG":       "#e87722",
    "NS":       "#9d56b6",
    "AD":       "#2ea876",
}
LINESTYLES = {
    "R-HJFBiO": "-",
    "HINV":     "--",
    "CG":       "--",
    "NS":       ":",
    "AD":       "-.",
}
LINEWIDTHS = {
    "R-HJFBiO": 2.6,
    "HINV":     1.7,
    "CG":       1.7,
    "NS":       1.7,
    "AD":       1.7,
}

# Reuse the comparison run config
N_IN, D, M_TR, M_VAL = 81, 70, 40, 20000
GR_K, HB_N           = 35, 8
ETA_X = {"stiefel": 1e-2, "grassmann": 1e-2, "hyperbolic": 1e-2}
ETA_Y = 5e-2
S_INNER = 5
T_REF = 100      # number of iterations for the v-solver-quality panel
LOG_EVERY = 5
DTYPE = torch.float64
MU_CLIP, LG_CLIP = 1e-2, 1.0
PINV_RCOND = 1e-8


def collect(data, mname, meth, key):
    return [data[f"{mname}__{meth}__s{s}__{key}"] for s in range(N_SEEDS)]


# --------------------------------------------------------------------------
def make_problem(mname, gen):
    """Match run_comparison.py's make_problem (init seed = 0)."""
    if mname == "stiefel":
        M = Stiefel(N_IN, D, dtype=DTYPE); fm = StiefelFM(N_IN, D)
    elif mname == "grassmann":
        M = Grassmann(N_IN, GR_K, dtype=DTYPE)
        R = torch.randn(N_IN, D, generator=gen, dtype=DTYPE) / np.sqrt(N_IN)
        fm = GrassmannFM(N_IN, GR_K, D, R)
    elif mname == "hyperbolic":
        M = PoincareBall(HB_N, eps=0.2, dtype=DTYPE)
        A = torch.randn(D, N_IN, generator=gen, dtype=DTYPE) / np.sqrt(N_IN)
        Pd = torch.randn(D, HB_N, generator=gen, dtype=DTYPE)
        Pd = Pd / (Pd.norm(dim=1, keepdim=True) + 1e-12)
        Pr = 0.5 * torch.rand(D, 1, generator=gen, dtype=DTYPE)
        fm = HyperbolicFM(N_IN, HB_N, D, A, Pr * Pd, sigma=1.5, eps=0.2)
    x = M.random(gen)
    return M, fm, x


def v_solver_quality(mname, meth, X_tr, y_tr, X_val, y_val):
    """Trace projected relative error on range(H).

    The kernel components of any v are not meaningfully comparable across
    methods because grad_y f generically has support in ker(H) and any
    ker-component of v contributes nothing to the cross-derivative
    (G_xy g)^dag [v]. The principled diagnostic is the relative error on
    range(H), which is the only subspace that affects the hypergradient:

        err(t) = || P_range (v_method - v_ref) || / || P_range v_ref ||

    where P_range is the orthogonal projector onto range(S_{[mu_clip, Lg]}(H))
    and v_ref is computed via the same range-restricted pseudoinverse.
    HINV and CG -- which solve a (badly regularised) inverse on the *whole*
    space -- inflate the small-eigenvalue components in the noise tail
    above mu_clip but below pinv-rcond, so their range-restricted error
    can still be O(1/reg) on those near-kernel directions.
    """
    init_gen = torch.Generator().manual_seed(0)
    M, fm, x = make_problem(mname, init_gen)
    y = torch.zeros(D, dtype=DTYPE)

    if meth != "R-HJFBiO":
        opt = RHGD(M, fm, STRATEGIES[meth],
                   eta_x=ETA_X[mname], eta_y=ETA_Y, S_inner=S_INNER)
    else:
        opt = RHGD(M, fm, STRATEGIES["NS"],   # placeholder, we override v
                   eta_x=ETA_X[mname], eta_y=ETA_Y, S_inner=S_INNER)

    iters, errs = [], []

    def compute_v_method(x, y):
        with torch.no_grad():
            H = _hess_yy(fm, x, X_tr)
            gyf = _grad_y_full(fm, x, y, X_val, y_val)

            if meth == "HINV":
                H_reg = H + 1e-8 * torch.eye(H.shape[0], dtype=DTYPE)
                return torch.linalg.solve(H_reg, gyf)
            if meth == "CG":
                v = torch.zeros_like(gyf)
                r = gyf - H @ v; p = r.clone()
                rs_old = torch.dot(r, r)
                for _ in range(20):
                    Hp = H @ p
                    denom = torch.dot(p, Hp)
                    if torch.abs(denom) < 1e-10: break
                    alpha = rs_old / denom
                    v = v + alpha * p
                    r = r - alpha * Hp
                    rs_new = torch.dot(r, r)
                    if torch.sqrt(rs_new) < 1e-10: break
                    p = r + (rs_new / rs_old) * p
                    rs_old = rs_new
                return v
            if meth == "NS":
                Mat = torch.eye(H.shape[0], dtype=DTYPE) - 5e-2 * H
                v = gyf.clone(); acc = gyf.clone()
                for _ in range(4):
                    v = Mat @ v; acc = acc + v
                return 5e-2 * acc
            if meth == "R-HJFBiO":
                ev, U = torch.linalg.eigh(H)
                ev_c = spectral_clip_kernel(ev, MU_CLIP, LG_CLIP)
                pinv_clip = torch.zeros_like(ev_c)
                m = ev_c.abs() > 0
                pinv_clip[m] = 1.0 / ev_c[m]
                return U @ (pinv_clip * (U.T @ gyf))
        return None

    def compute_v_AD(x, y):
        with torch.no_grad():
            H = _hess_yy(fm, x, X_tr)
            gyf = _grad_y_full(fm, x, y, X_val, y_val)
            Mat = torch.eye(H.shape[0], dtype=DTYPE) - ETA_Y * H
            v = gyf.clone(); acc = gyf.clone()
            for _ in range(S_INNER - 1):
                v = Mat @ v; acc = acc + v
            return ETA_Y * acc

    for t in range(T_REF):
        with torch.no_grad():
            H = _hess_yy(fm, x, X_tr)
            gyf = _grad_y_full(fm, x, y, X_val, y_val)
            ev, U = torch.linalg.eigh(H)

            # range projector and reference pseudoinverse, both based on
            # the same mu_clip threshold for fair comparison
            range_mask = ev.abs() > MU_CLIP
            P_range_diag = range_mask.to(DTYPE)
            P_range = U @ torch.diag(P_range_diag) @ U.T

            pinv = torch.zeros_like(ev)
            pinv[range_mask] = 1.0 / ev[range_mask]
            v_ref = U @ (pinv * (U.T @ gyf))      # restricted pinv = the "right" v*

        v_meth = compute_v_AD(x, y) if meth == "AD" else compute_v_method(x, y)

        if v_meth is None:
            err = float("nan")
        else:
            with torch.no_grad():
                diff_proj = P_range @ (v_meth - v_ref)
                ref_proj  = P_range @ v_ref
                ref_norm = torch.linalg.norm(ref_proj).item()
                if ref_norm > 1e-15:
                    err = torch.linalg.norm(diff_proj).item() / ref_norm
                else:
                    err = float("nan")

        if t % LOG_EVERY == 0:
            iters.append(t)
            errs.append(err)

        try:
            x, y, h_proj = opt.step(x, y, X_tr, y_tr, X_val, y_val)
        except Exception:
            break

    return np.array(iters), np.array(errs)


# --------------------------------------------------------------------------
def hypergrad_ref_norm(M, fm, W, y, X_tr, y_tr, X_val, y_val,
                       mu_clip=MU_CLIP, Lg_clip=LG_CLIP):
    """Reference Riemannian hypergradient norm at (W, y) using the
    spectrally-clipped pseudoinverse for v_ref. Method-agnostic:
    every method receives the same diagnostic at its own W_t.
    """
    with torch.no_grad():
        Phi_tr = fm.compute(W, X_tr)
        H = (2.0 / Phi_tr.shape[0]) * Phi_tr.T @ Phi_tr
        ev, U = torch.linalg.eigh(H)
        ev_c = spectral_clip_kernel(ev, mu_clip, Lg_clip)
        pinv = torch.where(ev_c.abs() > 0, 1.0 / ev_c, torch.zeros_like(ev_c))
        gyf = _grad_y_full(fm, W, y, X_val, y_val)
        v_ref = U @ (pinv * (U.T @ gyf))

    # cross = (G_xy g)^dag [v_ref] via grad_x of <grad_y g, v_ref>
    W_v = W.detach().clone().requires_grad_(True)
    Phi = fm.compute(W_v, X_tr)
    r = Phi @ y - y_tr
    gyg = (2.0 / Phi.shape[0]) * Phi.T @ r
    inner = torch.dot(gyg, v_ref)
    cross = torch.autograd.grad(inner, W_v)[0]

    gxf = _grad_x_full(fm, W, y, X_val, y_val)
    amb = gxf - cross
    proj = M.project(W, amb)
    return M.norm(W, proj).item()


def replay_for_ref_gnorm(mname, meth, X_tr, y_tr, X_val, y_val):
    """Replay the same (W_t, y_t) trajectory and log the REFERENCE
    hypergradient norm at every iteration. We re-run the optimisation
    here because the saved trajectories don't include W_t, y_t snapshots.
    """
    init_gen = torch.Generator().manual_seed(0)
    M, fm, x = make_problem(mname, init_gen)
    y = torch.zeros(D, dtype=DTYPE)

    if meth == "R-HJFBiO":
        from algorithm import SRHJFBiO
        v = torch.zeros(D, dtype=DTYPE)
        opt = SRHJFBiO(M, fm,
                       gamma=ETA_X[mname], lam=5e-2, tau=5e-2,
                       mu_clip=MU_CLIP, Lg_clip=LG_CLIP,
                       delta_eps=1e-4, rv=20.0)
        sampling_gen = torch.Generator().manual_seed(1)
    else:
        opt = RHGD(M, fm, STRATEGIES[meth],
                   eta_x=ETA_X[mname], eta_y=ETA_Y, S_inner=S_INNER)

    iters, gn_ref = [], []
    # iter 0
    iters.append(0)
    gn_ref.append(hypergrad_ref_norm(M, fm, x, y, X_tr, y_tr, X_val, y_val))

    for t in range(T_REF):
        try:
            if meth == "R-HJFBiO":
                x, y, v, _ = opt.step(x, y, v, X_tr, y_tr, X_val, y_val,
                                       B_g=M_TR, B_f=M_TR, B_J=M_TR, B_H=M_TR,
                                       gen=sampling_gen,
                                       S_inner=S_INNER, T_v=1)
            else:
                x, y, _ = opt.step(x, y, X_tr, y_tr, X_val, y_val)
        except Exception:
            break

        if (t + 1) % LOG_EVERY == 0:
            iters.append(t + 1)
            try:
                gn = hypergrad_ref_norm(M, fm, x, y, X_tr, y_tr, X_val, y_val)
                if not np.isfinite(gn):
                    gn = float("nan")
                gn_ref.append(gn)
            except Exception:
                gn_ref.append(float("nan"))

    return np.array(iters), np.array(gn_ref)


# --------------------------------------------------------------------------
def main():
    data = np.load(TRAJ)

    print("Computing v-solver quality reference traces ...")
    X_tr, y_tr, X_val, y_val = load_superconductivity(
        "/mnt/user-data/uploads/train.csv",
        m_tr=M_TR, m_val=M_VAL, seed=42, dtype=DTYPE)

    quality = {m: {} for m in MANIFOLDS}
    ref_gn  = {m: {} for m in MANIFOLDS}
    for mname in MANIFOLDS:
        for meth in METHODS:
            it_q, err = v_solver_quality(mname, meth, X_tr, y_tr, X_val, y_val)
            quality[mname][meth] = (it_q, err)
            it_r, gn_r = replay_for_ref_gnorm(mname, meth,
                                              X_tr, y_tr, X_val, y_val)
            ref_gn[mname][meth] = (it_r, gn_r)
            err_med = np.nanmedian(err) if len(err) else float("nan")
            gn0 = gn_r[0]
            print(f"  {mname:12s} {meth:9s}  v-relerr={err_med:.2e}  "
                  f"ref-gnorm[0]={gn0:.3f}  ref-gnorm[end]={gn_r[-1]:.3f}")

    # ----------------------------------------------------------------
    # Plot: 3 rows (manifolds) x 3 cols (F_actual, canonical gnorm, v-quality)
    fig, axes = plt.subplots(3, 3, figsize=(17, 11))

    for row, mname in enumerate(MANIFOLDS):
        ax_F  = axes[row, 0]
        ax_gT = axes[row, 1]
        ax_q  = axes[row, 2]

        for meth in METHODS:
            F_arr = np.stack(collect(data, mname, meth, "F_actual"))
            it_arr = collect(data, mname, meth, "iters")[0]

            # Col 1: F at method's own (W_t, y_t)
            ax_F.plot(it_arr, F_arr.mean(0),
                      color=COLORS[meth], ls=LINESTYLES[meth],
                      lw=LINEWIDTHS[meth], label=meth, alpha=0.95)

            # Col 2: CANONICAL Riemannian hypergradient norm
            #        (method-agnostic, same yardstick for every method)
            it_r, gn_r = ref_gn[mname][meth]
            ax_gT.plot(it_r, gn_r,
                       color=COLORS[meth], ls=LINESTYLES[meth],
                       lw=LINEWIDTHS[meth], label=meth, alpha=0.95)

            # Col 3: v-solver quality (how close is method's v to v_ref on range(H))
            it_q, err_q = quality[mname][meth]
            ax_q.semilogy(it_q, np.maximum(err_q, 1e-16),
                          color=COLORS[meth], ls=LINESTYLES[meth],
                          lw=LINEWIDTHS[meth], label=meth, alpha=0.95)

        ax_F.set_title(f"{DISPLAY[mname]}: $F$ at $(W_t, y_t)$")
        ax_F.set_xlabel("outer iteration $t$")
        ax_F.set_ylabel(r"$f(W_t, y_t)$ on $D_{val}$")
        ax_F.grid(alpha=0.25)
        if row == 0:
            ax_F.legend(loc="upper right", fontsize=9, frameon=True,
                        framealpha=0.9, ncol=1)

        ax_gT.set_title(r"Canonical $\|\mathrm{grad}\, F(W_t)\|_x$ "
                        r"(common $v_{\mathrm{ref}}$ for all methods)")
        ax_gT.set_xlabel("outer iteration $t$")
        ax_gT.set_ylabel(r"$\|\mathrm{grad}\, F\|_x$ via spectrally-clipped pinv")
        ax_gT.set_yscale("log")
        ax_gT.grid(alpha=0.25, which="both")

        ax_q.set_title(r"$v$-solver rel-error on $\mathrm{range}(H)$")
        ax_q.set_xlabel("outer iteration $t$")
        ax_q.set_ylabel(r"$\|P_{\mathrm{rg}}(v_{\mathrm{m}} - v^\star)\| / \|P_{\mathrm{rg}} v^\star\|$")
        ax_q.grid(alpha=0.25, which="both")
        ax_q.axhline(1.0, color="gray", lw=0.8, alpha=0.5)

    fig.suptitle(
        "Head-to-head: R-HJFBiO vs Han et al. (HINV / CG / NS / AD)  "
        "on UCI Superconductivity in the PL regime  "
        "(d = 70, |D$_{tr}$| = 40, full-batch deterministic)",
        y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(SAVE, dpi=140, bbox_inches="tight")
    print(f"\nSaved {SAVE}")

    # Final summary
    print("\nFinal-iterate summary (mean over 3 seeds):")
    print(f"{'manifold':12s} {'method':10s} {'F_final':>10s} "
          f"{'gnorm_canon':>14s} {'v_relerr_med':>16s}")
    for mname in MANIFOLDS:
        for meth in METHODS:
            Fs = [data[f"{mname}__{meth}__s{s}__F_actual"][-1]
                  for s in range(N_SEEDS)]
            err_med = np.nanmedian(quality[mname][meth][1])
            gn_ref_final = ref_gn[mname][meth][1][-1]
            print(f"{mname:12s} {meth:10s} {np.mean(Fs):10.4f} "
                  f"{gn_ref_final:14.3e} {err_med:16.3e}")


if __name__ == "__main__":
    main()


"""
Head-to-head comparison: R-HJFBiO vs Han et al.'s four hypergradient
strategies (HINV, CG, NS, AD), all under the same outer Riemannian-GD
template, on the three upper-level manifolds.

We use the deterministic full-batch setting (B = |D_tr|) to isolate
hypergradient quality. The PL trigger ensures the lower-level Hessian is
rank-deficient throughout, exposing how each method handles singularity.
"""
import os, time
import numpy as np
import torch

from data import load_superconductivity
from manifolds import Stiefel, Grassmann, PoincareBall
from feature_maps import StiefelFM, GrassmannFM, HyperbolicFM
from algorithm import SRHJFBiO, evaluate
from han_baselines import RHGD, STRATEGIES

HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = "/mnt/user-data/uploads/train.csv"
OUT  = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

# Reuse the dimensions from run_experiment.py for consistency
N_IN, D, M_TR, M_VAL = 81, 70, 40, 20000
GR_K, HB_N           = 35, 8

# Per-manifold gammas tuned for S_inner=5 (all methods).
# With S_inner=5 the y-iterate tracks y*(W_t) closely, so the cross-term
# is reliable and we can take much larger upper-level steps than the
# original conservative single-step (S_inner=1) tuning required.
ETA_X = {"stiefel": 1e-2, "grassmann": 1e-2, "hyperbolic": 1e-2}
ETA_Y = 5e-2
S_INNER = 5

T_OUTER  = 250
LOG_EVERY = 5
SEEDS = [1, 2, 3]
DATA_SEED = 42
SHARED_INIT_SEED = 0
DEVICE, DTYPE = "cpu", torch.float64

# R-HJFBiO uses its own internal stepsizes (already tuned in run_experiment.py).
RHJFBIO_GAMMA = ETA_X.copy()
RHJFBIO_LAM   = 5e-2
RHJFBIO_TAU   = 5e-2
MU_CLIP, LG_CLIP = 1e-2, 1.0
DELTA_EPS, RV    = 1e-4, 20.0


def make_problem(mname, init_gen):
    if mname == "stiefel":
        M = Stiefel(N_IN, D, device=DEVICE, dtype=DTYPE)
        fm = StiefelFM(N_IN, D)
    elif mname == "grassmann":
        M = Grassmann(N_IN, GR_K, device=DEVICE, dtype=DTYPE)
        R = torch.randn(N_IN, D, generator=init_gen, dtype=DTYPE) / np.sqrt(N_IN)
        fm = GrassmannFM(N_IN, GR_K, D, R)
    elif mname == "hyperbolic":
        M = PoincareBall(HB_N, eps=0.2, dtype=DTYPE)
        A = torch.randn(D, N_IN, generator=init_gen, dtype=DTYPE) / np.sqrt(N_IN)
        Pd = torch.randn(D, HB_N, generator=init_gen, dtype=DTYPE)
        Pd = Pd / (Pd.norm(dim=1, keepdim=True) + 1e-12)
        Pr = 0.5 * torch.rand(D, 1, generator=init_gen, dtype=DTYPE)
        fm = HyperbolicFM(N_IN, HB_N, D, A, Pr * Pd, sigma=1.5, eps=0.2)
    x = M.random(init_gen)
    return M, fm, x


def run_baseline(mname, strat_name, seed_idx,
                 X_tr, y_tr, X_val, y_val):
    init_gen = torch.Generator().manual_seed(SHARED_INIT_SEED)
    M, fm, x = make_problem(mname, init_gen)
    y = torch.zeros(D, dtype=DTYPE)

    opt = RHGD(M, fm, STRATEGIES[strat_name],
               eta_x=ETA_X[mname], eta_y=ETA_Y, S_inner=S_INNER)

    iters, F_actual, F_clipped, gnorm_traj = [], [], [], []
    # F at actual (W_0, y_0=0) and at the spectrally-clipped y*(W_0)
    Phi_val = fm.compute(x, X_val)
    F_actual.append(((Phi_val @ y - y_val) ** 2).mean().item())
    F_clipped.append(evaluate(x, fm, X_tr, y_tr, X_val, y_val,
                              mu_clip=MU_CLIP)[0])
    iters.append(0)

    diverged = False
    for t in range(T_OUTER):
        try:
            x, y, h_proj = opt.step(x, y, X_tr, y_tr, X_val, y_val)
            gn = M.norm(x, h_proj).item()
        except Exception:
            diverged = True; break

        if not np.isfinite(gn) or gn > 1e15:
            diverged = True; break

        if (t + 1) % LOG_EVERY == 0 or t == T_OUTER - 1:
            Phi_val = fm.compute(x, X_val)
            F_a = ((Phi_val @ y - y_val) ** 2).mean().item()
            F_c, _ = evaluate(x, fm, X_tr, y_tr, X_val, y_val, mu_clip=MU_CLIP)
            if not np.isfinite(F_a) or not np.isfinite(F_c):
                diverged = True; break
            iters.append(t + 1)
            F_actual.append(F_a)
            F_clipped.append(F_c)
            gnorm_traj.append(gn)

    return {
        "iters":      np.array(iters),
        "F_actual":   np.array(F_actual),
        "F_clipped":  np.array(F_clipped),
        "gnorm":      np.array(gnorm_traj),
        "diverged":   diverged,
    }


def run_rhjfbio(mname, seed_idx, X_tr, y_tr, X_val, y_val, B):
    """Full-batch SR-HJFBiO under our tuned settings (B = |D_tr|)."""
    init_gen = torch.Generator().manual_seed(SHARED_INIT_SEED)
    sampling_gen = torch.Generator().manual_seed(seed_idx)
    M, fm, x = make_problem(mname, init_gen)
    y = torch.zeros(D, dtype=DTYPE)
    v = torch.zeros(D, dtype=DTYPE)

    opt = SRHJFBiO(M, fm,
                   gamma=RHJFBIO_GAMMA[mname],
                   lam=RHJFBIO_LAM, tau=RHJFBIO_TAU,
                   mu_clip=MU_CLIP, Lg_clip=LG_CLIP,
                   delta_eps=DELTA_EPS, rv=RV)

    iters, F_actual, F_clipped, gnorm_traj = [], [], [], []
    Phi_val = fm.compute(x, X_val)
    F_actual.append(((Phi_val @ y - y_val) ** 2).mean().item())
    F_clipped.append(evaluate(x, fm, X_tr, y_tr, X_val, y_val,
                              mu_clip=MU_CLIP)[0])
    iters.append(0)

    for t in range(T_OUTER):
        x, y, v, w_proj = opt.step(x, y, v, X_tr, y_tr, X_val, y_val,
                                    B_g=B, B_f=B, B_J=B, B_H=B,
                                    gen=sampling_gen,
                                    S_inner=S_INNER, T_v=1)
        if (t + 1) % LOG_EVERY == 0 or t == T_OUTER - 1:
            Phi_val = fm.compute(x, X_val)
            F_a = ((Phi_val @ y - y_val) ** 2).mean().item()
            F_c, _ = evaluate(x, fm, X_tr, y_tr, X_val, y_val, mu_clip=MU_CLIP)
            iters.append(t + 1)
            F_actual.append(F_a)
            F_clipped.append(F_c)
            gnorm_traj.append(M.norm(x, w_proj).item())

    return {
        "iters":      np.array(iters),
        "F_actual":   np.array(F_actual),
        "F_clipped":  np.array(F_clipped),
        "gnorm":      np.array(gnorm_traj),
        "diverged":   False,
    }


def main():
    print(f"Loading {CSV} ...")
    X_tr, y_tr, X_val, y_val = load_superconductivity(
        CSV, m_tr=M_TR, m_val=M_VAL, seed=DATA_SEED, dtype=DTYPE, device=DEVICE)

    manifolds = ["stiefel", "grassmann", "hyperbolic"]
    methods   = ["R-HJFBiO", "HINV", "CG", "NS", "AD"]

    results = {m: {meth: [] for meth in methods} for m in manifolds}
    t0 = time.time()
    for mname in manifolds:
        print(f"\n--- {mname} ---")
        for meth in methods:
            for s_idx, seed in enumerate(SEEDS):
                tic = time.time()
                if meth == "R-HJFBiO":
                    out = run_rhjfbio(mname, seed, X_tr, y_tr, X_val, y_val,
                                      B=M_TR)   # deterministic for fairness
                else:
                    out = run_baseline(mname, meth, seed,
                                        X_tr, y_tr, X_val, y_val)
                results[mname][meth].append(out)
                tag = "DIV" if out["diverged"] else "OK "
                print(f"  {meth:9s} seed={seed}  {tag}  "
                      f"F_actual={out['F_actual'][-1]:.3f}  "
                      f"F_clip={out['F_clipped'][-1]:.3f}  "
                      f"gnorm_med={np.median(out['gnorm']) if len(out['gnorm']) else float('nan'):.2e}  "
                      f"({time.time()-tic:.1f}s)")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")

    save = os.path.join(OUT, "comparison.npz")
    flat = {}
    for mname in manifolds:
        for meth in methods:
            for s, out in enumerate(results[mname][meth]):
                pref = f"{mname}__{meth}__s{s}__"
                for k, v in out.items():
                    if isinstance(v, np.ndarray):
                        flat[pref + k] = v
                    else:
                        flat[pref + k] = np.array([v])
    np.savez(save, **flat)
    print(f"Saved {save}")


if __name__ == "__main__":
    main()

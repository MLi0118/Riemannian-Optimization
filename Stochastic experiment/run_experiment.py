"""
Main experiment driver. Runs SR-HJFBiO on UCI Superconductivity for
{Stiefel, Grassmann, Poincare} x {B in {1, 4, 16, 30}} x {seeds 0..S-1}
and saves the trajectories to .npz files for plotting.
"""
import os, time
import numpy as np
import torch

from data import load_superconductivity
from manifolds import Stiefel, Grassmann, PoincareBall
from feature_maps import StiefelFM, GrassmannFM, HyperbolicFM
from algorithm import SRHJFBiO, evaluate

# --------------------------------------------------------------------------
# Configuration
HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = "/mnt/user-data/uploads/train.csv"
OUT  = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

# Problem dimensions
N_IN   = 81          # UCI Superconductivity feature count
D      = 70          # lower-level dimension; Stiefel needs r <= n_in
M_TR   = 40          # PL trigger: m_tr < d guarantees rank-deficient lower Hessian
M_VAL  = 20000       # large val set so upper-level mini-batches are meaningful
GR_K   = 35          # Grassmannian subspace dimension
HB_N   = 8           # Poincare ball dimension

# Optimiser hyperparameters (per-manifold gamma to compensate for differing
# gradient magnitudes -- Stiefel and Grassmann grads scale as ||W|| and
# ||W||^2 respectively, hyperbolic features are bounded so a larger gamma
# is admissible)
GAMMA_BY_MANIFOLD = {
    "stiefel":    1e-3,
    "grassmann":  1e-5,
    "hyperbolic": 1e-2,
}
LAM        = 5e-2    # lower-level stepsize
TAU        = 5e-2    # auxiliary v stepsize
MU_CLIP    = 1e-2    # spectral clipping floor; chosen above the "noise"
                     # tail of the lower-level spectrum so the kernel/range
                     # partition is stable across iterations
LG_CLIP    = 1.0     # spectral clipping ceiling (max eigval is ~0.5)
DELTA_EPS  = 1e-4    # FD probe size
RV         = 20.0    # auxiliary v ball radius

# Sweep
T_OUTER  = 250
B_SWEEP  = [1, 4, 16, 40]
SEEDS    = list(range(1, 9))   # 8 seeds for cleaner variance bands
LOG_EVERY = 5

DATA_SEED = 42
SHARED_INIT_SEED = 0   # all seeds start from the SAME W0 / fmap params,
                       # so variance across seeds isolates the σ²/B term
DEVICE = "cpu"
DTYPE  = torch.float64


# --------------------------------------------------------------------------
def make_problem(manifold_name, init_gen):
    """Construct (manifold, fmap, x0, y0, v0) for a given manifold.

    init_gen is a separate generator for the initial point and any fixed
    fmap parameters (random sketches, anchors); the per-seed mini-batch
    sampling generator is created in run_one and is independent of this.
    Sharing init_gen across seeds isolates the σ²/B term predicted by
    Theorem 15 -- all seeds start from identical W0 and differ only in
    their stochastic batch realisations.
    """
    if manifold_name == "stiefel":
        M = Stiefel(N_IN, D, device=DEVICE, dtype=DTYPE)
        fm = StiefelFM(N_IN, D)
    elif manifold_name == "grassmann":
        M = Grassmann(N_IN, GR_K, device=DEVICE, dtype=DTYPE)
        R = torch.randn(N_IN, D, generator=init_gen,
                        device=DEVICE, dtype=DTYPE) / np.sqrt(N_IN)
        fm = GrassmannFM(N_IN, GR_K, D, R)
    elif manifold_name == "hyperbolic":
        M = PoincareBall(HB_N, eps=0.2, device=DEVICE, dtype=DTYPE)
        A = torch.randn(D, N_IN, generator=init_gen,
                        device=DEVICE, dtype=DTYPE) / np.sqrt(N_IN)
        Pd = torch.randn(D, HB_N, generator=init_gen,
                         device=DEVICE, dtype=DTYPE)
        Pd = Pd / (torch.linalg.norm(Pd, dim=1, keepdim=True) + 1e-12)
        Pr = 0.5 * torch.rand(D, 1, generator=init_gen,
                              device=DEVICE, dtype=DTYPE)
        P = Pr * Pd
        fm = HyperbolicFM(N_IN, HB_N, D, A, P, sigma=1.5, eps=0.2)
    else:
        raise ValueError(manifold_name)

    x0 = M.random(init_gen)
    y0 = torch.zeros(D, dtype=DTYPE, device=DEVICE)
    v0 = torch.zeros(D, dtype=DTYPE, device=DEVICE)
    return M, fm, x0, y0, v0


def run_one(manifold_name, B, sampling_seed,
            X_tr, y_tr, X_val, y_val):
    """Single trajectory: returns dict of logged scalars.

    The initial point and fmap parameters are derived from a fixed
    `init_gen` (shared across seeds for a given manifold). Only the
    mini-batch sampling RNG varies with `sampling_seed`.
    """
    init_gen = torch.Generator(device=DEVICE).manual_seed(SHARED_INIT_SEED)
    sampling_gen = torch.Generator(device=DEVICE).manual_seed(sampling_seed)

    M, fm, x, y, v = make_problem(manifold_name, init_gen)

    optimiser = SRHJFBiO(
        manifold=M, fmap=fm,
        gamma=GAMMA_BY_MANIFOLD[manifold_name],
        lam=LAM, tau=TAU,
        mu_clip=MU_CLIP, Lg_clip=LG_CLIP,
        delta_eps=DELTA_EPS, rv=RV,
    )

    iters, F_traj, gnorm_traj, mu_pl_traj = [], [], [], []

    F0, _ = evaluate(x, fm, X_tr, y_tr, X_val, y_val, mu_clip=MU_CLIP)
    iters.append(0); F_traj.append(F0)

    last_gnorm = float("nan")

    with torch.no_grad():
        Phi_tr = fm.compute(x, X_tr)
        H_full = (2.0 / Phi_tr.shape[0]) * Phi_tr.T @ Phi_tr
        ev = torch.linalg.eigvalsh(H_full).cpu().numpy()
    spec_init = ev.copy()
    nonzero = ev[ev > 1e-8]
    mu_pl_init = float(nonzero.min()) if nonzero.size > 0 else 0.0
    rank_init = int((ev > 1e-8).sum())

    for t in range(T_OUTER):
        x, y, v, w_proj = optimiser.step(
            x, y, v, X_tr, y_tr, X_val, y_val,
            B_g=B, B_f=B, B_J=B, B_H=B,
            gen=sampling_gen,
        )
        last_gnorm = M.norm(x, w_proj).item()

        if (t + 1) % LOG_EVERY == 0 or t == T_OUTER - 1:
            F, _ = evaluate(x, fm, X_tr, y_tr, X_val, y_val, mu_clip=MU_CLIP)
            iters.append(t + 1)
            F_traj.append(F)
            gnorm_traj.append(last_gnorm)

            with torch.no_grad():
                Phi = fm.compute(x, X_tr)
                Hf = (2.0 / Phi.shape[0]) * Phi.T @ Phi
                evs = torch.linalg.eigvalsh(Hf).cpu().numpy()
            nz = evs[evs > 1e-8]
            mu_pl_traj.append(float(nz.min()) if nz.size > 0 else 0.0)

    return {
        "iters":       np.array(iters),
        "F":           np.array(F_traj),
        "gnorm":       np.array(gnorm_traj),
        "mu_pl":       np.array(mu_pl_traj),
        "spec_init":   spec_init,
        "rank_init":   rank_init,
        "mu_pl_init":  mu_pl_init,
    }


# --------------------------------------------------------------------------
def main():
    print(f"Loading {CSV} ...")
    X_tr, y_tr, X_val, y_val = load_superconductivity(
        CSV, m_tr=M_TR, m_val=M_VAL, seed=DATA_SEED,
        dtype=DTYPE, device=DEVICE)
    print(f"  X_tr: {tuple(X_tr.shape)}  y_tr: {tuple(y_tr.shape)}")
    print(f"  X_val: {tuple(X_val.shape)}  y_val: {tuple(y_val.shape)}")
    print(f"  PL trigger: |D_tr|={M_TR}, d={D}, dim ker H >= {D - M_TR}")

    manifolds = ["stiefel", "grassmann", "hyperbolic"]
    results = {m: {} for m in manifolds}

    t0 = time.time()
    for mname in manifolds:
        for B in B_SWEEP:
            trajectories = []
            for seed in SEEDS:
                tic = time.time()
                traj = run_one(mname, B, seed, X_tr, y_tr, X_val, y_val)
                trajectories.append(traj)
                dt = time.time() - tic
                print(f"  {mname:10s}  B={B:3d}  seed={seed}  "
                      f"F_final={traj['F'][-1]:.4f}  ({dt:.1f}s)")
            results[mname][B] = trajectories
        print(f"-- {mname} done at t={time.time()-t0:.1f}s")

    # Persist
    save_path = os.path.join(OUT, "trajectories.npz")
    flat = {}
    for mname in manifolds:
        for B, trajs in results[mname].items():
            for s, tr in enumerate(trajs):
                pref = f"{mname}__B{B}__s{s}__"
                for k, v in tr.items():
                    flat[pref + k] = v
    np.savez(save_path, **flat)
    print(f"\nSaved trajectories to {save_path}")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

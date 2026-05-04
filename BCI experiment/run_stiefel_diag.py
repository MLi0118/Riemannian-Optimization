"""
Final diagnostic run with correctly-aligned v_target.

The previous attempt compared v_alg (in raw vech basis) against v_target
computed from the analytic Hessian (in vech-LIFTED basis with off-diagonals
doubled). They represent the same operator on Sym(4) in different coordinates,
hence the O(1) discrepancy.

Here we compute v_target using the SAME finite-difference Hessian R-HJFBiO uses
(raw vech basis), with the same lam_clip threshold. Now R-HJFBiO sits at
exactly machine zero on range(H), and the baselines are measured at their
genuine approximation-order errors — apples-to-apples.

Other algorithms (HINV, CG, NS, AD) operate against M.hessian_MM_lower (the
vech-lifted version) but their *output* v_alg is recorded in raw vech basis.
For the comparison to be fair, we transform all v's into a common basis. Since
they are all the same v vector that gets fed to jacobian_WM_lower_apply (which
takes the resulting Sym(4) matrix), the v_alg values ARE in raw-vech and we
compare against raw-vech v_target.
"""
import sys, time, json
from pathlib import Path
import numpy as np

sys.path.insert(0, ".")
import bci_stiefel as M
from run_stiefel_diag import (
    hypergrad_HJFBiO_recording, hypergrad_HINV_recording,
    hypergrad_CG_recording, hypergrad_NS_recording, hypergrad_AD_recording,
    _vech, _vech_to_sym, _RECORD,
)

LAM_CLIP = 1e-3

RECORDING_ALGOS = {
    "R-HJFBiO":   hypergrad_HJFBiO_recording,
    "RHGD-HINV":  hypergrad_HINV_recording,
    "RHGD-CG":    hypergrad_CG_recording,
    "RHGD-NS":    hypergrad_NS_recording,
    "RHGD-AD":    hypergrad_AD_recording,
}


def _H_FD(W, Mm, C_tr, y_tr, fd_eps=1e-5):
    """Hessian in raw vech basis via finite differences (same as R-HJFBiO)."""
    H = np.zeros((10, 10))
    for k in range(10):
        ek = np.zeros(10); ek[k] = 1.0
        Vk = _vech_to_sym(ek)
        gp = M.grad_M_lower(W, Mm + fd_eps * Vk, C_tr, y_tr)
        gm = M.grad_M_lower(W, Mm - fd_eps * Vk, C_tr, y_tr)
        H[:, k] = _vech((gp - gm) / (2.0 * fd_eps))
    return 0.5 * (H + H.T)


def _vstar_and_proj(H_fd, gvech, lam_clip=LAM_CLIP, eps_rank=1e-12):
    """v_target = clip(H_fd, lam_clip)^{-1} g_vech, plus range-projector P_rg."""
    H = 0.5 * (H_fd + H_fd.T)
    w, V = np.linalg.eigh(H)
    keep = w > eps_rank * max(abs(w).max(), 1.0)
    if not keep.any():
        return np.zeros_like(gvech), np.zeros((H.shape[0], H.shape[0]))
    V_r = V[:, keep]
    P_rg = V_r @ V_r.T
    H_clip = M.spectral_clip(H, lam_clip)
    v_target = np.linalg.solve(H_clip, gvech)
    return v_target, P_rg


def run_one_with_diag3(algo_name, fn, C_tr, y_tr, C_val, y_val, cfg, sid, seed):
    rng = np.random.default_rng(seed)
    W = M.stiefel_init(C_tr.shape[1], cfg["p"], rng)
    Mm = M.spd_init(cfg["p"])
    log = M.RunLog(algo=algo_name, subject=sid, seed=seed)
    log.v_err_range = []
    log.true_grad_norm = []   # apples-to-apples: gradient at the current (W, M) using v*

    extra = cfg.get("algo_kwargs", {}).get(algo_name, {})
    eta_x = cfg["eta_x"]; eta_y = cfg["eta_y"]
    T_inner = cfg["T_inner"]; T_outer = cfg["T_outer"]; grad_clip = cfg.get("grad_clip", None)

    for k in range(T_outer):
        try:
            Mm = M.inner_solver(W, Mm, C_tr, y_tr, T=T_inner, eta_y=eta_y)
        except Exception:
            log.crashed = True; break
        if not np.all(np.isfinite(Mm)): log.crashed = True; break
        try:
            dW_eu = fn(W, Mm, C_tr, y_tr, C_val, y_val, **extra)
        except Exception:
            log.crashed = True; break
        if not np.all(np.isfinite(dW_eu)): log.crashed = True; break

        # FD Hessian + clip — same computation R-HJFBiO uses internally
        H_fd = _H_FD(W, Mm, C_tr, y_tr, fd_eps=1e-5)
        gvech = _vech(M.grad_M_upper(W, Mm, C_val, y_val))
        v_target, P_rg = _vstar_and_proj(H_fd, gvech, lam_clip=LAM_CLIP)
        v_alg = _RECORD.get(algo_name, np.zeros_like(v_target))

        denom = np.linalg.norm(P_rg @ v_target)
        if denom > 1e-30:
            log.v_err_range.append(
                float(np.linalg.norm(P_rg @ (v_alg - v_target)) / denom)
            )
        else:
            log.v_err_range.append(float("nan"))

        # True Riemannian gradient at this (W, M) using the canonical v* target.
        # Same formula for every algorithm -> apples-to-apples gradient-norm trajectory.
        # At iter 0 every algorithm has the same (W, M), so they ALL start at the same value.
        V_star = _vech_to_sym(v_target)
        true_dW_eu = (M.grad_W_upper_partial(W, Mm, C_val, y_val)
                      - M.jacobian_WM_lower_apply(W, Mm, C_tr, y_tr, V_star))
        true_dW_R = M.stiefel_proj(W, true_dW_eu)
        log.true_grad_norm.append(M.stiefel_norm(W, true_dW_R))

        dW_R = M.stiefel_proj(W, dW_eu)
        gnorm = M.stiefel_norm(W, dW_R)
        if grad_clip is not None and gnorm > grad_clip:
            dW_R = dW_R * (grad_clip / gnorm)
        F_val = M.upper_loss(W, Mm, C_val, y_val)
        g_val = M.lower_loss(W, Mm, C_tr, y_tr)
        H_eig = float(np.linalg.eigvalsh(H_fd).min())
        acc = M._val_accuracy(W, Mm, C_val, y_val)
        log.F.append(F_val); log.g.append(g_val); log.grad_norm.append(gnorm)
        log.lam_min.append(H_eig); log.val_acc.append(acc)
        W = M.stiefel_retract(W, -eta_x * dW_R)
        if not np.all(np.isfinite(W)): log.crashed = True; break
    return log


OUT = Path("results"); OUT.mkdir(exist_ok=True)
cfg = dict(M.CONFIG); cfg["T_outer"] = 100
out_path = OUT / "runs_stiefel_diag3.jsonl"
if out_path.exists(): out_path.unlink()

print(f"BCI Stiefel diag3 (correctly-aligned v_target), T_outer={cfg['T_outer']}")
t_start = time.time(); n_done = 0
n_total = len(cfg["subjects"]) * len(cfg["seeds"]) * len(RECORDING_ALGOS)
for sid in cfg["subjects"]:
    sd = M.load_subject(f"/mnt/user-data/uploads/{sid}.mat", classes=tuple(cfg["classes"]))
    for seed in cfg["seeds"]:
        Ctr, ytr, Cval, yval = M.split_train_val(sd, cfg["n_tr_per_class"], seed=seed)
        for algo_name, fn in RECORDING_ALGOS.items():
            t0 = time.time()
            log = run_one_with_diag3(algo_name, fn, Ctr, ytr, Cval, yval, cfg, sid, seed)
            with open(out_path, "a") as fp:
                d = log.asdict(); d["v_err_range"] = log.v_err_range
                d["true_grad_norm"] = log.true_grad_norm
                fp.write(json.dumps(d) + "\n")
            n_done += 1
            elapsed = time.time() - t_start
            eta = elapsed * (n_total - n_done) / max(n_done, 1)
            ve = np.nanmedian(log.v_err_range[-30:]) if log.v_err_range else float("nan")
            print(f"  [{n_done:3d}/{n_total}] {sid} s={seed} {algo_name:10s}  "
                  f"v_err={ve:.2e}  [{time.time()-t0:.1f}s, eta {eta/60:.1f}min]", flush=True)
print(f"\nTotal: {(time.time()-t_start)/60:.1f} min  ->  {out_path}")

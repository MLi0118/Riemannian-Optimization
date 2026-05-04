"""
Grassmann diag run with full instrumentation:
  - F (upper-level)            -- already logged
  - g (lower-level)            -- already logged
  - true_grad_norm             -- already in run_one (consistent gradient at current x)
  - v_err_range                -- NEW: ‖P_rg(v_alg − v*)‖ / ‖P_rg v*‖

We monkey-patch the hypergrad_* functions so each one stashes its computed v
in a global _RECORD dict. Then in the outer loop, we recompute v* (the
spectrally-clipped target) and the range-projector P_rg from the same Hessian
the algorithm sees, and log the relative residual.
"""
import sys, time, json
from pathlib import Path
import numpy as np

sys.path.insert(0, ".")
import bci_hyperbolic as G

LAM_CLIP = 1e-3
_RECORD = {}   # filled per-call by the recording wrappers


def _hjfbio_rec(problem, x, theta, lam_clip=1e-3, fd_eps=1e-5):
    grad_f = problem.grad_th_upper(x, theta)
    H = problem.hess_thth(x); H = 0.5 * (H + H.T)
    H_clip = G._spectral_clip(H, lam_clip)
    v = np.linalg.solve(H_clip, grad_f)
    _RECORD["R-HJFBiO"] = v.copy()
    cross = problem.jac_xth_apply(x, theta, v)
    return problem.grad_x_upper_p(x, theta) - cross


def _hinv_rec(problem, x, theta, ridge=1e-8):
    grad_f = problem.grad_th_upper(x, theta)
    H = problem.hess_thth(x)
    try:
        v = np.linalg.solve(H + ridge * np.eye(H.shape[0]), grad_f)
    except np.linalg.LinAlgError:
        v = np.linalg.lstsq(H + ridge * np.eye(H.shape[0]), grad_f, rcond=None)[0]
    _RECORD["RHGD-HINV"] = v.copy()
    cross = problem.jac_xth_apply(x, theta, v)
    return problem.grad_x_upper_p(x, theta) - cross


def _cg_rec(problem, x, theta, cg_iters=20, tol=1e-10):
    grad_f = problem.grad_th_upper(x, theta)
    def Hv(v): return problem.hess_thth_apply(x, v)
    v = np.zeros_like(grad_f)
    r = grad_f - Hv(v); p = r.copy()
    rs_old = float(r @ r)
    for _ in range(cg_iters):
        Hp = Hv(p)
        denom = float(p @ Hp)
        if abs(denom) < 1e-30: break
        alpha = rs_old / denom
        v = v + alpha * p
        r = r - alpha * Hp
        rs_new = float(r @ r)
        if rs_new < tol: break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    _RECORD["RHGD-CG"] = v.copy()
    cross = problem.jac_xth_apply(x, theta, v)
    return problem.grad_x_upper_p(x, theta) - cross


def _ns_rec(problem, x, theta, ns_iters=5, ns_alpha=5e-2):
    grad_f = problem.grad_th_upper(x, theta)
    v_acc = np.zeros_like(grad_f)
    cur = grad_f.copy()
    for _ in range(ns_iters):
        v_acc = v_acc + cur
        cur = cur - ns_alpha * problem.hess_thth_apply(x, cur)
    v = ns_alpha * v_acc
    _RECORD["RHGD-NS"] = v.copy()
    cross = problem.jac_xth_apply(x, theta, v)
    return problem.grad_x_upper_p(x, theta) - cross


def _ad_rec(problem, x, theta_init, ad_T=5, eta_y=5e-2):
    # Mirror G.hypergrad_AD's reverse-mode unrolling
    thetas = [theta_init.copy()]; theta = theta_init.copy()
    for _ in range(ad_T):
        theta = theta - eta_y * problem.grad_th_lower(x, theta)
        thetas.append(theta.copy())
    lam = problem.grad_th_upper(x, thetas[-1])
    df_dx = problem.grad_x_upper_p(x, thetas[-1])
    v_eff = np.zeros_like(lam)
    for t in reversed(range(ad_T)):
        v_eff = v_eff + eta_y * lam
        cross = problem.jac_xth_apply(x, thetas[t], lam)
        df_dx = df_dx - eta_y * cross
        lam = lam - eta_y * problem.hess_thth_apply(x, lam)
    _RECORD["RHGD-AD"] = v_eff.copy()
    return df_dx


REC_ALGOS = {
    "R-HJFBiO":   _hjfbio_rec,
    "RHGD-HINV":  _hinv_rec,
    "RHGD-CG":    _cg_rec,
    "RHGD-NS":    _ns_rec,
    "RHGD-AD":    _ad_rec,
}


def _v_target_and_proj(H, grad_f, lam_clip=LAM_CLIP, eps_rank=1e-12):
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    keep = w > eps_rank * max(abs(w).max(), 1.0)
    if not keep.any():
        return np.zeros_like(grad_f), np.zeros((H.shape[0], H.shape[0]))
    V_r = V[:, keep]
    P_rg = V_r @ V_r.T
    H_clip = G._spectral_clip(H, lam_clip)
    v_target = np.linalg.solve(H_clip, grad_f)
    return v_target, P_rg


def run_one_diag(algo_name, fn, problem, manifold, label, seed, cfg):
    rng = np.random.default_rng(seed)
    x = manifold.init(rng)
    theta = np.zeros(cfg["d"])
    log = G.RunLog(algo=algo_name, label=label, seed=seed)
    log.v_err_range = []

    extra = cfg.get("algo_kwargs", {}).get(algo_name, {})
    eta_x = cfg["eta_x"]; eta_y = cfg["eta_y"]
    T_inner = cfg["T_inner"]; T_outer = cfg["T_outer"]; grad_clip = cfg.get("grad_clip", None)

    for k in range(T_outer):
        try:
            theta = G.inner_solver(problem, x, theta, T_inner, eta_y)
        except Exception:
            log.crashed = True; break
        if not np.all(np.isfinite(theta)): log.crashed = True; break
        try:
            dx_eu = fn(problem, x, theta, **extra)
        except Exception:
            log.crashed = True; break
        if not np.all(np.isfinite(dx_eu)): log.crashed = True; break

        # v_err diagnostic
        H = problem.hess_thth(x)
        grad_f = problem.grad_th_upper(x, theta)
        v_target, P_rg = _v_target_and_proj(H, grad_f, lam_clip=LAM_CLIP)
        v_alg = _RECORD.get(algo_name, np.zeros_like(v_target))
        denom = np.linalg.norm(P_rg @ v_target)
        if denom > 1e-30:
            log.v_err_range.append(float(np.linalg.norm(P_rg @ (v_alg - v_target)) / denom))
        else:
            log.v_err_range.append(float("nan"))

        # consistent true gradient norm at current x (uses closed-form theta*)
        true_gn = G._true_gradient_norm(problem, manifold, x, lam_clip=LAM_CLIP)

        dx_proj = manifold.proj(x, dx_eu)
        dx_R = manifold.egrad2rgrad(x, dx_proj)
        gnorm = manifold.norm(x, dx_R)
        if grad_clip is not None and gnorm > grad_clip:
            dx_R = dx_R * (grad_clip / gnorm)

        log.F.append(problem.upper_loss(x, theta))
        log.g.append(problem.lower_loss(x, theta))
        log.grad_norm.append(gnorm)
        log.true_grad_norm.append(true_gn)
        try:
            log.lam_min.append(float(np.linalg.eigvalsh(H).min()))
        except Exception:
            log.lam_min.append(float("nan"))

        x = manifold.retract(x, -eta_x * dx_R)
        if not np.all(np.isfinite(x)): log.crashed = True; break
    return log


OUT = Path("results"); OUT.mkdir(exist_ok=True)
cfg = dict(G.CONFIG)
cfg["T_outer"] = 100
out_path = OUT / "runs_hyperbolic_diag.jsonl"
if out_path.exists(): out_path.unlink()

print(f"Hyperbolic synthetic diag (with v_err and true_grad_norm), T_outer={cfg['T_outer']}")
manifold = G.HyperbolicManifold(n=cfg["n_dim"])

t_start = time.time(); n_done = 0
n_total_runs = len(cfg["data_seeds"]) * len(cfg["init_seeds"]) * len(REC_ALGOS)

for ds_seed in cfg["data_seeds"]:
    X, y, w_true, theta_true = G.synthesize_data(
        n_dim=cfg["n_dim"], d=cfg["d"], n_total=cfg["n_total"], seed=ds_seed,
        noise=cfg["noise"], w_radius=cfg["w_radius"], x_radius=cfg["x_radius"])
    n_tr = cfg["n_tr"]
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_val, y_val = X[n_tr:], y[n_tr:]
    problem = G.Problem(G, train_data=(X_tr, y_tr), val_data=(X_val, y_val), d=cfg["d"])
    for init_seed in cfg["init_seeds"]:
        for algo_name, fn in REC_ALGOS.items():
            t0 = time.time()
            log = run_one_diag(algo_name, fn, problem, manifold,
                               label=f"ds{ds_seed}", seed=init_seed, cfg=cfg)
            with open(out_path, "a") as fp:
                d = log.asdict()
                d["v_err_range"] = log.v_err_range
                fp.write(json.dumps(d) + "\n")
            n_done += 1
            elapsed = time.time() - t_start
            eta = elapsed * (n_total_runs - n_done) / max(n_done, 1)
            tg = np.median(log.true_grad_norm[-30:]) if log.true_grad_norm else float("nan")
            ve = np.nanmedian(log.v_err_range[-30:]) if log.v_err_range else float("nan")
            print(f"  [{n_done:3d}/{n_total_runs}] ds{ds_seed} init{init_seed} {algo_name:10s}  "
                  f"true|g|={tg:.2e}  v_err={ve:.2e}  [{time.time()-t0:.1f}s, eta {eta/60:.1f}min]",
                  flush=True)

print(f"\nTotal: {(time.time()-t_start)/60:.1f} min  ->  {out_path}")

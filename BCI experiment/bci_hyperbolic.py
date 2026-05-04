"""
Hyperbolic experiment — consolidated single-file version.

NOTE: This experiment uses a SYNTHETIC Poincaré-ball regression problem,
NOT BCI IV-2a. EEG covariance matrices live on the SPD manifold and are
not naturally hyperbolic, so we use a synthetic generator with a known
ground-truth w_true ∈ B^n_dim.

Bilevel formulation:
  min_{w in B^n}      F(w) := (1/n_val) Σ ( y_j - <φ(w; x_j), θ*(w)> )^2
  s.t. θ*(w) = argmin_{θ in R^d}  g(w, θ) = (1/n_tr) Σ ( y_i - <φ(w; x_i), θ> )^2

  φ_k(w; x) = (d_B(w, x) / D_SCALE)^{k+1}     polynomial in hyperbolic distance

Run:
  python3 bci_hyperbolic.py    # writes results/runs_hyperbolic.jsonl
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import json, sys, time

import numpy as np


# ============================================================================
# manifolds_hyperbolic.py — Poincaré ball B^n with Möbius-addition exp map
# ============================================================================

# Numerical guard so we never sit too close to the ideal boundary, where the
# conformal factor lambda(x) = 2/(1-||x||^2) and hyperbolic distances explode.
# A buffer of 0.2 (i.e. max-norm 0.8) is standard in Poincaré-ball optimisation
# (see e.g. Nickel & Kiela 2017, Ganea et al. 2018) — it costs nothing in modelling
# power but keeps every iterate in a numerically well-behaved region. We use
# 0.2 (rather than the literature's typical 1e-5) because 0.2 prevents the
# cascade of small eigenvalues in the lower-Hessian feature spectrum that make
# even spectrally-clipped solvers blow up over long horizons.
EPS_BOUNDARY = 0.2
MIN_NORM = 1e-15


def _conformal(x: np.ndarray) -> float:
    return 2.0 / max(1.0 - float(x @ x), MIN_NORM)


def _project_into_ball(x: np.ndarray) -> np.ndarray:
    """Re-project x into the open ball of radius 1 - eps if it has wandered out."""
    n = float(np.linalg.norm(x))
    if n >= 1.0 - EPS_BOUNDARY:
        return x * ((1.0 - EPS_BOUNDARY) / max(n, MIN_NORM))
    return x


def hyp_proj(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Tangent projection on Poincaré ball: identity (T_x B^n = R^n)."""
    return z


def hyp_egrad2rgrad(x: np.ndarray, g_eu: np.ndarray) -> np.ndarray:
    """Convert Euclidean gradient to Riemannian gradient on B^n."""
    factor = ((1.0 - float(x @ x)) / 2.0) ** 2
    return factor * g_eu


def _mobius_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Möbius addition x ⊕_M y (Ungar); curvature -1."""
    xy = float(x @ y)
    nx2 = float(x @ x)
    ny2 = float(y @ y)
    num = (1.0 + 2.0 * xy + ny2) * x + (1.0 - nx2) * y
    den = 1.0 + 2.0 * xy + nx2 * ny2
    return num / max(den, MIN_NORM)


def hyp_exp(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Exponential map at x on B^n with curvature -1."""
    vn = float(np.linalg.norm(v))
    if vn < MIN_NORM:
        return x
    lam = _conformal(x)
    direction = v / vn
    factor = np.tanh(lam * vn / 2.0)
    return _project_into_ball(_mobius_add(x, factor * direction))


def hyp_retract(x: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Use the true exponential map as the retraction (closed-form, stable)."""
    return hyp_exp(x, xi)


def hyp_norm(x: np.ndarray, xi: np.ndarray) -> float:
    """Riemannian norm of tangent xi at x."""
    return float(_conformal(x) * np.linalg.norm(xi))


def hyp_init(n: int, rng: np.random.Generator, radius: float = 0.3) -> np.ndarray:
    """Random initial point well inside the ball."""
    v = rng.standard_normal(n)
    v = v / np.linalg.norm(v)
    r = radius * rng.random()
    return r * v


def hyp_distance(x: np.ndarray, y: np.ndarray) -> float:
    nx2 = float(x @ x); ny2 = float(y @ y)
    diff2 = float((x - y) @ (x - y))
    arg = 1.0 + 2.0 * diff2 / max((1.0 - nx2) * (1.0 - ny2), MIN_NORM)
    return float(np.arccosh(max(arg, 1.0)))

# ============================================================================
# bilevel_hyperbolic.py — synthetic Poincaré-ball regression bilevel
# ============================================================================

# ---------- distance and gradient ----------

def _distances_and_grads(w: np.ndarray, X: np.ndarray):
    """
    Returns (d_H, dd_dw):
        d_H   shape (n,)            -- d_H(w, x_i) for each i
        dd_dw shape (n, n_dim)      -- gradient of d_H wrt w, for each i
    Stable for w, x_i strictly inside the ball.
    """
    a = 1.0 - float(w @ w)                         # scalar
    b = 1.0 - np.einsum("ni,ni->n", X, X)          # (n,)
    diff = w[None, :] - X                          # (n, n_dim)
    c = np.einsum("ni,ni->n", diff, diff)          # (n,)
    u = c / (a * np.maximum(b, 1e-30))             # (n,)
    z = 1.0 + 2.0 * u                              # >= 1
    d_H = np.arccosh(np.maximum(z, 1.0 + 1e-30))   # (n,)
    # gradient of d_H wrt w
    dz_dw = (4.0 / (a * a * np.maximum(b, 1e-30)))[:, None] * (a * diff + c[:, None] * w[None, :])
    denom = np.sqrt(np.maximum(z * z - 1.0, 1e-30))[:, None]
    dd_dw = dz_dw / denom                          # (n, n_dim)
    return d_H, dd_dw


D_SCALE = 2.0   # typical hyperbolic distance scale
TANH_SCALE = 1.0   # tanh argument scale; tanh(d_H/TANH_SCALE) saturates beyond ~3

# ---------- features ----------

def _phi_batch(w: np.ndarray, X: np.ndarray, d: int):
    """
    Returns (Phi (n,d), d_H (n,), dd_dw (n, n_dim), s (n,), ds_dw (n, n_dim)).

    Features:  phi_k(w; x) = s(w; x)^{k+1},  s(w; x) = tanh(d_H(w, x) / TANH_SCALE).

    Why tanh? Polynomial-in-distance features (d_H)^k explode for large d_H,
    which happens whenever w drifts toward the boundary of the Poincaré ball
    (where d_H → ∞). The exploded features cause catastrophic cancellation
    in the finite-difference Hessian and make every algorithm unstable. The
    tanh saturation is the standard practical fix in hyperbolic neural networks
    (Ganea et al. 2018, Chami et al. 2019) — it bounds features in [0, 1]
    while preserving the local geometric information for distances of order 1.
    """
    d_H, dd_dw = _distances_and_grads(w, X)
    s = np.tanh(d_H / TANH_SCALE)
    Phi = np.stack([s ** (k + 1) for k in range(d)], axis=1)
    # ds/dw = sech^2(d_H/T) * (1/T) * dd_H/dw
    sech2 = 1.0 - s * s
    ds_dw = (sech2 / TANH_SCALE)[:, None] * dd_dw
    return Phi, d_H, dd_dw, s, ds_dw


# ---------- losses ----------

def lower_loss(w, theta, X_tr, y_tr, d):
    Phi, *_ = _phi_batch(w, X_tr, d)
    r = Phi @ theta - y_tr
    return float(np.mean(r * r))


def upper_loss(w, theta_star, X_val, y_val, d):
    Phi, *_ = _phi_batch(w, X_val, d)
    r = Phi @ theta_star - y_val
    return float(np.mean(r * r))


# ---------- gradients (Euclidean wrt w) ----------

def grad_M_lower(w, theta, X_tr, y_tr, d):
    """Gradient of g wrt theta (renamed for consistency with bilevel.py / bilevel_grassmann.py)."""
    Phi, *_ = _phi_batch(w, X_tr, d)
    r = Phi @ theta - y_tr
    return (2.0 / len(y_tr)) * (Phi.T @ r)


def grad_M_upper(w, theta, X_val, y_val, d):
    Phi, *_ = _phi_batch(w, X_val, d)
    r = Phi @ theta - y_val
    return (2.0 / len(y_val)) * (Phi.T @ r)


def _dphiT_dw(w, X, T, d):
    """Compute  D[T^T phi]/dw  where T in R^d. Features phi_k = s^{k+1}, s = tanh(d_H/TANH_SCALE).
    d phi_k / dw = (k+1) * s^k * ds/dw.  Returns ((n, n_dim), Phi)."""
    Phi, d_H, dd_dw, s, ds_dw = _phi_batch(w, X, d)
    weights = np.zeros_like(s)
    for k in range(d):
        weights += T[k] * (k + 1) * (s ** k)
    return weights[:, None] * ds_dw, Phi


def grad_W_lower(w, theta, X_tr, y_tr, d):
    DM, Phi = _dphiT_dw(w, X_tr, theta, d)
    r = Phi @ theta - y_tr
    return (2.0 / len(y_tr)) * (r @ DM)             # (n_dim,)


def grad_W_upper_partial(w, theta, X_val, y_val, d):
    DM, Phi = _dphiT_dw(w, X_val, theta, d)
    r = Phi @ theta - y_val
    return (2.0 / len(y_val)) * (r @ DM)


# ---------- Hessian / Jacobian ----------

def hessian_MM_lower(w, X_tr, d):
    Phi, *_ = _phi_batch(w, X_tr, d)
    return (2.0 / Phi.shape[0]) * (Phi.T @ Phi)


def hessian_MM_lower_apply(w, X_tr, V, d):
    Phi, *_ = _phi_batch(w, X_tr, d)
    return (2.0 / Phi.shape[0]) * (Phi.T @ (Phi @ V))


def jacobian_WM_lower_apply(w, theta, X_tr, y_tr, V, d):
    """
    Apply  d/dw (d g / d theta)  to V in R^d. Returns shape (n_dim,).

       d/dtheta_k g = (2/n) sum_i (Phi_i theta - y_i) Phi_{i,k}
    => d/dw d/dtheta_k g = (2/n) sum_i [ d/dw(Phi_i theta) * Phi_{i,k}  +  r_i * d/dw Phi_{i,k} ]
    Contracting over k with V gives:
       (2/n) sum_i [ DphiT_i * (Phi_i^T V)  +  r_i * DphiV_i ]
    """
    DphiT, Phi = _dphiT_dw(w, X_tr, theta, d)
    DphiV, _   = _dphiT_dw(w, X_tr, V, d)
    r    = Phi @ theta - y_tr
    PhiV = Phi @ V
    n = len(y_tr)
    return (2.0 / n) * (PhiV @ DphiT + r @ DphiV)


def solve_inner_exact(w, X_tr, y_tr, d, ridge: float = 0.0):
    Phi, *_ = _phi_batch(w, X_tr, d)
    G = Phi.T @ Phi
    lam_min = float(np.linalg.eigvalsh(G).min())
    rhs = Phi.T @ y_tr
    theta = np.linalg.solve(G + ridge * np.eye(d), rhs)
    return theta, lam_min


# ---------- synthetic data generator ----------

def synthesize_data(n_dim: int, d: int, n_total: int, seed: int = 0,
                    noise: float = 0.05, w_radius: float = 0.3,
                    x_radius: float = 0.7):
    """Generate synthetic hyperbolic regression data with hidden anchor w_true.
    Targets are standardised (zero-mean, unit-std) so the lower-level Hessian is
    well-scaled relative to the feature design matrix.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n_dim); v /= np.linalg.norm(v)
    w_true = w_radius * rng.random() * v
    X = rng.standard_normal((n_total, n_dim))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    radii = x_radius * rng.random(n_total) ** (1.0 / n_dim)
    X = X * radii[:, None]
    theta_true = rng.standard_normal(d) / np.array([k + 1 for k in range(d)])
    Phi_true, *_ = _phi_batch(w_true, X, d)
    y = Phi_true @ theta_true + noise * rng.standard_normal(n_total)
    # standardise y for numerical scale
    y = (y - y.mean()) / max(y.std(), 1e-12)
    return X, y, w_true, theta_true

# ============================================================================
# Manifold interface
# ============================================================================

class HyperbolicManifold:
    def __init__(self, n: int):
        self.n = n
    def proj(self, x, z):           return hyp_proj(x, z)        # identity
    def retract(self, x, xi):       return hyp_retract(x, xi)
    def egrad2rgrad(self, x, g):    return hyp_egrad2rgrad(x, g) # ((1-||x||^2)/2)^2 * g
    def norm(self, x, xi):          return hyp_norm(x, xi)       # Riemannian norm
    def init(self, rng):            return hyp_init(self.n, rng)


# ============================================================================
# runner_v2.py — manifold-agnostic Problem + 5 algorithms + outer loop
# ============================================================================

# -------- Problem abstraction (lower variable theta in R^d) --------

class Problem:
    """Wraps a bilevel module so all data and kwargs are pre-bound; algorithms see a fixed API."""
    def __init__(self, bl_module, train_data: tuple, val_data: tuple, d: int, **bl_kwargs):
        self.bl = bl_module
        self.train = train_data
        self.val   = val_data
        self.d     = d
        self.kwargs = bl_kwargs

    def lower_loss(self, x, theta):     return self.bl.lower_loss(x, theta, *self.train, self.d, **self.kwargs)
    def upper_loss(self, x, theta):     return self.bl.upper_loss(x, theta, *self.val,   self.d, **self.kwargs)
    def grad_th_lower(self, x, theta):  return self.bl.grad_M_lower(x, theta, *self.train, self.d, **self.kwargs)
    def grad_th_upper(self, x, theta):  return self.bl.grad_M_upper(x, theta, *self.val,   self.d, **self.kwargs)
    def grad_x_lower(self, x, theta):   return self.bl.grad_W_lower(x, theta, *self.train, self.d, **self.kwargs)
    def grad_x_upper_p(self, x, theta): return self.bl.grad_W_upper_partial(x, theta, *self.val, self.d, **self.kwargs)
    def hess_thth(self, x):             return self.bl.hessian_MM_lower(x, self.train[0], self.d, **self.kwargs)
    def hess_thth_apply(self, x, V):    return self.bl.hessian_MM_lower_apply(x, self.train[0], V, self.d, **self.kwargs)
    def jac_xth_apply(self, x, theta, V): return self.bl.jacobian_WM_lower_apply(x, theta, *self.train, V, self.d, **self.kwargs)

    def solve_inner_exact(self, x, ridge: float = 0.0):
        """Closed-form inner minimiser if the bilevel module exposes one (used when g is quadratic
        in theta -- which is the case for all three of our regression-style problems)."""
        if hasattr(self.bl, "solve_inner_exact"):
            theta_star, _ = self.bl.solve_inner_exact(x, *self.train, self.d, ridge=ridge, **self.kwargs)
            return theta_star
        return None


# -------- Inner solver (Euclidean GD on theta) --------

def inner_solver(problem: Problem, x, theta0, T_inner: int, eta_y: float):
    theta = theta0.copy()
    for _ in range(T_inner):
        g = problem.grad_th_lower(x, theta)
        theta = theta - eta_y * g
    return theta


# -------- Spectral clipping helper --------

def _spectral_clip(H: np.ndarray, lam_clip: float) -> np.ndarray:
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    return (V * np.maximum(w, lam_clip)) @ V.T


# -------- Five hypergradient methods --------

def hypergrad_HJFBiO(problem: Problem, x, theta, lam_clip=1e-3, fd_eps=1e-5):
    """Build Hessian via finite differences of grad_theta, clip its spectrum, solve, contract."""
    d = problem.d
    grad_f = problem.grad_th_upper(x, theta)
    H = np.zeros((d, d))
    for k in range(d):
        e = np.zeros(d); e[k] = 1.0
        gp = problem.grad_th_lower(x, theta + fd_eps * e)
        gm = problem.grad_th_lower(x, theta - fd_eps * e)
        H[:, k] = (gp - gm) / (2.0 * fd_eps)
    H_clip = _spectral_clip(H, lam_clip)
    v = np.linalg.solve(H_clip, grad_f)
    cross = problem.jac_xth_apply(x, theta, v)
    return problem.grad_x_upper_p(x, theta) - cross


def hypergrad_HINV(problem: Problem, x, theta, ridge=1e-8):
    grad_f = problem.grad_th_upper(x, theta)
    H = problem.hess_thth(x)
    try:
        v = np.linalg.solve(H + ridge * np.eye(H.shape[0]), grad_f)
    except np.linalg.LinAlgError:
        v = np.linalg.lstsq(H + ridge * np.eye(H.shape[0]), grad_f, rcond=None)[0]
    cross = problem.jac_xth_apply(x, theta, v)
    return problem.grad_x_upper_p(x, theta) - cross


def hypergrad_CG(problem: Problem, x, theta, cg_iters=20, tol=1e-10):
    grad_f = problem.grad_th_upper(x, theta)
    v = np.zeros_like(grad_f)
    r = grad_f - problem.hess_thth_apply(x, v)
    p = r.copy()
    rs_old = float(r @ r)
    for _ in range(cg_iters):
        Hp = problem.hess_thth_apply(x, p)
        denom = float(p @ Hp)
        if abs(denom) < 1e-30:
            break
        alpha = rs_old / denom
        v = v + alpha * p
        r = r - alpha * Hp
        rs_new = float(r @ r)
        if rs_new < tol: break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    cross = problem.jac_xth_apply(x, theta, v)
    return problem.grad_x_upper_p(x, theta) - cross


def hypergrad_NS(problem: Problem, x, theta, ns_iters=5, ns_alpha=5e-2):
    grad_f = problem.grad_th_upper(x, theta)
    v_acc = np.zeros_like(grad_f)
    cur = grad_f.copy()
    for _ in range(ns_iters):
        v_acc = v_acc + cur
        Hv = problem.hess_thth_apply(x, cur)
        cur = cur - ns_alpha * Hv
    v = ns_alpha * v_acc
    cross = problem.jac_xth_apply(x, theta, v)
    return problem.grad_x_upper_p(x, theta) - cross


def hypergrad_AD(problem: Problem, x, theta_init, ad_T=5, eta_y=5e-2):
    """Truncated AD through ad_T Euclidean GD steps on theta (matches Han et al.'s baseline)."""
    thetas = [theta_init.copy()]
    th = theta_init.copy()
    for _ in range(ad_T):
        th = th - eta_y * problem.grad_th_lower(x, th)
        thetas.append(th.copy())
    lam = problem.grad_th_upper(x, thetas[-1])
    df_dx = problem.grad_x_upper_p(x, thetas[-1])
    for t in reversed(range(ad_T)):
        cross = problem.jac_xth_apply(x, thetas[t], lam)
        df_dx = df_dx - eta_y * cross
        Hlam = problem.hess_thth_apply(x, lam)
        lam = lam - eta_y * Hlam
    return df_dx


ALGORITHMS = {
    "R-HJFBiO":  hypergrad_HJFBiO,
    "RHGD-HINV": hypergrad_HINV,
    "RHGD-CG":   hypergrad_CG,
    "RHGD-NS":   hypergrad_NS,
    "RHGD-AD":   hypergrad_AD,
}


# -------- Run log + outer loop --------

@dataclass
class RunLog:
    algo: str
    label: str
    seed: int
    F:        List[float] = field(default_factory=list)
    g:        List[float] = field(default_factory=list)
    grad_norm:List[float] = field(default_factory=list)   # algorithm's own hypergradient norm
    true_grad_norm: List[float] = field(default_factory=list)   # consistent ground-truth: spectrally-clipped at the SAME w using closed-form M*(w)
    lam_min:  List[float] = field(default_factory=list)
    crashed:  bool = False

    def asdict(self):
        return {"algo": self.algo, "label": self.label, "seed": self.seed,
                "F": self.F, "g": self.g, "grad_norm": self.grad_norm,
                "true_grad_norm": self.true_grad_norm,
                "lam_min": self.lam_min, "crashed": self.crashed}


def _true_gradient_norm(problem: "Problem", manifold, x, lam_clip: float = 1e-3) -> float:
    """
    Compute a consistent 'true' Riemannian gradient norm at iterate x:
      1. solve closed-form theta* = M*(x)            (g is quadratic in theta)
      2. compute true Euclidean partial via implicit function theorem with
         spectrally-clipped Hessian inverse (same regularisation for everyone)
      3. project to manifold tangent and take Riemannian norm
    This is a pose-independent diagnostic — every algorithm gets the same
    metric, so plateau-at-bias for NS/AD and genuine descent for R-HJFBiO are
    compared on equal footing.
    """
    theta_star = problem.solve_inner_exact(x, ridge=1e-10)
    if theta_star is None or not np.all(np.isfinite(theta_star)):
        return float("nan")
    # form clipped Hessian inverse
    H = problem.hess_thth(x)
    H = 0.5 * (H + H.T)
    w_eig, V_eig = np.linalg.eigh(H)
    H_clip = (V_eig * np.maximum(w_eig, lam_clip)) @ V_eig.T
    grad_f_th = problem.grad_th_upper(x, theta_star)
    try:
        v = np.linalg.solve(H_clip, grad_f_th)
    except np.linalg.LinAlgError:
        return float("nan")
    cross = problem.jac_xth_apply(x, theta_star, v)
    df_dx = problem.grad_x_upper_p(x, theta_star) - cross
    if not np.all(np.isfinite(df_dx)):
        return float("nan")
    df_proj = manifold.proj(x, df_dx)
    df_R    = manifold.egrad2rgrad(x, df_proj)
    return float(manifold.norm(x, df_R))


def run_one(algo_name, hypergrad_fn, problem: Problem, manifold, label: str, seed: int, cfg: Dict) -> RunLog:
    rng = np.random.default_rng(seed)
    x = manifold.init(rng)
    theta = np.zeros(problem.d)
    log = RunLog(algo=algo_name, label=label, seed=seed)

    eta_x = cfg["eta_x"]; eta_y = cfg["eta_y"]
    T_inner = cfg["T_inner"]; T_outer = cfg["T_outer"]
    grad_clip = cfg.get("grad_clip", None)
    extra_kwargs = cfg.get("algo_kwargs", {}).get(algo_name, {})

    for k in range(T_outer):
        # Inner solve: T_inner steps of GD on theta. The iterative inner has an
        # implicit-regularisation effect that the closed-form solve does NOT —
        # important for these problems where ∇²_θ g is rank-deficient (PL regime).
        try:
            theta = inner_solver(problem, x, theta, T_inner, eta_y)
        except Exception:
            log.crashed = True; break
        if not np.all(np.isfinite(theta)):
            log.crashed = True; break

        try:
            dx_eu = hypergrad_fn(problem, x, theta, **extra_kwargs)
        except Exception:
            log.crashed = True; break
        if not np.all(np.isfinite(dx_eu)):
            log.crashed = True; break

        # Project to tangent and convert to Riemannian gradient
        dx_proj = manifold.proj(x, dx_eu)
        dx_R    = manifold.egrad2rgrad(x, dx_proj)
        gnorm   = manifold.norm(x, dx_R)
        if grad_clip is not None and gnorm > grad_clip:
            dx_R = dx_R * (grad_clip / gnorm)

        # Log
        F_val = problem.upper_loss(x, theta)
        g_val = problem.lower_loss(x, theta)
        H = problem.hess_thth(x)
        lam_min = float(np.linalg.eigvalsh(H).min())
        try:
            true_gn = _true_gradient_norm(problem, manifold, x, lam_clip=1e-3)
        except Exception:
            true_gn = float("nan")
        log.F.append(F_val); log.g.append(g_val); log.grad_norm.append(gnorm)
        log.true_grad_norm.append(true_gn)
        log.lam_min.append(lam_min)

        # Outer step
        x = manifold.retract(x, -eta_x * dx_R)
        if not np.all(np.isfinite(x)):
            log.crashed = True; break

    return log

# ============================================================================
# Main entry point (was run_hyperbolic.py)
# ============================================================================

CONFIG = {
    "n_dim":      4,                 # ambient dim of Poincaré ball
    "d":          8,                 # 8 polynomial-in-distance features
    "n_total":    206,               # 6 train + 200 val
    "n_tr":       6,                 # PL trigger: 6 < 8
    "noise":      0.05,
    "w_radius":   0.3,
    "x_radius":   0.7,

    "eta_x":      5e-2,
    "eta_y":      5e-2,              # stable now that features are bounded by D_SCALE
    "T_inner":    5,
    "T_outer":    150,
    "grad_clip":  1.0,

    "algo_kwargs": {
        "R-HJFBiO":  {"lam_clip": 1e-3, "fd_eps": 1e-5},
        "RHGD-HINV": {"ridge": 1e-8},
        "RHGD-CG":   {"cg_iters": 20, "tol": 1e-10},
        "RHGD-NS":   {"ns_iters": 5, "ns_alpha": 5e-2},
        "RHGD-AD":   {"ad_T": 5, "eta_y": 5e-2},
    },
    "data_seeds":   [0, 1, 2, 3, 4],   # 5 different data realisations
    "init_seeds":   [0, 1, 2],         # 3 inits per realisation
}


# Alias so original 'blh.foo' references resolve to this module
import sys as _sys
blh = _sys.modules[__name__]


def main(out_dir="results"):
    Path(out_dir).mkdir(exist_ok=True)
    out_path = Path(out_dir) / "runs_hyperbolic.jsonl"
    if out_path.exists(): out_path.unlink()

    print("=" * 72)
    print(f"Synthetic Poincaré ball B^{CONFIG['n_dim']}, polynomial-distance features (d={CONFIG['d']})")
    print(f"  data_seeds: {CONFIG['data_seeds']}, init_seeds: {CONFIG['init_seeds']}, T_outer: {CONFIG['T_outer']}")
    print(f"  PL: n_tr={CONFIG['n_tr']} < d={CONFIG['d']}  =>  Hessian rank <= {CONFIG['n_tr']}")
    print("=" * 72)

    manifold = HyperbolicManifold(n=CONFIG["n_dim"])
    t_start = time.time()
    n_done = 0
    n_total = len(CONFIG["data_seeds"]) * len(CONFIG["init_seeds"]) * len(ALGORITHMS)

    for ds in CONFIG["data_seeds"]:
        X, y, w_true, theta_true = blh.synthesize_data(
            n_dim=CONFIG["n_dim"], d=CONFIG["d"], n_total=CONFIG["n_total"],
            seed=ds, noise=CONFIG["noise"],
            w_radius=CONFIG["w_radius"], x_radius=CONFIG["x_radius"],
        )
        n_tr = CONFIG["n_tr"]
        X_tr, y_tr = X[:n_tr], y[:n_tr]
        X_val, y_val = X[n_tr:], y[n_tr:]
        problem = Problem(blh, train_data=(X_tr, y_tr), val_data=(X_val, y_val), d=CONFIG["d"])
        for seed in CONFIG["init_seeds"]:
            for algo_name, fn in ALGORITHMS.items():
                t0 = time.time()
                log = run_one(algo_name, fn, problem, manifold, label=f"ds{ds}", seed=seed, cfg=CONFIG)
                with open(out_path, "a") as fp:
                    fp.write(json.dumps(log.asdict()) + "\n")
                n_done += 1
                gn = np.median(log.grad_norm[-30:]) if log.grad_norm else float("nan")
                eta = (time.time() - t_start) * (n_total - n_done) / max(n_done, 1)
                print(f"  [{n_done:3d}/{n_total}] data={ds} init={seed} {algo_name:10s}  "
                      f"|grad|_med={gn:.2e}  [{time.time()-t0:.1f}s, eta {eta/60:.1f}min]", flush=True)

    print(f"\nTotal: {(time.time()-t_start)/60:.1f} min  ->  {out_path}")


if __name__ == "__main__":
    main()

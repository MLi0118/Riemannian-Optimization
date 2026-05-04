"""
BCI IV-2a Grassmann experiment — consolidated single-file version.

Bilevel formulation:
  min_{W in Gr(22, 8)}  F(W) := (1/n_val) Σ ( y_j - <φ(W; C_j), θ*(W)> )^2
  s.t. θ*(W) = argmin_{θ in R^8}  g(W, θ) = (1/n_tr) Σ ( y_i - <φ(W; C_i), θ> )^2

  φ(W; C) = trace-normalised power-sum features of B = W^T C W
            (Grassmann-invariant: φ(WQ; C) = φ(W; C) for Q in O(p))

Run:
  python3 bci_grassmann.py    # writes results/runs_grassmann.jsonl
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import json, sys, time

import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt


# Trial timing in BCI IV-2a (seconds): cue at t=2s, motor imagery 2-6s
EPOCH_START_SEC = 2.5    # 0.5s after cue
EPOCH_END_SEC = 4.5      # 2.5s after cue
FS = 250                 # sampling rate
N_EEG = 22               # EEG channels (drop 3 EOG)
BAND = (8.0, 30.0)       # bandpass Hz
RIDGE = 1e-3             # SPD ridge for numerical stability


@dataclass
class SubjectData:
    """Preprocessed binary-class trials for one subject."""
    subject_id: str
    covariances: np.ndarray   # (n_trials, 22, 22) SPD
    labels: np.ndarray        # (n_trials,) values in {-1, +1}: -1=left, +1=right


def _bandpass_sos(lo: float, hi: float, fs: int, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    return butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")


def _spd_ridge(C: np.ndarray, eps: float = RIDGE) -> np.ndarray:
    """Add small ridge to keep covariance well-conditioned (single-trial covariances often near-singular)."""
    return C + eps * np.trace(C) / C.shape[0] * np.eye(C.shape[0])


def load_subject(mat_path: str | Path, classes: Tuple[int, int] = (1, 2)) -> SubjectData:
    """
    Load and preprocess one BCI IV-2a subject.
    classes=(1,2) -> binary left-hand vs right-hand.
    """
    mat = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    runs = mat["data"]
    sos = _bandpass_sos(BAND[0], BAND[1], FS)

    cov_list, label_list = [], []
    for run in runs:
        if run is None or (hasattr(run, "size") and run.size == 0):
            continue
        if not hasattr(run, "trial") or not hasattr(run.trial, "__len__") or len(run.trial) == 0:
            continue

        X = np.asarray(run.X, dtype=np.float64)            # (T_total, 25)
        X = X[:, :N_EEG]                                    # drop EOG -> (T_total, 22)
        X = sosfiltfilt(sos, X, axis=0)                     # bandpass on each channel
        X = X - X.mean(axis=0, keepdims=True)               # remove DC

        trial_starts = np.asarray(run.trial, dtype=int)     # 1-indexed sample of each trial start
        ys = np.asarray(run.y, dtype=int)
        artifacts = np.asarray(run.artifacts, dtype=bool) if hasattr(run, "artifacts") else np.zeros_like(ys, dtype=bool)

        s0 = int(EPOCH_START_SEC * FS)
        s1 = int(EPOCH_END_SEC * FS)

        for ts, y, art in zip(trial_starts, ys, artifacts):
            if art or y not in classes:
                continue
            start = (ts - 1) + s0
            end   = (ts - 1) + s1
            if end > X.shape[0]:
                continue
            seg = X[start:end, :].T                          # (22, T_window)
            seg = seg - seg.mean(axis=1, keepdims=True)
            T = seg.shape[1]
            C = (seg @ seg.T) / T                            # (22,22) covariance
            C = _spd_ridge(C)
            C = C / np.trace(C) * C.shape[0]                  # normalize trace = n_channels (standard Riemannian-EEG)
            cov_list.append(C)
            label_list.append(-1 if y == classes[0] else +1)

    covariances = np.stack(cov_list, axis=0)
    labels = np.asarray(label_list, dtype=np.float64)
    sid = Path(mat_path).stem
    return SubjectData(subject_id=sid, covariances=covariances, labels=labels)


def load_all_subjects(uploads_dir: str | Path) -> List[SubjectData]:
    paths = sorted(Path(uploads_dir).glob("A0*T.mat"))
    return [load_subject(p) for p in paths]


def split_train_val(
    data: SubjectData,
    n_tr_per_class: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Random split: take n_tr_per_class trials per class for training, the rest for validation.
    Returns (C_tr, y_tr, C_val, y_val).
    """
    rng = np.random.default_rng(seed)
    C, y = data.covariances, data.labels
    idx_neg = np.where(y == -1)[0]
    idx_pos = np.where(y == +1)[0]
    rng.shuffle(idx_neg)
    rng.shuffle(idx_pos)
    tr_idx  = np.concatenate([idx_neg[:n_tr_per_class], idx_pos[:n_tr_per_class]])
    val_idx = np.concatenate([idx_neg[n_tr_per_class:], idx_pos[n_tr_per_class:]])
    rng.shuffle(tr_idx)
    rng.shuffle(val_idx)
    return C[tr_idx], y[tr_idx], C[val_idx], y[val_idx]

# ============================================================================
# manifolds_grassmann.py — Grassmann Gr(n, p) operations
# ============================================================================

def grassmann_proj(W: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Project ambient Z onto the horizontal tangent space at W:  xi = (I - W W^T) Z."""
    return Z - W @ (W.T @ Z)


def grassmann_retract(W: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """QR retraction (same as Stiefel) with sign-fix so retr(0) = W."""
    Q, R = np.linalg.qr(W + xi)
    s = np.sign(np.diag(R)); s[s == 0] = 1.0
    return Q * s


def grassmann_norm(W: np.ndarray, xi: np.ndarray) -> float:
    """Canonical Frobenius norm on the horizontal tangent space."""
    return float(np.linalg.norm(xi, "fro"))


def grassmann_init(n: int, p: int, rng: np.random.Generator) -> np.ndarray:
    """Random orthonormal frame; only the column-span matters."""
    A = rng.standard_normal((n, p))
    Q, _ = np.linalg.qr(A)
    return Q[:, :p]


def grassmann_distance(W1: np.ndarray, W2: np.ndarray) -> float:
    """Geodesic distance on Gr(n, p) via principal angles. Useful for diagnostics."""
    s = np.linalg.svd(W1.T @ W2, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return float(np.linalg.norm(np.arccos(s)))

# ============================================================================
# bilevel_grassmann.py — Grassmann bilevel objective
# ============================================================================

# ---------- features ----------

def _phi_batch(W: np.ndarray, C: np.ndarray, d: int):
    """
    Compute trace-normalised power-sum features for a batch of covariances.
    Returns:
       Phi : (n, d)                        feature matrix, phi_k = tr( (B/tau)^{k+1} )
       Bks_norm : list of d arrays (n,p,p)  (B/tau)^{k+1} for k=0..d-1  ['B', 'B^2', ..., 'B^d']
       CW : (n, n_ch, p)                    cached C_i W
       tau : (n,)                           trace(B)
       Bk_unnorm : list of d arrays (n,p,p) B^k for k=1..d  (the non-normalised matrix powers)
    """
    B  = np.einsum("ia, nij, jb -> nab", W, C, W)
    B  = 0.5 * (B + np.transpose(B, (0, 2, 1)))
    CW = np.einsum("nij, jb -> nib", C, W)
    n, p, _ = B.shape

    tau = np.einsum("nii->n", B)                         # (n,)
    Bk_unnorm = [B.copy()]                                # index k -> B^{k+1}, for k = 0 .. d
    cur = B.copy()
    for _ in range(1, d + 1):                             # need B^1 ... B^{d+1}
        cur = np.einsum("nij, njk -> nik", cur, B)
        Bk_unnorm.append(cur.copy())
    # Phi[:, k] = tr(B^{k+2}) / tau^{k+2}    (k = 0 ... d-1)  -- powers 2..d+1, skipping the trivial power 1
    Phi = np.zeros((n, d))
    for k in range(d):
        tr_pow = np.einsum("nii->n", Bk_unnorm[k + 1])    # tr(B^{k+2})
        Phi[:, k] = tr_pow / tau ** (k + 2)
    return Phi, Bk_unnorm, CW, tau


# ---------- losses ----------

def lower_loss(W, M, C_tr, y_tr, d):
    Phi, *_ = _phi_batch(W, C_tr, d)
    r = Phi @ M - y_tr
    return float(np.mean(r * r))


def upper_loss(W, M_star, C_val, y_val, d):
    Phi, *_ = _phi_batch(W, C_val, d)
    r = Phi @ M_star - y_val
    return float(np.mean(r * r))


# ---------- gradients ----------

def grad_M_lower(W, M, C_tr, y_tr, d):
    Phi, *_ = _phi_batch(W, C_tr, d)
    r = Phi @ M - y_tr
    return (2.0 / len(y_tr)) * (Phi.T @ r)


def grad_M_upper(W, M, C_val, y_val, d):
    Phi, *_ = _phi_batch(W, C_val, d)
    r = Phi @ M - y_val
    return (2.0 / len(y_val)) * (Phi.T @ r)


def _dphiM_dW(W, C, M, d):
    """
    Compute  D[M^T phi]/dW  for a batch (n, n_ch, p):
       phi_k = tr((B/tau)^{k+2}),   k = 0..d-1.
       d phi_k / dW = (2(k+2)/tau) * C W * ( (B/tau)^{k+1} - phi_k I_p ).
    Returns ((n, n_ch, p), (n, d)) -- the gradient and Phi.
    """
    Phi, Bk_unnorm, CW, tau = _phi_batch(W, C, d)
    n, p, _ = Bk_unnorm[0].shape
    n_ch = W.shape[0]
    out = np.zeros((n, n_ch, p))
    for k in range(d):                                     # power = k + 2
        # (B/tau)^{k+1} = Bk_unnorm[k] / tau^{k+1}
        BperTau_kp1 = Bk_unnorm[k] / (tau ** (k + 1))[:, None, None]
        phiI    = Phi[:, k][:, None, None] * np.eye(p)[None, :, :]
        bracket = BperTau_kp1 - phiI                       # (n, p, p)
        coef    = 2.0 * (k + 2) / tau                       # (n,)
        contrib = coef[:, None, None] * np.einsum("nij, njk -> nik", CW, bracket)
        out += M[k] * contrib
    return out, Phi


def grad_W_lower(W, M, C_tr, y_tr, d):
    DM, Phi = _dphiM_dW(W, C_tr, M, d)
    r = Phi @ M - y_tr
    return (2.0 / len(y_tr)) * np.einsum("n, nij -> ij", r, DM)


def grad_W_upper_partial(W, M, C_val, y_val, d):
    DM, Phi = _dphiM_dW(W, C_val, M, d)
    r = Phi @ M - y_val
    return (2.0 / len(y_val)) * np.einsum("n, nij -> ij", r, DM)


# ---------- Hessian / Jacobian ----------

def hessian_MM_lower(W, C_tr, d):
    Phi, *_ = _phi_batch(W, C_tr, d)
    return (2.0 / Phi.shape[0]) * (Phi.T @ Phi)


def hessian_MM_lower_apply(W, C_tr, V, d):
    Phi, *_ = _phi_batch(W, C_tr, d)
    return (2.0 / Phi.shape[0]) * (Phi.T @ (Phi @ V))


def jacobian_WM_lower_apply(W, M, C_tr, y_tr, V, d):
    """
    Apply  d/dW (d g / d M)  to V in R^d, returning shape (n_ch, p).

       d/dM_k g = (2/n) sum_i ( Phi_i M - y_i ) Phi_{i,k}
    => d/dW d/dM_k g = (2/n) sum_i [ d/dW(Phi_i M) * Phi_{i,k} + r_i * d/dW Phi_{i,k} ]
    Contracting over k with V:
       (2/n) sum_i [ DphiM_i * (Phi_i^T V) + r_i * DphiV_i ]
    where DphiM_i, DphiV_i are computed via _dphiM_dW with M and V respectively.
    """
    DphiM, Phi = _dphiM_dW(W, C_tr, M, d)
    DphiV, _   = _dphiM_dW(W, C_tr, V, d)
    r   = Phi @ M - y_tr
    PhiV = Phi @ V
    n = len(y_tr)
    return (2.0 / n) * (np.einsum("n, nij -> ij", PhiV, DphiM)
                        + np.einsum("n, nij -> ij", r,    DphiV))


def solve_inner_exact(W, C_tr, y_tr, d, ridge: float = 0.0):
    """Closed-form ridge regression on phi-features. Returns (M*, lam_min(Phi^T Phi))."""
    Phi, *_ = _phi_batch(W, C_tr, d)
    G = Phi.T @ Phi
    lam_min = float(np.linalg.eigvalsh(G).min())
    rhs = Phi.T @ y_tr
    M = np.linalg.solve(G + ridge * np.eye(d), rhs)
    return M, lam_min

# ============================================================================
# Manifold interface
# ============================================================================

class GrassmannManifold:
    def __init__(self, n: int, p: int):
        self.n = n; self.p = p
    def proj(self, W, Z):           return grassmann_proj(W, Z)
    def retract(self, W, xi):       return grassmann_retract(W, xi)
    def egrad2rgrad(self, W, g):    return g                       # canonical metric => identity
    def norm(self, W, xi):          return grassmann_norm(W, xi)
    def init(self, rng):            return grassmann_init(self.n, self.p, rng)


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
# Main entry point (was run_grassmann.py)
# ============================================================================

CONFIG = {
    "classes":         [1, 2],
    "p":               8,             # Gr(22, 8)
    "d":               8,             # 8 power-sum features (powers 2..9)
    "n_tr_per_class":  3,             # 6 trials < 8 features  => PL fires
    "eta_x":           5e-2,
    "eta_y":           5e-2,
    "T_inner":         5,
    "T_outer":         100,
    "grad_clip":       1.0,
    "algo_kwargs": {
        "R-HJFBiO":  {"lam_clip": 1e-3, "fd_eps": 1e-5},
        "RHGD-HINV": {"ridge": 1e-8},
        "RHGD-CG":   {"cg_iters": 20, "tol": 1e-10},
        "RHGD-NS":   {"ns_iters": 5, "ns_alpha": 5e-2},
        "RHGD-AD":   {"ad_T": 5, "eta_y": 5e-2},
    },
    "subjects": [f"A0{i}T" for i in range(1, 10)],
    "seeds":    [0, 1, 2],
}


def main(uploads_dir="/mnt/user-data/uploads", out_dir="results"):
    Path(out_dir).mkdir(exist_ok=True)
    out_path = Path(out_dir) / "runs_grassmann.jsonl"
    if out_path.exists(): out_path.unlink()

    print("=" * 72)
    print(f"BCI IV-2a, Grassmann manifold Gr(22, {CONFIG['p']})  ({CONFIG['d']}-dim features)")
    print(f"  Subjects: {CONFIG['subjects']}, seeds: {CONFIG['seeds']}, T_outer: {CONFIG['T_outer']}")
    print(f"  PL: n_tr={2*CONFIG['n_tr_per_class']} < d={CONFIG['d']}  =>  Hessian rank <= {2*CONFIG['n_tr_per_class']}")
    print("=" * 72)

    manifold = GrassmannManifold(n=22, p=CONFIG["p"])
    t_start = time.time()
    n_done = 0
    n_total = len(CONFIG["subjects"]) * len(CONFIG["seeds"]) * len(ALGORITHMS)

    for sid in CONFIG["subjects"]:
        sd = load_subject(Path(uploads_dir) / f"{sid}.mat", classes=tuple(CONFIG["classes"]))
        for seed in CONFIG["seeds"]:
            C_tr, y_tr, C_val, y_val = split_train_val(sd, CONFIG["n_tr_per_class"], seed=seed)
            problem = Problem(blg, train_data=(C_tr, y_tr), val_data=(C_val, y_val), d=CONFIG["d"])
            for algo_name, fn in ALGORITHMS.items():
                t0 = time.time()
                log = run_one(algo_name, fn, problem, manifold, label=sid, seed=seed, cfg=CONFIG)
                with open(out_path, "a") as fp:
                    fp.write(json.dumps(log.asdict()) + "\n")
                n_done += 1
                gn = np.median(log.grad_norm[-30:]) if log.grad_norm else float("nan")
                eta = (time.time() - t_start) * (n_total - n_done) / max(n_done, 1)
                print(f"  [{n_done:3d}/{n_total}] {sid} s={seed} {algo_name:10s}  "
                      f"|grad|_med={gn:.2e}  [{time.time()-t0:.1f}s, eta {eta/60:.1f}min]", flush=True)

    print(f"\nTotal: {(time.time()-t_start)/60:.1f} min  ->  {out_path}")


if __name__ == "__main__":
    main()

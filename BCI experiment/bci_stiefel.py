"""
BCI IV-2a Stiefel experiment — consolidated single-file version.

Bilevel formulation:
  min_{W in St(22, 4)}    F(W) := (1/n_val)  Σ ( y_j - <P_j(W), M*(W)>_F )^2
  s.t. M*(W) = argmin_{M in S^4_++}  g(W, M) = (1/n_tr) Σ ( y_i - <P_i(W), M>_F )^2
       P_i(W) = W^T C_i W

Run:
  python3 bci_stiefel.py    # writes results/runs_pl_regime.jsonl
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import json, sys, time

import numpy as np
import scipy.io as sio
import scipy.linalg as sla
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
# manifolds.py — Stiefel St(n, p) and SPD S^d_++ with affine-invariant metric
# ============================================================================

# ---------- generic helpers ----------

def sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def expm_sym(A: np.ndarray) -> np.ndarray:
    """Matrix exponential of a symmetric matrix via eigendecomposition (stable)."""
    w, V = np.linalg.eigh(sym(A))
    return (V * np.exp(w)) @ V.T


def logm_sym_pd(A: np.ndarray) -> np.ndarray:
    """Matrix logarithm of a symmetric positive-definite matrix."""
    w, V = np.linalg.eigh(sym(A))
    w = np.clip(w, 1e-12, None)
    return (V * np.log(w)) @ V.T


def sqrtm_sym_pd(A: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(sym(A))
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def inv_sqrtm_sym_pd(A: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(sym(A))
    w = np.clip(w, 1e-12, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


# ---------- Stiefel St(n, p) ----------

def stiefel_proj(W: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Project ambient Z onto tangent space at W."""
    return Z - W @ sym(W.T @ Z)


def stiefel_retract(W: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """QR retraction with sign correction (so that retr(0) = W)."""
    Q, R = np.linalg.qr(W + xi)
    # sign-correct so diagonal of R is positive (standard fix; ensures continuous retraction)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    return Q * s


def stiefel_norm(W: np.ndarray, xi: np.ndarray) -> float:
    """Frobenius norm on tangent at W (canonical-Euclidean Stiefel metric)."""
    return float(np.linalg.norm(xi, "fro"))


def stiefel_init(n: int, p: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((n, p))
    Q, _ = np.linalg.qr(A)
    return Q[:, :p]


# ---------- SPD S^d_++ with affine-invariant metric ----------

def spd_egrad2rgrad(M: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Convert Euclidean gradient G to Riemannian gradient at M."""
    return M @ sym(G) @ M


def spd_retract(M: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Exponential map (retraction) on S^d_++ at M with tangent xi."""
    Mh = sqrtm_sym_pd(M)
    Mhi = inv_sqrtm_sym_pd(M)
    inner = sym(Mhi @ xi @ Mhi)
    return Mh @ expm_sym(inner) @ Mh


def spd_inner(M: np.ndarray, U: np.ndarray, V: np.ndarray) -> float:
    Minv = np.linalg.inv(sym(M))
    return float(np.trace(Minv @ U @ Minv @ V))


def spd_norm(M: np.ndarray, xi: np.ndarray) -> float:
    return float(np.sqrt(max(spd_inner(M, xi, xi), 0.0)))


def spd_init(d: int) -> np.ndarray:
    return np.eye(d)


# ---------- spectral clipping (the R-HJFBiO core trick) ----------

def spectral_clip(H: np.ndarray, lam_clip: float) -> np.ndarray:
    """
    Replace eigenvalues of symmetric H below lam_clip with lam_clip.
    Used to regularize a singular/near-singular Hessian in the PL regime.
    """
    H = sym(H)
    w, V = np.linalg.eigh(H)
    w = np.maximum(w, lam_clip)
    return (V * w) @ V.T

# ============================================================================
# bilevel.py — F, g, gradients, Hessian/Jacobian, exact M*
# ============================================================================

# ---------- core scalar/matrix building blocks ----------

def _project_covariances(W: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Compute P_i = W^T C_i W for a batch C of shape (n, ch, ch). Returns (n, p, p)."""
    # einsum: 'ij, njk, kl -> nil'
    WC = np.einsum("ji,njk->nik", W, C)        # (n, p, ch)
    P  = np.einsum("nik,kl->nil", WC, W)       # (n, p, p)
    return 0.5 * (P + np.transpose(P, (0, 2, 1)))   # symmetrize


def _residuals(P: np.ndarray, M: np.ndarray, y: np.ndarray) -> np.ndarray:
    """r_i = <P_i, M>_F - y_i  (scalar per trial). Shape (n,)."""
    inner = np.einsum("nij,ij->n", P, M)
    return inner - y


# ---------- objective values ----------

def lower_loss(W: np.ndarray, M: np.ndarray, C_tr: np.ndarray, y_tr: np.ndarray) -> float:
    P = _project_covariances(W, C_tr)
    r = _residuals(P, M, y_tr)
    return float(np.mean(r * r))


def upper_loss(W: np.ndarray, M_star: np.ndarray, C_val: np.ndarray, y_val: np.ndarray) -> float:
    P = _project_covariances(W, C_val)
    r = _residuals(P, M_star, y_val)
    return float(np.mean(r * r))


# ---------- Euclidean gradients (input to Riemannian conversion) ----------

def grad_M_lower(W: np.ndarray, M: np.ndarray, C_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    """Euclidean gradient of g w.r.t. M (4x4)."""
    P = _project_covariances(W, C_tr)
    r = _residuals(P, M, y_tr)                          # (n,)
    G = (2.0 / len(y_tr)) * np.einsum("n,nij->ij", r, P)
    return sym(G)


def grad_W_lower(W: np.ndarray, M: np.ndarray, C_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    """Euclidean gradient of g w.r.t. W (ch, p).
       d/dW <W^T C_i W, M>_F = 2 C_i W M    (using symmetry of M and C_i)."""
    P = _project_covariances(W, C_tr)
    r = _residuals(P, M, y_tr)                          # (n,)
    n = len(y_tr)
    # sum_i 2 r_i C_i W M  /n  -> chain factor 2 from squared residual gives total 4/n
    CW = np.einsum("nij,jk->nik", C_tr, W)              # (n, ch, p)
    G  = (4.0 / n) * np.einsum("n,nik,kl->il", r, CW, M)
    return G


def grad_M_upper(W: np.ndarray, M: np.ndarray, C_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """Euclidean gradient of f (=upper objective) w.r.t. M, with M=M*(W)."""
    P = _project_covariances(W, C_val)
    r = _residuals(P, M, y_val)
    G = (2.0 / len(y_val)) * np.einsum("n,nij->ij", r, P)
    return sym(G)


def grad_W_upper_partial(W: np.ndarray, M: np.ndarray, C_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """Partial Euclidean gradient ∂f/∂W (treating M fixed) for the upper objective."""
    P = _project_covariances(W, C_val)
    r = _residuals(P, M, y_val)
    n = len(y_val)
    CW = np.einsum("nij,jk->nik", C_val, W)
    return (4.0 / n) * np.einsum("n,nik,kl->il", r, CW, M)


# ---------- Hessians and Jacobians (lower-level w.r.t. M ; cross W,M) ----------

def hessian_MM_lower(W: np.ndarray, C_tr: np.ndarray) -> np.ndarray:
    """
    Euclidean Hessian of g w.r.t. vech(M), as a (10x10) matrix.
    g is quadratic in M:   g = (1/n) sum_i ( <P_i, M>_F - y_i )^2
       =>  H = (2/n) sum_i  vech_lift(P_i) vech_lift(P_i)^T
    where vech_lift accounts for off-diagonals contributing twice in <P,M>_F.
    """
    P = _project_covariances(W, C_tr)              # (n, 4, 4)
    n, p, _ = P.shape
    # build "vech_lift" so that <P, M>_F = vech_lift(P)^T vech(M)
    iu = np.triu_indices(p)
    diag_mask = (iu[0] == iu[1]).astype(np.float64)
    # off-diagonal entries appear twice in <P,M>_F -> multiply by 2
    weights = diag_mask + 2.0 * (1.0 - diag_mask)  # 1 for diag, 2 for off-diag
    Phi = np.array([P[i][iu] * weights for i in range(n)])   # (n, 10)
    H = (2.0 / n) * Phi.T @ Phi                              # (10, 10)
    return H


def hessian_MM_lower_apply(W: np.ndarray, C_tr: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Apply Euclidean Hessian (∇^2_M g) to a symmetric matrix V (4x4):  H[V] = (2/n) sum_i <P_i, V>_F * P_i.
    Returns a 4x4 symmetric matrix.
    """
    P = _project_covariances(W, C_tr)              # (n, 4, 4)
    coeffs = np.einsum("nij,ij->n", P, V)           # (n,) inner products
    out = (2.0 / P.shape[0]) * np.einsum("n,nij->ij", coeffs, P)
    return sym(out)


def jacobian_WM_lower_apply(W: np.ndarray, M: np.ndarray, C_tr: np.ndarray,
                             y_tr: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Apply cross derivative  ∂_W (∂_M g)  to a 4x4 sym V: returns a (ch, p) matrix.
    
    ∂g/∂M_ab = (2/n) sum_i ( <P_i,M>_F - y_i ) [P_i]_ab
    Differentiating w.r.t. W and contracting with V (∈ Sym(4)):
      contraction = (2/n) sum_i [ d<P_i,M>_F/dW * <P_i, V>_F + (<P_i,M>-y_i) * d<P_i, V>_F/dW ]
                  = (2/n) sum_i [ 2 C_i W M * <P_i,V>_F + (<P_i,M>-y_i) * 2 C_i W V ]
    """
    P = _project_covariances(W, C_tr)
    n = P.shape[0]
    r = _residuals(P, M, y_tr)
    pv = np.einsum("nij,ij->n", P, V)                       # <P_i, V>_F
    CW = np.einsum("nij,jk->nik", C_tr, W)                  # (n, ch, p)
    term1 = np.einsum("n,nik,kl->il", pv, CW, M)
    term2 = np.einsum("n,nik,kl->il", r,  CW, V)
    return (4.0 / n) * (term1 + term2)


# ---------- inner solver: closed-form lower-level minimizer (when possible) ----------

def solve_inner_exact(W: np.ndarray, C_tr: np.ndarray, y_tr: np.ndarray,
                      ridge: float = 0.0) -> Tuple[np.ndarray, float]:
    """
    Closed-form Euclidean minimizer of g(W, M) over Sym(4):
      vech(M*) = ( Phi^T Phi + ridge I )^{-1} Phi^T y     (with vech_lift weighting)
    Returns (M*, lam_min(Phi^T Phi)) -- the latter exposes the PL trigger.
    """
    P = _project_covariances(W, C_tr)
    n, p, _ = P.shape
    iu = np.triu_indices(p)
    diag_mask = (iu[0] == iu[1]).astype(np.float64)
    weights = diag_mask + 2.0 * (1.0 - diag_mask)
    Phi = np.array([P[i][iu] * weights for i in range(n)])
    G = Phi.T @ Phi
    lam_min = float(np.linalg.eigvalsh(G).min())
    rhs = Phi.T @ y_tr
    sol = np.linalg.solve(G + ridge * np.eye(G.shape[0]), rhs)
    # rebuild M from vech (NOTE: regression coefficient is in vech(M), no weighting on M itself)
    M = np.zeros((p, p))
    M[iu] = sol
    M = M + M.T - np.diag(np.diag(M))
    return M, lam_min

# ============================================================================
# algorithms.py — five hypergradient methods
# ============================================================================

# Local alias so the original 'bl.foo' references work unchanged
import sys as _sys
bl = _sys.modules[__name__]

# ---------- inner solver (shared) ----------

def inner_solver(W: np.ndarray, M0: np.ndarray, C_tr: np.ndarray, y_tr: np.ndarray,
                  T: int, eta_y: float) -> np.ndarray:
    """T steps of Riemannian gradient descent on S^4_++ to approximate M*(W)."""
    M = M0.copy()
    for _ in range(T):
        gE = bl.grad_M_lower(W, M, C_tr, y_tr)
        gR = spd_egrad2rgrad(M, gE)            # affine-invariant Riemannian gradient
        M  = spd_retract(M, -eta_y * gR)
    return M


# ---------- helpers: vech / mat for 4x4 symmetric ----------

_P = 4
_IU = np.triu_indices(_P)
_DIAG = (_IU[0] == _IU[1]).astype(np.float64)
_W_VEC = _DIAG + 2.0 * (1.0 - _DIAG)            # weights for inner product <A,M>_F

def _vech_lift(A: np.ndarray) -> np.ndarray:
    """vech_lift(A) such that <A, M>_F = vech_lift(A)^T vech(M)."""
    return A[_IU] * _W_VEC

def _vech(A: np.ndarray) -> np.ndarray:
    return A[_IU]

def _vech_to_sym(v: np.ndarray) -> np.ndarray:
    M = np.zeros((_P, _P))
    M[_IU] = v
    return M + M.T - np.diag(np.diag(M))


def _euclidean_full_hessian(W: np.ndarray, C_tr: np.ndarray) -> np.ndarray:
    """10x10 Euclidean Hessian of g w.r.t. vech(M) (used by HINV)."""
    return bl.hessian_MM_lower(W, C_tr)


# ---------- 1. HINV : true Hessian inverse ----------

def hypergrad_HINV(W: np.ndarray, M: np.ndarray, C_tr, y_tr, C_val, y_val,
                   ridge: float = 1e-8) -> np.ndarray:
    """
    Form full Hessian; solve linear system (regularized to avoid breakage on truly singular H).
    The 'ridge' is a tiny floor to keep numerics finite -- with ridge=0 this would blow up
    in the PL regime, which IS the point of comparison.
    """
    grad_f = bl.grad_M_upper(W, M, C_val, y_val)
    H = _euclidean_full_hessian(W, C_tr)
    rhs = _vech(grad_f)
    # solve H v = rhs (in vech coords)
    try:
        v = np.linalg.solve(H + ridge * np.eye(H.shape[0]), rhs)
    except np.linalg.LinAlgError:
        v = np.linalg.lstsq(H + ridge * np.eye(H.shape[0]), rhs, rcond=None)[0]
    V = _vech_to_sym(v)
    cross_term = bl.jacobian_WM_lower_apply(W, M, C_tr, y_tr, V)
    df_dW = bl.grad_W_upper_partial(W, M, C_val, y_val)
    return df_dW - cross_term


# ---------- 2. CG : conjugate-gradient solve ----------

def hypergrad_CG(W: np.ndarray, M: np.ndarray, C_tr, y_tr, C_val, y_val,
                 cg_iters: int = 20, tol: float = 1e-10) -> np.ndarray:
    """Solve H v = grad_f via CG.  No regularization -- diverges/stalls when H singular."""
    grad_f_mat = bl.grad_M_upper(W, M, C_val, y_val)

    def Hv(V_mat: np.ndarray) -> np.ndarray:
        return bl.hessian_MM_lower_apply(W, C_tr, V_mat)

    # CG on vech-vector space (length 10); but operate in matrix form for clarity
    V = np.zeros_like(grad_f_mat)
    r = grad_f_mat - Hv(V)
    p = r.copy()
    rs_old = float(np.sum(r * r))
    for _ in range(cg_iters):
        Hp = Hv(p)
        denom = float(np.sum(p * Hp))
        if abs(denom) < 1e-30:
            # CG breakdown: pretend we have nothing useful and bail
            break
        alpha = rs_old / denom
        V = V + alpha * p
        r = r - alpha * Hp
        rs_new = float(np.sum(r * r))
        if rs_new < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    cross_term = bl.jacobian_WM_lower_apply(W, M, C_tr, y_tr, sym(V))
    df_dW = bl.grad_W_upper_partial(W, M, C_val, y_val)
    return df_dW - cross_term


# ---------- 3. NS : Neumann series ----------

def hypergrad_NS(W: np.ndarray, M: np.ndarray, C_tr, y_tr, C_val, y_val,
                 ns_iters: int = 5, ns_alpha: float = 0.05) -> np.ndarray:
    """
    Approximate H^{-1} g_f via   alpha sum_{k=0}^{T} (I - alpha H)^k g_f.
    Converges if eigenvalues of (I - alpha H) lie in (-1, 1), i.e. 0 < eig(H) < 2/alpha.
    Diverges otherwise (e.g. tiny eigenvalues -> contraction is weak; large eigs -> blowup).
    """
    grad_f_mat = bl.grad_M_upper(W, M, C_val, y_val)
    V_acc = np.zeros_like(grad_f_mat)
    cur = grad_f_mat.copy()
    for _ in range(ns_iters):
        V_acc = V_acc + cur
        Hv = bl.hessian_MM_lower_apply(W, C_tr, cur)
        cur = cur - ns_alpha * Hv
    V = ns_alpha * V_acc
    cross_term = bl.jacobian_WM_lower_apply(W, M, C_tr, y_tr, sym(V))
    df_dW = bl.grad_W_upper_partial(W, M, C_val, y_val)
    return df_dW - cross_term


# ---------- 4. AD : truncated backprop through inner steps ----------

def hypergrad_AD(W: np.ndarray, M_init: np.ndarray, C_tr, y_tr, C_val, y_val,
                 ad_T: int = 5, eta_y: float = 0.05) -> np.ndarray:
    """
    Compute total dF/dW by differentiating through T inner GD steps.
    Implementation: forward pass -> store M_t; backward pass via reverse-mode chain rule
    using the analytic blocks already derived.
    """
    Ms = [M_init.copy()]
    M = M_init.copy()
    for _ in range(ad_T):
        gE = bl.grad_M_lower(W, M, C_tr, y_tr)
        gR = spd_egrad2rgrad(M, gE)
        M = spd_retract(M, -eta_y * gR)
        Ms.append(M.copy())

    # Adjoint w.r.t. M_T  (gradient of upper f w.r.t. final M)
    lam_M = bl.grad_M_upper(W, Ms[-1], C_val, y_val)
    # Adjoint w.r.t. W : starts as partial of f
    df_dW = bl.grad_W_upper_partial(W, Ms[-1], C_val, y_val)

    # Reverse pass: at each step, M_{t+1} = M_t - eta_y * H_M_t * g_M_t (Euclidean approx for AD)
    # We use the *Euclidean* dynamics for the AD baseline (standard for HGD-AD in Han et al.):
    # M_{t+1} = M_t - eta_y * sym(grad_M_lower(W, M_t))
    # so dM_{t+1}/dM_t = I - eta_y * H_M_t  (sym block);  dM_{t+1}/dW depends on grad_W of grad_M.
    Ms_eu = [M_init.copy()]
    M = M_init.copy()
    for _ in range(ad_T):
        M = M - eta_y * bl.grad_M_lower(W, M, C_tr, y_tr)
        Ms_eu.append(M.copy())
    lam_M = bl.grad_M_upper(W, Ms_eu[-1], C_val, y_val)
    df_dW = bl.grad_W_upper_partial(W, Ms_eu[-1], C_val, y_val)

    for t in reversed(range(ad_T)):
        M_t = Ms_eu[t]
        # contribution to W gradient via dM_{t+1}/dW = -eta_y * d/dW grad_M g(W, M_t)
        cross = bl.jacobian_WM_lower_apply(W, M_t, C_tr, y_tr, sym(lam_M))
        df_dW = df_dW - eta_y * cross
        # update lam_M : lam_M <- (I - eta_y H_t) lam_M
        Hlam = bl.hessian_MM_lower_apply(W, C_tr, sym(lam_M))
        lam_M = lam_M - eta_y * Hlam
    return df_dW


# ---------- 5. R-HJFBiO (ours) : spectral clipping + finite-difference HVP ----------

def hypergrad_HJFBiO(W: np.ndarray, M: np.ndarray, C_tr, y_tr, C_val, y_val,
                     lam_clip: float = 1e-3, fd_eps: float = 1e-5,
                     cg_iters: int = 20, tol: float = 1e-10) -> np.ndarray:
    """
    Solve  H_clip v = grad_f    where H_clip = SpectralClip(H, lam_clip).
    Hessian-vector products use finite differences on the Euclidean gradient:
        H[V] ~ ( grad_M_lower(W, M+eps V) - grad_M_lower(W, M-eps V) ) / (2 eps)
    Combined with explicit spectral floor in the eigenbasis.

    For the 4x4 (10-dim vech) problem we form H once (cheap), clip its spectrum,
    and solve directly. This is the deterministic form of Algorithm 1.
    """
    grad_f_mat = bl.grad_M_upper(W, M, C_val, y_val)

    # Build Hessian via finite differences on grad_M (Hessian/Jacobian-FREE in spirit:
    # we never use the analytic 4-tensor; just gradient calls).
    H = np.zeros((10, 10))
    basis = []
    for k in range(10):
        ek = np.zeros(10); ek[k] = 1.0
        Vk = _vech_to_sym(ek)
        # Note: vech basis vectors correspond to symmetric-matrix bumps with weighted off-diagonals
        gp = bl.grad_M_lower(W, M + fd_eps * Vk, C_tr, y_tr)
        gm = bl.grad_M_lower(W, M - fd_eps * Vk, C_tr, y_tr)
        col = (gp - gm) / (2.0 * fd_eps)
        H[:, k] = _vech(col)
        basis.append(Vk)
    H = 0.5 * (H + H.T)

    # Spectral clipping
    H_clip = spectral_clip(H, lam_clip)

    rhs = _vech(grad_f_mat)
    v = np.linalg.solve(H_clip, rhs)
    V = _vech_to_sym(v)

    cross_term = bl.jacobian_WM_lower_apply(W, M, C_tr, y_tr, V)
    df_dW = bl.grad_W_upper_partial(W, M, C_val, y_val)
    return df_dW - cross_term

# ============================================================================
# experiment.py — outer-loop runner with logging
# ============================================================================

alg = _sys.modules[__name__]

ALGORITHMS = {
    "R-HJFBiO":   alg.hypergrad_HJFBiO,
    "RHGD-HINV":  alg.hypergrad_HINV,
    "RHGD-CG":    alg.hypergrad_CG,
    "RHGD-NS":    alg.hypergrad_NS,
    "RHGD-AD":    alg.hypergrad_AD,
}


@dataclass
class RunLog:
    algo: str
    subject: str
    seed: int
    F:        List[float] = field(default_factory=list)
    g:        List[float] = field(default_factory=list)
    grad_norm:List[float] = field(default_factory=list)
    lam_min:  List[float] = field(default_factory=list)
    val_acc:  List[float] = field(default_factory=list)
    crashed:  bool = False

    def asdict(self) -> Dict:
        return {
            "algo": self.algo, "subject": self.subject, "seed": self.seed,
            "F": self.F, "g": self.g, "grad_norm": self.grad_norm,
            "lam_min": self.lam_min, "val_acc": self.val_acc, "crashed": self.crashed,
        }


def _val_accuracy(W: np.ndarray, M: np.ndarray, C_val: np.ndarray, y_val: np.ndarray) -> float:
    P = np.einsum("ji,njk,kl->nil", W, C_val, W)
    P = 0.5 * (P + np.transpose(P, (0, 2, 1)))
    pred = np.einsum("nij,ij->n", P, M)
    return float(np.mean(np.sign(pred) == y_val))


def run_one(algo_name: str, hypergrad_fn: Callable,
             C_tr, y_tr, C_val, y_val,
             cfg: Dict, subject: str, seed: int) -> RunLog:
    rng = np.random.default_rng(seed)
    n_ch = C_tr.shape[1]
    p    = cfg["p"]

    W = stiefel_init(n_ch, p, rng)
    M = spd_init(p)

    log = RunLog(algo=algo_name, subject=subject, seed=seed)

    eta_x = cfg["eta_x"]
    eta_y = cfg["eta_y"]
    T_inner  = cfg["T_inner"]
    T_outer  = cfg["T_outer"]
    grad_clip = cfg.get("grad_clip", None)   # safety guard against numerical blowups

    extra_kwargs = cfg.get("algo_kwargs", {}).get(algo_name, {})

    for k in range(T_outer):
        # Inner solve: T_inner Riemannian GD steps on S^4_++
        try:
            M = alg.inner_solver(W, M, C_tr, y_tr, T=T_inner, eta_y=eta_y)
        except Exception:
            log.crashed = True
            break
        if not np.all(np.isfinite(M)):
            log.crashed = True
            break

        # Hypergradient (Euclidean partial dF/dW)
        try:
            dW_eu = hypergrad_fn(W, M, C_tr, y_tr, C_val, y_val, **extra_kwargs)
        except Exception:
            log.crashed = True
            break
        if not np.all(np.isfinite(dW_eu)):
            log.crashed = True
            break

        # Project to tangent space at W (Riemannian gradient on Stiefel)
        dW_R = stiefel_proj(W, dW_eu)
        gnorm = stiefel_norm(W, dW_R)
        if grad_clip is not None and gnorm > grad_clip:
            dW_R = dW_R * (grad_clip / gnorm)

        # Logging at this iterate
        F_val = bl.upper_loss(W, M, C_val, y_val)
        g_val = bl.lower_loss(W, M, C_tr, y_tr)
        H = bl.hessian_MM_lower(W, C_tr)
        lam_min = float(np.linalg.eigvalsh(H).min())
        acc = _val_accuracy(W, M, C_val, y_val)

        log.F.append(F_val); log.g.append(g_val); log.grad_norm.append(gnorm)
        log.lam_min.append(lam_min); log.val_acc.append(acc)

        # Outer step: retract on Stiefel
        W = stiefel_retract(W, -eta_x * dW_R)
        if not np.all(np.isfinite(W)):
            log.crashed = True
            break

    return log


def run_all(uploads_dir: str, cfg: Dict, subjects: List[str], seeds: List[int],
             out_dir: str | Path) -> List[RunLog]:
    from data import load_subject, split_train_val
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs: List[RunLog] = []

    for sid in subjects:
        path = Path(uploads_dir) / f"{sid}.mat"
        sd = load_subject(path, classes=tuple(cfg["classes"]))
        for seed in seeds:
            C_tr, y_tr, C_val, y_val = split_train_val(sd, cfg["n_tr_per_class"], seed=seed)
            for algo_name, fn in ALGORITHMS.items():
                print(f"[{sid} seed={seed} {algo_name}]", flush=True)
                log = run_one(algo_name, fn, C_tr, y_tr, C_val, y_val, cfg, sid, seed)
                logs.append(log)
                # save incrementally
                with open(out_dir / "all_runs.jsonl", "a") as fp:
                    fp.write(json.dumps(log.asdict()) + "\n")
    return logs

# ============================================================================
# config.py — pre-registered hyperparameters
# ============================================================================

CONFIG = {
    # --- problem ---
    "classes": [1, 2],                # 1=left hand, 2=right hand
    "p": 4,                           # Stiefel dimension St(22, 4)
    "n_tr_per_class": 3,              # PL trigger: 6 total trials < 10-dim vech(M)

    # --- shared optimization ---
    "eta_x": 5e-2,                    # Stiefel outer step size
    "eta_y": 5e-2,                    # SPD inner step size
    "T_inner": 5,                     # inner Riemannian-GD steps per outer iter
    "grad_clip": 1.0,                 # Stiefel-natural step bound

    # --- algorithm-specific (pre-registered, NOT tuned per dataset) ---
    "algo_kwargs": {
        "R-HJFBiO":  {"lam_clip": 1e-3, "fd_eps": 1e-5},
        "RHGD-HINV": {"ridge": 1e-8},
        "RHGD-CG":   {"cg_iters": 20, "tol": 1e-10},
        "RHGD-NS":   {"ns_iters": 5, "ns_alpha": 5e-2},
        "RHGD-AD":   {"ad_T": 5, "eta_y": 5e-2},
    },

    # --- experimental design ---
    "subjects": [f"A0{i}T" for i in range(1, 10)],   # A01T..A09T
    "seeds": [0, 1, 2],                               # 3 seeds per subject (15 trajectories per algo)
    "T_outer": 150,                                   # outer iterations
}

# ============================================================================
# Main entry point (was run_experiment.py)
# ============================================================================

OUT = Path("results")
OUT.mkdir(exist_ok=True)


def run_sweep(cfg, tag, uploads_dir="/mnt/user-data/uploads", subjects=None, seeds=None):
    """Run one full experiment under a given config and tag the output file."""
    subjects = subjects or cfg["subjects"]
    seeds = seeds or cfg["seeds"]
    out_path = OUT / f"runs_{tag}.jsonl"
    if out_path.exists():
        out_path.unlink()

    t_start = time.time()
    n_done = 0
    n_total = len(subjects) * len(seeds) * len(ALGORITHMS)

    for sid in subjects:
        sd = load_subject(Path(uploads_dir) / f"{sid}.mat", classes=tuple(cfg["classes"]))
        for seed in seeds:
            Ctr, ytr, Cval, yval = split_train_val(sd, cfg["n_tr_per_class"], seed=seed)
            for algo_name, fn in ALGORITHMS.items():
                t0 = time.time()
                log = run_one(algo_name, fn, Ctr, ytr, Cval, yval, cfg, sid, seed)
                with open(out_path, "a") as fp:
                    fp.write(json.dumps(log.asdict()) + "\n")
                n_done += 1
                eta = (time.time() - t_start) * (n_total - n_done) / max(n_done, 1)
                gn_med = np.median(log.grad_norm[-30:]) if log.grad_norm else float("nan")
                acc = max(log.val_acc) if log.val_acc else 0.0
                print(f"  [{n_done:3d}/{n_total}] {sid} s={seed} {algo_name:10s}  "
                      f"|grad|_med={gn_med:.2e}  acc_best={acc:.3f}  "
                      f"[{time.time()-t0:.1f}s, eta {eta/60:.1f}min]", flush=True)

    print(f"\n[{tag}] Total: {(time.time()-t_start)/60:.1f} min  ->  {out_path}")
    return out_path


if __name__ == "__main__":
    print("=" * 70)
    print("BCI IV-2a binary motor-imagery experiment")
    print(f"  Subjects: {CONFIG['subjects']}")
    print(f"  Seeds:    {CONFIG['seeds']}")
    print(f"  Outer iters: {CONFIG['T_outer']}")
    print(f"  PL regime: n_tr_per_class={CONFIG['n_tr_per_class']}  (Hessian rank <= 6 of 10)")
    print("=" * 70)

    cfg = dict(CONFIG)
    cfg["grad_clip"] = 1.0   # Stiefel-natural clip; preserves diagnostic in pre-clip log

    run_sweep(cfg, tag="pl_regime")

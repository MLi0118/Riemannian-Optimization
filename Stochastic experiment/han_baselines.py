"""
Han et al. (NeurIPS 2024) baselines.

Four hypergradient estimation strategies sharing a common outer loop:
  HINV: direct Riemannian Hessian inverse (well-defined only when H is
        non-singular, fails in our PL regime where rank H < d)
  CG  : conjugate gradient on H v = grad_y f
  NS  : truncated Neumann series approximation to H^{-1}
  AD  : automatic differentiation through S inner gradient steps

All four operate on the *Euclidean* lower-level here (M_y = R^d), so
parallel transport on M_y is identity and the analyses simplify
accordingly. The only Riemannian primitives used are the upper-level
manifold's tangent projection and retraction.

Note on the HINV behaviour in the PL regime: we use Tikhonov regularisation
H + reg*I with reg = 1e-8 to prevent an outright LinAlgError, which is the
same numerical floor used in the corresponding figure of the main paper
draft. This is too small to regularise the singular subspace meaningfully,
so HINV converges to a 1/reg-amplified estimate on the kernel; the
empirical effect (large hypergradient norms, divergent or oscillatory F)
is what we want to surface in the comparison.
"""
import torch


# ==========================================================================
#                   Hypergradient strategies (LS lower level)
# ==========================================================================
def _grad_y_full(fmap, x, y, X, y_target):
    """grad_y of (1/m) ||Phi y - y_target||^2 on a fixed batch."""
    Phi = fmap.compute(x, X)
    r = Phi @ y - y_target
    return (2.0 / Phi.shape[0]) * Phi.T @ r


def _grad_x_full(fmap, x, y, X, y_target):
    """grad_x via autograd."""
    x_v = x.detach().clone().requires_grad_(True)
    Phi = fmap.compute(x_v, X)
    r = Phi @ y - y_target
    loss = (r * r).mean()
    return torch.autograd.grad(loss, x_v)[0]


def _hess_yy(fmap, x, X):
    """Lower-level Hessian (constant in y for LS)."""
    Phi = fmap.compute(x, X)
    return (2.0 / Phi.shape[0]) * Phi.T @ Phi


# --------------------------------------------------------------------------
def hypergrad_HINV(fmap, x, y, X_tr, y_tr, X_val, y_val, reg=1e-8, **_):
    """grad_x f - (G_xy g)^dag [H^{-1} grad_y f]."""
    H = _hess_yy(fmap, x, X_tr)
    H_reg = H + reg * torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
    gyf = _grad_y_full(fmap, x, y, X_val, y_val)
    v = torch.linalg.solve(H_reg, gyf)
    return _hypergrad_assemble(fmap, x, y, v, X_tr, y_tr, X_val, y_val)


def hypergrad_CG(fmap, x, y, X_tr, y_tr, X_val, y_val,
                 T_cg=20, tol=1e-10, **_):
    """Conjugate gradient on H v = grad_y f, no regularisation."""
    H = _hess_yy(fmap, x, X_tr)
    gyf = _grad_y_full(fmap, x, y, X_val, y_val)
    v = torch.zeros_like(gyf)
    r = gyf - H @ v
    p = r.clone()
    rs_old = torch.dot(r, r)
    for _t in range(T_cg):
        Hp = H @ p
        denom = torch.dot(p, Hp)
        if torch.abs(denom) < tol:                         # CG breakdown
            break
        alpha = rs_old / denom
        v = v + alpha * p
        r = r - alpha * Hp
        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return _hypergrad_assemble(fmap, x, y, v, X_tr, y_tr, X_val, y_val)


def hypergrad_NS(fmap, x, y, X_tr, y_tr, X_val, y_val,
                 T_ns=5, alpha=5e-2, **_):
    """Truncated Neumann series: H^{-1} ~ alpha * sum_{k=0}^{T-1} (I - alpha H)^k."""
    H = _hess_yy(fmap, x, X_tr)
    gyf = _grad_y_full(fmap, x, y, X_val, y_val)
    M = torch.eye(H.shape[0], dtype=H.dtype, device=H.device) - alpha * H
    v = gyf.clone()
    acc = gyf.clone()
    for _ in range(T_ns - 1):
        v = M @ v
        acc = acc + v
    v = alpha * acc
    return _hypergrad_assemble(fmap, x, y, v, X_tr, y_tr, X_val, y_val)


def hypergrad_AD(fmap, x, y, X_tr, y_tr, X_val, y_val,
                 S_inner=5, eta_y=5e-2, **_):
    """Truncated automatic differentiation through S inner gradient steps.

    Run S Euclidean GD steps on g(x, ., D_tr) starting from y_init = y,
    then take grad_x of f at the resulting y_S. This is the same template
    as Han et al.'s AD strategy adapted to the Euclidean lower level.
    """
    x_v = x.detach().clone().requires_grad_(True)
    y_s = y.detach().clone()
    Phi_tr = fmap.compute(x_v, X_tr)
    for _ in range(S_inner):
        r = Phi_tr @ y_s - y_tr
        grad_y = (2.0 / Phi_tr.shape[0]) * Phi_tr.T @ r
        y_s = y_s - eta_y * grad_y
    Phi_val = fmap.compute(x_v, X_val)
    r_val = Phi_val @ y_s - y_val
    F_v = (r_val * r_val).mean()
    return torch.autograd.grad(F_v, x_v)[0]


# --------------------------------------------------------------------------
def _hypergrad_assemble(fmap, x, y, v, X_tr, y_tr, X_val, y_val):
    """Common: grad_x f(x, y) - (G_xy g)^dag [v] in the ambient space.

    For LS lower-level g = (1/m) ||Phi(x) y - y_tr||^2, the cross-derivative
    action (G_xy g)^dag [v] equals d/dx [ (2/m) Phi(x)^T r ]^T v evaluated
    on D_tr where r = Phi(x) y - y_tr. We compute this via autograd by
    noting it's the gradient of <grad_y g(x, y), v> w.r.t. x.
    """
    x_v = x.detach().clone().requires_grad_(True)

    # grad_x f at (x, y)
    Phi_val = fmap.compute(x_v, X_val)
    r_val = Phi_val @ y - y_val
    F = (r_val * r_val).mean()
    gxf = torch.autograd.grad(F, x_v, retain_graph=False, create_graph=False)[0]

    # (G_xy g)^dag [v] via grad_x of <grad_y g, v>
    x_v2 = x.detach().clone().requires_grad_(True)
    Phi_tr = fmap.compute(x_v2, X_tr)
    r_tr = Phi_tr @ y - y_tr
    gyg = (2.0 / Phi_tr.shape[0]) * Phi_tr.T @ r_tr
    inner = torch.dot(gyg, v)
    cross = torch.autograd.grad(inner, x_v2)[0]

    return gxf - cross


# ==========================================================================
#                          Outer loop driver
# ==========================================================================
class RHGD:
    """Riemannian hypergradient descent with one of the four strategies.

    Inner loop: S_inner Riemannian GD steps on g(x_t, .) (Euclidean here,
    so plain GD). Outer loop: assemble hypergradient via `strategy`,
    project to T_x M, retract.
    """

    def __init__(self, manifold, fmap, strategy,
                 eta_x=1e-3, eta_y=5e-2, S_inner=5,
                 strategy_kwargs=None):
        self.M = manifold
        self.fmap = fmap
        self.strategy = strategy
        self.eta_x = eta_x
        self.eta_y = eta_y
        self.S_inner = S_inner
        self.skw = strategy_kwargs or {}

    def step(self, x, y, X_tr, y_tr, X_val, y_val):
        # Inner: S_inner GD steps from current y (Euclidean lower level)
        with torch.no_grad():
            for _ in range(self.S_inner):
                grad_y = _grad_y_full(self.fmap, x, y, X_tr, y_tr)
                y = y - self.eta_y * grad_y

        # Hypergradient assembly
        h_amb = self.strategy(self.fmap, x, y, X_tr, y_tr, X_val, y_val,
                              **self.skw)

        # Outer: project + retract
        h_proj = self.M.project(x, h_amb)
        x_new = self.M.retract(x, -self.eta_x * h_proj)
        return x_new, y, h_proj


STRATEGIES = {
    "HINV": hypergrad_HINV,
    "CG":   hypergrad_CG,
    "NS":   hypergrad_NS,
    "AD":   hypergrad_AD,
}

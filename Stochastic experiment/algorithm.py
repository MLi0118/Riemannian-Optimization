"""
Stochastic Riemannian Hessian/Jacobian-Free Bilevel Optimisation
(SR-HJFBiO), Algorithm 2 of "Riemannian Bilevel Optimisation under the
Polyak-Lojasiewicz Condition".

Conventions:
- Upper-level x lives on a Riemannian manifold M_x (Stiefel / Grassmann /
  Poincare); lower-level y lives in R^d, so transports on M_y are identity.
- Cross-derivative term (G^2_xy g)^dag [v] is computed by FD probes in y
  (Eq. (7) of the paper).
- Hessian-vector with kernel-preserving spectral clipping (Eq. (3)+(9)).
  For a least-squares lower-level g(x, y) = (1/B) ||Phi(x) y - z||^2 the
  Hessian H = (2/B) Phi(x)^T Phi(x) is independent of y, so we materialise
  it on the H-batch (a 60x60 matrix) and clip its eigendecomposition.
  This is mathematically equivalent to the FD-then-clip prescription of
  the paper for quadratic g, and the "no Hessian inversion" guarantee is
  preserved -- we never form (Hess_y g)^{-1}, only S_{[mu, Lg]}.
"""
import torch


# --------------------------------------------------------------------------
def spectral_clip_kernel(eigvals, mu, Lg):
    """Kernel-preserving spectral clipping S_{[mu, Lg]} from Eq. (3).

    Eigenvalues with |lambda| < mu are mapped to 0 (preserving the kernel
    of the operator), eigenvalues with |lambda| in [mu, Lg] are kept,
    eigenvalues exceeding Lg are clipped to Lg with sign preserved.
    """
    abs_e = torch.abs(eigvals)
    mask = abs_e >= mu
    sign = torch.sign(eigvals)
    out = torch.zeros_like(eigvals)
    out[mask] = sign[mask] * torch.clamp(abs_e[mask], min=mu, max=Lg)
    return out


# --------------------------------------------------------------------------
class SRHJFBiO:
    """Single iteration of SR-HJFBiO; orchestrate from a driver script."""

    def __init__(self, manifold, fmap,
                 gamma, lam, tau,
                 mu_clip, Lg_clip,
                 delta_eps, rv):
        self.M = manifold
        self.fmap = fmap
        self.gamma = gamma
        self.lam = lam
        self.tau = tau
        self.mu = mu_clip
        self.Lg = Lg_clip
        self.delta_eps = delta_eps
        self.rv = rv

    # ---- gradient primitives ------------------------------------------
    def grad_y(self, x, y, X_b, yb_b):
        """Closed-form: (2/B) Phi^T (Phi y - yb)."""
        with torch.no_grad():
            Phi = self.fmap.compute(x, X_b)
            r = Phi @ y - yb_b
            return (2.0 / Phi.shape[0]) * Phi.T @ r

    def grad_x(self, x, y, X_b, yb_b):
        """Autograd-based grad_x of the same LS loss."""
        x_v = x.detach().clone().requires_grad_(True)
        Phi = self.fmap.compute(x_v, X_b)
        r = Phi @ y - yb_b
        loss = (r * r).mean()
        return torch.autograd.grad(loss, x_v)[0]

    def hess_yy(self, x, X_b):
        """Lower-level Hessian on the H-batch (constant in y for LS)."""
        with torch.no_grad():
            Phi = self.fmap.compute(x, X_b)
            return (2.0 / Phi.shape[0]) * Phi.T @ Phi

    # ---- one outer step ----------------------------------------------
    def step(self, x, y, v, X_tr, y_tr, X_val, y_val,
             B_g, B_f, B_J, B_H, gen, S_inner=1, T_v=1):

        n_tr, n_val = X_tr.shape[0], X_val.shape[0]

        # Sample independent fresh mini-batches.
        # For D_tr we sample WITHOUT replacement so B = m_tr really means
        # the full deterministic batch (under the same Generator state),
        # which is what the deterministic-limit comparison expects.
        # D_val is much larger than B, so with-replacement is fine.
        def perm_take(n, B):
            if B >= n:
                return torch.arange(n)
            return torch.randperm(n, generator=gen)[:B]

        # 1. Inner loop on y: S_inner stochastic GD steps on g(x_t, .)
        #    (Algorithm 2 of the paper allows S_inner > 1 in principle;
        #     we set S_inner = 1 for the strict single-loop variant and
        #     S_inner > 1 to match Han et al.'s baseline inner work.)
        for _ in range(S_inner):
            ig = perm_take(n_tr, B_g)
            u = self.grad_y(x, y, X_tr[ig], y_tr[ig])
            y = y - self.lam * u

        # Sample remaining batches now that y is at y^{S_inner}_t
        iJp = perm_take(n_tr, B_J)
        iJm = perm_take(n_tr, B_J)
        iH  = perm_take(n_tr, B_H)
        ifx = torch.randint(0, n_val, (B_f,), generator=gen)
        ify = torch.randint(0, n_val, (B_f,), generator=gen)

        # 2. Transport v from y_old to y_new; identity on Euclidean M_y
        v_p = v

        # 3. Jacobian-free cross-term via FD probes in y
        gp = self.grad_x(x, y + self.delta_eps * v_p, X_tr[iJp], y_tr[iJp])
        gm = self.grad_x(x, y - self.delta_eps * v_p, X_tr[iJm], y_tr[iJm])
        J_tilde = (gp - gm) / (2.0 * self.delta_eps)

        # 4. Upper-level direction: ambient gradient minus Jacobian-free term
        gxf = self.grad_x(x, y, X_val[ifx], y_val[ifx])
        w_amb = gxf - J_tilde

        # 5. Tangent projection + retraction on M_x
        w_proj = self.M.project(x, w_amb)
        x1 = self.M.retract(x, -self.gamma * w_proj)

        # 6. Auxiliary v update: T_v steps of the inner v iteration
        #    (Section 3 of the paper: v solves the spectrally-clipped
        #     linear system. Multiple v steps per outer step accelerate
        #     v_t's tracking of v_ref and are allowed by the analysis.)
        v_new = v_p
        for _ in range(T_v):
            H_B = self.hess_yy(x, X_tr[iH])
            eigvals, eigvecs = torch.linalg.eigh(H_B)
            eig_c = spectral_clip_kernel(eigvals, self.mu, self.Lg)
            Hclip_v = eigvecs @ (eig_c * (eigvecs.T @ v_new))
            gyf = self.grad_y(x, y, X_val[ify], y_val[ify])
            h_hat = Hclip_v - gyf
            v_new = v_new - self.tau * h_hat
            nv = torch.linalg.norm(v_new)
            if nv > self.rv:
                v_new = v_new * (self.rv / nv)

        return x1, y, v_new, w_proj


# --------------------------------------------------------------------------
def evaluate(x, fmap, X_tr, y_tr, X_val, y_val, mu_clip=1e-3, **kwargs):
    """Compute upper-level F(x) at the *spectrally-clipped* lower minimiser.

    For PL but rank-deficient quadratic g, the minimiser set is an affine
    subspace; the algorithm's auxiliary variable v_t tracks
    v*(x) = S_{[mu, Lg]}(H)^+ grad_y f(x, y), so the natural F-evaluation
    point matches: y*(x) = S_{[mu, Lg]}(H)^+ (Phi^T y_tr) / (-, +)-scaling
    consistent with the LS gradient. We use eigendecomposition with
    threshold mu_clip to keep this consistent across diagnostics and
    optimisation.
    """
    with torch.no_grad():
        Phi_tr = fmap.compute(x, X_tr)                # (m_tr, d)
        # Solve (Phi^T Phi) y = Phi^T y_tr via clipped pseudoinverse
        H = Phi_tr.T @ Phi_tr                         # (d, d), no 2/m factor needed
        rhs = Phi_tr.T @ y_tr                         # (d,)
        eigvals, eigvecs = torch.linalg.eigh(H)
        pinv = torch.zeros_like(eigvals)
        mask = eigvals.abs() > mu_clip
        pinv[mask] = 1.0 / eigvals[mask]
        y = eigvecs @ (pinv * (eigvecs.T @ rhs))

        Phi_val = fmap.compute(x, X_val)
        F = ((Phi_val @ y - y_val) ** 2).mean().item()
        return F, y


def riemannian_grad_norm(x, y, fmap, manifold, X_val, y_val, X_tr, y_tr,
                          delta_eps=1e-4):
    """||grad F(x)||_R using the implicit hypergradient formula.

    grad F = grad_x f - (G^2_xy g)^dag [H^dag grad_y f]. We compute this
    in closed form here (we have access to H since g is LS) for diagnostic
    purposes only -- the algorithm itself never inverts H.
    """
    with torch.no_grad():
        Phi_tr = fmap.compute(x, X_tr)
        H = (2.0 / Phi_tr.shape[0]) * Phi_tr.T @ Phi_tr
        eigvals, eigvecs = torch.linalg.eigh(H)
        # Pseudoinverse on the range cluster
        pinv = torch.zeros_like(eigvals)
        mask = eigvals.abs() > 1e-8
        pinv[mask] = 1.0 / eigvals[mask]
        H_pinv = eigvecs @ torch.diag(pinv) @ eigvecs.T

        Phi_val = fmap.compute(x, X_val)
        r_val = Phi_val @ y - y_val
        gyf = (2.0 / Phi_val.shape[0]) * Phi_val.T @ r_val
        v_star = H_pinv @ gyf

    # cross-derivative action via FD on a fresh autograd call
    x_v = x.detach().clone().requires_grad_(True)
    Phi = fmap.compute(x_v, X_tr)
    r = Phi @ (y + delta_eps * v_star) - y_tr
    Lp = (r * r).mean()
    gp = torch.autograd.grad(Lp, x_v, retain_graph=False)[0]

    x_v = x.detach().clone().requires_grad_(True)
    Phi = fmap.compute(x_v, X_tr)
    r = Phi @ (y - delta_eps * v_star) - y_tr
    Lm = (r * r).mean()
    gm = torch.autograd.grad(Lm, x_v, retain_graph=False)[0]

    cross = (gp - gm) / (2.0 * delta_eps)

    x_v = x.detach().clone().requires_grad_(True)
    Phi_val = fmap.compute(x_v, X_val)
    r_val = Phi_val @ y - y_val
    F = (r_val * r_val).mean()
    gxf = torch.autograd.grad(F, x_v)[0]

    grad_amb = gxf - cross
    grad_R = manifold.project(x, grad_amb)
    return manifold.norm(x, grad_R).item()

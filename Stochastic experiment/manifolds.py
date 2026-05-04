"""
Three Riemannian manifolds for the upper-level variable.

Each manifold exposes:
  - random(generator):  initial point on the manifold
  - project(x, z):      Euclidean ambient vector z -> Riemannian gradient at x
  - retract(x, v):      tangent vector v -> new point on the manifold
  - norm(x, v):         Riemannian norm of v in T_x M

For Stiefel and Grassmann we use the canonical (Frobenius) metric and
QR-based retraction; the only difference between them is the tangent-space
projector. For the Poincare ball we use the conformal metric and the
Mobius-addition retraction. Parallel transport is needed only on the
lower-level manifold M_y, which is Euclidean here, so no transport methods
are required on M_x.
"""
import torch


# --------------------------------------------------------------------------
class Stiefel:
    """St(n, r) = {W in R^{n x r} : W^T W = I_r} with Euclidean metric."""

    def __init__(self, n, r, device='cpu', dtype=torch.float64):
        self.n, self.r = n, r
        self.device = device
        self.dtype = dtype
        self.shape = (n, r)

    def random(self, gen):
        A = torch.randn(self.n, self.r, generator=gen,
                        device=self.device, dtype=self.dtype)
        Q, _ = torch.linalg.qr(A)
        return Q

    def project(self, W, Z):
        """Tangent projection: P_W(Z) = Z - W sym(W^T Z)."""
        sym = 0.5 * (W.T @ Z + Z.T @ W)
        return Z - W @ sym

    def retract(self, W, V):
        """QR retraction with sign correction so retract(W, 0) = W."""
        Q, R = torch.linalg.qr(W + V)
        d = torch.sign(torch.diagonal(R))
        d = torch.where(d == 0, torch.ones_like(d), d)
        return Q * d.unsqueeze(0)

    def norm(self, W, V):
        return torch.linalg.norm(V)  # Frobenius


# --------------------------------------------------------------------------
class Grassmann:
    """Gr(n, k) represented by a Stiefel rep modulo right O(k) action.

    The horizontal projector at W is P_W^h(Z) = Z - W W^T Z. This is the
    differential of the canonical quotient map and gives the Riemannian
    gradient of any O(k)-invariant function.
    """

    def __init__(self, n, k, device='cpu', dtype=torch.float64):
        self.n, self.k = n, k
        self.device = device
        self.dtype = dtype
        self.shape = (n, k)

    def random(self, gen):
        A = torch.randn(self.n, self.k, generator=gen,
                        device=self.device, dtype=self.dtype)
        Q, _ = torch.linalg.qr(A)
        return Q

    def project(self, W, Z):
        """Horizontal projector: P_W(Z) = Z - W W^T Z."""
        return Z - W @ (W.T @ Z)

    def retract(self, W, V):
        Q, R = torch.linalg.qr(W + V)
        d = torch.sign(torch.diagonal(R))
        d = torch.where(d == 0, torch.ones_like(d), d)
        return Q * d.unsqueeze(0)

    def norm(self, W, V):
        return torch.linalg.norm(V)


# --------------------------------------------------------------------------
class PoincareBall:
    """B^n = {z in R^n : ||z|| < 1} with the Poincare conformal metric.

    We use a buffered ball of radius 1-eps to avoid boundary instabilities,
    following the standard practice in Poincare-ball optimisation.
    Retraction uses Mobius addition with the tanh-scaled tangent vector,
    which is the closed-form exponential map on B^n.
    """

    def __init__(self, n, eps=0.2, device='cpu', dtype=torch.float64):
        self.n = n
        self.eps = eps
        self.boundary = 1.0 - eps
        self.device = device
        self.dtype = dtype
        self.shape = (n,)

    def random(self, gen, scale=0.3):
        z = torch.randn(self.n, generator=gen,
                        device=self.device, dtype=self.dtype)
        z = z / (torch.linalg.norm(z) + 1e-12)
        u = torch.rand(1, generator=gen, device=self.device,
                       dtype=self.dtype).item()
        return scale * u * z

    @staticmethod
    def _lambda(w):
        return 2.0 / (1.0 - torch.dot(w, w))

    def _project_to_ball(self, w):
        nrm = torch.linalg.norm(w)
        if nrm > self.boundary:
            w = w * (self.boundary / nrm)
        return w

    def project(self, w, z):
        """Convert ambient Euclidean gradient to Riemannian gradient.

        For the conformal metric g_w(u, v) = lambda(w)^2 <u, v>_E, the
        Riemannian gradient is grad_R F = (1 / lambda^2) grad_E F.
        """
        lam = self._lambda(w)
        return z / (lam * lam)

    def retract(self, w, v):
        """Exponential map (closed form on B^n)."""
        nv = torch.linalg.norm(v)
        if nv < 1e-12:
            return self._project_to_ball(w)
        lam = self._lambda(w)
        t = torch.tanh(lam * nv / 2.0)
        u = t * v / nv
        # Mobius addition w (+) u
        wu = torch.dot(w, u)
        ww = torch.dot(w, w)
        uu = torch.dot(u, u)
        num = (1.0 + 2.0 * wu + uu) * w + (1.0 - ww) * u
        den = 1.0 + 2.0 * wu + ww * uu
        return self._project_to_ball(num / den)

    def norm(self, w, v):
        return self._lambda(w) * torch.linalg.norm(v)

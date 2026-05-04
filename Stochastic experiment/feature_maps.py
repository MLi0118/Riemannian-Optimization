"""
Feature maps phi(x; .) : R^{n_in} -> R^d for each upper-level manifold.

The bilevel regression has the form
    g(x, y) = (1/B) sum_i (phi(x; x_i)^T y - y_i)^2,
    f(x, y) = (1/B') sum_j (phi(x; x_j)^T y - y_j)^2,
both least-squares quadratics in y. The lower-level Hessian H(x) =
(2/|D_tr|) Phi(x)^T Phi(x) has rank at most |D_tr|, so picking |D_tr| < d
forces dim ker H >= d - |D_tr| > 0 -- PL but not strongly convex.

Each feature map differs by how it produces the d features from x:
  Stiefel:    direct linear projection W^T x
  Grassmann:  random sketch of the projector P_W = W W^T  (O(k)-invariant)
  Hyperbolic: tanh distance polynomials w.r.t. an embedded copy of x
"""
import torch


# --------------------------------------------------------------------------
class StiefelFM:
    """phi(W; x) = W^T x in R^d,  W in St(n_in, d)."""

    def __init__(self, n_in, d):
        self.n_in = n_in
        self.d = d

    def compute(self, W, X_batch):
        # X_batch: (B, n_in), W: (n_in, d) -> (B, d)
        return X_batch @ W


# --------------------------------------------------------------------------
class GrassmannFM:
    """phi(W; x) = R^T W W^T x in R^d,  W in Gr(n_in, k).

    The fixed Gaussian R in R^{n_in x d} sketches the rank-k projector
    P_W = W W^T (which is the gauge-invariant object on Gr) into d features.
    Since P_W is O(k)-invariant, the whole map descends to the Grassmannian
    and is admissible as an upper-level loss on Gr(n_in, k).
    """

    def __init__(self, n_in, k, d, R):
        self.n_in = n_in
        self.k = k
        self.d = d
        self.R = R              # (n_in, d), fixed at construction

    def compute(self, W, X_batch):
        # phi = X (W W^T) R = (X W) (W^T R)
        WtR = W.T @ self.R                      # (k, d)
        XW = X_batch @ W                        # (B, k)
        return XW @ WtR                         # (B, d)


# --------------------------------------------------------------------------
class HyperbolicFM:
    """phi_k(w; x) = (a_k^T x) * exp(-d_H(w, p_k)^2 / (2 sigma^2)).

    The k-th feature is a fixed Euclidean projection of x along a_k,
    *gated* by a hyperbolic radial-basis weight: how close w is to the
    fixed anchor point p_k in the Poincare ball. So w controls a soft
    selection over d Euclidean directions, and the feature matrix
    Phi(w) = (X A^T) diag(rho(w)) is well-conditioned whenever the
    Gaussian directions A are.

    A simple Vandermonde design phi_k = tanh(d)^(k+1) was rejected
    because it has condition number exponential in d for any single x_i,
    so the lower-level Hessian becomes numerically singular even before
    the |D_tr| < d trigger.
    """

    def __init__(self, n_in, n_ball, d, A, P, sigma=1.0, eps=0.2):
        self.n_in = n_in
        self.n = n_ball
        self.d = d
        self.A = A              # (d, n_in)  Euclidean directions
        self.P = P              # (d, n_ball) anchor points in B^{n_ball}
        self.sigma = sigma
        self.eps = eps
        self.boundary = 1.0 - eps

    def _rho(self, w):
        """Hyperbolic-RBF anchor weights rho_k(w) in (0, 1]."""
        diffs = self.P - w.unsqueeze(0)
        d2 = (diffs * diffs).sum(dim=1)
        nw2 = torch.dot(w, w)
        np2 = (self.P * self.P).sum(dim=1)
        denom = torch.clamp((1.0 - nw2) * (1.0 - np2), min=1e-12)
        arg = torch.clamp(1.0 + 2.0 * d2 / denom, min=1.0 + 1e-12)
        d_H = torch.acosh(arg)
        return torch.exp(-d_H * d_H / (2.0 * self.sigma * self.sigma))

    def compute(self, w, X_batch):
        rho = self._rho(w)                      # (d,)
        XA = X_batch @ self.A.T                 # (B, d)
        return XA * rho.unsqueeze(0)            # (B, d)

"""
Inverse integral transforms (Laplace and Fourier) for real-valued functions

Functions
    laplinv(F, t, gamma, Nk=50, Nq=100)   inverse Laplace transform (cosine Fourier series + Euler-Knopp)
    euler_knopp(a, p=1, K=None)           Euler-Knopp acceleration of sum_k (-1)^k a_k
    fourealinv(F, x, Nj=50, Nq=100)       inverse Fourier transform of real-valued functions
    precompute_freqs(x, Nj=50, Nq=100)    frequencies at which F must be evaluated by fourealinv
    lgwt(N, a, b)                         Gauss-Legendre nodes and weights on [a, b]
    gaulagwt(N, alpha=0)                  Gauss-Laguerre nodes and weights (weight x^alpha e^-x)
    double_inversion(F, x, t, gamma, Nj=30, Nq=40, Nk=30, Nq_t=40, assume_real=False)
        double inverse transform f(x, t) = F^{-1}_{k->x} L^{-1}_{s->t} [ F(k, s) ],  x in R,  t > 0, for a real-valued f(x, t)

"""
import warnings
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import gammaln, roots_genlaguerre


# ---------------------------------------------------------------------------
# quadrature helpers
# ---------------------------------------------------------------------------
def lgwt(N, a, b):
    """Gauss-Legendre nodes x and weights w on [a, b] (1D arrays)."""
    t, w = leggauss(N)
    x = 0.5 * (b - a) * t + 0.5 * (b + a)
    return x, 0.5 * (b - a) * w


def gaulagwt(N, alpha=0.0):
    """Gauss-(generalised) Laguerre nodes and weights for the weight x^alpha exp(-x) on [0, inf)."""
    x, w = roots_genlaguerre(N, alpha)
    return x, w


def _eval(F, s):
    """Evaluate F on the array s; falls back to np.vectorize if F is not array-aware."""
    s = np.asarray(s, dtype=complex)
    try:
        out = np.asarray(F(s), dtype=complex)
        if out.shape == s.shape:
            return out
        if out.size == 1:                       # constant function
            return np.full(s.shape, out.item(), dtype=complex)
    except (TypeError, ValueError):
        pass
    return np.vectorize(F, otypes=[complex])(s)


# ---------------------------------------------------------------------------
# Euler-Knopp acceleration
# ---------------------------------------------------------------------------
def euler_knopp(a, p=1.0, K=None):
    """
    Euler-Knopp acceleration for the convergent series  S = sum_{k>=0} (-1)^k a_k.

    a : vector of length K, or a callable a(k) (then K, the number of terms, is required).
        An array of shape (..., K) is also accepted: the series index is the LAST axis and one sum is
        returned per leading index (used by the double-inversion test below).
    p : acceleration parameter (default 1)
    """
    if callable(a):
        if K is None:
            raise ValueError("If a is a function handle, K (number of terms) must be provided.")
        a_vals = np.array([a(k) for k in range(K)], dtype=float)
    else:
        a_vals = np.asarray(a, dtype=float)
        if a_vals.ndim == 0:
            a_vals = a_vals.reshape(1)
        K = a_vals.shape[-1]

    S = np.zeros(a_vals.shape[:-1])
    for k in range(K):
        j = np.arange(k + 1)
        # binomial coefficients via gammaln for numerical stability
        binom = np.exp(gammaln(k + 1) - gammaln(j + 1) - gammaln(k - j + 1))
        j_sum = np.sum((-1.0) ** j * binom * p ** j * a_vals[..., k - j], axis=-1)
        S = S + (-1.0) ** k * j_sum / (1.0 + p) ** (k + 1)
    return float(S) if S.ndim == 0 else S


# ---------------------------------------------------------------------------
# inverse Laplace transform
# ---------------------------------------------------------------------------
def laplinv(F, t, gamma, Nk=50, Nq=100):
    """
    Inverse Laplace transform by the cosine Fourier series method with Euler-Knopp acceleration.

    F     : function F(s) of the Laplace transform (vectorised over complex arrays)
    t     : time(s) at which to evaluate the inverse transform (t > 0)
    gamma : real part of the Bromwich contour, larger than the real parts of all singularities of F
    Nk    : number of series terms (default 50)
    Nq    : number of quadrature points (default 100)
    """
    t_arr = np.asarray(t, dtype=float)
    shape = t_arr.shape
    t_flat = t_arr.ravel()
    f = np.zeros(t_flat.size)

    omega0, w0 = lgwt(Nq, 0.0, np.pi / 2)            # nodes for the first term
    omegak, wk = lgwt(Nq, -np.pi / 2, np.pi / 2)     # nodes for the other terms
    kk = np.arange(1, Nk)[:, None]                   # series index k = 1..Nk-1 (column)

    for idx, ti in enumerate(t_flat):
        a = np.zeros(Nk)                             # alternating series coefficients

        # first term
        a[0] = np.sum(np.real(_eval(F, gamma + 1j * omega0 / ti)) * np.cos(omega0) * w0)

        # remaining terms, all k at once: array of shape (Nk-1, Nq)
        s = gamma + 1j * (omegak[None, :] + kk * np.pi) / ti
        vals = np.real(_eval(F, s))
        a[1:] = np.sum(vals * np.cos(omegak)[None, :] * wk[None, :], axis=1)

        f[idx] = 2.0 * np.exp(gamma * ti) / (np.pi * ti) * euler_knopp(a)

    f = f.reshape(shape)
    if f.size == 1:                                  # scalar in -> scalar out
        return float(f.ravel()[0])
    return f


# ---------------------------------------------------------------------------
# inverse Fourier transform of real-valued functions
# ---------------------------------------------------------------------------
def precompute_freqs(x, Nj=50, Nq=100):
    """
    Frequencies k at which F must be evaluated by fourealinv, as a row of length Nj*Nq per point x
    (a 2D array of shape (len(x), Nj*Nq) if x is an array).

    Reconstructed from fourealinv (precompute_freqs.m was not provided):
      x == 0          : first Nq entries are the Gauss-Laguerre nodes (half-space integral for f(0));
                        the rest repeats them (never used)
      0 < |x| < 0.1   : first block = Gauss-Legendre nodes on [0, pi/|x|] (positive k); then the blocks
                        j = 1..Nj-1 are k = (u + j pi)/x
      |x| >= 0.1      : blocks j = 0..Nj-1,  k = (u + j pi)/x,  u = Gauss-Legendre nodes on [0, pi]
    where k = (u + j pi)/x is NEGATIVE for x < 0; fourealinv's integrand (without a sign(x) in front of
    the imaginary part) is exactly right for those signed frequencies, because F(-k) = conj(F(k)).
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    u, _ = lgwt(Nq, 0.0, np.pi)
    out = np.zeros((x_arr.size, Nj * Nq))
    for i, xi in enumerate(x_arr):
        if xi == 0.0:
            k_lg, _ = gaulagwt(Nq, 0.0)
            out[i] = np.tile(k_lg, Nj)
            continue
        blocks = []
        if abs(xi) < 0.1:
            k0, _ = lgwt(Nq, 0.0, np.pi / abs(xi))
            blocks.append(k0)
            start_j = 1
        else:
            start_j = 0
        for j in range(start_j, Nj):
            blocks.append((u + j * np.pi) / xi)
        out[i] = np.concatenate(blocks)
    return out[0] if np.ndim(x) == 0 else out


def fourealinv(F, x, Nj=50, Nq=100):
    """
    Inverse Fourier transform for real-valued functions:  f(x) = 1/pi int_0^inf [Re F cos(kx) - Im F sin(kx)] dk
    evaluated segment by segment (alternating series of integrals over [j pi, (j+1) pi]/|x|).

    F  : function handle F(k) (vectorised), or a matrix of values of shape (Nx, Nj*Nq) evaluated at
         precompute_freqs(x, Nj, Nq)
    x  : evaluation point(s)
    Nj : number of segments (default 50)
    Nq : number of quadrature nodes per segment (default 100)
    """
    x_arr = np.asarray(x, dtype=float)
    shape = x_arr.shape
    x_flat = x_arr.ravel()
    f = np.zeros(x_flat.size)

    u, w_u = lgwt(Nq, 0.0, np.pi)

    for i, xi in enumerate(x_flat):
        if callable(F):
            k_row = precompute_freqs(xi, Nj, Nq)
            F_vals = _eval(F, k_row)
        else:
            F_vals = np.asarray(F)[i, :]

        if xi == 0.0:
            # Gauss-Laguerre quadrature for the half-space integral
            warnings.warn("Zero point detected; using half-space integral for f(0).")
            k_lg, w_lg = gaulagwt(Nq, 0.0)
            F0 = F_vals[:Nq]
            f[i] = (1.0 / np.pi) * np.sum(np.real(F0) * np.exp(k_lg) * w_lg)
            continue

        f_i = 0.0
        pos = 0

        if abs(xi) < 0.1:
            k0, w_k = lgwt(Nq, 0.0, np.pi / abs(xi))
            F_first = F_vals[:Nq]
            integrand = np.real(F_first) * np.cos(k0 * xi) - np.imag(F_first) * np.sin(k0 * xi)
            f_i = np.sum(integrand * w_k) / np.pi
            pos = Nq
            start_j = 1
        else:
            start_j = 0

        for j in range(start_j, Nj):
            idx_start = pos
            idx_end = pos + Nq
            if idx_end > Nj * Nq:
                break
            Fj = F_vals[idx_start:idx_end]
            integrand = np.real(Fj) * np.cos(u) - np.imag(Fj) * np.sin(u)
            f_i += np.sign(xi) * ((-1.0) ** j) * np.sum(integrand * w_u) / (np.pi * xi)
            pos = idx_end

        f[i] = f_i

    f = f.reshape(shape)
    return float(f.ravel()[0]) if (np.ndim(x) == 0) else f


# ---------------------------------------------------------------------------
# double inversion: F(k, s) -> f(x, t),  x in R,  t in R+
# ---------------------------------------------------------------------------
def _laplace_in_s(F, t, gamma, Nk, Nq, assume_real):
    """
    Return G(k) = L^{-1}_s[F(k, .)](t) as a function of a 1D array k (vectorised over k).

    assume_real=True : F(k, s) is the Laplace transform (in s) of a REAL function of t for every real k
                       (e.g. even-in-x problems); the real-time formula of laplinv is then used directly.
    assume_real=False: general case, G(k, t) complex.  F is split into the transforms of the real and
                       imaginary parts of the time function,
                           Fr = (F(k,s) + conj(F(k, conj s))) / 2,   Fi = (F(k,s) - conj(F(k, conj s))) / (2i),
                       each inverted with the real-time formula, G = L^-1[Fr] + i L^-1[Fi]  (2x the cost).
    """
    omega0, w0 = lgwt(Nq, 0.0, np.pi / 2)
    omegak, wk = lgwt(Nq, -np.pi / 2, np.pi / 2)
    s0 = gamma + 1j * omega0 / t                                                      # (Nq,)
    s_rest = gamma + 1j * (omegak[None, :] + np.arange(1, Nk)[:, None] * np.pi) / t   # (Nk-1, Nq)
    chunk = max(1, int(2e6 // (Nk * Nq)))                                             # memory control

    def series(Fp, k):
        a = np.zeros((k.size, Nk))
        a[:, 0] = np.sum(np.real(Fp(k[:, None], s0[None, :])) * np.cos(omega0) * w0, axis=1)
        vals = np.real(Fp(k[:, None, None], s_rest[None, :, :]))
        a[:, 1:] = np.sum(vals * np.cos(omegak) * wk, axis=2)
        return a

    if assume_real:
        parts = [(F, 1.0)]
    else:
        Fr = lambda k, s: 0.5 * (F(k, s) + np.conj(F(k, np.conj(s))))
        Fi = lambda k, s: (F(k, s) - np.conj(F(k, np.conj(s)))) / 2j
        parts = [(Fr, 1.0), (Fi, 1j)]

    def G(k):
        k = np.asarray(k, dtype=float).ravel()
        out = np.zeros(k.size, dtype=complex)
        for lo in range(0, k.size, chunk):
            kc = k[lo:lo + chunk]
            for Fp, phase in parts:
                out[lo:lo + chunk] += phase * 2.0 * np.exp(gamma * t) / (np.pi * t) * euler_knopp(series(Fp, kc))
        return out.real if assume_real else out

    return G


def double_inversion(F, x, t, gamma, Nj=30, Nq=40, Nk=30, Nq_t=40, assume_real=False):
    """
    Double inverse transform  f(x, t) = F^{-1}_{k->x} L^{-1}_{s->t} [ F(k, s) ],  x in R,  t > 0,
    for a real-valued f(x, t).

    F           : F(k, s), array-aware with broadcasting between k and s (k real, s complex)
    x, t        : evaluation points (scalars or arrays; t > 0)
    gamma       : real part of the Bromwich contour (> real parts of all singularities of F in s)
    Nj, Nq      : number of segments / quadrature nodes per segment of fourealinv (spatial inversion)
    Nk, Nq_t    : number of series terms / quadrature points of the Laplace inversion
    assume_real : True if F(k, s) is the Laplace transform of a real function of t for every real k
                  (faster, 2x); False (default) handles complex-in-k transforms, e.g. shifted profiles

    Returns an array of shape np.shape(x) + np.shape(t) (a float if both are scalars).
    Cost: for every t and every x the Laplace inversion is evaluated at Nj*Nq frequencies.
    """
    x_arr = np.asarray(x, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    if np.any(t_arr <= 0):
        raise ValueError("t must be positive")
    x_flat, t_flat = x_arr.ravel(), t_arr.ravel()

    f = np.zeros((x_flat.size, t_flat.size))
    for i, ti in enumerate(t_flat):
        G = _laplace_in_s(F, ti, gamma, Nk, Nq_t, assume_real)
        f[:, i] = np.atleast_1d(fourealinv(G, x_flat, Nj, Nq))

    f = f.reshape(x_arr.shape + t_arr.shape)
    return float(f) if f.ndim == 0 else f


# ---------------------------------------------------------------------------
# self-tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    warnings.simplefilter("ignore")

    # Euler-Knopp: sum (-1)^k/(k+1) = ln 2 (converges only like 1/K without acceleration)
    S = euler_knopp(lambda k: 1.0 / (k + 1), K=40)
    print(f"euler_knopp  : {S:.12f}  (ln 2 = {np.log(2):.12f})  err {abs(S - np.log(2)):.2e}")

    # inverse Laplace
    t = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    for name, F, ex, g in [("1/(s+1)    -> exp(-t)", lambda s: 1 / (s + 1), np.exp(-t), 0.5),
                           ("1/(s^2+1)  -> sin(t)", lambda s: 1 / (s ** 2 + 1), np.sin(t), 0.5),
                           ("1/s^2      -> t", lambda s: 1 / s ** 2, t, 0.5)]:
        err = np.max(np.abs(laplinv(F, t, g) - ex))
        print(f"laplinv {name:22s}: max error {err:.2e}")

    # inverse Fourier (real f): even, odd (imaginary F) and shifted (complex F) Gaussians,
    # points on both sides of 0, inside |x| < 0.1, and x = 0
    x = np.array([-3.0, -1.2, -0.5, -0.05, 0.0, 0.03, 0.4, 1.0, 2.5])
    c = np.sqrt(2 * np.pi)
    tests = [("exp(-x^2/2)", lambda k: c * np.exp(-k ** 2 / 2), np.exp(-x ** 2 / 2)),
             ("x exp(-x^2/2)", lambda k: -1j * k * c * np.exp(-k ** 2 / 2), x * np.exp(-x ** 2 / 2)),
             ("exp(-(x-1)^2/2)", lambda k: c * np.exp(-k ** 2 / 2) * np.exp(-1j * k),
              np.exp(-(x - 1) ** 2 / 2))]
    for name, F, ex in tests:
        f = fourealinv(F, x)
        print(f"fourealinv {name:16s}: max error {np.max(np.abs(f - ex)):.2e}  "
              f"(excluding x=0: {np.max(np.abs(np.delete(f - ex, 4))):.2e})")

    # matrix input (precomputed frequencies) gives the same result
    K = precompute_freqs(x)
    Fmat = _eval(tests[2][1], K)
    print("matrix input vs handle input:", np.max(np.abs(fourealinv(Fmat, x) - fourealinv(tests[2][1], x))))

    # -----------------------------------------------------------------------
    # DOUBLE INVERSION: Laplace in s (time) and Fourier in k (space)
    #   f(r,t) = exp(-r^2/(2 sigma^2)) / (2 pi sigma^2) * lambda * exp(-lambda t)      (as written)
    #   F(k,s) = exp(-0.5 sigma^2 k^2) * lambda / (s + lambda)
    # The pair as written is the 2D radial (Hankel) one, whose Gaussian normalisation is 1/(2 pi sigma^2).
    # fourealinv is the 1D inverse Fourier transform, so it returns the 1D normalisation
    # 1/sqrt(2 pi sigma^2): f_1d = f_written * sqrt(2 pi) * sigma.  Both references are printed.
    # -----------------------------------------------------------------------
    sig, lam, gam = 0.5, 2.0, 1.0
    F2 = lambda k, s: np.exp(-0.5 * sig ** 2 * k ** 2) * lam / (s + lam)
    f_written = lambda r, t: np.exp(-r ** 2 / (2 * sig ** 2)) / (2 * np.pi * sig ** 2) * lam * np.exp(-lam * t)
    f_1d = lambda r, t: np.exp(-r ** 2 / (2 * sig ** 2)) / np.sqrt(2 * np.pi * sig ** 2) * lam * np.exp(-lam * t)

    r = np.array([-1.2, -0.4, 0.0, 0.05, 0.3, 0.8, 1.5])
    tt_ = np.array([0.2, 0.5, 1.0, 2.0])
    print("\ndouble inversion (wrapper), even Gaussian, F real in s:")
    f_num = double_inversion(F2, r, tt_, gam)                       # shape (len(r), len(tt_))
    f_fast = double_inversion(F2, r, tt_, gam, assume_real=True)
    ref = f_1d(r[:, None], tt_[None, :])
    print(f"  shape {f_num.shape};  max |f_num - f_1d| = {np.abs(f_num - ref).max():.2e};  "
          f"assume_real=True differs by {np.abs(f_fast - f_num).max():.1e}")
    print(f"  f_num / f_written = {np.mean(f_num / f_written(r[:, None], tt_[None, :])):.6f}  "
          f"(sqrt(2 pi) sigma = {np.sqrt(2 * np.pi) * sig:.6f})")

    # complex-in-k case: Gaussian shifted by x0 (not even in x); needs assume_real=False
    x0 = 1.0
    F3 = lambda k, s: np.exp(-0.5 * sig ** 2 * k ** 2) * np.exp(-1j * k * x0) * lam / (s + lam)
    ref3 = f_1d(r[:, None] - x0, tt_[None, :])
    f3 = double_inversion(F3, r, tt_, gam)
    print(f"double inversion, shifted Gaussian (complex F): max error {np.abs(f3 - ref3).max():.2e}")

    # scalar x and t -> float
    print("scalar call:", double_inversion(F2, 0.3, 0.5, gam), "vs", f_1d(0.3, 0.5))
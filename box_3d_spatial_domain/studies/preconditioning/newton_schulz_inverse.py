"""
Newton-Schulz iteration for the inverse of the BoxTransport operator, started from the rank-1 Kronecker inverse.

    X_0     = A_1^{-1},   A_1 = A rounded to TT-rank 1 (a Kronecker product, inverted core by core, exact),
    X_{k+1} = 2 X_k - X_k A X_k,          E_k = I - A X_k,   E_{k+1} = E_k^2,

so the iteration converges (quadratically) iff rho(I - A X_0) < 1. Both TTM-TTM products per step by AMEn
(torchtt.amen_mm), then X rounded at EPS (max rank RMAX). Before iterating, rho(I - A X_0) is estimated by power
iteration on random TT vectors (matvecs, rank capped at R_POWER: an estimate). After K_ITER steps: A @ X_K by amen_mm again, ||A X - I||_F / ||I||_F,
and the action on a random vector, ||A X b - b|| / ||b||.

Problem: N = 64 cells per axis, p = 1 (128 dofs per axis), box [0, 10]^3, N_mu = N_phi = 16, G = 4 groups, N_t = 128
backward-Euler steps (time-dependent), isotropic scattering with down-scatter, c ~ 0.5 per group.
TTM cores in space and time are 128 x 128: a rank-r core has r^2 * 16384 entries (r = 40: 0.2 GB), hence RMAX.

    EPS=1e-6 RMAX=20 K_ITER=10 R_POWER=8 python newton_schulz_inverse.py
"""
import os
import time

import numpy as np
import torch as tn
import torchtt as tntt
from tt_transport_box.box import BoxTransport

tn.set_default_dtype(tn.float64)
EPS = float(os.environ.get("EPS", 1e-6))
RMAX = int(os.environ.get("RMAX", 20))
K_ITER = int(os.environ.get("K_ITER", 10))
N_POWER = int(os.environ.get("N_POWER", 30))
R_POWER = int(os.environ.get("R_POWER", 8))       # rank cap of the power-iteration vectors
NAMES = ["x", "y", "z", "mu", "phi", "g", "t"]


def rel_dist_to_identity(P, I, nI):
    return float((P - I).norm()) / nI


def apply(X, v, rmax=R_POWER):
    """Exact matvec, then rounding with a rank cap (a random vector has growing ranks under E^k)."""
    return (X @ v).round(1e-10, rmax)


def spectral_radius(A, X, n_iter, seed=0):
    """Power iteration on E = I - A X with exact matvecs + rounding; returns the last Rayleigh-type ratio."""
    tn.manual_seed(seed)
    v = tntt.randn(A.N, [1] + [2] * (len(A.N) - 1) + [1])
    v = v / v.norm()
    ratios = []
    for _ in range(n_iter):
        w = (v - apply(A, apply(X, v))).round(1e-10, R_POWER)
        nw = float(w.norm())
        ratios.append(nw)
        v = w / nw
    return ratios


def main():
    G = 4
    sigma_t = np.array([0.5, 1.0, 1.5, 2.0])
    S0 = np.zeros((G, G))                          # S0[g, g'] = sigma_{s,0,g -> g'} (row = incoming)
    for g in range(G):
        S0[g, g] = 0.3 * sigma_t[g]
        if g + 1 < G:
            S0[g, g + 1] = 0.2 * sigma_t[g]
    model = BoxTransport(N_x=64, N_y=64, N_z=64, p=1, L_x=10, L_y=10, L_z=10, N_mu=16, N_phi=16,
                         sigma_t=sigma_t, sigma_s=[S0], v=np.array([10.0, 5.0, 2.0, 1.0]), N_t=128, T=1.0,
                         steady_state=False, N_recon=2)
    t0 = time.time()
    model.construct_tt_operator()
    A = model.A
    print(f"A: ranks {list(A.R)}, mode sizes {list(A.N)}, built in {time.time() - t0:.1f} s", flush=True)
    print(f"EPS = {EPS}, RMAX = {RMAX}, K_ITER = {K_ITER}", flush=True)

    I = tntt.eye(A.N)
    nI = float(I.norm())

    # ---- X_0: rank-1 rounding of A, inverted core by core ---------------------------------------
    A_1 = A.round(1e-14, rmax=1)
    print(f"||A - A_1||_F / ||A||_F = {float((A - A_1).norm()) / float(A.norm()):.3e}")
    X_cores = []
    for name, core in zip(NAMES, A_1.cores):
        M = core[0, :, :, 0]
        s = tn.linalg.svdvals(M)
        print(f"  core {name:3s}: {M.shape[0]:4d} x {M.shape[1]:4d}, cond = {float(s[0] / s[-1]):.3e}")
        X_cores.append(tn.linalg.inv(M)[None, :, :, None])
    X = tntt.TT(X_cores)
    print(f"||A_1 X_0 - I||_F / ||I||_F = {rel_dist_to_identity(A_1 @ X, I, nI):.3e}  (check)", flush=True)

    # ---- convergence condition: rho(I - A X_0) -----------------------------------------------
    t0 = time.time()
    rat = spectral_radius(A, X, N_POWER)
    print(f"power iteration on I - A X_0 ({N_POWER} its, {time.time() - t0:.1f} s): "
          f"||E^k v|| / ||E^(k-1) v|| last 5 = {[f'{r:.4f}' for r in rat[-5:]]}  -> rho ~ {rat[-1]:.4f}"
          f"{'  (< 1: Newton-Schulz converges)' if rat[-1] < 1 else '  (>= 1: Newton-Schulz diverges)'}",
          flush=True)

    # ---- Newton-Schulz ---------------------------------------------------------------------------
    AX = tntt.amen_mm(A, X, eps=EPS, rmax=RMAX)
    res = rel_dist_to_identity(AX, I, nI)
    print(f"\n k  {'||AX - I||/||I||':>17s}  {'max rank X':>10s}  {'time [s]':>9s}")
    print(f" 0  {res:17.3e}  {max(X.R):10d}  {0.0:9.1f}", flush=True)
    for k in range(1, K_ITER + 1):
        t0 = time.time()
        XAX = tntt.amen_mm(X, AX, eps=EPS, rmax=RMAX, X0=X)
        X = (2 * X - XAX).round(EPS, RMAX)
        AX = tntt.amen_mm(A, X, eps=EPS, rmax=RMAX)
        res = rel_dist_to_identity(AX, I, nI)
        print(f"{k:2d}  {res:17.3e}  {max(X.R):10d}  {time.time() - t0:9.1f}   ranks X {list(X.R)}", flush=True)
        if not np.isfinite(res) or res > 1e3:
            print("diverged, stopping")
            break

    # ---- final check: A @ A^{-1} -------------------------------------------------------------
    t0 = time.time()
    P = tntt.amen_mm(A, X, eps=EPS, rmax=RMAX)
    print(f"\nA @ X (amen_mm, {time.time() - t0:.1f} s): ranks {list(P.R)}, "
          f"||A X - I||_F / ||I||_F = {rel_dist_to_identity(P, I, nI):.3e}")
    tn.manual_seed(1)
    b = tntt.randn(A.N, [1] + [2] * (len(A.N) - 1) + [1])
    print(f"random vector: ||A X b - b|| / ||b|| = {float((apply(A, apply(X, b)) - b).norm()) / float(b.norm()):.3e}")


if __name__ == "__main__":
    main()

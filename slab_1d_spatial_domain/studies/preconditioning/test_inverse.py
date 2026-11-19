"""
Exponential-sum approximate inverse of the streaming + collision operator (Grasedyck, Computing 72 (2004)), in TT.

Operator (steady, no scattering), 4D TT-matrix over (z-dof, mu, phi, g), from SlabTransport:
    L = M (x) I (x) I (x) diag(sigma_t)  +  mu-streaming (upwind DG),
diagonal in (mu, phi, g):  L = sum_{n,g} L_{n,g} (x) e_n e_n^T (x) I (x) e_g e_g^T,
    M^{-1} L_{n,g} = sigma_g I + mu_n K_n          (K_n: upwind DG streaming for the sign of mu_n).
For each (n, g) this is a Kronecker sum (z) + (g), so with the Stenger nodes t_j, weights w_j (j = -k..k) and the
scale c = alpha / lambda_min (alpha = 2: Grasedyck's choice, spectrum of c M^{-1} L in Re >= 2)
    L^{-1} ~ A_inv = sum_j c w_j  sum_n  [e^{-c t_j mu_n K_n} M^{-1}] (x) e_n e_n^T (x) I (x) diag(e^{-c t_j sigma_g}).
The (mu, phi, g) cores are diagonal, so their exponential is elementwise; the z core needs expm, and since mu d_z
couples z and mu, it is one z matrix per (node, ordinate): (z|mu) rank (2k+1) N_mu before rounding.
Same mode sizes as L, so A_inv has the TT shape of the direct operator.

Check: ||L A_inv - I||_F / ||I||_F in TT.

Run with the TT environment:
    /home/ale234/JRC/tt_python/tt_env/bin/python test_inverse.py [--k 15] [--alpha 2.0]
"""
import argparse
import time

import numpy as np
import scipy.linalg as sla
import torch as tn
import torchtt as tntt
import tt_transport.slab as tt

N_Z, P, L_Z, N_MU, N_PHI, G = 100, 1, 30.0, 24, 24, 10
SIGMA_T = np.linspace(0.1, 1.0, G)          # cm^-1, synthetic
ap = argparse.ArgumentParser()
ap.add_argument('--k', type=int, default=15, help='Stenger nodes j = -k..k')
ap.add_argument('--alpha', type=float, default=2.0, help='scale c = alpha / lambda_min (Grasedyck: 2)')
args = ap.parse_args()
K_QUAD, ALPHA = args.k, args.alpha


def stream_coll_op_tt(N_z, p, L_z, N_mu, N_phi, sigma_t):
    """Streaming + collision operator of SlabTransport (steady, no scattering) as a 4D TT-matrix (z, mu, phi, g)."""
    G = len(sigma_t)
    model = tt.SlabTransport(N_z=N_z, p=p, L_z=L_z, N_mu=N_mu, N_phi=N_phi, sigma_t=np.asarray(sigma_t, float),
                             sigma_s=[np.zeros((G, G))], steady_state=True)
    model.construct_tt_operator()
    cores = list(model.A.cores)
    cores[3] = tn.einsum('agbr,rcds->agbs', cores[3], cores[4])   # absorb the size-1 time core
    return tntt.TT(cores[:4]), model


def z_block(A, n, g, k_phi=0):
    """Dense z block of a TT-matrix that is diagonal in (mu, phi, g), at ordinate n, azimuth k_phi, group g."""
    c = A.cores
    v = c[1][:, n, n, :] @ c[2][:, k_phi, k_phi, :] @ c[3][:, g, g, :]
    return tn.einsum('abr,r->ab', c[0][0], v[:, 0]).numpy()


def stenger(k):
    h = np.pi * np.sqrt(2.0 / k)
    j = np.arange(-k, k + 1)
    return np.arcsinh(np.exp(j * h)), h / np.sqrt(1 + np.exp(-2 * j * h))


A, model = stream_coll_op_tt(N_Z, P, L_Z, N_MU, N_PHI, SIGMA_T)
A_s, _ = stream_coll_op_tt(N_Z, P, L_Z, N_MU, N_PHI, np.zeros(G))      # streaming only
M_inv = np.linalg.inv(model.M)
ndof, mu = model.ndof_z, model.mu
print(f"L: N = {list(A.N)}, ranks {list(A.R)}")

# streaming matrices mu_n K_n = M^{-1} (streaming block), and lambda_min = min Re sigma(M^{-1} L)
muK = [M_inv @ z_block(A_s, n, 0) for n in range(N_MU)]
lam_min = SIGMA_T.min() + min(np.linalg.eigvals(B).real.min() for B in muK)
print(f"lambda_min(M^-1 L) = {lam_min:.4e} (sigma_min = {SIGMA_T.min()}, "
      f"streaming part {lam_min - SIGMA_T.min():.3e}),  lambda_max = "
      f"{SIGMA_T.max() + max(np.linalg.eigvals(B).real.max() for B in muK):.3e}")


def build_inverse_tt(k, alpha):
    """A_inv as a TT-matrix over (z, mu, phi, g) with ranks [(2k+1) N_mu, 2k+1, 2k+1]."""
    t, w = stenger(k)
    c = alpha / lam_min
    J = len(t)
    Z = np.zeros((1, ndof, ndof, J * N_MU))
    U = np.zeros((J * N_MU, N_MU, N_MU, J))
    F = np.zeros((J, N_PHI, N_PHI, J))
    E = np.zeros((J, G, G, 1))
    for j in range(J):
        for n in range(N_MU):
            Z[0, :, :, j * N_MU + n] = sla.expm(-c * t[j] * muK[n]) @ M_inv
            U[j * N_MU + n, n, n, j] = 1.0
        F[j, :, :, j] = np.eye(N_PHI)
        E[j, :, :, 0] = np.diag(c * w[j] * np.exp(-c * t[j] * SIGMA_T))
    return tntt.TT([tn.tensor(x) for x in (Z, U, F, E)])


# ---------------------------------------------------------------------------
# TT: A @ A_inv vs identity
# ---------------------------------------------------------------------------
t0 = time.time()
A_inv = build_inverse_tt(K_QUAD, ALPHA)
print(f"\nk = {K_QUAD} ({2 * K_QUAD + 1} nodes), alpha = {ALPHA}: A_inv ranks {list(A_inv.R)}", end='')
print(f", built in {time.time() - t0:.1f} s", flush=True)
t0 = time.time()
A_inv = A_inv.round(1e-10)
print(f"rounded (1e-10): ranks {list(A_inv.R)}, {time.time() - t0:.1f} s", flush=True)
t0 = time.time()
AA = A @ A_inv
print(f"L @ A_inv: ranks {list(AA.R)}, {time.time() - t0:.1f} s", flush=True)
t0 = time.time()
Id = tntt.eye([ndof, N_MU, N_PHI, G])
R_tt = AA - Id
print(f"TT: ||L A_inv - I||_F / ||I||_F = {float(R_tt.norm() / Id.norm()):.3e}, {time.time() - t0:.1f} s", flush=True)

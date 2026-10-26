"""
Graphite P0tr, 10 groups (setup of graphite_precond.py): finer scan of the exponential-sum preconditioner P ~ L^{-1}
over (k, alpha), looking for ||P L - I||_F / ||I||_F ~ 1e-2; then the incremental TT build of the best point, timed.

P and L are block diagonal in (mu, phi, g), so the error is evaluated block by block (exact, no TT):
    ||P L - I||_F^2 = N_phi sum_{n,g} ||r(B_{n,g}) B_{n,g} - I||_F^2,   B_{n,g} = sigma_g I + mu_n K,
    r(z) = sum_j s w_j e^{-s t_j z},  s = alpha / lambda_min,  Stenger nodes t_j, j = -k..k.
Speed-ups: K (upwind DG streaming, mu > 0, uniform h) is block lower-triangular Toeplitz, so e^{-s mu K} is too and only
its first block column is computed (expm_multiply); mu < 0 blocks are the mirrored mu > 0 ones (same Frobenius norm).
Both are checked at start. The best point is then built as a TT, node by node with rounding (1e-10), and checked in TT.

Run with the TT environment:
    /home/ale234/JRC/tt_python/tt_env/bin/python -u expsum_scan_fine.py [--ks 60 120 240 480] \
        [--alphas 0.02 0.03 0.05 0.07 0.1 0.15 0.2] [--target 1e-2] > run_expsum_scan_fine.log
"""
import argparse
import contextlib
import io
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
import torch as tn
import torchtt as tntt
import tt_transport.slab as tt

ap = argparse.ArgumentParser()
ap.add_argument('--ks', type=int, nargs='+', default=[60, 120, 240, 480])
ap.add_argument('--alphas', type=float, nargs='+', default=[0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2])
ap.add_argument('--target', type=float, default=1e-2)
args = ap.parse_args()

HERE = Path(__file__).resolve().parent
DATA = HERE / 'graphite_P0tr10g10_mg'
xs, mg = np.load(DATA / 'graphite_mgxs.npz'), np.load(DATA / 'graphite_mg_flux.npz')
L_Z, SIGMA_T = float(xs['L']), xs['sigma_tr']
G, N_Z = len(SIGMA_T), len(mg['z'])
P_DG, N_MU, N_PHI = 1, 16, 3
NB = P_DG + 1


def z_block(A, n, g):
    """Dense z block of a TT-matrix (z, mu, phi, g, t) diagonal in (mu, phi, g), at ordinate n, phi = 0, group g."""
    c = A.cores
    v = c[1][:, n, n, :] @ c[2][:, 0, 0, :] @ c[3][:, g, g, :] @ c[4][:, 0, 0, :]
    return tn.einsum('abr,r->ab', c[0][0], v[:, 0]).numpy()


def stenger(k):
    h = np.pi * np.sqrt(2.0 / k)
    j = np.arange(-k, k + 1)
    return np.arcsinh(np.exp(j * h)), h / np.sqrt(1 + np.exp(-2 * j * h))


def tt5(cores):
    """TT-matrix over (z, mu, phi, g, t) from four numpy cores, plus the size-1 time core."""
    return tntt.TT([tn.tensor(x) for x in cores] + [tn.ones((1, 1, 1, 1), dtype=tn.float64)])


# ---------------------------------------------------------------------------
# L, streaming matrix K (mu > 0), checks
# ---------------------------------------------------------------------------
mL = tt.SlabTransport(N_z=N_Z, p=P_DG, L_z=L_Z, N_mu=N_MU, N_phi=N_PHI, sigma_t=SIGMA_T,
                      sigma_s=[np.zeros((G, G))], steady_state=True)
with contextlib.redirect_stdout(io.StringIO()):
    mL.construct_tt_operator()
L_tt, ndof, M_inv, mu = mL.A, mL.ndof_z, np.linalg.inv(mL.M), mL.mu
muK = [M_inv @ z_block(L_tt, n, 0) - SIGMA_T[0] * np.eye(ndof) for n in range(N_MU)]
pos = [n for n in range(N_MU) if mu[n] > 0]
K = muK[pos[0]] / mu[pos[0]]
assert all(np.allclose(muK[n], mu[n] * K, atol=1e-12 * np.abs(K).max()) for n in pos), "mu_n K_n != mu_n K"
Kb = K.reshape(N_Z, NB, N_Z, NB)
toeplitz_err = max(np.abs(Kb[i, :, i - d, :] - Kb[d, :, 0, :]).max() for d in range(N_Z) for i in range(d, N_Z))
assert toeplitz_err < 1e-10 * np.abs(K).max() and np.allclose(np.triu(K, NB), 0), "K is not lower block Toeplitz"
lam_min = SIGMA_T.min() + min(np.linalg.eigvals(muK[n]).real.min() for n in range(N_MU))
K_sp = sp.csr_matrix(K)
E1 = np.eye(ndof)[:, :NB]


def expm_toeplitz(a):
    """e^{-a K} for a >= 0 from its first block column (block lower-triangular Toeplitz)."""
    col = expm_multiply(-a * K_sp, E1).reshape(N_Z, NB, NB)
    E4 = np.zeros((N_Z, NB, N_Z, NB))
    for d in range(N_Z):
        i = np.arange(d, N_Z)
        E4[i, :, i - d, :] = col[d]
    return E4.reshape(ndof, ndof)


a_test = 3.0 / lam_min * mu[pos[-1]]
check = np.abs(expm_toeplitz(a_test) - sla.expm(-a_test * K)).max()
assert check < 1e-10, f"Toeplitz exponential check failed: {check:.2e}"
print(f"N_z = {N_Z}, p = {P_DG}, N_mu = {N_MU}, N_phi = {N_PHI}, G = {G}; lambda_min = {lam_min:.4e}; "
      f"Toeplitz check {toeplitz_err:.1e}, expm check {check:.1e}", flush=True)

I = np.eye(ndof)


def block_error(k, alpha):
    """||P L - I||_F / ||I||_F over all (n, g), mu < 0 by mirror symmetry."""
    t, w = stenger(k)
    s = alpha / lam_min
    err2 = 0.0
    for n in pos:
        Pg = np.zeros((G, ndof, ndof))
        for tj, wj in zip(t, w):
            Ej = expm_toeplitz(s * tj * mu[n])
            Pg += (s * wj * np.exp(-s * tj * SIGMA_T))[:, None, None] * Ej[None]
        for g in range(G):
            err2 += 2 * np.sum((Pg[g] @ (SIGMA_T[g] * I + muK[n]) - I) ** 2)
    return np.sqrt(err2 / (N_MU * G * ndof))


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------
print(f"\n{'k':>4} {'alpha':>6} {'nodes':>5} {'alpha*t_max':>11} {'||PL-I||/||I||':>15} {'time[s]':>8}", flush=True)
table = []
for k in args.ks:
    t_max = stenger(k)[0].max()
    for alpha in args.alphas:
        t0 = time.time()
        e = block_error(k, alpha)
        table.append((k, alpha, e))
        print(f"{k:4d} {alpha:6.3f} {2 * k + 1:5d} {alpha * t_max:11.2f} {e:15.3e} {time.time() - t0:8.1f}", flush=True)

ok = [x for x in table if x[2] <= args.target]
best = min(ok, key=lambda x: (x[0], x[2])) if ok else min(table, key=lambda x: x[2])
print(f"\nbest: k = {best[0]}, alpha = {best[1]}, error {best[2]:.3e} "
      f"({'meets' if ok else 'does not meet'} target {args.target:.0e})", flush=True)

# ---------------------------------------------------------------------------
# TT build of the best point, node by node with rounding
# ---------------------------------------------------------------------------
k, alpha = best[0], best[1]
t, w = stenger(k)
s = alpha / lam_min
R = np.kron(np.fliplr(np.eye(N_Z)), np.diag((-1.0) ** np.arange(NB)))
U = np.zeros((N_MU, N_MU, N_MU, 1))
U[np.arange(N_MU), np.arange(N_MU), np.arange(N_MU), 0] = 1.0
F = np.eye(N_PHI)[None, :, :, None]
t_exp = t_rnd = 0.0
P = None
for tj, wj in zip(t, w):
    t0 = time.time()
    Z = np.zeros((ndof, ndof, N_MU))
    for n in pos:
        Z[:, :, n] = expm_toeplitz(s * tj * mu[n]) @ M_inv
        Z[:, :, N_MU - 1 - n] = R @ Z[:, :, n] @ R
    E = np.diag(s * wj * np.exp(-s * tj * SIGMA_T))[None, :, :, None]
    t_exp += time.time() - t0
    t0 = time.time()
    term = tt5([Z[None], U, F, E])
    P = term if P is None else (P + term).round(1e-10)
    t_rnd += time.time() - t0
Id = tntt.eye(list(L_tt.N))
err_tt = float((P @ L_tt - Id).norm() / Id.norm())
print(f"TT build (k = {k}, alpha = {alpha}, {len(t)} nodes): exponentials {t_exp:.1f} s, additions + roundings "
      f"{t_rnd:.1f} s, total {t_exp + t_rnd:.1f} s; ranks {list(P.R)}; TT ||P L - I||_F/||I||_F = {err_tt:.3e}",
      flush=True)

"""
Graphite P0tr, 10 groups (setup of graphite_precond.py): accuracy of the exponential-sum preconditioner P ~ L^{-1}
versus the number of Stenger nodes (k) and the scale (alpha), and the cost of building P with incremental rounding.

For every (k, alpha): P = sum_j s w_j sum_n [e^{-s t_j mu_n K_n} M^{-1}] (x) e_n e_n^T (x) I (x) diag(e^{-s t_j sigma_g}),
s = alpha / lambda_min, built node by node, P <- round(P + term_j, 1e-10). Reported: build time (matrix exponentials,
TT additions + roundings), ranks of P, ||P L - I||_F / ||I||_F.
The exponentials for mu < 0 are the mirrored mu > 0 ones (z -> L - z, Legendre mode a -> (-1)^a), checked at start.

Run with the TT environment:
    /home/ale234/JRC/tt_python/tt_env/bin/python -u expsum_scan.py [--ks 30 60] [--alphas 0.02 0.05 0.1] \
        > run_expsum_scan.log
"""
import argparse
import contextlib
import io
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import torch as tn
import torchtt as tntt
import tt_transport.slab as tt

ap = argparse.ArgumentParser()
ap.add_argument('--ks', type=int, nargs='+', default=[30, 60])
ap.add_argument('--alphas', type=float, nargs='+', default=[0.02, 0.05, 0.1])
args = ap.parse_args()

HERE = Path(__file__).resolve().parent
DATA = HERE / 'graphite_P0tr10g10_mg'
xs, mg = np.load(DATA / 'graphite_mgxs.npz'), np.load(DATA / 'graphite_mg_flux.npz')
L_Z, SIGMA_T = float(xs['L']), xs['sigma_tr']
G, N_Z = len(SIGMA_T), len(mg['z'])
P_DG, N_MU, N_PHI = 1, 16, 3


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


# L = streaming + collision (SlabTransport without scattering) and its z blocks
mL = tt.SlabTransport(N_z=N_Z, p=P_DG, L_z=L_Z, N_mu=N_MU, N_phi=N_PHI, sigma_t=SIGMA_T,
                      sigma_s=[np.zeros((G, G))], steady_state=True)
with contextlib.redirect_stdout(io.StringIO()):         # silence the operator printout
    mL.construct_tt_operator()
L_tt, ndof, M_inv = mL.A, mL.ndof_z, np.linalg.inv(mL.M)
muK = [M_inv @ z_block(L_tt, n, 0) - SIGMA_T[0] * np.eye(ndof) for n in range(N_MU)]
lam_min = SIGMA_T.min() + min(np.linalg.eigvals(B).real.min() for B in muK)

# mirror R: cell i -> N_z - 1 - i, mode a -> (-1)^a; mu_{N-1-n} = -mu_n (symmetric Gauss nodes)
R = np.kron(np.fliplr(np.eye(N_Z)), np.diag((-1.0) ** np.arange(P_DG + 1)))
mirror_err = max(np.abs(muK[N_MU - 1 - n] - R @ muK[n] @ R).max() / np.abs(muK[n]).max() for n in range(N_MU // 2))
assert mirror_err < 1e-12, f"mirror check failed: {mirror_err:.2e}"
pos = [n for n in range(N_MU) if mL.mu[n] > 0]
print(f"N_z = {N_Z}, p = {P_DG}, N_mu = {N_MU}, N_phi = {N_PHI}, G = {G}; lambda_min = {lam_min:.4e}; "
      f"mirror check {mirror_err:.1e}", flush=True)

U = np.zeros((N_MU, N_MU, N_MU, 1))
U[np.arange(N_MU), np.arange(N_MU), np.arange(N_MU), 0] = 1.0
F = np.eye(N_PHI)[None, :, :, None]
Id = tntt.eye(list(L_tt.N))
print(f"\n{'k':>4} {'alpha':>6} {'nodes':>5} {'t_expm[s]':>9} {'t_round[s]':>10} {'t_total[s]':>10} "
      f"{'ranks P':>22} {'||PL-I||/||I||':>15}", flush=True)
for k in args.ks:
    t, w = stenger(k)
    for alpha in args.alphas:
        s = alpha / lam_min
        t_exp = t_rnd = 0.0
        P = None
        for j in range(len(t)):
            t0 = time.time()
            Z = np.zeros((ndof, ndof, N_MU))
            for n in pos:
                Z[:, :, n] = sla.expm(-s * t[j] * muK[n]) @ M_inv
                Z[:, :, N_MU - 1 - n] = R @ Z[:, :, n] @ R          # M^{-1} commutes with R
            E = np.diag(s * w[j] * np.exp(-s * t[j] * SIGMA_T))[None, :, :, None]
            t_exp += time.time() - t0
            t0 = time.time()
            term = tt5([Z[None], U, F, E])
            P = term if P is None else (P + term).round(1e-10)
            t_rnd += time.time() - t0
        err = float((P @ L_tt - Id).norm() / Id.norm())
        print(f"{k:4d} {alpha:6.3f} {len(t):5d} {t_exp:9.1f} {t_rnd:10.1f} {t_exp + t_rnd:10.1f} "
              f"{str(list(P.R)):>22} {err:15.3e}", flush=True)

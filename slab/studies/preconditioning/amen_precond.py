"""
AMEn with a left sweep preconditioner: solve P A x = P b instead of A x = b, A = L - S.

Slab, steady, P0 scattering, flat isotropic source in group 0, vacuum boundaries; N_z = 100, p = 1, N_mu = N_phi = 24,
G = 10, sigma_t = 0.1..1 cm^-1 (synthetic). Scattering ratio c: per group sigma_s,g->g = 0.7 c sigma_t,g and
sigma_s,g->g+1 = 0.3 c sigma_t,g (last group: all in-group).
L = streaming + collision is diagonal in (mu, phi, g); P approximates L^{-1}:
    none   : P = I
    exact  : P = sum_{n,g} L_{n,g}^{-1} (x) e_n e_n^T (x) I (x) e_g e_g^T     (dense z blocks, exact sweep)
    expsum : P = sum_j c w_j sum_n [e^{-c t_j mu_n K_n} M^{-1}] (x) e_n e_n^T (x) I (x) diag(e^{-c t_j sigma_g})
             (Stenger nodes j = -k..k, scale c = alpha / lambda_min; see test_inverse.py)
P is independent of the scattering ratio, so both are built once. P A and P b are rounded (1e-12) before AMEn.
Reported per case: AMEn sweeps and wall time, solution ranks, true residual ||A x - b|| / ||b||.
The verbose AMEn output goes to stdout; summary lines start with RESULT.

Run with the TT environment:
    /home/ale234/JRC/tt_python/tt_env/bin/python -u amen_precond.py [--k 30] [--alpha 0.1] > run_amen_precond.log
"""
import argparse
import time

import numpy as np
import scipy.linalg as sla
import torch as tn
import torchtt as tntt
import tt_transport.slab as tt

ap = argparse.ArgumentParser()
ap.add_argument('--k', type=int, default=30, help='Stenger nodes j = -k..k')
ap.add_argument('--alpha', type=float, default=0.1, help='scale c = alpha / lambda_min')
args = ap.parse_args()

N_Z, P_DG, L_Z, N_MU, N_PHI, G = 100, 1, 30.0, 24, 24, 10
SIGMA_T = np.linspace(0.1, 1.0, G)
C_SCAT = [0.5, 0.9, 0.99]
AMEN = dict(nswp=40, eps=1e-6, kickrank=4, max_full=15000, verbose=True)


def model_for(c):
    """SlabTransport with scattering ratio c (P0), source and right-hand side set; operator built."""
    S = np.zeros((G, G))                          # S[g, g'] = sigma_s,g->g'
    for g in range(G):
        if g < G - 1:
            S[g, g], S[g, g + 1] = 0.7 * c * SIGMA_T[g], 0.3 * c * SIGMA_T[g]
        else:
            S[g, g] = c * SIGMA_T[g]
    m = tt.SlabTransport(N_z=N_Z, p=P_DG, L_z=L_Z, N_mu=N_MU, N_phi=N_PHI, sigma_t=SIGMA_T, sigma_s=[S],
                         steady_state=True)
    m.set_source([lambda z: 1.0 + 0 * z, lambda mu: 1.0 + 0 * mu, lambda phi: 1.0 + 0 * phi, np.eye(G)[0]])
    m.construct_tt_operator()
    m.construct_tt_rhs()
    return m


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
# preconditioners from L (c = 0: streaming + collision only)
# ---------------------------------------------------------------------------
mL = model_for(0.0)
ndof, M_inv = mL.ndof_z, np.linalg.inv(mL.M)
L_blk = [[z_block(mL.A, n, g) for g in range(G)] for n in range(N_MU)]

t0 = time.time()
Z = np.zeros((1, ndof, ndof, N_MU * G))
U = np.zeros((N_MU * G, N_MU, N_MU, G))
for n in range(N_MU):
    for g in range(G):
        Z[0, :, :, n * G + g] = np.linalg.inv(L_blk[n][g])
        U[n * G + g, n, n, g] = 1.0
F = np.stack([np.eye(N_PHI)] * G)[:, :, :, None] * np.eye(G)[:, None, None, :]
E = np.zeros((G, G, G, 1))
E[np.arange(G), np.arange(G), np.arange(G), 0] = 1.0
P_exact = tt5([Z, U, F, E]).round(1e-12)
print(f"P_exact: ranks {list(P_exact.R)}, {time.time() - t0:.1f} s")

t0 = time.time()
muK = [M_inv @ L_blk[n][0] - SIGMA_T[0] * np.eye(ndof) for n in range(N_MU)]
lam_min = SIGMA_T.min() + min(np.linalg.eigvals(B).real.min() for B in muK)
t, w = stenger(args.k)
sc, J = args.alpha / lam_min, len(t)
Z = np.zeros((1, ndof, ndof, J * N_MU))
U = np.zeros((J * N_MU, N_MU, N_MU, J))
F = np.zeros((J, N_PHI, N_PHI, J))
E = np.zeros((J, G, G, 1))
for j in range(J):
    for n in range(N_MU):
        Z[0, :, :, j * N_MU + n] = sla.expm(-sc * t[j] * muK[n]) @ M_inv
        U[j * N_MU + n, n, n, j] = 1.0
    F[j, :, :, j] = np.eye(N_PHI)
    E[j, :, :, 0] = np.diag(sc * w[j] * np.exp(-sc * t[j] * SIGMA_T))
P_exp = tt5([Z, U, F, E]).round(1e-10)
Id = tntt.eye(list(mL.A.N))
err = float((mL.A @ P_exp - Id).norm() / Id.norm())
print(f"P_expsum (k = {args.k}, alpha = {args.alpha}): ranks {list(P_exp.R)}, ||L P - I||_F/||I||_F = {err:.2e}, "
      f"{time.time() - t0:.1f} s")

# ---------------------------------------------------------------------------
# AMEn on A x = b and on P A x = P b
# ---------------------------------------------------------------------------
for c in C_SCAT:
    m = model_for(c)
    A, b = m.A, m.F
    for name, P in (('none', None), ('exact', P_exact), ('expsum', P_exp)):
        AA, bb = (A, b) if P is None else ((P @ A).round(1e-12), (P @ b).round(1e-12))
        print(f"\n===== c = {c}, P = {name}: system ranks {list(AA.R)}", flush=True)
        t0 = time.time()
        x = tntt.solvers.amen_solve(AA, bb, **AMEN)
        wall = time.time() - t0
        res = float((A @ x - b).norm() / b.norm())
        print(f"RESULT c={c} P={name} wall={wall:.1f}s ranks={list(x.R)} true_res={res:.2e} "
              f"system_ranks={list(AA.R)}", flush=True)

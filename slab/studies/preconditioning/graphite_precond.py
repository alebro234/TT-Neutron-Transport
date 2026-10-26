"""
Graphite slab, P0tr, 10 groups: AMEn on A x = b (none) vs P A x = P b with P = exact sweep or exponential sum.

Same problem as ~/openmc/graphite/tt_graphite_slab.py P0tr: 30 cm graphite, normal 2 MeV beam on z = 0 split off as
a first-collision source, vacuum boundaries, transport-corrected sigma_tr and isotropic sigma~_s0 (with upscatter);
N_z = 200 (OpenMC tally mesh), p = 1, N_mu = 16, N_phi = 3, AMEn eps = 1e-7, nswp = 40, kickrank = 4.
Data (copied from ~/openmc/graphite/results_P0tr): graphite_P0tr10g10_mg/graphite_mgxs.npz, graphite_mg_flux.npz.

Preconditioners of L = streaming + collision (SlabTransport without scattering), as in amen_precond.py:
    exact  : P = sum_{n,g} L_{n,g}^{-1} (x) e_n e_n^T (x) I (x) e_g e_g^T
    expsum : P = sum_j s w_j sum_n [e^{-s t_j mu_n K_n} M^{-1}] (x) e_n e_n^T (x) I (x) diag(e^{-s t_j sigma_g})
expsum is built either stacked and rounded once (--round once) or node by node with a rounding after each node
(--round incremental), both at 1e-10. P A and P b are rounded at 1e-12. Per case: AMEn wall time, solution ranks and compression, true residual
||A x - b|| / ||b||, integrated TT/MG - 1 (total and per group) and ||TT - MG||_F / ||MG||_F of the cell-averaged
scalar flux (collided + uncollided). Summary lines start with RESULT; fluxes in graphite_precond_flux_<round>.npz.

Run with the TT environment:
    /home/ale234/JRC/tt_python/tt_env/bin/python -u graphite_precond.py [--k 30] [--alpha 0.1] [--round once|incremental] \
        > run_graphite_precond_<round>.log
"""
import argparse
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
ap.add_argument('--k', type=int, default=30, help='Stenger nodes j = -k..k')
ap.add_argument('--alpha', type=float, default=0.1, help='scale s = alpha / lambda_min')
ap.add_argument('--round', choices=['once', 'incremental'], default='once',
                help='expsum: stack all terms and round once, or add one node at a time and round each time')
args = ap.parse_args()

HERE = Path(__file__).resolve().parent
DATA = HERE / 'graphite_P0tr10g10_mg'
xs, mg = np.load(DATA / 'graphite_mgxs.npz'), np.load(DATA / 'graphite_mg_flux.npz')
L_Z, SIGMA_T = float(xs['L']), xs['sigma_tr']
S0 = xs['scatter_legendre'][0]                 # S0[g, g'] = sigma~_s0(g' -> g), row = receiving group
G, N_Z = len(SIGMA_T), len(mg['z'])
P_DG, N_MU, N_PHI = 1, 16, 3
AMEN = dict(nswp=40, eps=1e-7, kickrank=4, max_full=15000, verbose=True)


def make_model(sigma_s0):
    return tt.SlabTransport(N_z=N_Z, p=P_DG, L_z=L_Z, N_mu=N_MU, N_phi=N_PHI, sigma_t=SIGMA_T,
                            sigma_s=[sigma_s0.T], steady_state=True)   # SlabTransport: row = incoming group


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


def tt_entries(T):
    return sum(c.numel() for c in T.cores)


# ---------------------------------------------------------------------------
# full problem: operator and first-collision right-hand side (as tt_graphite_slab.py, P0)
# ---------------------------------------------------------------------------
model = make_model(S0)
model.construct_tt_operator()
Q_z = model._project_z(lambda z: np.exp(-SIGMA_T[0] * z))
model.q = tt.vector_kron([Q_z, 1 / (4 * np.pi) * np.ones(N_MU), np.ones(N_PHI), S0[:, 0], np.ones(1)])
model.set_BC_left(0)
model.set_BC_right(0)
model.construct_tt_rhs()
A, b = model.A, model.F
n_dof = int(np.prod(b.N))

# ---------------------------------------------------------------------------
# preconditioners from L (no scattering)
# ---------------------------------------------------------------------------
mL = make_model(np.zeros((G, G)))
mL.construct_tt_operator()
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
print(f"P_exact: ranks {list(P_exact.R)}, {time.time() - t0:.1f} s", flush=True)

t0 = time.time()
muK = [M_inv @ L_blk[n][0] - SIGMA_T[0] * np.eye(ndof) for n in range(N_MU)]
lam_min = SIGMA_T.min() + min(np.linalg.eigvals(B).real.min() for B in muK)
t, w = stenger(args.k)
sc, J = args.alpha / lam_min, len(t)
if args.round == 'once':                        # all J N_mu terms stacked, one rounding
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
else:                                           # one node at a time: P <- round(P + term_j)
    # mu > 0: mu_n K_n = mu_n K with K block lower-triangular Toeplitz (uniform h), so e^{-a K} is too and only its
    # first block column is computed; mu < 0: mirror z -> L - z, mode b -> (-1)^b (see expsum_scan_fine.py checks)
    NB = P_DG + 1
    pos = [n for n in range(N_MU) if model.mu[n] > 0]
    K_sp = sp.csr_matrix(muK[pos[0]] / model.mu[pos[0]])
    E1 = np.eye(ndof)[:, :NB]
    R = np.kron(np.fliplr(np.eye(N_Z)), np.diag((-1.0) ** np.arange(NB)))

    def expm_toeplitz(a):
        col = expm_multiply(-a * K_sp, E1).reshape(N_Z, NB, NB)
        E4 = np.zeros((N_Z, NB, N_Z, NB))
        for d in range(N_Z):
            i = np.arange(d, N_Z)
            E4[i, :, i - d, :] = col[d]
        return E4.reshape(ndof, ndof)

    P_exp = None
    U = np.zeros((N_MU, N_MU, N_MU, 1))
    U[np.arange(N_MU), np.arange(N_MU), np.arange(N_MU), 0] = 1.0
    for j in range(J):
        Z = np.zeros((ndof, ndof, N_MU))
        for n in pos:
            Z[:, :, n] = expm_toeplitz(sc * t[j] * model.mu[n]) @ M_inv
            Z[:, :, N_MU - 1 - n] = R @ Z[:, :, n] @ R
        E = np.diag(sc * w[j] * np.exp(-sc * t[j] * SIGMA_T))[None, :, :, None]
        term = tt5([Z[None], U, np.eye(N_PHI)[None, :, :, None], E])
        P_exp = term if P_exp is None else (P_exp + term).round(1e-10)
Id = tntt.eye(list(mL.A.N))
err = float((mL.A @ P_exp - Id).norm() / Id.norm())
err_r = float((P_exp @ mL.A - Id).norm() / Id.norm())
print(f"P_expsum (k = {args.k}, alpha = {args.alpha}, round = {args.round}, lambda_min = {lam_min:.3e}): ranks {list(P_exp.R)}, "
      f"||L P - I||_F/||I||_F = {err:.2e}, ||P L - I||_F/||I||_F = {err_r:.2e}, {time.time() - t0:.1f} s", flush=True)

# ---------------------------------------------------------------------------
# AMEn, scalar flux, comparison with OpenMC MG
# ---------------------------------------------------------------------------
a = mg['phi']
dz = L_Z / N_Z
z_lo = np.arange(N_Z) * dz
phi_u = (np.exp(-SIGMA_T[0] * z_lo) - np.exp(-SIGMA_T[0] * (z_lo + dz))) / (SIGMA_T[0] * dz)
out = {}
for name, P in (('none', None), ('exact', P_exact), ('expsum', P_exp)):
    AA, bb = (A, b) if P is None else ((P @ A).round(1e-12), (P @ b).round(1e-12))
    print(f"\n===== P = {name}: system ranks {list(AA.R)}, TT entries {tt_entries(AA)}", flush=True)
    t0 = time.time()
    x = tntt.solvers.amen_solve(AA, bb, **AMEN)
    wall = time.time() - t0
    x = x.round(1e-8)
    res = float((A @ x - b).norm() / b.norm())
    model.psi_modal = x
    phi_r = model.compute_scalar_flux().full().numpy()[:, :, 0]
    phi = np.einsum('irg,r->ig', phi_r.reshape(N_Z, model.N_recon, G), model.w_xi) / 2
    phi[:, 0] += phi_u
    out[name] = phi
    with np.errstate(divide='ignore', invalid='ignore'):
        per_g = phi.sum(0) / a.sum(0) - 1
    print(f"RESULT P={name} wall={wall:.1f}s ranks={list(x.R)} compression={tt_entries(x) / n_dof:.3e} "
          f"true_res={res:.2e} int_TT/MG-1={phi.sum() / a.sum() - 1:+.4e} "
          f"fro={np.linalg.norm(phi - a) / np.linalg.norm(a):.4e}", flush=True)
    print(f"RESULT P={name} int_TT/MG-1 per group: " + " ".join(f"{v:+.4f}" for v in per_g), flush=True)
np.savez(HERE / f'graphite_precond_flux_{args.round}.npz', z=mg['z'], phi_mg=a, **{f'phi_{k}': v for k, v in out.items()})

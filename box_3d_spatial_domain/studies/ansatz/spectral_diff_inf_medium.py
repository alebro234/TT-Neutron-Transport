import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch as tn
import torchtt as tntt
import tt_transport_box.box as ttbox

tn.set_default_dtype(tn.float64)
HERE = Path(__file__).resolve().parent

# Spectral diffusion solution (Dirichlet box eigenfunctions) as P1 ansatz x0 for the infinite-medium transport study
# (../infinite_medium/steady_state.py, same model and source).

# transport model: identical to ../infinite_medium/steady_state.py
c = 0.5
sigma_t = 1.0
model = ttbox.BoxTransport(
     N_x = 200,
     N_y = 200,
     N_z = 200,
     L_x = 10,
     L_y = 10,
     L_z = 10,
     p = 1,
     N_mu = 64,
     N_phi = 64,
     sigma_t = sigma_t,
     sigma_s = c,
     steady_state = True,
     N_recon = 2
        )
alpha = 0.5
x0, y0, z0 = model.L_x / 2, model.L_y / 2, model.L_z / 2
model.set_source(
     [
    lambda x: 1 / (4*np.pi * (2*np.pi*alpha**2)**(3/2)) * np.exp(-(x - x0)**2/(2*alpha**2)),
    lambda y: np.exp(-(y - y0)**2/(2*alpha**2)),
    lambda z: np.exp(-(z - z0)**2/(2*alpha**2)),
    lambda mu: 1 + 0*mu,
    lambda phi: 1 + 0*phi,
    [1]
 ]
        )

# diffusion data: isotropic scattering -> sigma_tr = sigma_t
D = 1 / (3 * sigma_t)
sigma_a = sigma_t - c
L_x, L_y, L_z = model.L_x, model.L_y, model.L_z
N_x, N_y, N_z = model.N_x, model.N_y, model.N_z

Norm = np.sqrt( 8 /(L_x*L_y*L_z) )

# grids: the transport cell midpoints, so phi gives the cell values of the p = 0 ansatz
x, y, z = model.x_mid, model.y_mid, model.z_mid

# number of modes (sine coefficients of the source decay as exp(-alpha^2 k^2 / 2): ~1e-9 at n = 41)
modes_x = 41
modes_y = 41
modes_z = 41 # final ranks no rounding = modes_x * modes_y * modes_z

max_rank_round = 30
eps_round = 1e-6

def psi(x, y, z, n_x, n_y, n_z):
    k_squared = (n_x * np.pi / L_x) ** 2 + (n_y * np.pi / L_y) ** 2 + (n_z * np.pi / L_z) ** 2
    psi = Norm * ttbox.vector_kron([
        np.sin(n_x * np.pi * x / L_x),
        np.sin(n_y * np.pi * y / L_y),
        np.sin(n_z * np.pi * z / L_z)
        ])
    return psi, k_squared

# gaussian source term, consistent with the transport source: q = int Q dOmega = 4 pi Q (unit total)
q = ttbox.vector_kron([
    (2*np.pi*alpha**2)**(-3/2) * np.exp(-(x - x0) ** 2 / (2*alpha**2)),
    np.exp(-(y - y0) ** 2 / (2*alpha**2)),
    np.exp(-(z - z0) ** 2 / (2*alpha**2))
    ])

# midpoint weights on the cell midpoints: int_V f psi_n dV ~ sum f psi_n W
W = ttbox.vector_kron([np.full(N_x, model.h_x), np.full(N_y, model.h_y), np.full(N_z, model.h_z)])

def project(f, n_x, n_y, n_z):
    psi_n, k_squared = psi(x, y, z, n_x, n_y, n_z)
    q_coeff = float(tntt.dot(f * W, psi_n))   # rank-1 TTs: dot contracts core by core
    return q_coeff # scalar, n_i passed as single integers

# build solution in TT format
t0 = time.time()
phi = tntt.zeros([N_x, N_y, N_z], dtype=tn.float64)
for n_x in range(1, modes_x + 1):
    for n_y in range(1, modes_y + 1):
        for n_z in range(1, modes_z + 1):
            q_coeff = project(q, n_x, n_y, n_z)
            k_squared = (n_x * np.pi / L_x) ** 2 + (n_y * np.pi / L_y) ** 2 + (n_z * np.pi / L_z) ** 2
            phi_n = q_coeff / (D * k_squared + sigma_a)
            phi += phi_n * psi(x, y, z, n_x, n_y, n_z)[0]
            if max(phi.R) > max_rank_round:
                phi = phi.round(eps_round)

phi = phi.round(eps_round)
print(f"modes {modes_x} x {modes_y} x {modes_z}, eps_round = {eps_round:g}: phi ranks {phi.R}, "
      f"{time.time() - t0:.1f} s")

# ---------------------------------------------------------------------------
# P1 ansatz psi0 = phi/(4 pi) - 3D/(4 pi) (d_x phi eta cos(varphi) + d_y phi eta sin(varphi) + d_z phi mu)
# ---------------------------------------------------------------------------
def pderiv(core_ith, dx=model.h_x):
    """Forward difference along the mode index of one spatial core: B = (-delta_ij + delta_{i,j-1}) / dx,
    last row copied from the previous one (d phi at the last cell = d phi at the one before)."""
    n = core_ith.shape[1]
    B = (np.eye(n, k=1) - np.eye(n)) / dx
    B[-1] = B[-2]
    return tn.einsum('ij,ajb->aib', tn.tensor(B, dtype=core_ith.dtype), core_ith)

# cell value -> DG Legendre mode 0 of the cell (mode 1 = 0: p = 0 ansatz in the p = 1 space)
def embed(core):
    n = core.shape[1]
    E = np.zeros((n * (model.p + 1), n))
    E[np.arange(n) * (model.p + 1), np.arange(n)] = 1.0
    return tn.einsum('ij,ajb->aib', tn.tensor(E, dtype=core.dtype), core)

def ansatz_term(space_cores, v_mu, v_phi):
    """space (x) v_mu (x) v_phi (x) [1]_g (x) [1]_t; the last spatial rank of phi is 1."""
    cores = [embed(cr) for cr in space_cores]
    cores += [tn.tensor(v, dtype=tn.float64)[None, :, None] for v in (v_mu, v_phi)]
    cores += [tn.ones((1, 1, 1), dtype=tn.float64), tn.ones((1, 1, 1), dtype=tn.float64)]
    return tntt.TT(cores)

c_x, c_y, c_z = phi.cores
ones_mu, ones_phi = np.ones(model.N_mu), np.ones(model.N_phi)
coef = -3 * D / (4 * np.pi)
psi0 = (ansatz_term([c_x, c_y, c_z], ones_mu / (4 * np.pi), ones_phi)
        + ansatz_term([pderiv(c_x, model.h_x), c_y, c_z], coef * model.eta, model.cos_phi)
        + ansatz_term([c_x, pderiv(c_y, model.h_y), c_z], coef * model.eta, model.sin_phi)
        + ansatz_term([c_x, c_y, pderiv(c_z, model.h_z)], coef * model.mu, ones_phi))
psi0 = psi0.round(eps_round)
print(f"ansatz psi0: ranks {psi0.R}")

# ---------------------------------------------------------------------------
# transport solve with x0 = psi0 (AMEn settings of the nswp = 40, kickrank = 2 run, without the loose stage)
# ---------------------------------------------------------------------------
NSWP, KICKRANK = 40, 2
TAG = f"_ansatz_nswp{NSWP}_kr{KICKRANK}"
model.construct_tt_operator()
model.construct_tt_rhs(round_eps=1e-13)
A, F = model.A, model.F
x_ones = tntt.ones(list(F.N), dtype=tn.float64)
print(f"initial residual ||A x0 - F||/||F||: ones {float((A @ x_ones - F).norm() / F.norm()):.3e}, "
      f"ansatz {float((A @ psi0 - F).norm() / F.norm()):.3e}")
t0 = time.time()
model.run_amen(nswp=NSWP, eps=1e-6, kickrank=KICKRANK, round_eps=1e-8, x0=psi0)
print(f"total solve time {time.time() - t0:.1f} s, true residual {model.residual():.3e}")
Phi_tt = model.compute_scalar_flux()                      # TT over (x_r, y_r, z_r, g, t)
ttbox.save_tt(Phi_tt, HERE / f"phi_tt_c{c}{TAG}.pt", vars=[model.x_r, model.y_r, model.z_r])

# ---------------------------------------------------------------------------
# radial comparison with the Fourier-inversion reference (as in ../infinite_medium/steady_state.py)
# ---------------------------------------------------------------------------
def Phi_hat(k, c):
    return np.arctan(k) / (k - c*np.arctan(k)) * np.exp(-0.5*alpha**2 * k**2)

def Phi_ref(r, c, k_max=30.0, n_k=600):
    xk, wk = np.polynomial.legendre.leggauss(n_k)
    k, wk = k_max/2*(xk + 1), k_max/2*wk
    return (wk * k * np.sin(np.outer(r, k)) * Phi_hat(k, c)).sum(axis=1) / (2*np.pi**2 * r)

r_bins = np.linspace(0.01, 5, 100)
r_mid = 0.5*(r_bins[1:] + r_bins[:-1])
Phi_amen = Phi_tt.full().reshape(len(model.x_r), len(model.y_r), len(model.z_r)).numpy()
rho2 = (model.x_r[:, None] - x0)**2 + (model.y_r[None, :] - y0)**2
sums, counts = np.zeros(len(r_mid)), np.zeros(len(r_mid))
for kz, zz in enumerate(model.z_r):
    idx = np.digitize(np.sqrt(rho2 + (zz - z0)**2), r_bins) - 1
    inside = (idx >= 0) & (idx < len(r_mid))
    sums += np.bincount(idx[inside], weights=Phi_amen[:, :, kz][inside], minlength=len(r_mid))
    counts += np.bincount(idx[inside], minlength=len(r_mid))
Phi_bin = sums / counts
xr, wr = np.polynomial.legendre.leggauss(8)
Phi_ref_bin = np.array([
    np.sum(wr * rr**2 * Phi_ref(rr, c)) / np.sum(wr * rr**2)
    for rr in (0.5*(r_bins[:-1, None] + r_bins[1:, None]) + 0.5*(r_bins[1:, None] - r_bins[:-1, None])*xr[None, :])])
dr = np.diff(r_bins)
for r_cut in (2.0, 4.0, r_bins[-1]):
    m = r_mid < r_cut
    e = np.sqrt(np.sum((Phi_bin[m] - Phi_ref_bin[m])**2 * r_mid[m]**2 * dr[m]) / np.sum(Phi_ref_bin[m]**2 * r_mid[m]**2 * dr[m]))
    print(f"relative L2 error (r^2 dr), r < {r_cut}: {e:.3e}")
np.savez(HERE / f"Phi_amen_c{c}{TAG}.npz", r_mid=r_mid, Phi_bin=Phi_bin, Phi_ref_bin=Phi_ref_bin, counts=counts)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.semilogy(r_mid, Phi_ref_bin, "k-", label="reference (Fourier inversion)")
ax1.semilogy(r_mid, Phi_bin, "r--", label="TT AMEn (P1 ansatz x0), bin average")
ax1.set_xlabel("r [mfp]"); ax1.set_ylabel(r"$\Phi(r)$"); ax1.legend(); ax1.set_title(f"c = {c}, alpha = {alpha}")
ax2.semilogy(r_mid, np.abs(Phi_bin - Phi_ref_bin) / Phi_ref_bin)
ax2.set_xlabel("r [mfp]"); ax2.set_ylabel("relative error")
fig.tight_layout()
fig.savefig(HERE / f"steady_state_c{c}{TAG}.png", dpi=150)

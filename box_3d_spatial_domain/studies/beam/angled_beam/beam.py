"""
Angled beam into the box [-1, 1]^3, no scattering (c = 0), steady state: purely uncollided transport.

The beam enters through the face x = -1 on the square |y|, |z| < 1/4 along the ordinate closest to mu = 0.5,
phi = pi/4 (N_phi = 248, so phi = pi/4 is on the azimuthal grid; mu = mu_n0 is the Gauss node nearest 0.5), i.e.
Omega = (eta cos phi, eta sin phi, mu). The face term carries Omega_x, so psi_in = J / (Omega_x w_n w_q) on (n0, q0)
gives the incoming partial current J = 1 per unit spot area.
Exact solution (path length s = (x + 1) / Omega_x back to the face x = -1; the box is convex, so the back-traced
segment stays inside it):
    psi = psi_in e^{-sigma_t s} 1(|y - Omega_y s| < 1/4) 1(|z - Omega_z s| < 1/4)   on (n0, q0), zero elsewhere,
    Phi = J / Omega_x e^{-sigma_t s} 1(|y - Omega_y s| < 1/4) 1(|z - Omega_z s| < 1/4).
The beam leaves through the faces y = 1 and z = 1. The spot is now a sheared band in (x, y) and (x, z): not low-rank.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch as tn
import tt_transport_box.box as ttbox

tn.set_default_dtype(tn.float64)
HERE = Path(__file__).resolve().parent

# logging: Python prints and the C++ AMEn output (file descriptors 1 and 2) all go to beam.log
_log = open(HERE / "beam.log", "w", buffering=1)
os.dup2(_log.fileno(), 1)
os.dup2(_log.fileno(), 2)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {Path(__file__).name} in {HERE.name}, torch threads {tn.get_num_threads()}")

model = ttbox.BoxTransport(
        N_x = 250,
        N_y = 250,
        N_z = 250,
        p = 1,
        L_x = (-1.0, 1.0),
        L_y = (-1.0, 1.0),
        L_z = (-1.0, 1.0),
        N_mu = 251,
        N_phi = 248,               # divisible by 8: phi = pi/4 is an ordinate
        sigma_t = 1.0,
        sigma_s = 0.0,             # c = 0: no scattering
        steady_state = True
        )

# beam ordinate: Gauss node closest to mu = 0.5, phi = pi/4 exactly
n0 = int(np.argmin(np.abs(model.mu - 0.5)))
q0 = int(np.argmin(np.abs(model.phi - np.pi / 4)))
assert abs(model.phi[q0] - np.pi / 4) < 1e-14
Om = np.array([model.eta[n0] * model.cos_phi[q0], model.eta[n0] * model.sin_phi[q0], model.mu[n0]])
J = 1.0
sig = model.sigma_t[0]
print(f"beam ordinate: mu = {model.mu[n0]:.6f}, phi = {model.phi[q0]:.6f}, Omega = ({Om[0]:.6f}, {Om[1]:.6f}, {Om[2]:.6f}), "
      f"w_mu = {model.w_mu[n0]:.4e}, w_phi = {model.w_phi[q0]:.4e}")

def beam_mu(mu):
    out = np.zeros_like(mu)
    out[n0] = 1.0 / (model.w_mu[n0] * model.eta[n0])
    return out

def beam_phi(phi):
    out = np.zeros_like(phi)
    out[q0] = J / (model.w_phi[q0] * model.cos_phi[q0])
    return out

# BC: beam along Omega on the spot |y|, |z| < 0.25 of the face x = -1
model.set_BC('L_x', [
             lambda y: np.where(np.abs(y) < 0.25, 1.0, 0.0),
             lambda z: np.where(np.abs(z) < 0.25, 1.0, 0.0),
             beam_mu,
             beam_phi,
             [1]
             ])

# solve
t0 = time.time()
model.construct_tt_operator()
model.construct_tt_rhs(round_eps=1e-12)
model.run_amen(nswp=15, eps=1e-6, kickrank=2, round_eps=1e-8)
print(f"total solve time {time.time() - t0:.1f} s, true residual {model.residual():.3e}, "
      f"psi ranks {list(model.psi_modal.R)}")

# scalar flux as a TT over (x_r, y_r, z_r, g, t); never expanded in full (750^3 points)
Phi = model.compute_scalar_flux()
ttbox.save_tt(Phi, HERE / "phi_tt_beam.pt", vars=[model.x_r, model.y_r, model.z_r])

# ---------------------------------------------------------------------------
# comparison with the analytical uncollided solution
# ---------------------------------------------------------------------------
tntt = ttbox.tntt

def Phi_exact(x, y, z):
    s = (x + 1.0) / Om[0]
    return J / Om[0] * np.exp(-sig * s) * (np.abs(y - Om[1] * s) < 0.25) * (np.abs(z - Om[2] * s) < 0.25)

c_x, c_y, c_z, c_g, c_t = Phi.cores
c_z1 = tn.einsum('aib,bc,cd->aid', c_z, c_g[:, 0, :], c_t[:, 0, :])[..., 0]     # (r_2, N_z_r): z core with g, t absorbed
XY = tn.einsum('aib,bjc->ijc', c_x, c_y)                                          # (N_x_r, N_y_r, r_2)

def plane_xy(kz):
    """Phi(x_r, y_r) at z_r[kz]."""
    return (XY @ c_z1[:, kz]).numpy()

# off-ordinate content of psi
cc = list(model.psi_modal.cores)
c_mu, c_phi = tn.zeros_like(cc[3]), tn.zeros_like(cc[4])
c_mu[:, n0, :], c_phi[:, q0, :] = cc[3][:, n0, :], cc[4][:, q0, :]
psi_off = model.psi_modal - tntt.TT(cc[:3] + [c_mu, c_phi] + cc[5:])
print(f"psi off the beam ordinate: ||psi_off|| / ||psi|| = {float(psi_off.norm() / model.psi_modal.norm()):.3e}")

# 3D errors, one z plane at a time (Gauss points of equal weight per cell: plain sums are the L2 / L1 integrals)
X, Y = np.meshgrid(model.x_r, model.y_r, indexing="ij")
e2 = n2 = e1 = n1 = 0.0
neg = 0.0
col = np.zeros_like(X); col_ex = np.zeros_like(X)                                # z-integrated flux int Phi dz
w_xi_z = model.h_z / 2 * np.tile(model.w_xi, model.N_z)
for kz, zz in enumerate(model.z_r):
    P, Pe = plane_xy(kz), Phi_exact(X, Y, zz)
    e2 += np.sum((P - Pe) ** 2); n2 += np.sum(Pe ** 2)
    e1 += np.sum(np.abs(P - Pe)); n1 += np.sum(np.abs(Pe))
    neg = min(neg, P.min())
    col += w_xi_z[kz] * P; col_ex += w_xi_z[kz] * Pe
print(f"||Phi - Phi_exact||_L2 / ||Phi_exact||_L2 = {np.sqrt(e2 / n2):.3e}, L1: {e1 / n1:.3e}  (all {len(model.x_r)}^3 points)")
print(f"min Phi over the box = {neg:.4f}  (exact max J/Omega_x = {J / Om[0]:.4f})")
print(f"z-integrated flux: ||col - col_exact|| / ||col_exact|| = {np.linalg.norm(col - col_ex) / np.linalg.norm(col_ex):.3e}")
# TT rank the exact solution needs in the (x | y) unfolding of the z-integrated flux
sv = np.linalg.svd(col_ex, compute_uv=False)
print(f"exact z-integrated flux, (x | y) singular values: rank at 1e-2 / 1e-4 / 1e-6 = "
      f"{[int(np.sum(sv > tol * sv[0])) for tol in (1e-2, 1e-4, 1e-6)]}")

# beam centreline: (x, y_c, z_c) = (x, Omega_y s, Omega_z s), s = (x + 1) / Omega_x, until the beam leaves the box
s_r = (model.x_r + 1.0) / Om[0]
inside = (Om[1] * s_r < model.y_r[-1]) & (Om[2] * s_r < model.z_r[-1])
ix = np.where(inside)[0]
jy = np.searchsorted(model.y_r, Om[1] * s_r[ix]).clip(0, len(model.y_r) - 1)
kz_c = np.searchsorted(model.z_r, Om[2] * s_r[ix]).clip(0, len(model.z_r) - 1)
# nearest reconstruction point to the centreline: use the nearer of the two neighbours
jy = np.where(np.abs(model.y_r[jy - 1] - Om[1] * s_r[ix]) < np.abs(model.y_r[jy] - Om[1] * s_r[ix]), jy - 1, jy)
kz_c = np.where(np.abs(model.z_r[kz_c - 1] - Om[2] * s_r[ix]) < np.abs(model.z_r[kz_c] - Om[2] * s_r[ix]), kz_c - 1, kz_c)
line = tn.einsum('nb,nb->n', XY[ix, jy, :], c_z1[:, kz_c].T).numpy()
line_ex = Phi_exact(model.x_r[ix], model.y_r[jy], model.z_r[kz_c])
x_line = model.x_r[ix]
rel = np.abs(line - line_ex) / line_ex
print(f"centreline (until the beam leaves at x = {x_line[-1]:.3f}): max rel. error {rel.max():.3e}, "
      f"median {np.median(rel):.3e}")
for xv in (-0.9, -0.6, -0.3, 0.0):
    i = int(np.argmin(np.abs(x_line - xv)))
    print(f"  x = {x_line[i]:+.4f}: Phi = {line[i]:.6f}, exact = {line_ex[i]:.6f}")

# transverse profile in y at x ~ -0.5, on the z plane through the beam centre
kx = int(np.argmin(np.abs(model.x_r + 0.5)))
kz_p = int(np.argmin(np.abs(model.z_r - Om[2] * s_r[kx])))
prof = plane_xy(kz_p)[kx, :]
prof_ex = Phi_exact(model.x_r[kx], model.y_r, model.z_r[kz_p])
np.savez(HERE / "beam_flux.npz", x_r=model.x_r, y_r=model.y_r, col=col, col_ex=col_ex, x_line=x_line, line=line,
         line_ex=line_ex, prof=prof, prof_ex=prof_ex, Omega=Om)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))
im = ax1.imshow(col.T, origin="lower", aspect="equal",
                extent=[model.x_r[0], model.x_r[-1], model.y_r[0], model.y_r[-1]], cmap="viridis")
for yb in (-0.25, 0.25):
    ax1.plot(model.x_r, yb + Om[1] * s_r, "w--", lw=0.8)
ax1.set_ylim(model.y_r[0], model.y_r[-1])
fig.colorbar(im, ax=ax1, label=r"$\int \Phi\, dz$")
ax1.set_xlabel("x"); ax1.set_ylabel("y")
ax1.set_title(rf"z-integrated flux, c = 0, $\mu$ = {model.mu[n0]:.3f}, $\varphi = \pi/4$")
ax2.semilogy(x_line, rel)
ax2.set_xlabel("x"); ax2.set_ylabel(r"$|\Phi - \Phi_{exact}| / \Phi_{exact}$ on the beam centreline"); ax2.grid(alpha=0.4)
ax3.plot(model.y_r, prof, ".-", ms=2, label="TT")
ax3.plot(model.y_r, prof_ex, "k--", label="exact")
yc = Om[1] * s_r[kx]
ax3.set_xlim(yc - 0.4, yc + 0.4); ax3.set_xlabel("y")
ax3.set_ylabel(rf"$\Phi(x = {model.x_r[kx]:.3f}, y, z = {model.z_r[kz_p]:.3f})$"); ax3.legend()
fig.tight_layout()
fig.savefig(HERE / "beam_flux.png", dpi=150)

"""
Normal beam along +x into the box [-1, 1]^3, P5 Henyey-Greenstein scattering (gamma = 0.3, c = 0.3),
steady state, no first-collision splitting.

The beam enters through the face x = -1 on the square |y|, |z| < 0.25 along Omega = (1, 0, 0). With N_mu odd the
Gauss-Legendre rule has a mu = 0 node (eta = 1), and phi = 0 is always on the azimuthal grid, so Omega = (1, 0, 0) is
an ordinate of the S_N set: the beam is put on that single ordinate,
    psi_in(mu_n, phi_q) = J / (w_n w_q)  on (n, q) = (n_0, 0),  0 elsewhere,
so that the incoming partial current int Omega_x psi_in dOmega = sum_nq eta_n cos(phi_q) w_n w_q psi_in = J = 1
(per unit area of the beam spot). On that ordinate Omega_y = Omega_z = 0 exactly, so the uncollided beam streams only
along x; scattering spreads the collided flux. All other faces are vacuum.
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

gamma = 0.3                # Henyey-Greenstein anisotropy
c = 0.3                    # scattering ratio sigma_s,0 / sigma_t, as in P0

model = ttbox.BoxTransport(
        N_x = 250,
        N_y = 250,
        N_z = 250,
        p = 1,
        L_x = (-1.0, 1.0),
        L_y = (-1.0, 1.0),
        L_z = (-1.0, 1.0),
        N_mu = 251,                # odd: mu = 0 is a node, so Omega = (1, 0, 0) is an ordinate
        N_phi = 250,
        sigma_t = 1.0,
        sigma_s = c * gamma**np.arange(0, 6),   # P5 HG kernel: sigma_s,l = c gamma^l, l = 0..5
        steady_state = True
        )

# beam ordinate: mu_n0 = 0 (eta = 1) and phi_0 = 0 (cos = 1)
n0 = int(np.argmin(np.abs(model.mu)))
q0 = 0
assert abs(model.mu[n0]) < 1e-12 and model.phi[q0] == 0.0
J = 1.0
print(f"beam ordinate: mu = {model.mu[n0]:.1e}, eta = {model.eta[n0]:.6f}, phi = {model.phi[q0]}, "
      f"w_mu = {model.w_mu[n0]:.4e}, w_phi = {model.w_phi[q0]:.4e}")

def beam_mu(mu):
    out = np.zeros_like(mu)
    out[n0] = 1.0 / model.w_mu[n0]
    return out

def beam_phi(phi):
    out = np.zeros_like(phi)
    out[q0] = J / model.w_phi[q0]
    return out

# BC = normal beam along x-axis on the spot |y|, |z| < 0.25 of the face x = -1
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

def plane_xy(Phi, kz):
    """Phi(x_r, y_r) at the reconstruction point z_r[kz]: contract the z, g, t cores."""
    c_x, c_y, c_z, c_g, c_t = Phi.cores
    right = c_z[:, kz, :] @ c_g[:, 0, :] @ c_t[:, 0, :]          # (r_2, 1)
    return tn.einsum('aib,bjc,cd->ij', c_x, c_y, right).numpy()

kz = int(np.argmin(np.abs(model.z_r)))                          # plane closest to z = 0
ky = int(np.argmin(np.abs(model.y_r)))
plane = plane_xy(Phi, kz)
line = plane[:, ky]                                             # centreline y ~ 0, z ~ 0
np.savez(HERE / "beam_flux.npz", x_r=model.x_r, y_r=model.y_r, z_plane=model.z_r[kz], plane=plane, line=line)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
im = ax1.imshow(np.log10(np.maximum(plane.T, 1e-12)), origin="lower", aspect="equal",
                extent=[model.x_r[0], model.x_r[-1], model.y_r[0], model.y_r[-1]], cmap="viridis")
fig.colorbar(im, ax=ax1, label=r"$\log_{10}\Phi$")
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_title(f"scalar flux, z = {model.z_r[kz]:.3f}")
ax2.semilogy(model.x_r, line, label="TT")
ax2.semilogy(model.x_r, J * np.exp(-model.sigma_t[0] * (model.x_r + 1.0)), "k--", label=r"uncollided $J e^{-\sigma_t (x+1)}$")
ax2.set_xlabel("x"); ax2.set_ylabel(r"$\Phi(x, 0, 0)$"); ax2.legend(); ax2.grid(alpha=0.4)
fig.tight_layout()
fig.savefig(HERE / "beam_flux.png", dpi=150)

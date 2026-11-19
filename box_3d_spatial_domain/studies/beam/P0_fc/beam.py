"""
Normal beam along +x into the box [-1, 1]^3, isotropic scattering (c = 0.3), steady state, WITH first-collision splitting.

psi = psi_u + psi_c. The uncollided flux is analytic (beam on the ordinate Omega = (1, 0, 0), see ../no_scattering):
    psi_u = J / (w_n0 w_q0) e^{-sigma_t (x+1)} 1(|y| < 1/4) 1(|z| < 1/4)  on (n0, q0),
    Phi_u = int psi_u dOmega = J e^{-sigma_t (x+1)} 1(|y| < 1/4) 1(|z| < 1/4)            (rank 1).
The collided flux solves the same transport problem with vacuum BCs on every face and the first-collision source
    Q_c = sigma_s / (4 pi) Phi_u                                                        (isotropic scattering),
so AMEn sees a rank-1 isotropic source and no boundary beam. Phi = Phi_u + Phi_c, compared with the unsplit run ../P0.
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
        N_mu = 251,                # odd: mu = 0 is a node, so Omega = (1, 0, 0) is an ordinate
        N_phi = 250,
        sigma_t = 1.0,
        sigma_s = 0.3,             # = c, isotropic scattering
        steady_state = True
        )

# beam ordinate (only for the uncollided normalisation: Phi_u = sum_nq w_n w_q psi_u = J e^{-sigma_t (x+1)} in the spot)
n0 = int(np.argmin(np.abs(model.mu)))
q0 = 0
assert abs(model.mu[n0]) < 1e-12 and model.phi[q0] == 0.0
J = 1.0
sig_t, sig_s = model.sigma_t[0], model.sigma_s[0, 0, 0]
spot = lambda u: np.where(np.abs(u) < 0.25, 1.0, 0.0)
decay = lambda x: np.exp(-sig_t * (x + 1.0))

# first-collision source Q_c = sigma_s / (4 pi) Phi_u, isotropic; BCs: none set = vacuum on every face
model.set_source([
             lambda x: sig_s / (4 * np.pi) * J * decay(x),
             spot,
             spot,
             lambda mu: 1 + 0 * mu,
             lambda phi: 1 + 0 * phi,
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
Phi_c = model.compute_scalar_flux()
tntt = ttbox.tntt
Phi_u = tntt.TT([tn.tensor(v, dtype=tn.float64)[None, :, None] for v in
                 (J * decay(model.x_r), spot(model.y_r), spot(model.z_r), np.ones(1), np.ones(1))])
Phi = (Phi_u + Phi_c).round(1e-10)
print(f"Phi_c ranks {list(Phi_c.R)}, Phi = Phi_u + Phi_c ranks {list(Phi.R)}")
ttbox.save_tt(Phi_c, HERE / "phi_c_tt_beam.pt", vars=[model.x_r, model.y_r, model.z_r])
ttbox.save_tt(Phi, HERE / "phi_tt_beam.pt", vars=[model.x_r, model.y_r, model.z_r])

def plane_xy(Phi, kz):
    """Phi(x_r, y_r) at the reconstruction point z_r[kz]: contract the z, g, t cores."""
    c_x, c_y, c_z, c_g, c_t = Phi.cores
    right = c_z[:, kz, :] @ c_g[:, 0, :] @ c_t[:, 0, :]          # (r_2, 1)
    return tn.einsum('aib,bjc,cd->ij', c_x, c_y, right).numpy()

kz = int(np.argmin(np.abs(model.z_r)))                          # plane closest to z = 0
ky = int(np.argmin(np.abs(model.y_r)))
plane = plane_xy(Phi, kz)
plane_c = plane_xy(Phi_c, kz)
line, line_c = plane[:, ky], plane_c[:, ky]                     # centreline y ~ 0, z ~ 0
np.savez(HERE / "beam_flux.npz", x_r=model.x_r, y_r=model.y_r, z_plane=model.z_r[kz], plane=plane, line=line,
         plane_c=plane_c, line_c=line_c)

# comparison with the unsplit run ../P0 (same grid; its uncollided part is the DG solution, with the Gibbs undershoot)
ref = np.load(HERE.parent / "P0" / "beam_flux.npz")
print(f"plane z = {model.z_r[kz]:.4f}: min Phi = {plane.min():.4f}, min Phi_c = {plane_c.min():.3e}  "
      f"(unsplit P0: min Phi = {ref['plane'].min():.4f})")
out = (np.abs(model.y_r) > 0.3)[None, :] | np.zeros_like(plane, dtype=bool)
print(f"||Phi - Phi_P0|| / ||Phi_P0|| on the plane: all {np.linalg.norm(plane - ref['plane']) / np.linalg.norm(ref['plane']):.3e}, "
      f"outside the spot (|y| > 0.3) {np.linalg.norm((plane - ref['plane'])[out]) / np.linalg.norm(ref['plane'][out]):.3e}")
for xv in (-0.9, -0.5, 0.0, 0.5, 0.9):
    i = int(np.argmin(np.abs(model.x_r - xv)))
    print(f"  x = {model.x_r[i]:+.4f}: Phi = {line[i]:.5f} (Phi_u {J * decay(model.x_r[i]):.5f} + Phi_c {line_c[i]:.5f}), "
          f"unsplit P0 {ref['line'][i]:.5f}")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.5))
im = ax1.imshow(np.log10(np.maximum(plane_c.T, 1e-12)), origin="lower", aspect="equal",
                extent=[model.x_r[0], model.x_r[-1], model.y_r[0], model.y_r[-1]], cmap="viridis")
fig.colorbar(im, ax=ax1, label=r"$\log_{10}\Phi_c$")
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_title(f"collided scalar flux, z = {model.z_r[kz]:.3f}")
ax2.semilogy(model.x_r, line, label=r"$\Phi_u + \Phi_c$ (split)")
ax2.semilogy(model.x_r, ref["line"], ":", label="unsplit P0")
ax2.semilogy(model.x_r, line_c, label=r"$\Phi_c$")
ax2.semilogy(model.x_r, J * decay(model.x_r), "k--", label=r"$\Phi_u = J e^{-\sigma_t (x+1)}$")
ax2.set_xlabel("x"); ax2.set_ylabel(r"$\Phi(x, 0, 0)$"); ax2.legend(); ax2.grid(alpha=0.4)
kx = int(np.argmin(np.abs(model.x_r)))
ax3.plot(model.y_r, plane[kx, :], label="split")
ax3.plot(model.y_r, ref["plane"][kx, :], ":", label="unsplit P0")
ax3.plot(model.y_r, plane_c[kx, :], label=r"$\Phi_c$")
ax3.set_xlabel("y"); ax3.set_ylabel(rf"$\Phi(x = {model.x_r[kx]:.3f}, y, 0)$"); ax3.legend(); ax3.grid(alpha=0.4)
fig.tight_layout()
fig.savefig(HERE / "beam_flux.png", dpi=150)

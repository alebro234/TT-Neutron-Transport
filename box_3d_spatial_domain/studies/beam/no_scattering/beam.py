"""
Normal beam along +x into the box [-1, 1]^3, no scattering (c = 0), steady state: purely uncollided transport.
Exact solution: psi = psi_in e^{-sigma_t (x+1)} 1(|y| < 1/4) 1(|z| < 1/4) on the beam ordinate Omega = (1, 0, 0), zero on
every other ordinate, so Phi = J e^{-sigma_t (x+1)} 1(|y| < 1/4) 1(|z| < 1/4). Compared both with the exact Phi and with the
exact psi projected onto the DG space (in y, z the solution is the L2 projection of the spot, since Omega_y = Omega_z = 0).

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
        sigma_s = 0.0,             # c = 0: no scattering
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

# ---------------------------------------------------------------------------
# comparison with the analytical uncollided solution
# ---------------------------------------------------------------------------
tntt = ttbox.tntt
spot = lambda u: np.where(np.abs(u) < 0.25, 1.0, 0.0)
decay = lambda x: np.exp(-model.sigma_t[0] * (x + 1.0))

def modal(f, d):
    """L2 projection of f onto the p-th degree DG space of axis d: c_ia = (2a+1)/h int_{K_i} f P_a."""
    N, h, _ = model._axis(d)
    return model._project(f, d).reshape(N, model.p + 1) * (2 * np.arange(model.p + 1) + 1) / h

def rank1(vectors):
    return tntt.TT([tn.tensor(np.asarray(v, dtype=float).ravel())[None, :, None] for v in vectors])

e_mu = np.zeros(model.N_mu); e_mu[n0] = 1.0 / model.w_mu[n0]
e_phi = np.zeros(model.N_phi); e_phi[q0] = J / model.w_phi[q0]
psi_h = rank1([modal(decay, 0), modal(spot, 1), modal(spot, 2), e_mu, e_phi, [1], [1]])   # exact psi in the DG space
print(f"exact psi (DG projection): true residual ||A psi_h - F||/||F|| = "
      f"{float((model.A @ psi_h - model.F).norm() / model.F.norm()):.3e}  (DG error in x only)")
print(f"||psi_AMEn - psi_h|| / ||psi_h|| = {float((model.psi_modal - psi_h).norm() / psi_h.norm()):.3e}")
# off-ordinate content: psi_off = psi - psi restricted to the beam ordinate (other mu, phi slices zeroed)
c = list(model.psi_modal.cores)
c_mu, c_phi = tn.zeros_like(c[3]), tn.zeros_like(c[4])
c_mu[:, n0, :], c_phi[:, q0, :] = c[3][:, n0, :], c[4][:, q0, :]
psi_off = model.psi_modal - tntt.TT(c[:3] + [c_mu, c_phi] + c[5:])
print(f"psi off the beam ordinate: ||psi_off|| / ||psi|| = {float(psi_off.norm() / model.psi_modal.norm()):.3e}")

# scalar flux at the reconstruction points, vs exact (discontinuous spot) and vs the DG projection of the spot
Phi_ex = rank1([decay(model.x_r), spot(model.y_r), spot(model.z_r), [1], [1]])
Phi_h = model._reconstruct(rank1([modal(decay, 0), modal(spot, 1), modal(spot, 2), [1], [1]]) * J)
for name, ref in (("exact", Phi_ex), ("DG-projected exact", Phi_h)):
    print(f"||Phi - Phi_{name}|| / ||Phi_{name}|| = {float((Phi - ref).norm() / ref.norm()):.3e}  (reconstruction points)")
print(f"||Phi_h - Phi_exact|| / ||Phi_exact|| = {float((Phi_h - Phi_ex).norm() / Phi_ex.norm()):.3e}  (DG approximation of the spot)")

kz = int(np.argmin(np.abs(model.z_r)))                          # plane closest to z = 0
ky = int(np.argmin(np.abs(model.y_r)))
kx = int(np.argmin(np.abs(model.x_r)))
plane = plane_xy(Phi, kz)
line = plane[:, ky]                                             # centreline y ~ 0, z ~ 0
prof = plane[kx, :]                                             # transverse profile at x ~ 0
line_ex = J * decay(model.x_r)
print(f"centreline: max |Phi - J e^-(x+1)| / J e^-(x+1) = {np.max(np.abs(line - line_ex) / line_ex):.3e}")
for xv in (-0.9, -0.5, 0.0, 0.5, 0.9):
    i = int(np.argmin(np.abs(model.x_r - xv)))
    print(f"  x = {model.x_r[i]:+.4f}: Phi = {line[i]:.6f}, exact = {line_ex[i]:.6f}")
print(f"plane z = {model.z_r[kz]:.4f}: min {plane.min():.4f}, max {plane.max():.4f}; "
      f"outside the spot max |Phi| = {np.abs(plane[:, np.abs(model.y_r) > 0.3]).max():.3e}")
np.savez(HERE / "beam_flux.npz", x_r=model.x_r, y_r=model.y_r, z_plane=model.z_r[kz], plane=plane, line=line, prof=prof)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))
im = ax1.imshow(plane.T, origin="lower", aspect="equal",
                extent=[model.x_r[0], model.x_r[-1], model.y_r[0], model.y_r[-1]], cmap="viridis")
fig.colorbar(im, ax=ax1, label=r"$\Phi$")
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_title(f"scalar flux, c = 0, z = {model.z_r[kz]:.3f}")
ax2.semilogy(model.x_r, np.abs(line - line_ex) / line_ex)
ax2.set_xlabel("x"); ax2.set_ylabel(r"$|\Phi - J e^{-(x+1)}| / J e^{-(x+1)}$ at $y, z \approx 0$"); ax2.grid(alpha=0.4)
ax3.plot(model.y_r, prof, ".-", ms=2, label="TT")
ax3.plot(model.y_r, J * decay(model.x_r[kx]) * spot(model.y_r), "k--", label="exact")
ax3.set_xlim(-0.4, 0.4); ax3.set_xlabel("y"); ax3.set_ylabel(rf"$\Phi(x = {model.x_r[kx]:.3f}, y, 0)$"); ax3.legend()
fig.tight_layout()
fig.savefig(HERE / "beam_flux.png", dpi=150)

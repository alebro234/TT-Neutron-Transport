"""
Time-dependent one-speed normal beam ("light pulse") into the box [-1, 1]^3, isotropic scattering.

sigma_t = v = 1, sigma_s = c = 0.9, zero initial condition, vacuum everywhere except the beam spot |y|, |z| < 1/4 on
the face x = -1, along the ordinate Omega = (1, 0, 0) (N_mu odd -> mu = 0 is a node, phi = 0 is a node), with unit
partial current J = 1 while 0 < t <= 0.2 T and zero afterwards. Backward Euler all-at-once in time, AMEn on the whole
space-angle-time tensor.

    python light_pulse.py [T nswp kickrank threads]      outputs in light_pulse/
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import torch as tn
import torchtt as tntt
import tt_transport_box.box as ttbox

tn.set_default_dtype(tn.float64)
T = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0          # final time; the front crosses the box at t = 2
NSWP = int(sys.argv[2]) if len(sys.argv) > 2 else 20
KICKRANK = int(sys.argv[3]) if len(sys.argv) > 3 else 4
N_THREADS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
tn.set_num_threads(N_THREADS)
HERE = Path(__file__).resolve().parent / "light_pulse"
HERE.mkdir(exist_ok=True)

# logging: Python prints and the C++ AMEn output (file descriptors 1 and 2) all go to light_pulse.log
_log = open(HERE / "light_pulse.log", "w", buffering=1)
os.dup2(_log.fileno(), 1)
os.dup2(_log.fileno(), 2)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {Path(__file__).name}, torch threads {tn.get_num_threads()}, "
      f"T = {T}, nswp = {NSWP}, kickrank = {KICKRANK}")

c = 0.9
T_BEAM = 0.2 * T
model = ttbox.BoxTransport(
        N_x = 150,
        N_y = 150,
        N_z = 150,
        p = 1,
        L_x = (-1.0, 1.0),
        L_y = (-1.0, 1.0),
        L_z = (-1.0, 1.0),
        N_mu = 125,                # odd: mu = 0 is a node, so Omega = (1, 0, 0) is an ordinate
        N_phi = 125,
        sigma_t = 1.0,
        sigma_s = c,               # isotropic scattering
        v = 1.0,
        N_t = 500,
        T = T,
        steady_state = False
        )
print(f"dt = {model.dt:.4e}, h = {model.h_x:.4e}, v dt / h = {model.v[0] * model.dt / model.h_x:.3f}, "
      f"beam on for t <= {T_BEAM}")

# beam ordinate: mu_n0 = 0 (eta = 1) and phi_0 = 0 (cos = 1)
n0 = int(np.argmin(np.abs(model.mu)))
q0 = 0
assert abs(model.mu[n0]) < 1e-12 and model.phi[q0] == 0.0
J = 1.0

def beam_mu(mu):
    out = np.zeros_like(mu)
    out[n0] = 1.0 / model.w_mu[n0]
    return out

def beam_phi(phi):
    out = np.zeros_like(phi)
    out[q0] = J / model.w_phi[q0]
    return out

model.set_IC(0)
model.set_BC('L_x', [
             lambda y: np.where(np.abs(y) < 0.25, 1.0, 0.0),
             lambda z: np.where(np.abs(z) < 0.25, 1.0, 0.0),
             beam_mu,
             beam_phi,
             [1],
             lambda t: np.where(t <= T_BEAM + 1e-12, 1.0, 0.0)     # pulse: on for 0 < t <= 0.2 T
             ])

# solve
t0 = time.time()
model.construct_tt_operator()
model.construct_tt_rhs(round_eps=1e-12)
model.run_amen(nswp=NSWP, eps=1e-6, kickrank=KICKRANK, round_eps=1e-8)
print(f"total solve time {time.time() - t0:.1f} s, psi ranks {list(model.psi_modal.R)}")
ttbox.save_tt(model.psi_modal, HERE / "psi_modal.pt")
# true residual; A @ psi in full would have ranks rank(A) * rank(psi), so the product is formed by AMEn (eps 1e-10)
Apsi = tntt.amen_mv(model.A, model.psi_modal, eps=1e-10)
print(f"true residual ||A psi - F|| / ||F|| = {float((Apsi - model.F).norm() / model.F.norm()):.3e}")
del Apsi

# scalar flux as a TT over (x_r, y_r, z_r, g, t)
Phi = model.compute_scalar_flux()
ttbox.save_tt(Phi, HERE / "phi_tt.pt", vars=[model.x_r, model.y_r, model.z_r, model.t])
c_x, c_y, c_z, c_g, c_t = Phi.cores
g0 = c_g[:, 0, :]

def plane_xy(kz, kt):
    right = c_z[:, kz, :] @ g0 @ c_t[:, kt, :]                    # (r_2, 1)
    return tn.einsum('aib,bjc,cd->ij', c_x, c_y, right).numpy()

def plane_yz(kx, kt):
    left = c_x[:, kx, :]                                           # (1, r_1)
    right = g0 @ c_t[:, kt, :]
    return tn.einsum('ab,bjc,ckd,de->jk', left, c_y, c_z, right).numpy()

def point_t(kx, ky, kz):
    """Phi(x_r[kx], y_r[ky], z_r[kz], t) for all t."""
    v = c_x[:, kx, :] @ c_y[:, ky, :] @ c_z[:, kz, :] @ g0         # (1, r_4)
    return (v @ c_t[:, :, 0]).ravel()

kx1 = len(model.x_r) - 1                                           # reconstruction point closest to x = +1
ky0 = int(np.argmin(np.abs(model.y_r)))
kz0 = int(np.argmin(np.abs(model.z_r)))
ext_yz = [model.y_r[0], model.y_r[-1], model.z_r[0], model.z_r[-1]]
ext_xy = [model.x_r[0], model.x_r[-1], model.y_r[0], model.y_r[-1]]

# --- scalar flux on the exit face x = +1 at several times, and on the axis vs time ---
t_show = [0.25 * T_BEAM, 0.5 * T_BEAM, T_BEAM, 1.5, 2.0, 2.0 + T_BEAM, 0.75 * T, T]
kts = [int(np.argmin(np.abs(model.t - ts))) for ts in t_show]
planes = [plane_yz(kx1, kt) for kt in kts]
vmax = np.log10(max(p.max() for p in planes))
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
for ax, kt, pl in zip(axs.ravel(), kts, planes):
    im = ax.imshow(np.log10(np.maximum(pl.T, 1e-12)), origin="lower", aspect="equal", extent=ext_yz,
                   cmap="viridis", vmin=vmax - 6, vmax=vmax)
    ax.set_title(f"t = {model.t[kt]:.3f}"); ax.set_xlabel("y"); ax.set_ylabel("z")
fig.colorbar(im, ax=axs, label=r"$\log_{10}\Phi$", shrink=0.8)
fig.suptitle(f"scalar flux on the face x = {model.x_r[kx1]:.4f}")
fig.savefig(HERE / "exit_face.png", dpi=130)
plt.close(fig)

phi_axis = point_t(kx1, ky0, kz0)
ix_mid = int(np.argmin(np.abs(model.x_r)))
phi_mid = point_t(ix_mid, ky0, kz0)
np.savez(HERE / "light_pulse.npz", t=model.t, phi_exit_axis=phi_axis, phi_center=phi_mid,
         x_r=model.x_r, y_r=model.y_r, z_r=model.z_r, kt_exit=kts, planes_exit=np.array(planes))
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(model.t, np.maximum(phi_axis, 1e-16), label=f"x = {model.x_r[kx1]:.3f}")
ax.semilogy(model.t, np.maximum(phi_mid, 1e-16), label=f"x = {model.x_r[ix_mid]:.3f}")
for ts, lab in [(T_BEAM, "beam off"), (2.0, "front at x=1"), (2.0 + T_BEAM, "tail at x=1")]:
    ax.axvline(ts, color="k", lw=0.6, ls=":")
ax.set_xlabel("t"); ax.set_ylabel(r"$\Phi(x, 0, 0, t)$"); ax.grid(alpha=0.4, which="both"); ax.legend()
fig.tight_layout()
fig.savefig(HERE / "axis_vs_t.png", dpi=150)
plt.close(fig)

# --- movie of the xy plane at z ~ 0 ---
fig, ax = plt.subplots(figsize=(6.5, 5.5))
pl = plane_xy(kz0, 0)
vmax = 0.0                                                         # log10 of the beam flux J = 1
im = ax.imshow(np.log10(np.maximum(pl.T, 1e-12)), origin="lower", aspect="equal", extent=ext_xy,
               cmap="viridis", vmin=vmax - 6, vmax=vmax)
fig.colorbar(im, ax=ax, label=r"$\log_{10}\Phi$")
ax.set_xlabel("x"); ax.set_ylabel("y")
title = ax.set_title("")
writer = FFMpegWriter(fps=25, bitrate=4000)
with writer.saving(fig, HERE / "xy_plane.mp4", dpi=120):
    for kt in range(model.N_t):
        im.set_data(np.log10(np.maximum(plane_xy(kz0, kt).T, 1e-12)))
        title.set_text(f"z = {model.z_r[kz0]:.3f}, t = {model.t[kt]:.3f}")
        writer.grab_frame()
plt.close(fig)
print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  done, total {time.time() - t0:.1f} s")

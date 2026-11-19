"""
Normal 2 MeV beam along +x into the box [-1, 1]^3 (cm) filled with graphite, P20 scattering, 128 groups,
steady state, no first-collision splitting.  Same geometry and beam as ../c_convergence/beam.py.

Data: graphite_mgxs.npz, copied from openmc/graphite/results_P20_g_128 (same md5): sigma_t (G,), g = 0 highest,
scatter_legendre[l, g, g'] = sigma_s^l(g' -> g) (row = receiving group), l = 0..20.  BoxTransport takes
sigma_s[l][g', g] (row = incoming group), so the transpose is passed, as in the slab script
slab/studies/montecarlo/P20g128/tt_graphite_slab.py.

The beam enters through the face x = -1 on the square |y|, |z| < 0.25 along Omega = (1, 0, 0), in group 0
([1.9, 2.1] MeV, contains E0 = 2 MeV).  With N_mu odd the Gauss-Legendre rule has a mu = 0 node (eta = 1), and
phi = 0 is always on the azimuthal grid, so Omega = (1, 0, 0) is an ordinate of the S_N set:
    psi_in(mu_n, phi_q, g) = J / (w_n w_q) delta_g0  on (n, q) = (n_0, 0),  0 elsewhere,
so the incoming partial current per unit spot area is J = 1.  All other faces are vacuum.

Run with the TT environment (args optional):
    python beam.py [N_cells] [N_mu] [N_phi] [nswp] [threads] [kickrank] [prec]
  prec = 'c': local central-Jacobi preconditioner, 80 GMRES iterations x 4 resets (amen_solve called directly,
  BoxTransport.run_amen does not expose these options); default: run_amen as in the dummy-material runs
Outputs in run_N{N_cells}_mu{N_mu}_phi{N_phi}[_kr{kickrank}][_prec]/ (suffixes only if non-default): beam.log (AMEn output incl. Maxres per sweep), phi_tt_beam.pt,
beam_flux.npz, beam_flux.png, maxres.png.
"""
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch as tn
import torchtt as tntt
import tt_transport_box.box as ttbox

tn.set_default_dtype(tn.float64)
argv = sys.argv[1:]
N_CELLS = int(argv[0]) if len(argv) > 0 else 40
N_MU = int(argv[1]) if len(argv) > 1 else 21              # odd (mu = 0 node) and >= L_max + 1 = 21
N_PHI = int(argv[2]) if len(argv) > 2 else 44             # >= 2 L_max + 1 = 41
NSWP = int(argv[3]) if len(argv) > 3 else 15
N_THREADS = int(argv[4]) if len(argv) > 4 else 6
KICK = int(argv[5]) if len(argv) > 5 else 2               # 2 as in the dummy-material runs
PREC = argv[6] if len(argv) > 6 else None                 # None or 'c'
tn.set_num_threads(N_THREADS)
SRC = Path(__file__).resolve().parent
HERE = SRC / (f"run_N{N_CELLS}_mu{N_MU}_phi{N_PHI}" + (f"_kr{KICK}" if KICK != 2 else "") + ("_prec" if PREC else ""))
HERE.mkdir(exist_ok=True)

# logging: Python prints and the C++ AMEn output (file descriptors 1 and 2) all go to beam.log
_log = open(HERE / "beam.log", "w", buffering=1)
os.dup2(_log.fileno(), 1)
os.dup2(_log.fileno(), 2)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {Path(__file__).name} in {HERE.name}, torch threads {tn.get_num_threads()}")

# ---------------- graphite P20, 128 groups ----------------
xs = np.load(SRC / "graphite_mgxs.npz")
sigma_t = xs["sigma_t"]
S = xs["scatter_legendre"]                                  # S[l, g, g'] = sigma_s^l(g' -> g)
L_MAX = S.shape[0] - 1
G = len(sigma_t)
print(f"graphite {xs['case']}: G = {G}, L_max = {L_MAX}, density {float(xs['density'])} g/cm3, "
      f"sigma_t in [{sigma_t.min():.3f}, {sigma_t.max():.3f}] 1/cm, "
      f"sigma_s0/sigma_t in [{(S[0].sum(0) / sigma_t).min():.6f}, {(S[0].sum(0) / sigma_t).max():.6f}]")
print(f"box width 2 cm = {2 * sigma_t[0]:.3f} mfp in group 0, {2 * sigma_t.max():.3f} mfp at most")

model = ttbox.BoxTransport(
        N_x = N_CELLS,
        N_y = N_CELLS,
        N_z = N_CELLS,
        p = 1,
        L_x = (-1.0, 1.0),
        L_y = (-1.0, 1.0),
        L_z = (-1.0, 1.0),
        N_mu = N_MU,
        N_phi = N_PHI,
        sigma_t = sigma_t,
        sigma_s = [S[l].T for l in range(L_MAX + 1)],       # BoxTransport: row = incoming group
        E_edges = xs["group_edges_desc"],
        steady_state = True
        )

# beam ordinate: mu_n0 = 0 (eta = 1) and phi_0 = 0 (cos = 1)
n0 = int(np.argmin(np.abs(model.mu)))
q0 = 0
assert abs(model.mu[n0]) < 1e-12 and model.phi[q0] == 0.0
J = 1.0
print(f"beam ordinate: mu = {model.mu[n0]:.1e}, eta = {model.eta[n0]:.6f}, phi = {model.phi[q0]}, "
      f"w_mu = {model.w_mu[n0]:.4e}, w_phi = {model.w_phi[q0]:.4e}, group 0 = "
      f"[{model.E_edges[1]:.3e}, {model.E_edges[0]:.3e}] eV")

def beam_mu(mu):
    out = np.zeros_like(mu)
    out[n0] = 1.0 / model.w_mu[n0]
    return out

def beam_phi(phi):
    out = np.zeros_like(phi)
    out[q0] = J / model.w_phi[q0]
    return out

beam_E = np.zeros(G)
beam_E[0] = 1.0

# BC = normal beam along x-axis on the spot |y|, |z| < 0.25 of the face x = -1, group 0
model.set_BC('L_x', [
             lambda y: np.where(np.abs(y) < 0.25, 1.0, 0.0),
             lambda z: np.where(np.abs(z) < 0.25, 1.0, 0.0),
             beam_mu,
             beam_phi,
             beam_E
             ])

# solve, same AMEn settings as the dummy-material beam runs (c_convergence, P0, P5)
t0 = time.time()
model.construct_tt_operator()
print(f"operator built in {time.time() - t0:.1f} s, raw ranks {model.A_ranks_raw}")
model.construct_tt_rhs(round_eps=1e-12)
if PREC is None:
    model.run_amen(nswp=NSWP, eps=1e-6, kickrank=KICK, round_eps=1e-8)
else:
    t_amen = time.time()
    psi = tntt.solvers.amen_solve(model.A, model.F, eps=1e-6, nswp=NSWP, kickrank=KICK, max_full=15000,
                                  local_iterations=80, resets=4, preconditioner=PREC, verbose=True)
    print(f"AMEn solver completed in {time.time() - t_amen:.2f} seconds.")
    model.psi_modal = psi.round(1e-8)
    model.psi = model._reconstruct(model.psi_modal)
print(f"total solve time {time.time() - t0:.1f} s, psi ranks {list(model.psi_modal.R)}")

tntt.save(model.psi_modal, str(HERE / "psi_modal_beam.pt"))          # keep the solution before anything else

# true residual ||A psi - F|| / ||F||; the exact product has (mu|phi) rank ~ 128 * rank(psi), well below 1 GB here
# (A.fast_matvec crashes in the torchtt C++ backend: "Cannot access data pointer of Tensor that doesn't have storage")
t1 = time.time()
res = model.residual()
print(f"true residual {res:.3e} ({time.time() - t1:.1f} s)")

# scalar flux as a TT over (x_r, y_r, z_r, g, t)
Phi = model.compute_scalar_flux()
ttbox.save_tt(Phi, HERE / "phi_tt_beam.pt", vars=[model.x_r, model.y_r, model.z_r])

def plane_xy(Phi, kz, g_weights):
    """sum_g g_weights[g] Phi_g(x_r, y_r) at the reconstruction point z_r[kz]: contract the z, g, t cores."""
    c_x, c_y, c_z, c_g, c_t = Phi.cores
    w = tn.tensor(g_weights, dtype=c_g.dtype)
    right = c_z[:, kz, :] @ tn.einsum('agb,g->ab', c_g, w) @ c_t[:, 0, :]     # (r_2, 1)
    return tn.einsum('aib,bjc,cd->ij', c_x, c_y, right).numpy()

kz = int(np.argmin(np.abs(model.z_r)))                          # plane closest to z = 0
ky = int(np.argmin(np.abs(model.y_r)))
e0 = np.eye(G)[0]
plane_tot = plane_xy(Phi, kz, np.ones(G))
plane_g0 = plane_xy(Phi, kz, e0)
line_tot, line_g0 = plane_tot[:, ky], plane_g0[:, ky]          # centreline y ~ 0, z ~ 0

# group spectrum at the centre of the box (contract x, y, z at the points closest to 0)
kx = int(np.argmin(np.abs(model.x_r)))
c_x, c_y, c_z, c_g, c_t = Phi.cores
spec = tn.einsum('a,ab,bc,cgd,d->g', c_x[0, kx, :], c_y[:, ky, :], c_z[:, kz, :], c_g, c_t[:, 0, 0]).numpy()
np.savez(HERE / "beam_flux.npz", x_r=model.x_r, y_r=model.y_r, z_plane=model.z_r[kz],
         plane_tot=plane_tot, plane_g0=plane_g0, line_tot=line_tot, line_g0=line_g0,
         E_edges=model.E_edges, spectrum_centre=spec, true_residual=res)

fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))
im = axs[0].imshow(np.log10(np.maximum(plane_tot.T, 1e-12)), origin="lower", aspect="equal",
                   extent=[model.x_r[0], model.x_r[-1], model.y_r[0], model.y_r[-1]], cmap="viridis")
fig.colorbar(im, ax=axs[0], label=r"$\log_{10}\sum_g\Phi_g$")
axs[0].set_xlabel("x [cm]"); axs[0].set_ylabel("y [cm]"); axs[0].set_title(f"total scalar flux, z = {model.z_r[kz]:.3f}")
axs[1].semilogy(model.x_r, line_g0, label="TT, group 0")
axs[1].semilogy(model.x_r, line_tot - line_g0, label="TT, groups 1..127")
axs[1].semilogy(model.x_r, J * np.exp(-sigma_t[0] * (model.x_r + 1.0)), "k--",
                label=r"uncollided $J e^{-\sigma_{t,0} (x+1)}$")
axs[1].set_xlabel("x [cm]"); axs[1].set_ylabel(r"$\Phi(x, 0, 0)$"); axs[1].legend(); axs[1].grid(alpha=0.4)
E = model.E
axs[2].loglog(E, np.maximum(spec, 1e-30), drawstyle="steps-mid")
axs[2].set_xlabel("E [eV]"); axs[2].set_ylabel(r"$\Phi_g$ at box centre"); axs[2].grid(alpha=0.4, which="both")
fig.tight_layout()
fig.savefig(HERE / "beam_flux.png", dpi=150)

# AMEn convergence: Maxres per sweep from this log
log = (HERE / "beam.log").read_text()
maxres = [float(m) for m in re.findall(r"Maxres\s+([0-9.eE+-]+)", log)]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.semilogy(range(1, len(maxres) + 1), maxres, "o-", ms=4, label=f"Maxres (true res. {res:.1e})")
ax.axhline(1e-4, color="k", ls=":", lw=0.8, label="1e-4")
ax.set_xlabel("sweep"); ax.set_ylabel("Maxres"); ax.grid(alpha=0.4, which="both"); ax.legend()
ax.set_title(f"graphite P20 g128 beam, N={N_CELLS}, N_mu={N_MU}, N_phi={N_PHI}, kickrank {KICK}")
fig.tight_layout()
fig.savefig(HERE / "maxres.png", dpi=150)
print(f"Maxres per sweep: {' '.join(f'{m:.2e}' for m in maxres)}")

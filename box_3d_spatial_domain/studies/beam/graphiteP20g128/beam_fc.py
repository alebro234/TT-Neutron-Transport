"""
Graphite P20 g128 normal beam (same problem as beam.py) with FIRST-COLLISION splitting, steady state.

psi = psi_unc + psi_c. The uncollided flux of the beam (Omega_0 = (1, 0, 0), group 0, spot |y|, |z| < 1/4 on x = -1,
partial current J = 1) is analytic:
    psi_unc(x, Omega, g) = J e^{-sigma_t0 (x + 1)} chi(y) chi(z) delta(Omega - Omega_0) delta_g0,
    phi_unc = J e^{-sigma_t0 (x + 1)} chi(y) chi(z)  (group 0),
and the collided flux solves the same transport problem with VACUUM boundaries and the first-collision source
    q_1(x, Omega, g) = sum_l sigma_{s,l}(0 -> g) (2l+1)/(4 pi) P_l(Omega . Omega_0) phi_unc(x),   Omega . Omega_0 = Omega_x.
On the ordinates this is exactly what the discrete scattering operator sum_l K_l (x) Sigma_l gives when applied to the
discrete beam psi = phi_unc / (w_n0 w_q0) on the ordinate Omega_0 (addition theorem, pointwise on every ordinate).
Space part: rank 1 (load vectors of e^{-sigma_t0 (x+1)}, chi(y), chi(z)); (mu, phi, g) part: a 3-core TT of rank <= 21.

    python beam_fc.py [N_cells N_mu N_phi nswp threads kickrank]       outputs in run_fc_N{..}_mu{..}_phi{..}_kr{..}/
"""
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch as tn
import torchtt as tntt
from scipy.special import eval_legendre
import tt_transport_box.box as ttbox

tn.set_default_dtype(tn.float64)
argv = sys.argv[1:]
N_CELLS = int(argv[0]) if len(argv) > 0 else 40
N_MU = int(argv[1]) if len(argv) > 1 else 21
N_PHI = int(argv[2]) if len(argv) > 2 else 44
NSWP = int(argv[3]) if len(argv) > 3 else 20
N_THREADS = int(argv[4]) if len(argv) > 4 else 8
KICK = int(argv[5]) if len(argv) > 5 else 2
EPS = 1e-6
tn.set_num_threads(N_THREADS)
SRC = Path(__file__).resolve().parent
HERE = SRC / f"run_fc_N{N_CELLS}_mu{N_MU}_phi{N_PHI}_kr{KICK}"
HERE.mkdir(exist_ok=True)

_log = open(HERE / "beam.log", "w", buffering=1)
os.dup2(_log.fileno(), 1)
os.dup2(_log.fileno(), 2)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {Path(__file__).name} in {HERE.name}, torch threads {tn.get_num_threads()}, "
      f"nswp {NSWP}, kickrank {KICK}, eps {EPS}")

# ---------------- graphite P20, 128 groups ----------------
xs = np.load(SRC / "graphite_mgxs.npz")
sigma_t = xs["sigma_t"]
S = xs["scatter_legendre"]                                  # S[l, g, g'] = sigma_s^l(g' -> g)
L_MAX = S.shape[0] - 1
G = len(sigma_t)

model = ttbox.BoxTransport(
        N_x = N_CELLS, N_y = N_CELLS, N_z = N_CELLS, p = 1,
        L_x = (-1.0, 1.0), L_y = (-1.0, 1.0), L_z = (-1.0, 1.0),
        N_mu = N_MU, N_phi = N_PHI,
        sigma_t = sigma_t,
        sigma_s = [S[l].T for l in range(L_MAX + 1)],       # BoxTransport: row = incoming group
        E_edges = xs["group_edges_desc"],
        steady_state = True
        )

# ---------------- first-collision source ----------------
J = 1.0
s0 = sigma_t[0]
chi = lambda u: np.where(np.abs(u) < 0.25, 1.0, 0.0)
phi_unc_x = lambda x: J * np.exp(-s0 * (x + 1.0))
Omega_x = np.outer(model.eta, model.cos_phi)                # Omega . Omega_0 on the ordinates (N_mu, N_phi)
T = np.zeros((N_MU, N_PHI, G))
for l in range(L_MAX + 1):
    T += ((2 * l + 1) / (4 * np.pi) * eval_legendre(l, Omega_x))[:, :, None] * S[l][:, 0][None, None, :]
T_tt = tntt.TT(tn.tensor(T), eps=1e-12)                     # (mu, phi, g) part
space = [model._project(phi_unc_x, 0), model._project(chi, 1), model._project(chi, 2)]
cores = [tn.tensor(v)[None, :, None] for v in space] + list(T_tt.cores) + [tn.ones(1, 1, 1)]
model.q = tntt.TT(cores)
print(f"first-collision source: (mu, phi, g) ranks {list(T_tt.R)}, source ranks {list(model.q.R)}, "
      f"max |q_ang| {np.abs(T).max():.3e}")

# ---------------- solve the collided problem (vacuum everywhere) ----------------
t0 = time.time()
model.construct_tt_operator()
print(f"operator built in {time.time() - t0:.1f} s, raw ranks {model.A_ranks_raw}")
model.construct_tt_rhs(round_eps=1e-12)
model.run_amen(nswp=NSWP, eps=EPS, kickrank=KICK, round_eps=1e-8)
print(f"total solve time {time.time() - t0:.1f} s, psi_c ranks {list(model.psi_modal.R)}")
tntt.save(model.psi_modal, str(HERE / "psi_c_modal.pt"))

t1 = time.time()
res = model.residual()
print(f"true residual (collided system) {res:.3e} ({time.time() - t1:.1f} s)")

# ---------------- scalar flux: collided (TT) + uncollided (analytic, group 0) ----------------
Phi = model.compute_scalar_flux()
ttbox.save_tt(Phi, HERE / "phi_c_tt.pt", vars=[model.x_r, model.y_r, model.z_r])

def plane_xy(Phi, kz, g_weights):
    c_x, c_y, c_z, c_g, c_t = Phi.cores
    w = tn.tensor(g_weights, dtype=c_g.dtype)
    right = c_z[:, kz, :] @ tn.einsum('agb,g->ab', c_g, w) @ c_t[:, 0, :]
    return tn.einsum('aib,bjc,cd->ij', c_x, c_y, right).numpy()

kz = int(np.argmin(np.abs(model.z_r)))
ky = int(np.argmin(np.abs(model.y_r)))
kx = int(np.argmin(np.abs(model.x_r)))
e0 = np.eye(G)[0]
unc_plane = phi_unc_x(model.x_r)[:, None] * chi(model.y_r)[None, :] * chi(model.z_r[kz])
plane_c_tot = plane_xy(Phi, kz, np.ones(G))
plane_c_g0 = plane_xy(Phi, kz, e0)
plane_tot = plane_c_tot + unc_plane
line_c_g0, line_c_rest = plane_c_g0[:, ky], plane_c_tot[:, ky] - plane_c_g0[:, ky]
line_unc = unc_plane[:, ky]
c_x, c_y, c_z, c_g, c_t = Phi.cores
spec_c = tn.einsum('a,ab,bc,cgd,d->g', c_x[0, kx, :], c_y[:, ky, :], c_z[:, kz, :], c_g, c_t[:, 0, 0]).numpy()
np.savez(HERE / "beam_flux.npz", x_r=model.x_r, y_r=model.y_r, z_plane=model.z_r[kz], plane_tot=plane_tot,
         plane_c_tot=plane_c_tot, plane_c_g0=plane_c_g0, line_unc=line_unc, line_c_g0=line_c_g0,
         line_c_rest=line_c_rest, E_edges=model.E_edges, spectrum_c_centre=spec_c, true_residual=res)

fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))
im = axs[0].imshow(np.log10(np.maximum(plane_tot.T, 1e-12)), origin="lower", aspect="equal",
                   extent=[model.x_r[0], model.x_r[-1], model.y_r[0], model.y_r[-1]], cmap="viridis")
fig.colorbar(im, ax=axs[0], label=r"$\log_{10}\sum_g\Phi_g$")
axs[0].set_xlabel("x [cm]"); axs[0].set_ylabel("y [cm]")
axs[0].set_title(f"total scalar flux (unc. + coll.), z = {model.z_r[kz]:.3f}")
axs[1].semilogy(model.x_r, line_unc, "k--", label=r"uncollided $J e^{-\sigma_{t,0}(x+1)}$ (analytic)")
axs[1].semilogy(model.x_r, np.maximum(line_c_g0, 1e-16), label="collided, group 0")
axs[1].semilogy(model.x_r, np.maximum(line_c_rest, 1e-16), label="collided, groups 1..127")
axs[1].set_xlabel("x [cm]"); axs[1].set_ylabel(r"$\Phi(x, 0, 0)$"); axs[1].legend(); axs[1].grid(alpha=0.4)
axs[2].loglog(model.E, np.maximum(spec_c, 1e-30), drawstyle="steps-mid")
axs[2].set_xlabel("E [eV]"); axs[2].set_ylabel(r"collided $\Phi_g$ at box centre"); axs[2].grid(alpha=0.4, which="both")
fig.tight_layout()
fig.savefig(HERE / "beam_flux.png", dpi=150)

log = (HERE / "beam.log").read_text()
maxres = [float(m) for m in re.findall(r"Maxres\s+([0-9.eE+-]+)", log)]
old = SRC / "run_N40_mu21_phi44_kr4" / "beam.log"
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.semilogy(range(1, len(maxres) + 1), maxres, "o-", ms=4, label=f"first collision, kr {KICK} (true res. {res:.1e})")
if old.exists() and N_CELLS == 40 and N_MU == 21 and N_PHI == 44:
    mo = [float(m) for m in re.findall(r"Maxres\s+([0-9.eE+-]+)", old.read_text())]
    ax.semilogy(range(1, len(mo) + 1), mo, "s-", ms=4, label="beam BC, kr 4 (true res. 1.7e-3)")
ax.set_xlabel("sweep"); ax.set_ylabel("Maxres"); ax.grid(alpha=0.4, which="both"); ax.legend()
ax.set_title(f"graphite P20 g128 beam, N={N_CELLS}, N_mu={N_MU}, N_phi={N_PHI}")
fig.tight_layout()
fig.savefig(HERE / "maxres.png", dpi=150)
print(f"Maxres per sweep: {' '.join(f'{m:.2e}' for m in maxres)}")
print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  done")

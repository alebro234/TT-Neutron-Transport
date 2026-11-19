"""
Graphite slab, step 3: TT (SlabTransport) solution with the MG data of graphite_ce.py, compared with OpenMC MG and CE.

Same solver setup as ../water_slab/tt_water_slab.py. For case P0tr the data are transport corrected:
sigma_t -> sigma_tr (the "sigma_t" stored in mgxs.h5 / used below) and the isotropic matrix is the corrected
sigma~_s0 = sigma_s0 - diag(sigma_s1) (in-scatter form, as OpenMC builds it); the uncollided beam then decays
with sigma_tr,0, consistently with OpenMC MG.

Problem: steady state, slab 0 < z < L, vacuum on both faces, normal 2 MeV beam of unit intensity on z = 0.

The beam delta(mu - 1) cannot be represented on the Gauss-Legendre ordinates, so it is split off analytically
(first-collision source):
    uncollided :  psi_u = delta(Omega - e_z) exp(-sigma_t,0 z)       ->  phi_u,0(z) = exp(-sigma_t,0 z)  (group 0)
    collided   :  solved by TT with vacuum BCs and the first-collision source
                  q_g(z, mu) = sum_l (2l+1)/(4 pi) sigma_s^l(0 -> g) P_l(mu) exp(-sigma_t,0 z)
    phi_g = phi_collided,g + delta_g0 phi_u,0

Inputs : results_<case>/graphite_mgxs.npz (cross sections), graphite_mg_flux.npz / graphite_ce_flux.npz (OpenMC)
Outputs: tt_runs/<case>_Nmu{N_mu}_p{p}[_az]/  with  run.log, tt_graphite_slab_flux.npz, tt_graphite_slab_vs_openmc.png

Run with the TT environment:  /home/ale234/JRC/tt_python/tt_env/bin/python tt_graphite_slab.py <case> [N_mu] [p] [az]   (case = P0tr | P7)
  az: use SlabTransport(azimuthal_symmetry=True) -> N_phi = 1, only the m = 0 scattering terms (output dir *_az)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import eval_legendre
import tt_transport.slab as tt

HERE = Path(__file__).resolve().parent
CASE = sys.argv[1] if len(sys.argv) > 1 else 'P0tr'
RES = HERE / f'results_{CASE}'
argv = sys.argv[1:]                        # argv[1:] below = [N_mu, p, az] as in the water script


class Tee:
    """Write to the terminal and to the run log at the same time."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
xs = np.load(RES / 'graphite_mgxs.npz')
L = float(xs['L'])
sigma_t = xs['sigma_tr'] if CASE == 'P0tr' else xs['sigma_t']   # (G,), g = 0 highest; transport-corrected total for P0tr
S = xs['scatter_legendre']                 # S[l, g, g'] = sigma_s^l(g' -> g), row = receiving group
L_MAX = S.shape[0] - 1
G = len(sigma_t)
E_edges = xs['group_edges_desc']

# reference for the comparison: OpenMC MG if it was run for this case, otherwise the CE flux
REF = 'MG' if (RES / 'graphite_mg_flux.npz').exists() else 'CE'
ref = np.load(RES / ('graphite_mg_flux.npz' if REF == 'MG' else 'graphite_ce_flux.npz'))
ce = np.load(RES / 'graphite_ce_flux.npz')
N_z = len(ce['z'])                         # same mesh as the OpenMC tallies -> cell averages compare directly

# ---------------------------------------------------------------------------
# discretisation
# ---------------------------------------------------------------------------
P_DEG = int(argv[2]) if len(argv) > 2 else 1                                  # DG degree in z
N_MU = int(argv[1]) if len(argv) > 1 else 16
AZ_SYM = len(argv) > 3 and argv[3] == 'az'
N_PHI = 1 if AZ_SYM else L_MAX + 3         # problem is azimuthally symmetric; N_phi > L_max makes the m != 0
                                           # kernel terms vanish exactly on a phi-independent solution
N_RECON = 3                                # Gauss points per cell (cell averages are exact for p <= 2N_RECON-1)

OUT = HERE / 'tt_runs' / (f'{CASE}_Nmu{N_MU}_p{P_DEG}' + ('_az' if AZ_SYM else ''))
OUT.mkdir(parents=True, exist_ok=True)
_log = open(OUT / 'run.log', 'w')
sys.stdout = Tee(sys.__stdout__, _log)
sys.stderr = Tee(sys.__stderr__, _log)

model = tt.SlabTransport(N_z=N_z, p=P_DEG, L_z=L, N_mu=N_MU, N_phi=N_PHI,
                         sigma_t=sigma_t,
                         sigma_s=[S[l].T for l in range(L_MAX + 1)],   # SlabTransport: row = incoming group
                         E_edges=E_edges, steady_state=True, N_recon=N_RECON,
                         azimuthal_symmetry=AZ_SYM)

# ---------------------------------------------------------------------------
# first-collision source (sum of L_MAX + 1 rank-one terms) and vacuum boundaries
# ---------------------------------------------------------------------------
Q_z = model._project_z(lambda z: np.exp(-sigma_t[0] * z))
q = None
for l in range(L_MAX + 1):
    term = tt.vector_kron([Q_z,
                           (2 * l + 1) / (4 * np.pi) * eval_legendre(l, model.mu), # nice simmetry
                           np.ones(N_PHI),
                           S[l][:, 0],                          # sigma_s^l(0 -> g) for every receiving g
                           np.ones(1)])
    q = term if q is None else q + term
model.q = q.round(1e-12)
model.set_BC_left(0)
model.set_BC_right(0)

AMEN = dict(nswp=40, eps=1e-7, kickrank=4)
model.run_amen(**AMEN, verbose=True)

# ---------------------------------------------------------------------------
# solver report: AMEn settings, size of the operator A and of the solution psi, true residual
# ---------------------------------------------------------------------------
def tt_entries(T):
    return sum(c.numel() for c in T.cores)


A_tt, F_tt, psi_tt = model.A, model.F, model.psi_modal
print(f"\nAMEn: nswp={AMEN['nswp']}, eps={AMEN['eps']:.0e}, kickrank={AMEN['kickrank']}, max_full=15000, "
      f"round_eps=1e-8 (SlabTransport.run_amen defaults); torchtt amen_solve defaults: local solver GMRES "
      f"(local_iterations=40, resets=2, no preconditioner), x0 = ones, trunc_norm='res', C++ backend")
n_dof = int(np.prod(psi_tt.N))
n_A = tt_entries(A_tt)
print(f"operator A: mode sizes {list(A_tt.M)} x {list(A_tt.N)}  (z-dof, mu, phi, g, t)")
print(f"  ranks R = {list(A_tt.R)}")
print(f"  TT entries = {n_A},  dense = {n_dof}^2 = {float(n_dof)**2:.3e}  ->  compression {n_A / float(n_dof)**2:.2e}")
n_psi = tt_entries(psi_tt)
print(f"solution psi_modal: mode sizes N = {list(psi_tt.N)}  (z-dof = N_z (p+1), mu, phi, g, t)")
print(f"  ranks R = {list(psi_tt.R)}")
print("  core shapes = " + ", ".join(str(tuple(c.shape)) for c in psi_tt.cores))
print(f"  TT entries = {n_psi},  full tensor = {n_dof}  ->  compression {n_psi / n_dof:.2e}"
      f"  ({n_psi * 8 / 1024:.0f} KiB vs {n_dof * 8 / 1024**2:.1f} MiB in float64)")
res = float((A_tt @ psi_tt - F_tt).norm() / F_tt.norm())
print(f"true relative residual ||A psi - F|| / ||F|| = {res:.2e}  (psi rounded at 1e-8)")

# azimuthal symmetry check: psi must not depend on phi
psi_full = model.psi_modal.full().numpy()                                  # (z-dof, mu, phi, g, 1)
phi_dev = np.abs(psi_full - psi_full.mean(axis=2, keepdims=True)).max() / np.abs(psi_full).max()
print(f"max |psi - <psi>_phi| / max|psi| = {phi_dev:.2e}")

# ---------------------------------------------------------------------------
# cell-averaged scalar flux, plus the uncollided beam in group 0
# ---------------------------------------------------------------------------
phi_r = model.compute_scalar_flux().full().numpy()[:, :, 0]              # (N_z * N_recon, G) at z_r
phi_c = np.einsum('irg,r->ig', phi_r.reshape(N_z, N_RECON, G), model.w_xi) / 2  # cell averages

dz = L / N_z
z_lo = np.arange(N_z) * dz
phi_u = (np.exp(-sigma_t[0] * z_lo) - np.exp(-sigma_t[0] * (z_lo + dz))) / (sigma_t[0] * dz)
phi_tt = phi_c.copy()
phi_tt[:, 0] += phi_u

np.savez(OUT / 'tt_graphite_slab_flux.npz', z=model.z_mid, dz=dz, L=L, group_edges_desc=E_edges,
         phi=phi_tt, phi_collided=phi_c, phi_uncollided=phi_u, N_mu=N_MU, N_phi=N_PHI, az_sym=AZ_SYM, p=P_DEG,
         ranks=np.array(model.psi_modal.R))

# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
a, sa = ref['phi'], ref['phi_sd']
c = ce['phi']
with np.errstate(divide='ignore', invalid='ignore'):
    rel = phi_tt / a - 1
    zsc = (phi_tt - a) / sa
idx = [0, N_z // 6, N_z // 3, N_z // 2, 2 * N_z // 3, 5 * N_z // 6, N_z - 1]
print(f"\nTT: N_z={N_z}, p={P_DEG}, N_mu={N_MU}, N_phi={N_PHI}, az_sym={AZ_SYM}, L_max={L_MAX}, ranks={model.psi_modal.R}")
print(f"reference: OpenMC {REF}; global ||TT-{REF}||_F/||{REF}||_F = {np.linalg.norm(phi_tt - a) / np.linalg.norm(a):.4f}, "
      f"||TT-CE||_F/||CE||_F = {np.linalg.norm(phi_tt - c) / np.linalg.norm(c):.4f}")
tot_rel = phi_tt.sum(1) / c.sum(1) - 1
print("total flux sum_g phi_g, TT/CE-1 at z = " + ", ".join(f"{model.z_mid[i]:.1f}" for i in idx) + ": "
      + " ".join(f"{tot_rel[i]:+.4f}" for i in idx) + f"   (integrated {phi_tt.sum() / c.sum() - 1:+.4f})")
print(f"g | int TT/{REF}-1 | int TT/CE-1 | TT/{REF}-1 at z = " + ", ".join(f"{model.z_mid[i]:.1f}" for i in idx)
      + " | median|z| | frac|z|>3")
for g in range(G):
    print(f"{g} | {phi_tt[:, g].sum() / a[:, g].sum() - 1:+.4f}     | {phi_tt[:, g].sum() / c[:, g].sum() - 1:+.4f}     | "
          + " ".join(f"{rel[i, g]:+.3f}" for i in idx)
          + f" | {np.nanmedian(np.abs(zsc[:, g])):.1f} | {np.nanmean(np.abs(zsc[:, g]) > 3):.2f}")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

z = model.z_mid
PLOT_G = [g for g in range(G) if np.any(c[:, g] > 0)]   # groups with flux (g9 is empty in graphite)
cols = plt.cm.viridis(np.linspace(0, 0.95, G))
fig, ax = plt.subplots(2, 1, figsize=(10, 9), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
for g in PLOT_G:
    ax[0].semilogy(z, phi_tt[:, g], color=cols[g],
                   label=f'g={g}: {E_edges[g+1]:.1e}-{E_edges[g]:.1e} eV' if G <= 12 else None)
    ax[0].semilogy(z, a[:, g], '--', color=cols[g])
    if g < G - 1:
        with np.errstate(divide='ignore', invalid='ignore'):
            s = sa[:, g] / a[:, g]
        ax[1].plot(z, rel[:, g], color=cols[g])
        ax[1].fill_between(z, -2 * s, 2 * s, color=cols[g], alpha=0.12)
ax[0].plot([], [], 'k-', label='TT'); ax[0].plot([], [], 'k--', label=f'OpenMC {REF}')
ax[0].set_ylabel(r'$\phi_g(z)$ per unit beam intensity'); ax[0].grid(True, which='both', alpha=0.4)
ax[0].legend(fontsize=7, ncol=2)
ax[0].set_title(f'Graphite slab, 2 MeV beam: TT (N_mu={N_MU}, p={P_DEG}, {CASE}) vs OpenMC {REF}')
ax[1].axhline(0, color='k', lw=0.8)
ax[1].set_ylabel(f'TT/{REF} - 1  (band = MC 2σ)'); ax[1].set_xlabel('z [cm]'); ax[1].grid(alpha=0.4)
ax[1].set_ylim(-0.15, 0.15)
plt.tight_layout()
plt.savefig(OUT / 'tt_graphite_slab_vs_openmc.png', dpi=130)

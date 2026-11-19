"""
Graphite slab with a He-3 detector gas region: forward TT (SlabTransport, heterogeneous) solution, 2 MeV normal beam.

Problem: steady state, slab 0 < z < L = 30 cm, vacuum on both faces, normal 2 MeV beam of unit intensity on z = 0,
graphite everywhere except He-3 gas in Z_DET < z < Z_DET + 2 cm (same geometry as het_ce.py / het_mg.py).

Cross sections (--xs):
  endf  graphite from graphite/results_P20_g_128 (copied to graphite_mgxs.npz), He-3 collapsed from ENDF by
        he3_mgxs.py (he3_mgxs.npz)                                           [default, until the MC tallies exist]
  mc    both materials as tallied by het_ce.py (results_P20_g_128/het_mgxs.npz)

First-collision split as in graphite/tt_graphite_slab.py, now with the optical depth through the materials:
    tau(z) = int_0^z sigma_t,0(z') dz'     (piecewise linear, sigma_t,0 constant per cell)
    uncollided :  phi_u,0(z) = exp(-tau(z))  (group 0)
    collided   :  TT solve with vacuum BCs and the first-collision source
                  q_g(z, mu) = sum_c 1_c(z) exp(-tau(z)) sum_l (2l+1)/(4 pi) sigma^c_s,l(0 -> g) P_l(mu)
Also reported: the He-3 absorption rate R = sum_g int_det sigma^He_a,g phi_g dz (detector response per beam neutron).

Outputs: tt_runs/<xs>_Nmu{N_mu}_p{p}[_eps{eps}_r{round_eps} if not default]/  run.log, het_tt_flux.npz, het_tt_flux_3groups.png

Run with the TT environment:
    /home/ale234/JRC/tt_python/tt_env/bin/python het_tt_slab.py [--xs endf] [--N_mu 24] [--p 1] [--eps 1e-7] [--round_eps 1e-8]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import eval_legendre
import tt_transport.slab as tt

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--xs', choices=['endf', 'mc'], default='endf')
ap.add_argument('--N_mu', type=int, default=24)
ap.add_argument('--p', type=int, default=1)
ap.add_argument('--N_z', type=int, default=300, help='cells (only used with --xs endf; mc takes the tally mesh)')
ap.add_argument('--z_det', type=float, default=14.0, help='left face of the 2 cm He-3 region [cm] (endf only)')
ap.add_argument('--eps', type=float, default=1e-7, help='AMEn tolerance')
ap.add_argument('--round_eps', type=float, default=1e-8, help='rounding of the AMEn solution')
args = ap.parse_args()
W_DET = 2.0


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
# data: tables (C, G), (C, L+1, G, G) with S[c, l, g, g'] = sigma_s^l(g' -> g) (row = receiving), region map
# ---------------------------------------------------------------------------
if args.xs == 'endf':
    xs = [np.load(HERE / f'{m}_mgxs.npz') for m in ('graphite', 'he3')]
    L = float(xs[0]['L'])
    E_edges = xs[0]['group_edges_desc']
    assert np.allclose(xs[1]['group_edges_desc'], E_edges)
    sigma_t = np.stack([x['sigma_t'] for x in xs])
    sigma_a = np.stack([x['sigma_a'] for x in xs])
    S = np.stack([x['scatter_legendre'] for x in xs])
    N_z, z_det = args.N_z, args.z_det
    h = L / N_z
    z_mid = (np.arange(N_z) + 0.5) * h
    region_map = ((z_mid > z_det) & (z_mid < z_det + W_DET)).astype(int)
else:
    xs = np.load(HERE / 'results_P20_g_128' / 'het_mgxs.npz')
    L, E_edges = float(xs['L']), xs['group_edges_desc']
    sigma_t, sigma_a, S = xs['sigma_t'], xs['sigma_a'], xs['scatter_legendre']
    N_z, z_det, region_map = int(xs['N_z']), float(xs['z_det']), xs['region_map']
    h = L / N_z
C, L1, G, _ = S.shape
L_MAX = L1 - 1
for z_b in (z_det, z_det + W_DET):
    assert abs(z_b / h - round(z_b / h)) < 1e-9, f"material interface z = {z_b} is not a cell edge (h = {h})"

OUT = HERE / 'tt_runs' / f'{args.xs}_Nmu{args.N_mu}_p{args.p}'
if (args.eps, args.round_eps) != (1e-7, 1e-8):
    OUT = OUT.with_name(OUT.name + f'_eps{args.eps:.0e}_r{args.round_eps:.0e}')
OUT.mkdir(parents=True, exist_ok=True)
_log = open(OUT / 'run.log', 'w')
sys.stdout = Tee(sys.__stdout__, _log)
sys.stderr = Tee(sys.__stderr__, _log)
print(f"xs = {args.xs}: C = {C}, G = {G}, L_max = {L_MAX}, L = {L} cm, N_z = {N_z} (h = {h:g} cm), "
      f"He-3 in [{z_det}, {z_det + W_DET}] cm (cells {np.flatnonzero(region_map)[[0, -1]]})")

model = tt.SlabTransport(N_z=N_z, p=args.p, L_z=L, N_mu=args.N_mu, N_phi=1,
                         sigma_t=sigma_t, region_map=np.eye(C)[region_map],   # (C, G) table + 0/1 matrix chi
                         sigma_s=np.swapaxes(S, 2, 3),          # SlabTransport: row = incoming group
                         E_edges=E_edges, steady_state=True, azimuthal_symmetry=True)

# ---------------------------------------------------------------------------
# optical depth of the beam group and first-collision source (C (L_MAX + 1) rank-one terms)
# ---------------------------------------------------------------------------
sig0 = sigma_t[region_map, 0]                                   # sigma_t,0 per cell
tau_e = np.concatenate(([0.0], np.cumsum(sig0 * h)))            # tau at the cell edges


def cell_of(z):
    return np.clip((np.asarray(z) / h).astype(int), 0, N_z - 1)


def tau(z):
    i = cell_of(z)
    return tau_e[i] + sig0[i] * (z - i * h)


q = None
for c in range(C):
    Q_z = model._project_z(lambda z, c=c: (region_map[cell_of(z)] == c) * np.exp(-tau(z)))
    for l in range(L_MAX + 1):
        term = tt.vector_kron([Q_z, (2 * l + 1) / (4 * np.pi) * eval_legendre(l, model.mu), np.ones(1),
                               S[c, l][:, 0], np.ones(1)])
        q = term if q is None else q + term
model.q = q.round(1e-12)
model.set_BC_left(0)
model.set_BC_right(0)
print(f"first-collision source: ranks {list(model.q.R)}")

AMEN = dict(nswp=40, eps=args.eps, kickrank=4, round_eps=args.round_eps)
t0 = time.time()
model.run_amen(**AMEN, verbose=True)
wall = time.time() - t0


# ---------------------------------------------------------------------------
# solver report
# ---------------------------------------------------------------------------
def tt_entries(T):
    return sum(c.numel() for c in T.cores)


A_tt, psi_tt = model.A, model.psi_modal
n_dof = int(np.prod(psi_tt.N))
print(f"\nAMEn: nswp={AMEN['nswp']}, eps={AMEN['eps']:.0e}, kickrank={AMEN['kickrank']}, round_eps={AMEN['round_eps']:.0e}; wall {wall:.1f} s")
print(f"operator A: ranks {list(A_tt.R)} (term by term {model.A_ranks_raw}), TT entries {tt_entries(A_tt)}, "
      f"compression {float(n_dof) ** 2 / tt_entries(A_tt):.2e}")
print(f"solution psi_modal: N = {list(psi_tt.N)}, ranks {list(psi_tt.R)}, TT entries {tt_entries(psi_tt)}, "
      f"full {n_dof} -> compression {n_dof / tt_entries(psi_tt):.2e}")
print(f"true relative residual ||A psi - F|| / ||F|| = {model.residual():.2e}")

# ---------------------------------------------------------------------------
# cell-averaged scalar flux, plus the uncollided beam in group 0
# ---------------------------------------------------------------------------
phi_r = model.compute_scalar_flux().full().numpy()[:, :, 0]
phi_c = np.einsum('irg,r->ig', phi_r.reshape(N_z, model.N_recon, G), model.w_xi) / 2
phi_u = (np.exp(-tau_e[:-1]) - np.exp(-tau_e[1:])) / (sig0 * h)
phi = phi_c.copy()
phi[:, 0] += phi_u

det = region_map == 1
R_det = h * np.sum(phi[det] * sigma_a[1][None, :])
R_det_th = h * np.sum(phi[det][:, E_edges[1:] < 0.625] * sigma_a[1][E_edges[1:] < 0.625][None, :])
print(f"He-3 absorptions per beam neutron: R = {R_det:.6e}  (below 0.625 eV: {R_det_th / R_det:.3f})")

np.savez(OUT / 'het_tt_flux.npz', z=model.z_mid, dz=h, L=L, group_edges_desc=E_edges, region_map=region_map,
         phi=phi, phi_collided=phi_c, phi_uncollided=phi_u, N_mu=args.N_mu, p=args.p,
         ranks=np.array(psi_tt.R), R_det=R_det, xs=args.xs)

# ---------------------------------------------------------------------------
# plot: one fast, one epithermal, one thermal group; homogeneous graphite TT run (200 cells) for reference
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt

E_lo, E_hi = E_edges[1:], E_edges[:-1]
group_of = lambda E: int(np.flatnonzero((E_lo <= E) & (E < E_hi))[0])
PICK = [('fast', group_of(1.0e6)), ('epithermal', group_of(10.0)), ('thermal', group_of(0.0253))]
hom_file = HERE.parent / 'P20g128' / 'tt_runs' / f'P20_g_128_Nmu{args.N_mu}_p{args.p}_az' / 'tt_graphite_slab_flux.npz'
hom = np.load(hom_file) if hom_file.exists() else None
z = model.z_mid

# OpenMC reference (het_ce.py / het_mg.py), same 300-cell mesh: MG = same multigroup problem, CE = continuous energy
MC_DIR = HERE / 'results_P20_g_128'
mc = {k: np.load(MC_DIR / f'het_{k.lower()}_flux.npz') for k in ('MG', 'CE') if (MC_DIR / f'het_{k.lower()}_flux.npz').exists()}
for k, d in mc.items():
    assert len(d['z']) == N_z and np.allclose(d['group_edges_desc'], E_edges), f"{k} tally mesh/groups differ"
    a = d['phi']
    with np.errstate(divide='ignore', invalid='ignore'):
        R_mc = h * np.sum(a[det] * sigma_a[1][None, :])
    print(f"\nTT vs OpenMC {k}: global ||TT - {k}||_F / ||{k}||_F = {np.linalg.norm(phi - a) / np.linalg.norm(a):.4f}, "
          f"He-3 absorptions R_{k} = {R_mc:.6e} (TT/{k} - 1 = {R_det / R_mc - 1:+.4f}, sigma_a from --xs {args.xs})")
    for name, g in PICK:
        sd = d['phi_sd'][:, g] / a[:, g]
        print(f"  {name:10s} g={g:3d}: int TT/{k} - 1 = {phi[:, g].sum() / a[:, g].sum() - 1:+.4f}, "
              f"max |TT/{k} - 1| = {np.max(np.abs(phi[:, g] / a[:, g] - 1)):.4f} (He-3 cells "
              f"{np.max(np.abs(phi[det, g] / a[det, g] - 1)):.4f}), median MC 1 sd = {np.median(sd):.4f}")

fig, ax = plt.subplots(2, 2, figsize=(12, 9))
for a, (name, g) in zip(ax.flat[:3], PICK):
    a.axvspan(z_det, z_det + W_DET, color='0.88', zorder=0, label='He-3')
    a.semilogy(z, phi[:, g], color='C0', label='graphite + He-3 (TT)')
    if hom is not None:
        a.semilogy(hom['z'], hom['phi'][:, g], '--', color='C1', label='graphite only (TT, 200 cells)')
    if 'MG' in mc:
        m, sd = mc['MG']['phi'][:, g], mc['MG']['phi_sd'][:, g]
        a.semilogy(z, m, color='C3', lw=0.9, label=r'graphite + He-3 (OpenMC MG, $\pm1\sigma$)')
        a.fill_between(z, m - sd, m + sd, color='C3', alpha=0.3, lw=0)
    a.set_title(f'{name}: g = {g}, {E_lo[g]:.3g}-{E_hi[g]:.3g} eV')
    a.set_xlabel('z [cm]'); a.set_ylabel(r'$\phi_g(z)$ per unit beam intensity')
    a.grid(True, which='both', alpha=0.4); a.legend(fontsize=8)
a = ax[1, 1]
a.axvspan(z_det, z_det + W_DET, color='0.88', zorder=0)
if hom is not None:
    for (name, g), col in zip(PICK, ('C3', 'C2', 'C4')):
        ref = np.interp(z, hom['z'], hom['phi'][:, g])
        a.plot(z, phi[:, g] / ref, color=col, lw=2, label=f'{name} (g = {g})')
        if 'MG' in mc:
            m, sd = mc['MG']['phi'][:, g] / ref, mc['MG']['phi_sd'][:, g] / ref
            a.plot(z, m, color=col, lw=0.8)
            a.fill_between(z, m - sd, m + sd, color=col, alpha=0.3, lw=0)
    a.plot([], [], 'k-', lw=2, label='TT'); a.plot([], [], 'k-', lw=0.8, label=r'OpenMC MG $\pm1\sigma$')
    a.axhline(1, color='k', lw=0.8)
    a.set_title('flux depression: heterogeneous / graphite only (TT)')
    a.set_ylabel('ratio'); a.legend(fontsize=8)
a.set_xlabel('z [cm]'); a.grid(alpha=0.4)
fig.suptitle(f'Graphite slab + 2 cm He-3, 2 MeV beam, P{L_MAX}, G = {G} (xs: {args.xs}); TT ranks {list(psi_tt.R)}, '
             f'R_He3 = {R_det:.3e}' + (f' (MG {h * np.sum(mc["MG"]["phi"][det] * sigma_a[1][None, :]):.3e})'
                                         if 'MG' in mc else ''), fontsize=11)
plt.tight_layout()
plt.savefig(OUT / 'het_tt_flux_3groups.png', dpi=130)
plt.show()

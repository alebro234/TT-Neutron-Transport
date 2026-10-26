"""
Graphite slab: statistical uncertainty of the MGXS Legendre moments, read back from an existing CE statepoint
(no new transport run).

The openmc.mgxs.Library of graphite_ce.py is rebuilt with the same definition (material, geometry, groups, MGXS types,
Legendre order) and loaded from <results>/ce_run/statepoint.50.h5. openmc.mgxs returns, for every entry, the mean and the
standard deviation propagated from the batch statistics (first order, operands assumed uncorrelated: conservative).

Outputs (in <results>/):
  graphite_mgxs_sd.npz      sigma_t, sigma_t_sd, scatter_legendre, scatter_legendre_sd  [l, g, g'], g = 0 highest
  mgxs_uncertainty.png      per Legendre order l:
                              (a) ||Sigma^l||_F / ||Sigma^0||_F and the noise ||sd(Sigma^l)||_F / ||Sigma^0||_F
                              (b) fraction of transfers with |sigma^l| > 3 sd, unweighted and weighted by the
                                  scattering rate sigma^0(g'->g) Phi_g' (Phi = slab-integrated CE flux)
Run:  python mgxs_uncertainty.py [case] [G]        (default: P20 128 -> results_P20_g_128)
"""
import sys
from pathlib import Path

import numpy as np
import openmc
import openmc.mgxs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CASE = sys.argv[1] if len(sys.argv) > 1 else 'P20'
G = int(sys.argv[2]) if len(sys.argv) > 2 else 128
RES = HERE / (f'results_{CASE}' + (f'_g_{G}' if G != 10 else ''))
L_MAX = int(CASE[1:])

# ---------------------------------------------------------------------------
# same definitions as graphite_ce.py (material first, so it gets the same id)
# ---------------------------------------------------------------------------
L, W, DENSITY, TEMPERATURE, E0 = 30.0, 50.0, 1.7, 294.0, 2.0e6
graphite = openmc.Material(name='graphite')
graphite.add_nuclide('C0', 1.0)
graphite.set_density('g/cm3', DENSITY)
graphite.temperature = TEMPERATURE
graphite.add_s_alpha_beta('c_Graphite')
planes = [openmc.ZPlane(z0=0.0, boundary_type='vacuum'), openmc.ZPlane(z0=L, boundary_type='vacuum'),
          openmc.XPlane(x0=-W, boundary_type='reflective'), openmc.XPlane(x0=W, boundary_type='reflective'),
          openmc.YPlane(y0=-W, boundary_type='reflective'), openmc.YPlane(y0=W, boundary_type='reflective')]
region = +planes[0] & -planes[1] & +planes[2] & -planes[3] & +planes[4] & -planes[5]
geometry = openmc.Geometry([openmc.Cell(name='slab', fill=graphite, region=region)])

group_edges = np.append(np.logspace(np.log10(1.0e-5), np.log10(0.95 * E0), G), 1.05 * E0)
mgxs_lib = openmc.mgxs.Library(geometry)
mgxs_lib.energy_groups = openmc.mgxs.EnergyGroups(group_edges)
mgxs_lib.mgxs_types = ['total', 'absorption', 'inverse-velocity', 'consistent scatter matrix',
                       'consistent nu-scatter matrix']
mgxs_lib.scatter_format = 'legendre'
if L_MAX > 10:
    openmc.mgxs.mgxs._MAX_LEGENDRE = L_MAX
    mgxs_lib._legendre_order = L_MAX
else:
    mgxs_lib.legendre_order = L_MAX
mgxs_lib.correction = None
mgxs_lib.domain_type = 'material'
mgxs_lib.domains = [graphite]
mgxs_lib.by_nuclide = False
mgxs_lib.build_library()

with openmc.StatePoint(RES / 'ce_run' / 'statepoint.50.h5') as sp:
    mgxs_lib.load_from_statepoint(sp)


def matrix(value):
    m = mgxs_lib.get_mgxs(graphite, 'consistent nu-scatter matrix')
    S = np.asarray(m.get_xs(order_groups='increasing', row_column='inout', value=value)).reshape(G, G, L_MAX + 1)
    return np.transpose(S, (2, 1, 0))        # [l, g, g'], row = receiving group


S, Ssd = matrix('mean'), matrix('std_dev')
tot = mgxs_lib.get_mgxs(graphite, 'total')
st = np.asarray(tot.get_xs(order_groups='increasing')).reshape(G)
st_sd = np.asarray(tot.get_xs(order_groups='increasing', value='std_dev')).reshape(G)

# consistency with the saved means
saved = np.load(RES / 'graphite_mgxs.npz')['scatter_legendre']
print(f"max |S - saved| / max|S| = {np.abs(S - saved).max() / np.abs(S).max():.1e}")
np.savez(RES / 'graphite_mgxs_sd.npz', sigma_t=st, sigma_t_sd=st_sd, scatter_legendre=S, scatter_legendre_sd=Ssd,
         group_edges_desc=group_edges[::-1])

# ---------------------------------------------------------------------------
# per-order statistics
# ---------------------------------------------------------------------------
Phi = np.load(RES / 'graphite_ce_flux.npz')['phi'].sum(0)           # slab-integrated flux per group
rate = S[0] * Phi[None, :]                                           # sigma^0(g'->g) Phi_g'
nz = S[0] > 0
n0 = np.linalg.norm(S[0])
ls = np.arange(L_MAX + 1)
sig = np.array([np.linalg.norm(S[l]) / n0 for l in ls])
noise = np.array([np.linalg.norm(Ssd[l]) / n0 for l in ls])
frac_n, frac_w = [], []
for l in ls:
    signif = nz & (np.abs(S[l]) > 3 * Ssd[l])
    frac_n.append(signif.sum() / nz.sum())
    frac_w.append(rate[signif].sum() / rate[nz].sum())
print(f"{G} groups, P{L_MAX}: {nz.sum()} nonzero transfers; sigma_t rel. sd: median {np.median(st_sd[st > 0] / st[st > 0]):.1e}")
print(" l | ||S^l||/||S^0|| | noise ||sd||/||S^0|| | signal/noise | frac |s|>3sd (unweighted, rate-weighted)")
for l in ls:
    print(f"{l:>2} | {sig[l]:.3e}       | {noise[l]:.3e}            | {sig[l] / noise[l]:7.1f}      | {frac_n[l]:.3f}  {frac_w[l]:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].semilogy(ls, sig, 'o-', color='#2a78d6', lw=2, ms=6, label=r'$\|\Sigma^l\|_F/\|\Sigma^0\|_F$')
ax[0].semilogy(ls, noise, 's--', color='#eb6834', lw=2, ms=6, mfc='white', label=r'$\|\mathrm{sd}(\Sigma^l)\|_F/\|\Sigma^0\|_F$ (noise)')
ax[0].set_xlabel('Legendre order $l$'); ax[0].set_title('(a) moment size and statistical noise')
ax[1].plot(ls, frac_w, 'o-', color='#2a78d6', lw=2, ms=6, label='weighted by scattering rate')
ax[1].plot(ls, frac_n, 's--', color='#eb6834', lw=2, ms=6, mfc='white', label='unweighted (all nonzero transfers)')
ax[1].set_xlabel('Legendre order $l$'); ax[1].set_ylabel(r'fraction with $|\sigma^l|>3\,\mathrm{sd}$')
ax[1].set_ylim(0, 1.02); ax[1].set_title('(b) statistically significant moments')
for a in ax:
    a.set_xticks(range(0, L_MAX + 1, 2)); a.grid(True, color='#d9d8d2', lw=0.6); a.set_axisbelow(True)
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)
    a.legend(frameon=False, fontsize=9)
fig.suptitle(f'Graphite, {G} groups, $P_{{{L_MAX}}}$ (CE, 50 batches): uncertainty of the Legendre scattering moments')
fig.tight_layout()
fig.savefig(RES / 'mgxs_uncertainty.png', dpi=140)
print('saved', RES / 'mgxs_uncertainty.png')

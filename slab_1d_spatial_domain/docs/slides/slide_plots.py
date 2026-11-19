"""
TT/CE-only figures for the slides (no OpenMC MG curves).

graphite_groups_tt_ce.png : graphite slab, P20, TT/CE - 1 of the total flux vs z for 16-256 groups
                            (data of ~/openmc/graphite/group_convergence.py)
graphite_g128_tt_ce.png   : graphite slab, 128 groups, P20, phi_g and TT/CE - 1 vs z for one fast, epithermal, thermal group
graphite_he3_tt_ce.png    : graphite + 2 cm He-3, 128 groups, P20, TT/CE - 1 of the total flux and of groups 4, 60, 89 vs z,
                            and per group integrated over the slab (data of studies/montecarlo/P20g128_het)
graphite_he3_tt_vs_ce.png : as notes/images/graphite_he3_tt_vs_mg.png (het_tt_slab.py) with OpenMC CE instead of MG
Output: ../images/

Run: python slide_plots.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'images'
GRAPHITE = Path.home() / 'openmc' / 'graphite'
HET = HERE.parents[1] / 'studies' / 'montecarlo' / 'P20g128_het'


def style(a):
    a.grid(True, color='#e3e2dc', lw=0.6); a.set_axisbelow(True)
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)


# ---------------------------------------------------------------------------
# group collapse study
# ---------------------------------------------------------------------------
CASES = [(256, 'P20_g_256'), (128, 'P20_g_128'), (64, 'P20_g_128_to_64'), (32, 'P20_g_128_to_32'), (16, 'P20_g_128_to_16')]
COLORS = {256: '#e87ba4', 128: '#1f1f1f', 64: '#2a78d6', 32: '#1baf7a', 16: '#eb6834'}

fig, a = plt.subplots(figsize=(7.0, 4.6))
for G, case in CASES:
    ce = np.load(GRAPHITE / f'results_{case}' / 'graphite_ce_flux.npz')
    tt = np.load(GRAPHITE / 'tt_runs' / f'{case}_Nmu24_p1_az' / 'tt_graphite_slab_flux.npz')['phi']
    a_tot, b_tot = tt.sum(1), ce['phi'].sum(1)
    a.plot(ce['z'], a_tot / b_tot - 1, color=COLORS[G], lw=1.6, label=f'{G} groups')
    print(f'{G:4d} groups: integrated TT/CE - 1 = {a_tot.sum() / b_tot.sum() - 1:+.2%}')
a.axhline(0, color='#8a8a85', lw=0.8)
a.set_xlabel('z [cm]'); a.set_ylabel('TT/CE $-$ 1, total flux')
a.legend(frameon=False, fontsize=9)
style(a)
fig.tight_layout()
fig.savefig(OUT / 'graphite_groups_tt_ce.png', dpi=200)
print('saved', OUT / 'graphite_groups_tt_ce.png')

# ---------------------------------------------------------------------------
# graphite, 128 groups: one fast, one epithermal and one thermal group
# ---------------------------------------------------------------------------
ce = np.load(GRAPHITE / 'results_P20_g_128' / 'graphite_ce_flux.npz')
T = np.load(GRAPHITE / 'tt_runs' / 'P20_g_128_Nmu24_p1_az' / 'tt_graphite_slab_flux.npz')['phi']
C, sC, z, E = ce['phi'], ce['phi_sd'], ce['z'], ce['group_edges_desc']
i_mc = np.linspace(0, len(z) - 1, 10).round().astype(int)       # CE markers at 10 cells
SHOW = [(4, 'fast', '#2a78d6'), (60, 'epithermal', '#eb6834'), (89, 'thermal', '#1baf7a')]


def e_label(e):
    for unit, f in (('MeV', 1e6), ('keV', 1e3), ('eV', 1.0), ('meV', 1e-3)):
        if e >= f:
            return f'{e / f:.3g} {unit}'


fig, ax = plt.subplots(2, 1, figsize=(7.0, 6.6), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
for g, band, c in SHOW:
    ax[0].semilogy(z, T[:, g], '-', color=c, lw=1.6, label=f'{band}, g={g}: {e_label(E[g + 1])} - {e_label(E[g])}')
    ax[0].semilogy(z[i_mc], C[i_mc, g], 'o', color=c, ms=5, mfc='none', mew=1.3)
    s = sC[:, g] / C[:, g]
    ax[1].plot(z, T[:, g] / C[:, g] - 1, color=c, lw=1.4)
    ax[1].fill_between(z, -2 * s, 2 * s, color=c, alpha=0.15, lw=0)
    print(f'g={g:3d} ({band}): integrated TT/CE - 1 = {T[:, g].sum() / C[:, g].sum() - 1:+.2%}')
ax[0].plot([], [], 'k-', lw=1.6, label='TT'); ax[0].plot([], [], 'ko', ms=5, mfc='none', mew=1.3, label='CE')
ax[0].set_ylabel(r'$\phi_g(z)$ per unit beam intensity')
ax[0].legend(frameon=False, fontsize=8)
ax[1].axhline(0, color='#8a8a85', lw=0.8)
ax[1].set_xlabel('z [cm]'); ax[1].set_ylabel('TT/CE $-$ 1  (band: CE $2\\sigma$)')
for a in ax:
    style(a)
fig.tight_layout()
fig.savefig(OUT / 'graphite_g128_tt_ce.png', dpi=200)
print('saved', OUT / 'graphite_g128_tt_ce.png')

# ---------------------------------------------------------------------------
# graphite + He-3
# ---------------------------------------------------------------------------
RES = HET / 'results_P20_g_128'
ce, xs = np.load(RES / 'het_ce_flux.npz'), np.load(RES / 'het_mgxs.npz')
T = np.load(HET / 'tt_runs' / 'mc_Nmu24_p1_eps1e-05_r1e-06' / 'het_tt_flux.npz')['phi']
C, sC, z, E = ce['phi'], ce['phi_sd'], ce['z'], ce['group_edges_desc']
z_det, w_det = float(xs['z_det']), float(xs['w_det'])
Ec = np.sqrt(E[:-1] * E[1:])

fig, ax = plt.subplots(2, 1, figsize=(7.0, 6.6))
ax[0].axvspan(z_det, z_det + w_det, color='#e6e5df', zorder=0, label='He-3')
A, B = T.sum(1), C.sum(1)
s = np.sqrt((sC ** 2).sum(1)) / B
ax[0].plot(z, A / B - 1, '-', color='#1f1f1f', lw=1.5, label='total')
ax[0].fill_between(z, -2 * s, 2 * s, color='#8a8a85', alpha=0.25, lw=0, label='CE $2\\sigma$')
print(f'total: integrated TT/CE - 1 = {A.sum() / B.sum() - 1:+.2%}')
for g, band, c in SHOW:
    ax[0].plot(z, T[:, g] / C[:, g] - 1, '-', color=c, lw=1.1, label=f'g={g}: {e_label(E[g + 1])} - {e_label(E[g])}')
    ax[0].fill_between(z, -2 * sC[:, g] / C[:, g], 2 * sC[:, g] / C[:, g], color=c, alpha=0.12, lw=0)
    print(f'g={g:3d} ({band}): integrated TT/CE - 1 = {T[:, g].sum() / C[:, g].sum() - 1:+.2%}')
ax[0].axhline(0, color='#8a8a85', lw=0.8)
ax[0].set_xlabel('z [cm]'); ax[0].set_ylabel('TT/CE $-$ 1')
ax[0].set_ylim(-0.10, 0.08)
ax[0].legend(frameon=False, fontsize=8, ncol=2, loc='lower left')

with np.errstate(divide='ignore', invalid='ignore'):
    gi = T.sum(0) / C.sum(0) - 1
    gs = np.sqrt((sC ** 2).sum(0)) / C.sum(0)
ok = (C.sum(0) > 0) & (T.sum(0) > 0)
ax[1].errorbar(Ec[ok], gi[ok], yerr=2 * gs[ok], fmt='o', ms=3, color='#8a8a85', ecolor='#d0cfc9', elinewidth=1,
               capsize=0, label='per group, integrated over the slab (CE $2\\sigma$)')
for g, band, c in SHOW:
    ax[1].plot(Ec[g], gi[g], 'o', ms=6, color=c, zorder=3)
ax[1].axhline(0, color='#8a8a85', lw=0.8)
ax[1].set_xscale('log'); ax[1].set_xlabel('group mid energy [eV]'); ax[1].set_ylabel('TT/CE $-$ 1')
ax[1].set_ylim(-0.08, 0.08)
ax[1].legend(frameon=False, fontsize=8, loc='upper right')
for a in ax:
    style(a)
fig.tight_layout()
fig.savefig(OUT / 'graphite_he3_tt_ce.png', dpi=200)
print('saved', OUT / 'graphite_he3_tt_ce.png')

# ---------------------------------------------------------------------------
# graphite + He-3: phi_g for g = 4, 60, 89 and flux depression, as notes/images/graphite_he3_tt_vs_mg.png
# (het_tt_slab.py) with OpenMC CE in place of OpenMC MG
# ---------------------------------------------------------------------------
tt_het = np.load(HET / 'tt_runs' / 'mc_Nmu24_p1_eps1e-05_r1e-06' / 'het_tt_flux.npz')
hom = np.load(GRAPHITE / 'tt_runs' / 'P20_g_128_Nmu24_p1_az' / 'tt_graphite_slab_flux.npz')
det = ce['region_map'] == 1
R_ce = float(ce['dz']) * np.sum(C[det] * xs['sigma_a'][1][None, :])
E_lo, E_hi = E[1:], E[:-1]
PICK = [(band, g) for g, band, _ in SHOW]

fig, ax = plt.subplots(2, 2, figsize=(12, 9))
for a, (name, g) in zip(ax.flat[:3], PICK):
    a.axvspan(z_det, z_det + w_det, color='0.88', zorder=0, label='He-3')
    a.semilogy(z, T[:, g], color='C0', label='graphite + He-3 (TT)')
    a.semilogy(hom['z'], hom['phi'][:, g], '--', color='C1', label='graphite only (TT, 200 cells)')
    m, sd = C[:, g], sC[:, g]
    a.semilogy(z, m, color='C3', lw=0.9, label=r'graphite + He-3 (OpenMC CE, $\pm1\sigma$)')
    a.fill_between(z, m - sd, m + sd, color='C3', alpha=0.3, lw=0)
    a.set_title(f'{name}: g = {g}, {E_lo[g]:.3g}-{E_hi[g]:.3g} eV')
    a.set_xlabel('z [cm]'); a.set_ylabel(r'$\phi_g(z)$ per unit beam intensity')
    a.grid(True, which='both', alpha=0.4); a.legend(fontsize=8)
a = ax[1, 1]
a.axvspan(z_det, z_det + w_det, color='0.88', zorder=0)
for (name, g), col in zip(PICK, ('C3', 'C2', 'C4')):
    ref = np.interp(z, hom['z'], hom['phi'][:, g])
    a.plot(z, T[:, g] / ref, color=col, lw=2, label=f'{name} (g = {g})')
    m, sd = C[:, g] / ref, sC[:, g] / ref
    a.plot(z, m, color=col, lw=0.8)
    a.fill_between(z, m - sd, m + sd, color=col, alpha=0.3, lw=0)
a.plot([], [], 'k-', lw=2, label='TT'); a.plot([], [], 'k-', lw=0.8, label=r'OpenMC CE $\pm1\sigma$')
a.axhline(1, color='k', lw=0.8)
a.set_title('flux depression: heterogeneous / graphite only (TT)')
a.set_ylabel('ratio'); a.legend(fontsize=8)
a.set_xlabel('z [cm]'); a.grid(alpha=0.4)
fig.suptitle(f'Graphite slab + 2 cm He-3, 2 MeV beam, P20, G = {T.shape[1]} (xs: mc); TT ranks {list(tt_het["ranks"])}, '
             f'R_He3 = {float(tt_het["R_det"]):.3e} (CE {R_ce:.3e})', fontsize=11)
fig.tight_layout()
fig.savefig(OUT / 'graphite_he3_tt_vs_ce.png', dpi=130)
print('saved', OUT / 'graphite_he3_tt_vs_ce.png')

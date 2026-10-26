"""
Graphite slab, 128 groups, P20: TT versus the continuous-energy (CE) and the multigroup (MG) OpenMC flux.

Panels: (a) TT/CE - 1 vs z for the total flux sum_g phi_g and for the fast / epithermal / thermal bands
            (sums of the groups inside; CE 2-sigma band on the total);
        (b) the same for TT/MG - 1 (MG 2-sigma band on the total). TT and MG solve the same multigroup problem;
            MG samples a clipped (non-negative) version of the truncated P20 kernels.
Inputs : results_P20_g_128/graphite_{ce,mg}_flux.npz, tt_runs/P20_g_128_Nmu24_p1_az/tt_graphite_slab_flux.npz
Output : compare_tt_ce_g128.png (next to this script)
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ce = np.load(HERE / 'results_P20_g_128' / 'graphite_ce_flux.npz')
mg = np.load(HERE / 'results_P20_g_128' / 'graphite_mg_flux.npz')
tt = np.load(HERE / 'tt_runs' / 'P20_g_128_Nmu24_p1_az' / 'tt_graphite_slab_flux.npz')
T, z, E = tt['phi'], ce['z'], ce['group_edges_desc']
Ec = np.sqrt(E[:-1] * E[1:])
BANDS = [('total', np.ones_like(Ec, bool), '#1f1f1f', '-'),
         ('fast (> 0.1 MeV)', Ec > 1e5, '#2a78d6', '-'),
         ('epithermal (1 eV - 0.1 MeV)', (Ec <= 1e5) & (Ec > 1.0), '#eb6834', '-'),
         ('thermal (< 1 eV)', Ec <= 1.0, '#1baf7a', '-')]

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for a, (ref, lab, n) in zip(ax, ((ce, 'CE', '(a)'), (mg, 'MG', '(b)'))):
    C, sC = ref['phi'], ref['phi_sd']
    print(f"TT/{lab} - 1:")
    for name, m, c, ls in BANDS:
        num, den = T[:, m].sum(1), C[:, m].sum(1)
        r = num / den - 1
        a.plot(z, r, ls, color=c, lw=1.8, label=f'{name} (int {num.sum() / den.sum() - 1:+.2%})')
        print(f"  {name:28s} int {num.sum() / den.sum() - 1:+.2%}   rms(z<=25) {np.sqrt(np.mean(r[z <= 25] ** 2)):.2%}")
        if name == 'total':
            s = np.sqrt((sC[:, m]**2).sum(1)) / den
            a.fill_between(z, -2 * s, 2 * s, color='#8a8a85', alpha=0.25, lw=0, label=f'{lab} $2\\sigma$ (total)')
    print(f"  global ||TT - {lab}||_F / ||{lab}||_F = {np.linalg.norm(T - C) / np.linalg.norm(C):.4f}")
    a.axhline(0, color='#8a8a85', lw=0.8)
    a.set_xlabel('z [cm]'); a.set_ylabel(f'TT/{lab} $-$ 1')
    a.set_title(f'{n} TT versus OpenMC {lab}' + (' (continuous energy)' if lab == 'CE' else ' (same multigroup data)'))
ax[0].set_ylim(-0.02, 0.02)
for a in ax:
    a.yaxis.set_tick_params(labelleft=True)
    a.grid(True, color='#d9d8d2', lw=0.6); a.set_axisbelow(True)
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)
    a.legend(frameon=False, fontsize=8)
fig.suptitle('Graphite slab, 128 groups, $P_{20}$: TT ($N_\\mu=24$, $p=1$) versus OpenMC CE and MG ($5\\cdot10^5$ particles/batch)')
fig.tight_layout()
fig.savefig(HERE / 'compare_tt_ce_g128.png', dpi=140)
print('saved', HERE / 'compare_tt_ce_g128.png')
plt.show()

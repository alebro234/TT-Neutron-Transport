"""
Graphite slab with a 2 cm He-3 region, 128 groups, P20: TT versus the continuous-energy (CE) OpenMC flux,
with OpenMC MG on the same axes (MG/CE), so that both references can be judged.

Panels: (a) TT/CE - 1 (solid) and MG/CE - 1 (dashed) vs z for the total flux and the fast / epithermal / thermal
            bands (sums of the groups inside); CE 2-sigma band on the total; He-3 region shaded;
        (b) per group, TT/CE - 1 and MG/CE - 1 integrated over the slab vs group mid energy (CE 2-sigma error bars).
Printed: TT/CE, TT/MG, MG/CE by band (integrated, rms over z, He-3 cells) and for the He-3 absorption rate R.
Inputs : results_P20_g_128/het_{ce,mg}_flux.npz, het_mgxs.npz, tt_runs/mc_Nmu24_p1_eps1e-05_r1e-06/het_tt_flux.npz
Output : compare_tt_ce_het.png (next to this script)

Run: python compare_tt_ce_het.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / 'results_P20_g_128'
ce, mg = np.load(RES / 'het_ce_flux.npz'), np.load(RES / 'het_mg_flux.npz')
xs = np.load(RES / 'het_mgxs.npz')
tt = np.load(HERE / 'tt_runs' / 'mc_Nmu24_p1_eps1e-05_r1e-06' / 'het_tt_flux.npz')
T, M, C, sC = tt['phi'], mg['phi'], ce['phi'], ce['phi_sd']
z, E, h, det = ce['z'], ce['group_edges_desc'], float(ce['dz']), ce['region_map'] == 1
z_det, w_det = float(xs['z_det']), float(xs['w_det'])
Ec = np.sqrt(E[:-1] * E[1:])
sig_a_he = xs['sigma_a'][1]
BANDS = [('total', np.ones_like(Ec, bool), '#1f1f1f'),
         ('fast (> 0.1 MeV)', Ec > 1e5, '#2a78d6'),
         ('epithermal (1 eV - 0.1 MeV)', (Ec <= 1e5) & (Ec > 1.0), '#eb6834'),
         ('thermal (< 1 eV)', Ec <= 1.0, '#1baf7a')]

# ---------------------------------------------------------------------------
# table: three comparisons by band
# ---------------------------------------------------------------------------
PAIRS = [('TT/CE', T, C), ('TT/MG', T, M), ('MG/CE', M, C)]
print(f"{'band':28s}" + "".join(f"{p:>27s}" for p, _, _ in PAIRS))
print(f"{'':28s}" + "   int %   rms(z<=25) %  He-3 %" * 3)
for name, m, _ in BANDS:
    row = f"{name:28s}"
    for _, a, b in PAIRS:
        A, B = a[:, m].sum(1), b[:, m].sum(1)
        r = A / B - 1
        row += (f"  {100 * (A.sum() / B.sum() - 1):+7.2f}  {100 * np.sqrt(np.mean(r[z <= 25] ** 2)):9.2f}"
                f"  {100 * (A[det].sum() / B[det].sum() - 1):+7.2f}")
    print(row)
R = {k: h * np.sum(f[det] * sig_a_he[None, :]) for k, f in (('TT', T), ('MG', M), ('CE', C))}
R_sd = h * np.sqrt(np.sum((ce['phi_sd'][det] * sig_a_he[None, :]) ** 2))     # cells treated as independent
print(f"He-3 absorptions R:  TT {R['TT']:.5e}  MG {R['MG']:.5e}  CE {R['CE']:.5e} (+- {R_sd:.1e}, CE 1 sigma)")
print(f"  TT/CE - 1 = {R['TT'] / R['CE'] - 1:+.3%}   TT/MG - 1 = {R['TT'] / R['MG'] - 1:+.3%}   "
      f"MG/CE - 1 = {R['MG'] / R['CE'] - 1:+.3%}   (CE 1 sigma = {R_sd / R['CE']:.2%})")
for k, a, b in PAIRS:
    print(f"global ||{k[:2]} - {k[3:]}||_F / ||{k[3:]}||_F = {np.linalg.norm(a - b) / np.linalg.norm(b):.4f}")

# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for a in ax[:1]:
    a.axvspan(z_det, z_det + w_det, color='#e6e5df', zorder=0, label='He-3')
for name, m, c in BANDS:
    A, B, Mg = T[:, m].sum(1), C[:, m].sum(1), M[:, m].sum(1)
    ax[0].plot(z, A / B - 1, '-', color=c, lw=1.8, label=f'{name} (int {A.sum() / B.sum() - 1:+.2%})')
    ax[0].plot(z, Mg / B - 1, '--', color=c, lw=1.0)
    if name == 'total':
        s = np.sqrt((sC[:, m] ** 2).sum(1)) / B
        ax[0].fill_between(z, -2 * s, 2 * s, color='#8a8a85', alpha=0.25, lw=0, label='CE $2\\sigma$ (total)')
ax[0].plot([], [], 'k-', lw=1.8, label='TT/CE'); ax[0].plot([], [], 'k--', lw=1.0, label='MG/CE')
ax[0].axhline(0, color='#8a8a85', lw=0.8)
ax[0].set_xlabel('z [cm]'); ax[0].set_ylabel('ratio $-$ 1')
ax[0].set_title('(a) total flux and energy bands')
ax[0].set_ylim(-0.04, 0.03)

with np.errstate(divide='ignore', invalid='ignore'):
    gi_t, gi_m = T.sum(0) / C.sum(0) - 1, M.sum(0) / C.sum(0) - 1
    gs = np.sqrt((sC ** 2).sum(0)) / C.sum(0)
ok = (C.sum(0) > 0) & (T.sum(0) > 0)
ax[1].errorbar(Ec[ok], gi_t[ok], yerr=2 * gs[ok], fmt='o', ms=3.5, color='#2a78d6', ecolor='#9ec5f4', elinewidth=1,
               capsize=0, label='TT/CE per group, CE $2\\sigma$')
ax[1].plot(Ec[ok], gi_m[ok], 'x', ms=4, color='#eb6834', label='MG/CE per group')
ax[1].axhline(0, color='#8a8a85', lw=0.8)
ax[1].axvline(1.8e-3, color='#8a8a85', lw=1, ls=':', label='last Bragg edge (~1.8 meV)')
ax[1].set_xscale('log'); ax[1].set_xlabel('group mid energy [eV]'); ax[1].set_ylabel('integrated over the slab, ratio $-$ 1')
ax[1].set_title('(b) per group')
ax[1].set_ylim(-0.08, 0.08)
n_out = int(np.sum(ok & (np.abs(gi_t) > 0.08)))
if n_out:
    ax[1].text(0.02, 0.02, f'{n_out} TT/CE group(s) outside the axis range', transform=ax[1].transAxes, fontsize=8)
for a in ax:
    a.grid(True, color='#d9d8d2', lw=0.6); a.set_axisbelow(True)
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)
    a.legend(frameon=False, fontsize=8)
fig.suptitle('Graphite slab + 2 cm He-3, 128 groups, $P_{20}$: TT and OpenMC MG versus continuous-energy OpenMC '
             f'(R$_{{He3}}$: TT/CE {R["TT"] / R["CE"] - 1:+.2%}, MG/CE {R["MG"] / R["CE"] - 1:+.2%})', fontsize=11)
fig.tight_layout()
fig.savefig(HERE / 'compare_tt_ce_het.png', dpi=140)
print('saved', HERE / 'compare_tt_ce_het.png')
plt.show()

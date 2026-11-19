"""
He-3 detector gas (pure He-3, 0.001 g/cm^3, 294 K) multigroup cross sections on the 128-group structure of
graphite/results_P20_g_128, in the same convention as graphite_mgxs.npz:
    sigma_t, sigma_a (G,) [1/cm];  scatter_legendre[l, g, g'] = sigma_s^l(g' -> g) (row = receiving group,
    no (2l+1) factor);  g = 0 the highest group.

Collapsed directly from the ENDF/B pointwise data (nndc_hdf5/He3.h5), not by an OpenMC tally run:
  - intra-group weight 1/E (flat in lethargy), Gauss points in lethargy within every group;
  - absorption = MT 102 + 103 + 104 (He-3 has no other neutron-emitting reactions in the file), total = MT 2 + absorption;
  - elastic transfer for a target at rest (no thermal motion, no up-scatter): CM cosine pdf from the file,
    E' = E [(1 + alpha) + (1 - alpha) mu_cm] / 2,  mu_lab = (1 + A mu_cm) / sqrt(1 + A^2 + 2 A mu_cm);
    E' is linear in mu_cm, so the mu_cm range is split at the receiving-group edges and each piece integrated by Gauss.
At this density Sigma_s ~ 1e-3 /cm while Sigma_a ~ 1 /cm thermal (1/v), so the scattering approximations hardly matter.
inv_velocity is copied from graphite (a group property of the common spectrum).

Run: python he3_mgxs.py   (openmc env)  ->  he3_mgxs.npz
"""
from pathlib import Path

import numpy as np
import openmc.data
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre

HERE = Path(__file__).resolve().parent
DENSITY = 1e-3                                   # g/cm^3
M_HE3 = 3.0160293                                # g/mol
N = DENSITY * openmc.data.AVOGADRO / M_HE3 * 1e-24   # atoms / (barn cm)

gr = np.load(HERE / 'graphite_mgxs.npz')
E_edges = gr['group_edges_desc']                 # descending, eV
G, L_MAX = len(E_edges) - 1, int(gr['L_max'])

nuc = openmc.data.IncidentNeutron.from_hdf5('/home/ale234/nuclear_data/nndc_hdf5/He3.h5')
A = nuc.atomic_weight_ratio
alpha = ((A - 1) / (A + 1)) ** 2
xs = lambda mt, E: nuc[mt].xs['294K'](E)
ang = nuc[2].products[0].distribution[0].angle   # CM cosine pdfs at tabulated incident energies
assert nuc[2].center_of_mass


def pdf_cm(mu, E):
    """CM cosine pdf at incident energy E, linear in E between the tabulated energies."""
    k = np.clip(np.searchsorted(ang.energy, E) - 1, 0, len(ang.energy) - 2)
    f = (E - ang.energy[k]) / (ang.energy[k + 1] - ang.energy[k])
    p = lambda i: np.interp(mu, ang.mu[i].x, ang.mu[i].p)
    return (1 - f) * p(k) + f * p(k + 1)


x_u, w_u = leggauss(64)                          # lethargy points per source group
x_m, w_m = leggauss(16)                          # cosine points per piece
sigma_t, sigma_a = np.zeros(G), np.zeros(G)
S = np.zeros((L_MAX + 1, G, G))
for gp in range(G):
    u0, u1 = np.log(E_edges[gp + 1]), np.log(E_edges[gp])
    E_src = np.exp((u0 + u1) / 2 + (u1 - u0) / 2 * x_u)
    w = w_u / w_u.sum()                          # 1/E weight = uniform in lethargy
    s_el = xs(2, E_src)
    s_ab = xs(102, E_src) + xs(103, E_src) + xs(104, E_src)
    sigma_a[gp] = N * np.sum(w * s_ab)
    sigma_t[gp] = N * np.sum(w * (s_el + s_ab))
    for E, wi, se in zip(E_src, w, s_el):
        # mu_cm at which E' crosses a group edge, inside (-1, 1)
        mu_b = (2 * E_edges / E - (1 + alpha)) / (1 - alpha)
        cuts = np.concatenate(([-1.0], np.sort(mu_b[(mu_b > -1) & (mu_b < 1)]), [1.0]))
        for a, b in zip(cuts[:-1], cuts[1:]):
            mu = (a + b) / 2 + (b - a) / 2 * x_m
            wm = (b - a) / 2 * w_m
            E_out = E * ((1 + alpha) + (1 - alpha) * mu) / 2
            g = min(np.searchsorted(-E_edges, -E_out[len(mu) // 2]) - 1, G - 1)   # whole piece in one group;
            # below the lowest edge (no thermal motion here) -> kept in the last group
            mu_lab = (1 + A * mu) / np.sqrt(1 + A ** 2 + 2 * A * mu)
            f = wm * pdf_cm(mu, E)
            for l in range(L_MAX + 1):
                S[l, g, gp] += N * wi * se * np.sum(f * eval_legendre(l, mu_lab))

# pdf normalisation check: sum_g S0(g' -> g) = sigma_s,g'
print("max |sum_g S0 - sigma_s| / sigma_s:", np.max(np.abs(S[0].sum(0) / (sigma_t - sigma_a) - 1)))
print("thermal g = G-1: sigma_t = %.4f, sigma_a = %.4f /cm;  g = 0: sigma_t = %.3e /cm" % (sigma_t[-1], sigma_a[-1], sigma_t[0]))
print("max Sigma_s = %.3e /cm, N = %.4e /b-cm" % ((sigma_t - sigma_a).max(), N))
np.savez(HERE / 'he3_mgxs.npz', case='He3', density=DENSITY, temperature=294.0, L_max=L_MAX,
         group_edges_desc=E_edges, sigma_t=sigma_t, sigma_a=sigma_a, inv_velocity=gr['inv_velocity'],
         scatter_legendre=S)

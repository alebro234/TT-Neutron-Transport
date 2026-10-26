"""
Heterogeneous SlabTransport: graphite slab (30 cm, P20, 128 groups, data copied from graphite/results_P20_g_128)
with a 2 cm He-3 detector gas region (pure He-3, 0.001 g/cm^3, he3_mgxs.py) at z in [Z_DET, Z_DET + 2].
Only the constructor and the operator: prints the TT ranks of A (expected (z|mu) rank C + 2).

Cross-section input:  sigma_t = (C, G) table, region_map = chi (N_z, C) 0/1 matrix, one material per cell
                      sigma_s = (C, L_max+1, G, G)                    (row = incoming group, SlabTransport convention)

Run with the TT environment:
    /home/ale234/JRC/tt_python/tt_env/bin/python het_operator_ranks.py [--N_z 300] [--z_det 14] [--N_mu 24]
"""
import argparse
import time
from pathlib import Path

import contextlib
import io

import numpy as np
import tt_transport.slab as tt

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--N_z', type=int, default=300, help='cells; detector edges must fall on cell edges')
ap.add_argument('--z_det', type=float, default=14.0, help='left edge of the 2 cm detector [cm]')
ap.add_argument('--N_mu', type=int, default=24)
ap.add_argument('--p', type=int, default=1)
ap.add_argument('--N_t', type=int, default=20, help='time steps of the transient operator')
args = ap.parse_args()

MATERIALS = ['graphite', 'he3']                        # c = 0, 1
xs = [np.load(HERE / f'{m}_mgxs.npz') for m in MATERIALS]
L = float(xs[0]['L'])
E_edges = xs[0]['group_edges_desc']
assert np.allclose(xs[1]['group_edges_desc'], E_edges)
sigma_t_tab = np.stack([x['sigma_t'] for x in xs])                                 # (C, G)
sigma_s = np.stack([np.swapaxes(x['scatter_legendre'], 1, 2) for x in xs])         # (C, L+1, G, G), row = incoming
with np.errstate(divide='ignore'):
    v = 1 / xs[0]['inv_velocity']                      # lowest group empty in graphite: 1/v = 0 -> v = inf (no time term)
C, L1, G, _ = sigma_s.shape

h = L / args.N_z
z_edges = np.arange(args.N_z + 1) * h
for z_b in (args.z_det, args.z_det + 2.0):
    if np.min(np.abs(z_edges - z_b)) > 1e-9 * L:
        raise SystemExit(f"detector edge z = {z_b} is not a cell edge (h = {h}); change --N_z or --z_det")
z_mid = (np.arange(args.N_z) + 0.5) * h
region_map = ((z_mid > args.z_det) & (z_mid < args.z_det + 2.0)).astype(int)       # 0 graphite, 1 He-3

print(f"L = {L} cm, N_z = {args.N_z} (h = {h:g} cm), p = {args.p}, N_mu = {args.N_mu}, G = {G}, L_max = {L1 - 1}")
print(f"He-3 cells {np.flatnonzero(region_map)[[0, -1]]}, z in [{args.z_det}, {args.z_det + 2}] cm")
for c, m in enumerate(MATERIALS):
    print(f"  {m:9s} sigma_t: g=0 {sigma_t_tab[c, 0]:.3e}, max {sigma_t_tab[c].max():.3e} /cm;  "
          f"sum_l ||S_l|| = {np.abs(sigma_s[c]).sum():.3e}")
chi = np.eye(C)[region_map]                            # region map as the (N_z, C) 0/1 matrix
print(f"region map chi: shape {chi.shape}, cells per material {chi.sum(0).astype(int).tolist()}")


def operator(mode, tkw, **xs_kw):
    """Constructor + operator only; returns the model and the wall time (the class prints A: silenced)."""
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        m = tt.SlabTransport(N_z=args.N_z, p=args.p, L_z=L, N_mu=args.N_mu, N_phi=1, E_edges=E_edges,
                             azimuthal_symmetry=True, mode=mode, **tkw, **xs_kw)
        m.construct_tt_operator()
    return m, time.time() - t0


def compression(A):
    """Dense entries (prod N)^2 of the operator, TT entries (sum of core sizes), their ratio, TT memory in MB."""
    n_full = float(np.prod([float(n) for n in A.N])) ** 2
    n_tt = sum(c.numel() for c in A.cores)
    return n_full, n_tt, n_full / n_tt, n_tt * 8 / 2 ** 20


print("\nTT ranks of A over (z-dof | mu | phi | g | t), after rounding (term by term in brackets);")
print("compression = dense entries / TT entries (the dense A would be (prod N)^2 doubles):")
for tag, tkw in (('steady', dict(steady_state=True)),
                 (f'transient N_t={args.N_t}', dict(steady_state=False, N_t=args.N_t, T=1e-3, v=v))):
    for mode in ('frw', 'adj'):
        for name, xs_kw in (('graphite only, C=1', dict(sigma_t=sigma_t_tab[0], sigma_s=sigma_s[0])),
                            ('graphite + He-3, C=2', dict(sigma_t=sigma_t_tab, sigma_s=sigma_s, region_map=chi))):
            m, wall = operator(mode, tkw, **xs_kw)
            n_full, n_tt, ratio, mb = compression(m.A)
            print(f"  {name:21s} {mode} {tag:15s} {str(list(m.A.R)):24s} ({list(m.A_ranks_raw)})  {wall:.1f} s",
                  flush=True)
            print(f"  {'':21s} dense {n_full:.2e}, TT {n_tt:.3e} entries ({mb:.1f} MB), compression {ratio:.2e}",
                  flush=True)

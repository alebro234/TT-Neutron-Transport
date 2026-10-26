"""
Time-dependent diffusion (P1) ansatz as AMEn initial guess, graphite slab, P7, 10 groups.

Problem: slab 0 < z < L, zero initial condition, no volume source, vacuum at z = L, and at z = 0 a normal 2 MeV beam
(group 0) switched on for the first tenth of the run:
    J_in(t) = 1 for 0 < t < 0.1 T, 0 otherwise        (incoming partial current per unit area)
  transport: the beam enters as inflow on the ordinate closest to mu = 1 (the Gauss rule has no mu = 1),
             psi_in(mu_N) = J_in / (2 pi mu_N w_N), constant in the azimuth, zero on the other ordinates, so that
             sum_{mu_n > 0} mu_n w_n int dphi psi_in = J_in;
  diffusion: the same J_in through the Marshak condition phi/4 + J/2 = J_in at z = 0 (phi/4 - J/2 = 0 at z = L).
Both solve for the total flux (no uncollided split), so the P1 ansatz approximates the transport unknown itself.
The time grid is backward Euler all at once, t_k = k dt, dt = T/N_t; the step enters as its average over
(t_{k-1}, t_k) (exact integral of J_in over the step, 0.1 N_t full steps when 10 | N_t).

  1. SlabTransport without ansatz (torchtt default x0 = ones, rank 1).
  2. SlabDiffusion on the same cells, groups and time levels (continuous Q1, D_g = 1/(3 sigma_tr,g),
     sigma_tr,g = sigma_t,g - sum_g' sigma^1_{g'->g}); checked against sparse backward-Euler time marching.
  3. psi0 = phi/(4 pi) - 3/(4 pi) mu D_g d_z phi over (z, mu, phi, g, t) (tt.p1_angular_flux, p = 0 ansatz in the
     p = 1 modes), then SlabTransport again with x0 = psi0.
Group 9 has no data (sigma_t = 0, 1/v = 0, no transfer into it): left out of the diffusion solve, zero in psi0, and
given v_9 = v_8 in transport (its flux is 0 whatever v is).

Data: graphite_P7g10/ (v_g = 1/inv_velocity from graphite_mgxs.npz).
Outputs: runs/td_P7g10_Nmu{N_mu}_p1_Nt{N_t}_T{T}[_az]/  run.log, amen_*.log, ansatz_td_results.npz, *.png

Run with the TT environment:
  /home/ale234/JRC/tt_python/tt_env/bin/python diffusion_ansatz_graphite_td.py [--N_t 50] [--T 1e-3] [--N_mu 16] [--az]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import torch as tn
import torchtt as tntt

HERE = Path(__file__).resolve().parent
import tt_transport.slab as tt
from amen_monitor import start_log, capture_fd1, parse_amen, report_amen, tt_entries, rel_res   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--N_t', type=int, default=50)
ap.add_argument('--T', type=float, default=1e-3, help='final time [s]')
ap.add_argument('--N_mu', type=int, default=16)
ap.add_argument('--az', action='store_true', help='azimuthal_symmetry=True, N_phi = 1')
ap.add_argument('--eps', type=float, default=1e-7, help='AMEn tolerance of the transport solves')
ap.add_argument('--eps_diff', type=float, default=1e-10, help='AMEn tolerance of the diffusion solve')
ap.add_argument('--nswp', type=int, default=40)
ap.add_argument('--kickrank', type=int, default=4)
ap.add_argument('--lumped', action='store_true', help='lumped mass in SlabDiffusion')
ap.add_argument('--x0_round', type=float, default=None, help='round the ansatz psi0 at this tolerance before AMEn')
args = ap.parse_args()

P_DEG = 1
DATA = HERE / 'graphite_P7g10'
OUT = HERE / 'runs' / (f'td_P7g10_Nmu{args.N_mu}_p{P_DEG}_Nt{args.N_t}_T{args.T:g}' + ('_az' if args.az else '')
                      + (f'_kr{args.kickrank}' if args.kickrank != 4 else '')
                      + (f'_x0r{args.x0_round:g}' if args.x0_round else ''))
OUT.mkdir(parents=True, exist_ok=True)
start_log(OUT / 'run.log')

# ---------------------------------------------------------------------------
# data, grids, pulse
# ---------------------------------------------------------------------------
xs = np.load(DATA / 'graphite_mgxs.npz')
ce = np.load(DATA / 'graphite_ce_flux.npz')
L = float(xs['L'])
sigma_t = xs['sigma_t']                    # (G,), g = 0 highest energy
S = xs['scatter_legendre']                 # S[l, g, g'] = sigma_s^l(g' -> g), row = receiving group
inv_v = xs['inv_velocity']
L_MAX = S.shape[0] - 1
G = len(sigma_t)
E_edges = xs['group_edges_desc']
N_z = len(ce['z'])
N_PHI = 1 if args.az else L_MAX + 3
N_RECON = 3
N_t, T = args.N_t, args.T
dt = T / N_t
AMEN = dict(nswp=args.nswp, eps=args.eps, kickrank=args.kickrank)

act = np.flatnonzero(sigma_t > 0)
Ga = len(act)
v = np.empty(G)
v[act] = 1.0 / inv_v[act]
v[sigma_t == 0] = v[act[-1]]

t_k = dt * np.arange(1, N_t + 1)
J_avg = np.clip((np.minimum(t_k, 0.1 * T) - (t_k - dt)) / dt, 0.0, 1.0)     # step average over (t_{k-1}, t_k)
print(f"graphite P{L_MAX}, G = {G}, L = {L} cm, N_z = {N_z}, p = {P_DEG}, N_mu = {args.N_mu}, N_phi = {N_PHI}, "
      f"az = {args.az}; N_t = {N_t}, T = {T:g} s, dt = {dt:.3g} s; AMEn {AMEN}")
print("v_g [cm/s] = " + np.array2string(v, precision=3))
print(f"J_in step averages (nonzero): {J_avg[J_avg > 0]}")

# ---------------------------------------------------------------------------
# transport model, beam inflow on the ordinate closest to mu = 1
# ---------------------------------------------------------------------------
model = tt.SlabTransport(N_z=N_z, p=P_DEG, L_z=L, N_mu=args.N_mu, N_phi=N_PHI, sigma_t=sigma_t,
                         sigma_s=[S[l].T for l in range(L_MAX + 1)],   # SlabTransport: row = incoming group
                         v=v, E_edges=E_edges, N_t=N_t, T=T, steady_state=False, N_recon=N_RECON,
                         azimuthal_symmetry=args.az)
n_beam = int(np.argmax(model.mu))
psi_beam = np.zeros(args.N_mu)
psi_beam[n_beam] = 1.0 / (2 * np.pi * model.mu[n_beam] * model.w_mu[n_beam])
e0 = np.zeros(G)
e0[0] = 1.0
print(f"beam ordinate mu_N = {model.mu[n_beam]:.6f} (w_N = {model.w_mu[n_beam]:.4e}), "
      f"partial current per unit J_in = {2 * np.pi * np.sum(np.maximum(model.mu, 0) * model.w_mu * psi_beam):.6f}")
model.set_source(0)
model.set_IC(0)
model.set_BC_left([lambda mu: psi_beam, lambda phi: np.ones_like(phi), e0, lambda t: J_avg])
model.set_BC_right(0)
t0 = time.time()
model.construct_tt_operator()
model.construct_tt_rhs(round_eps=1e-12)
t_build_tr = time.time() - t0
A_tr, F_tr = model.A, model.F
print(f"transport operator: ranks {list(A_tr.R)} (raw {model.A_ranks_raw}), F ranks {list(F_tr.R)}, "
      f"dofs {int(np.prod(F_tr.N))}, build {t_build_tr:.2f} s")

# ---------------------------------------------------------------------------
# 1. transport without ansatz
# ---------------------------------------------------------------------------
x_ones = tntt.ones(list(F_tr.N), dtype=F_tr.cores[0].dtype)
res0_ones = rel_res(A_tr, x_ones, F_tr)
print("\n=== transport solve 1: x0 = ones (torchtt default) ===")
with capture_fd1(OUT / 'amen_transport_no_ansatz.log') as cap:
    t0 = time.time()
    model.run_amen(**AMEN, verbose=True)
    wall_1 = time.time() - t0
sw_1, nl_1, its_1 = parse_amen(cap['text'])
psi_1 = model.psi_modal
res_1 = rel_res(A_tr, psi_1, F_tr)
phi_r = model.compute_scalar_flux().full().numpy()                          # (N_z N_recon, G, N_t) at z_r
phi_tr = np.einsum('irgk,r->igk', phi_r.reshape(N_z, N_RECON, G, N_t), model.w_xi) / 2   # cell averages
report_amen('transport, x0 = ones', sw_1, nl_1, its_1, wall_1)
print(f"  solution ranks {list(psi_1.R)}, entries {tt_entries(psi_1)}, true residual {res_1:.2e}")

# ---------------------------------------------------------------------------
# 2. diffusion on the same cells, groups (with data) and time levels
# ---------------------------------------------------------------------------
sig_s_d = [S[l].T[np.ix_(act, act)] for l in (0, 1)]          # row = incoming group, as SlabTransport
print(f"\n=== diffusion: groups {act.tolist()} (G = {Ga}), {N_z + 1} nodes, N_t = {N_t}, "
      f"{'lumped' if args.lumped else 'consistent'} mass ===")
t0 = time.time()
diff = tt.SlabDiffusion(N_z=N_z, L_z=L, sigma_t=sigma_t[act], sigma_s=sig_s_d, v=v[act], N_t=N_t, T=T,
                        steady_state=False, lumped_mass=args.lumped)
diff.set_source(0)
diff.set_IC(0)
diff.set_BC_left([e0[act], lambda t: J_avg])
diff.set_BC_right(0)
diff.construct_tt_operator()
diff.construct_tt_rhs(round_eps=1e-12)
t_build_d = time.time() - t0
print("sigma_tr = " + np.array2string(diff.sigma_tr_c[0], precision=4))
print("D_g      = " + np.array2string(diff.D, precision=4))
with capture_fd1(OUT / 'amen_diffusion.log') as cap:
    t0 = time.time()
    diff.run_amen(nswp=args.nswp, eps=args.eps_diff, kickrank=args.kickrank, round_eps=1e-12, verbose=True)
    wall_d = time.time() - t0
sw_d, nl_d, its_d = parse_amen(cap['text'])
res_d = rel_res(diff.A, diff.phi, diff.F)
report_amen('diffusion', sw_d, nl_d, its_d, wall_d)
print(f"  operator ranks {list(diff.A.R)} (raw {diff.A_ranks_raw}), F ranks {list(diff.F.R)}, "
      f"solution ranks {list(diff.phi.R)}, true residual {res_d:.2e}, build {t_build_d:.2f} s")

# reference: sparse backward-Euler time marching of the same discrete system
n_s = (N_z + 1) * Ga
Vinv = sp.diags(1.0 / v[act])
A_step = sp.csc_matrix(sp.kron(sp.csr_matrix(diff.M), Vinv) / dt + sp.kron(sp.csr_matrix(diff.B), sp.eye(Ga)))
for K_c, M_c, D_c, s_t, s_s in zip(diff.K_c, diff.M_c, diff.D_c, diff.sigma_t_c, diff.sigma_s):
    A_step += sp.kron(sp.csr_matrix(K_c), sp.diags(D_c)) + sp.kron(sp.csr_matrix(M_c), sp.diags(s_t) - s_s[0].T)
M_hist = sp.csr_matrix(sp.kron(sp.csr_matrix(diff.M), Vinv) / dt)
F_full = diff.F.full().numpy().reshape(n_s, N_t)           # index (z, g) row-major = kron order
t0 = time.time()
lu = spl.splu(sp.csc_matrix(A_step))
x_ref = np.zeros((n_s, N_t))
x_prev = np.zeros(n_s)
for k in range(N_t):
    x_prev = lu.solve(F_full[:, k] + M_hist @ x_prev)
    x_ref[:, k] = x_prev
t_march = time.time() - t0
x_amen = diff.phi.full().numpy().reshape(n_s, N_t)
err_d = np.linalg.norm(x_amen - x_ref) / np.linalg.norm(x_ref)
phi_d_nodes = np.zeros((N_z + 1, G, N_t))
phi_d_nodes[:, act, :] = x_ref.reshape(N_z + 1, Ga, N_t)
sv = np.linalg.svd(x_ref.reshape(N_z + 1, Ga * N_t), compute_uv=False)
print(f"  sparse time marching {t_march:.3f} s; ||phi_AMEn - phi_march|| / ||phi_march|| = {err_d:.2e}; "
      f"min phi = {x_ref.min():.3e}")
print("  z|(g,t) singular values / s_0: " + " ".join(f"{s / sv[0]:.0e}" for s in sv[:14]))

# TT phi(z, g, t) on all G groups (zero in the empty ones), D_g padded likewise
c_z, c_g, c_t = diff.phi.cores
c_g_full = tn.zeros((c_g.shape[0], G, c_g.shape[2]), dtype=c_g.dtype)
c_g_full[:, act, :] = c_g
phi_zgt = tntt.TT([c_z, c_g_full, c_t])
D_full = np.zeros(G)
D_full[act] = diff.D
phi_d = 0.5 * (phi_d_nodes[:-1] + phi_d_nodes[1:])          # cell averages (N_z, G, N_t)

# ---------------------------------------------------------------------------
# 3. P1 ansatz and transport with x0 = psi0
# ---------------------------------------------------------------------------
t0 = time.time()
psi0 = tt.p1_angular_flux(phi_zgt, D_full, model)
ranks_psi0_full = list(psi0.R)
if args.x0_round:
    psi0 = psi0.round(args.x0_round)
t_ansatz = time.time() - t0
print(f"\nansatz psi0: ranks {ranks_psi0_full}"
      + (f" -> {list(psi0.R)} after rounding at {args.x0_round:g}" if args.x0_round else "")
      + f", built in {t_ansatz:.3f} s")


def weighted_rel_err(x, ref, drop_beam=False):
    """||x - ref||_W / ||ref||_W with the physical weights: DG mass h/(2a+1) in z, w_mu w_phi in angle (dt cancels);
    drop_beam = True also gives the beam ordinate mu_N zero weight (uncollided beam excluded)."""
    w_z = np.tile(model.h / (2 * np.arange(P_DEG + 1) + 1), N_z)
    w_mu = model.w_mu.copy()
    if drop_beam:
        w_mu[n_beam] = 0.0
    scale = [np.sqrt(w_z), np.sqrt(w_mu), np.sqrt(model.w_phi)]

    def weigh(T):
        cores = list(T.cores)
        for d, s in enumerate(scale):
            cores[d] = cores[d] * tn.tensor(s, dtype=cores[d].dtype)[None, :, None]
        return tntt.TT(cores)
    return float(weigh(x - ref).norm() / weigh(ref).norm())


res0_ans = rel_res(A_tr, psi0, F_tr)
err0_ans = float((psi0 - psi_1).norm() / psi_1.norm())
err0_ones = float((x_ones - psi_1).norm() / psi_1.norm())
werr0_ans, werr0_ans_nb = weighted_rel_err(psi0, psi_1), weighted_rel_err(psi0, psi_1, drop_beam=True)
werr0_ones = weighted_rel_err(x_ones, psi_1)
print(f"  initial residual ||A x0 - F||/||F||:   ones {res0_ones:.3e}   ansatz {res0_ans:.3e}")
print(f"  initial error ||x0 - psi_1||/||psi_1||: ones {err0_ones:.3e}   ansatz {err0_ans:.3e}   (unweighted)")
print(f"  initial error, weighted (mass, w_mu w_phi): ones {werr0_ones:.3e}   ansatz {werr0_ans:.3e}   "
      f"ansatz without the beam ordinate {werr0_ans_nb:.3e}")

print("\n=== transport solve 2: x0 = P1 ansatz ===")
with capture_fd1(OUT / 'amen_transport_ansatz.log') as cap:
    t0 = time.time()
    model.run_amen(**AMEN, verbose=True, x0=psi0)
    wall_2 = time.time() - t0
sw_2, nl_2, its_2 = parse_amen(cap['text'])
psi_2 = model.psi_modal
res_2 = rel_res(A_tr, psi_2, F_tr)
report_amen('transport, x0 = P1 ansatz', sw_2, nl_2, its_2, wall_2)
print(f"  solution ranks {list(psi_2.R)}, entries {tt_entries(psi_2)}, true residual {res_2:.2e}")
diff_12 = float((psi_2 - psi_1).norm() / psi_1.norm())

# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
t_pre = t_build_d + wall_d + t_ansatz
print("\n================ summary ================")
print(f"{'':28s} | {'x0 = ones':>22s} | {'x0 = P1 ansatz':>22s}")
print(f"{'sweeps':28s} | {len(sw_1):22d} | {len(sw_2):22d}")
print(f"{'AMEn wall time [s]':28s} | {wall_1:22.2f} | {wall_2:22.2f}")
print(f"{'  + diffusion/ansatz [s]':28s} | {'':22s} | {t_pre:22.2f}")
print(f"{'local GMRES / direct':28s} | {f'{nl_1[0]} / {nl_1[1]}':>22s} | {f'{nl_2[0]} / {nl_2[1]}':>22s}")
print(f"{'GMRES local its (total)':28s} | {sum(its_1):22d} | {sum(its_2):22d}")
print(f"{'max rank during solve':28s} | {max(max(s['ranks']) for s in sw_1):22d} | "
      f"{max(max(s['ranks']) for s in sw_2):22d}")
print(f"{'final ranks':28s} | {str(list(psi_1.R)):>22s} | {str(list(psi_2.R)):>22s}")
print(f"{'true residual':28s} | {res_1:22.2e} | {res_2:22.2e}")
print(f"{'initial residual':28s} | {res0_ones:22.2e} | {res0_ans:22.2e}")
print(f"{'initial error vs psi_1':28s} | {err0_ones:22.2e} | {err0_ans:22.2e}")
print(f"||psi_2 - psi_1|| / ||psi_1|| = {diff_12:.2e}")

# diffusion vs transport, scalar flux
pop_tr = phi_tr.sum(axis=0) * (L / N_z)                     # group populations int phi dz, (G, N_t)
pop_d = phi_d.sum(axis=0) * (L / N_z)
k_show = sorted({0, N_t // 10 - 1, N_t // 5, N_t // 2, N_t - 1} - {-1})
print("\nscalar flux, diffusion vs transport, per group:")
print("g | ||d-t||/||t|| over (z,t) | int dz: d/t - 1 at t = " + ", ".join(f"{t_k[k]:.2e}" for k in k_show))
for g in act:
    num = np.linalg.norm(phi_d[:, g] - phi_tr[:, g])
    den = np.linalg.norm(phi_tr[:, g])
    with np.errstate(divide='ignore', invalid='ignore'):
        r_pop = pop_d[g, k_show] / pop_tr[g, k_show] - 1
    print(f"{g} | {num / den:.4f}                   | " + " ".join(f"{x:+.3f}" for x in r_pop))

np.savez(OUT / 'ansatz_td_results.npz', z=model.z_mid, z_nodes=diff.z, t=t_k, J_in=J_avg, E_edges=E_edges, act=act,
         v=v, D=D_full, sigma_tr=diff.sigma_tr_c[0], phi_transport=phi_tr, phi_diffusion=phi_d,
         phi_diffusion_nodes=phi_d_nodes,
         maxres_1=np.array([s['maxres'] for s in sw_1]), maxres_2=np.array([s['maxres'] for s in sw_2]),
         maxrank_1=np.array([max(s['ranks']) for s in sw_1]), maxrank_2=np.array([max(s['ranks']) for s in sw_2]),
         time_1=np.array([s['time'] for s in sw_1]), time_2=np.array([s['time'] for s in sw_2]),
         maxres_d=np.array([s['maxres'] for s in sw_d]),
         wall_1=wall_1, wall_2=wall_2, wall_d=wall_d, t_ansatz=t_ansatz, res_1=res_1, res_2=res_2, res_d=res_d,
         res0_ones=res0_ones, res0_ans=res0_ans, err0_ones=err0_ones, err0_ans=err0_ans, werr0_ones=werr0_ones, werr0_ans=werr0_ans,
         werr0_ans_nobeam=werr0_ans_nb, ranks_psi0_full=np.array(ranks_psi0_full), diff_12=diff_12,
         err_d_march=err_d, ranks_1=np.array(psi_1.R), ranks_2=np.array(psi_2.R), ranks_psi0=np.array(psi0.R),
         ranks_d=np.array(diff.phi.R))

# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
import matplotlib                                          # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for sw, lab, st in ((sw_1, 'x0 = ones', 'o-'), (sw_2, 'x0 = P1 ansatz', 's-')):
    k = np.arange(1, len(sw) + 1)
    ax[0].semilogy(k, [s['maxres'] for s in sw], st, ms=4, label=lab)
    ax[1].plot(k, [max(s['ranks']) for s in sw], st, ms=4, label=lab)
    ax[2].plot(k, np.cumsum([s['time'] for s in sw]), st, ms=4, label=lab)
ax[0].axhline(args.eps, color='k', lw=0.8, ls=':')
ax[0].set_ylabel('max local residual'); ax[1].set_ylabel('max solution rank'); ax[2].set_ylabel('cumulative time [s]')
for a_ in ax:
    a_.set_xlabel('sweep'); a_.grid(alpha=0.4); a_.legend()
fig.suptitle(f'Graphite P{L_MAX} g{G}, time-dependent transport AMEn (N_mu={args.N_mu}, N_phi={N_PHI}, p={P_DEG}, '
             f'N_t={N_t}, T={T:g} s, eps={args.eps:.0e})')
plt.tight_layout()
plt.savefig(OUT / 'amen_convergence.png', dpi=130)

cols = plt.cm.viridis(np.linspace(0, 0.95, G))
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for g in act:
    ax[0].semilogy(t_k, pop_tr[g], color=cols[g], label=f'g={g}')
    ax[0].semilogy(t_k, pop_d[g], '--', color=cols[g])
ax[0].axvline(0.1 * T, color='k', lw=0.8, ls=':')
ax[0].plot([], [], 'k-', label='transport'); ax[0].plot([], [], 'k--', label='diffusion')
ax[0].set_xlabel('t [s]'); ax[0].set_ylabel(r'$\int_0^L\phi_g\,dz$'); ax[0].grid(True, which='both', alpha=0.4)
ax[0].legend(fontsize=7, ncol=2); ax[0].set_title('group populations')
tot_tr, tot_d = phi_tr.sum(axis=1), phi_d.sum(axis=1)       # total flux (N_z, N_t)
tcols = plt.cm.plasma(np.linspace(0, 0.85, len(k_show)))
for c, k in zip(tcols, k_show):
    ax[1].semilogy(model.z_mid, tot_tr[:, k], color=c, label=f't = {t_k[k]:.2e} s')
    ax[1].semilogy(model.z_mid, tot_d[:, k], '--', color=c)
ax[1].set_xlabel('z [cm]'); ax[1].set_ylabel(r'$\sum_g\phi_g$'); ax[1].grid(True, which='both', alpha=0.4)
ax[1].legend(fontsize=8); ax[1].set_title('total flux, transport (solid) vs diffusion (dashed)')
plt.tight_layout()
plt.savefig(OUT / 'diffusion_vs_transport.png', dpi=130)
print(f"\noutputs in {OUT}")

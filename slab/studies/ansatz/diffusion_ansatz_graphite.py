"""
Diffusion (P1) ansatz as AMEn initial guess for the graphite slab, P7, 10 groups.

  1. SlabTransport, steady state, without ansatz (torchtt default x0 = ones, rank 1).
  2. SlabDiffusion on the same cells and groups (continuous Q1, Marshak vacuum), same first-collision source
     (isotropic part, sum_g' -> int dOmega), D_g = 1/(3 sigma_tr,g), sigma_tr,g = sigma_t,g - sum_g' sigma^1_{g'->g}.
  3. P1 angular flux psi0 = phi/(4 pi) - 3/(4 pi) mu D_g d_z phi (tt.p1_angular_flux; p = 0 ansatz in the
     p = 1 DG modes: mode 1 set to 0), then SlabTransport again with x0 = psi0.
Monitors the two transport solves (per-sweep max residual, ranks and time from the C++ AMEn output, wall time,
final ranks, true residual, initial residual/error of x0) and analyses the diffusion solve (operator/solution
ranks, AMEn vs a dense direct solve, diffusion vs transport scalar flux).

Problem (as ../montecarlo/P0trg10/tt_graphite_slab.py): slab 0 < z < L, vacuum faces, normal 2 MeV beam on z = 0,
uncollided beam split off analytically; the solvers see the collided flux with the first-collision source
    q_g(z, mu) = sum_l (2l+1)/(4 pi) sigma_s^l(0 -> g) P_l(mu) exp(-sigma_t,0 z).
Group 9 has no data (sigma_t = 0, no transfer into it): it is left out of the diffusion solve (D undefined) and
padded with zeros in psi0.

Data: graphite_P7g10/ (copied from ~/openmc/graphite/results_P7_g10).
Outputs: runs/P7g10_Nmu{N_mu}_p1[_az]/  run.log, amen_*.log (raw C++ AMEn output), ansatz_results.npz, *.png

Run with the TT environment:
  /home/ale234/JRC/tt_python/tt_env/bin/python diffusion_ansatz_graphite.py [--N_mu 16] [--az] [--eps 1e-7]
"""
import argparse
import ctypes
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch as tn
import torchtt as tntt
from scipy.special import eval_legendre

HERE = Path(__file__).resolve().parent
import tt_transport.slab as tt

ap = argparse.ArgumentParser()
ap.add_argument('--N_mu', type=int, default=16)
ap.add_argument('--az', action='store_true', help='azimuthal_symmetry=True, N_phi = 1')
ap.add_argument('--eps', type=float, default=1e-7, help='AMEn tolerance of the transport solves')
ap.add_argument('--eps_diff', type=float, default=1e-10, help='AMEn tolerance of the diffusion solve')
ap.add_argument('--nswp', type=int, default=40)
ap.add_argument('--kickrank', type=int, default=4)
ap.add_argument('--lumped', action='store_true', help='lumped mass in SlabDiffusion')
args = ap.parse_args()

P_DEG = 1
DATA = HERE / 'graphite_P7g10'
OUT = HERE / 'runs' / (f'P7g10_Nmu{args.N_mu}_p{P_DEG}' + ('_az' if args.az else ''))
OUT.mkdir(parents=True, exist_ok=True)


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


_log = open(OUT / 'run.log', 'w')
sys.stdout = Tee(sys.__stdout__, _log)
sys.stderr = Tee(sys.__stderr__, _log)
_libc = ctypes.CDLL(None)


@contextmanager
def capture_fd1(path):
    """Redirect file descriptor 1 (where the C++ AMEn prints) to `path`; Python prints inside go there too.
    The captured text is returned in the yielded dict under 'text'."""
    out = {}
    sys.stdout.flush()
    saved_py, saved_fd = sys.stdout, os.dup(1)
    with open(path, 'w+b') as f:
        os.dup2(f.fileno(), 1)
        sys.stdout = sys.__stdout__
        try:
            yield out
        finally:
            sys.__stdout__.flush()
            _libc.fflush(None)
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
            sys.stdout = saved_py
            f.seek(0)
            out['text'] = f.read().decode(errors='replace')


def parse_amen(text):
    """Per-sweep history from the C++ AMEn verbose output: sweep, solution ranks, max residual, time [s]."""
    sweeps = []
    for block in re.split(r'Starting sweep ', text)[1:]:
        m_r = re.search(r'Solution rank \[ ([\d ]+)\]', block)
        m_res = re.search(r'Maxres ([0-9.eE+-]+|nan|inf)', block)
        m_t = re.search(r'\nTime ([0-9.eE+-]+) ms', block)
        if m_r and m_res:
            sweeps.append(dict(ranks=[int(x) for x in m_r.group(1).split()], maxres=float(m_res.group(1)),
                               time=float(m_t.group(1)) / 1e3 if m_t else np.nan))
    n_local = len(re.findall(r'Choosing iterative solver', text)), len(re.findall(r'Choosing direct solver', text))
    gmres_its = [int(x) for x in re.findall(r'after (\d+) iterations', text)]
    return sweeps, n_local, gmres_its


def tt_entries(T):
    return sum(c.numel() for c in T.cores)


def rel_res(A, x, F):
    return float((A @ x - F).norm() / F.norm())


def report_amen(tag, sweeps, n_local, gmres_its, wall):
    print(f"\n[{tag}] AMEn history ({len(sweeps)} sweeps, wall {wall:.2f} s; local solves: "
          f"{n_local[0]} GMRES ({sum(gmres_its)} its total), {n_local[1]} direct)")
    print("  sweep | max res    | max rank | ranks                     | time [s]")
    for k, s in enumerate(sweeps):
        print(f"  {k + 1:5d} | {s['maxres']:.3e}  | {max(s['ranks']):8d} | {str(s['ranks']):25s} | {s['time']:.2f}")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
xs = np.load(DATA / 'graphite_mgxs.npz')
ref = np.load(DATA / 'graphite_mg_flux.npz')
ce = np.load(DATA / 'graphite_ce_flux.npz')
L = float(xs['L'])
sigma_t = xs['sigma_t']                    # (G,), g = 0 highest energy
S = xs['scatter_legendre']                 # S[l, g, g'] = sigma_s^l(g' -> g), row = receiving group
L_MAX = S.shape[0] - 1
G = len(sigma_t)
E_edges = xs['group_edges_desc']
N_z = len(ce['z'])                         # OpenMC tally mesh
N_PHI = 1 if args.az else L_MAX + 3
N_RECON = 3
AMEN = dict(nswp=args.nswp, eps=args.eps, kickrank=args.kickrank)
print(f"graphite P{L_MAX}, G = {G}, L = {L} cm, N_z = {N_z}, p = {P_DEG}, N_mu = {args.N_mu}, N_phi = {N_PHI}, "
      f"az = {args.az}; AMEn {AMEN}")

# ---------------------------------------------------------------------------
# transport model (operator and first-collision source built once, reused by both solves)
# ---------------------------------------------------------------------------
model = tt.SlabTransport(N_z=N_z, p=P_DEG, L_z=L, N_mu=args.N_mu, N_phi=N_PHI, sigma_t=sigma_t,
                         sigma_s=[S[l].T for l in range(L_MAX + 1)],   # SlabTransport: row = incoming group
                         E_edges=E_edges, steady_state=True, N_recon=N_RECON, azimuthal_symmetry=args.az)
Q_z = model._project_z(lambda z: np.exp(-sigma_t[0] * z))
q = None
for l in range(L_MAX + 1):
    term = tt.vector_kron([Q_z, (2 * l + 1) / (4 * np.pi) * eval_legendre(l, model.mu), np.ones(N_PHI),
                           S[l][:, 0], np.ones(1)])
    q = term if q is None else q + term
model.q = q.round(1e-12)
model.set_BC_left(0)
model.set_BC_right(0)
t0 = time.time()
model.construct_tt_operator()
model.construct_tt_rhs()
t_build_tr = time.time() - t0
A_tr, F_tr = model.A, model.F
n_dof_tr = int(np.prod(F_tr.N))
print(f"transport operator: ranks {list(A_tr.R)} (raw {model.A_ranks_raw}), F ranks {list(F_tr.R)}, "
      f"dofs {n_dof_tr}, build {t_build_tr:.2f} s")

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
phi_r = model.compute_scalar_flux().full().numpy()[:, :, 0]
phi_tr_c = np.einsum('irg,r->ig', phi_r.reshape(N_z, N_RECON, G), model.w_xi) / 2      # collided cell averages
report_amen('transport, x0 = ones', sw_1, nl_1, its_1, wall_1)
print(f"  solution ranks {list(psi_1.R)}, entries {tt_entries(psi_1)}, true residual {res_1:.2e}")

# ---------------------------------------------------------------------------
# 2. diffusion on the same cells and groups (groups with data only)
# ---------------------------------------------------------------------------
act = np.flatnonzero(sigma_t > 0)
Ga = len(act)
sig_s_d = [S[l].T[np.ix_(act, act)] for l in (0, 1)]          # row = incoming group, as SlabTransport
print(f"\n=== diffusion: groups {act.tolist()} (G = {Ga}), N_z = {N_z} cells -> {N_z + 1} nodes, "
      f"{'lumped' if args.lumped else 'consistent'} mass ===")
t0 = time.time()
diff = tt.SlabDiffusion(N_z=N_z, L_z=L, sigma_t=sigma_t[act], sigma_s=sig_s_d, E_edges=None,
                        steady_state=True, lumped_mass=args.lumped)
diff.set_source([lambda z: np.exp(-sigma_t[0] * z), S[0][act, 0]])   # isotropic part: sigma_s^0(0 -> g)
diff.set_BC_left(0)
diff.set_BC_right(0)
diff.construct_tt_operator()
diff.construct_tt_rhs()
t_build_d = time.time() - t0
print("sigma_tr =", np.array2string(diff.sigma_tr_c[0], precision=4))
print("D_g      =", np.array2string(diff.D, precision=4))
with capture_fd1(OUT / 'amen_diffusion.log') as cap:
    t0 = time.time()
    diff.run_amen(nswp=args.nswp, eps=args.eps_diff, kickrank=args.kickrank, round_eps=1e-12, verbose=True)
    wall_d = time.time() - t0
sw_d, nl_d, its_d = parse_amen(cap['text'])
res_d = rel_res(diff.A, diff.phi, diff.F)
report_amen('diffusion', sw_d, nl_d, its_d, wall_d)
print(f"  operator ranks {list(diff.A.R)} (raw {diff.A_ranks_raw}), F ranks {list(diff.F.R)}, "
      f"solution ranks {list(diff.phi.R)}, true residual {res_d:.2e}, build {t_build_d:.2f} s")

# dense direct solve of the same discrete system (small: (N_z+1) G unknowns)
n_d = (N_z + 1) * Ga
A_dense = diff.A.full().reshape(n_d, n_d).numpy()
F_dense = diff.F.full().reshape(n_d).numpy()
t0 = time.time()
x_dense = np.linalg.solve(A_dense, F_dense)
t_dense = time.time() - t0
x_amen = diff.phi.full().reshape(n_d).numpy()
err_d = np.linalg.norm(x_amen - x_dense) / np.linalg.norm(x_dense)
sv = np.linalg.svd(x_dense.reshape(N_z + 1, Ga), compute_uv=False)
print(f"  dense solve {t_dense:.3f} s, cond(A) = {np.linalg.cond(A_dense):.2e}; ||phi_AMEn - phi_dense|| / ||phi_dense|| "
      f"= {err_d:.2e}; min phi = {x_dense.min():.3e}")
print("  singular values of phi(z, g) / s_0: " + " ".join(f"{s / sv[0]:.1e}" for s in sv))

# 2-core TT phi(z, g) on all G groups (zero in the empty ones), D_g padded likewise
c_z, c_g, c_t = diff.phi.cores
c_g = tn.einsum('agb,b->ag', c_g, c_t[:, 0, 0])[:, :, None]
c_g_full = tn.zeros((c_g.shape[0], G, 1), dtype=c_g.dtype)
c_g_full[:, act, :] = c_g
phi_zg = tntt.TT([c_z, c_g_full])
D_full = np.zeros(G)
D_full[act] = diff.D

# diffusion vs transport: collided cell averages
phi_nodes = np.zeros((N_z + 1, G))
phi_nodes[:, act] = x_dense.reshape(N_z + 1, Ga)
phi_d_c = 0.5 * (phi_nodes[:-1] + phi_nodes[1:])

# ---------------------------------------------------------------------------
# 3. P1 ansatz and transport with x0 = psi0
# ---------------------------------------------------------------------------
t0 = time.time()
psi0 = tt.p1_angular_flux(phi_zg, D_full, model)
t_ansatz = time.time() - t0
res0_ans = rel_res(A_tr, psi0, F_tr)
err0_ans = float((psi0 - psi_1).norm() / psi_1.norm())
err0_ones = float((x_ones - psi_1).norm() / psi_1.norm())
print(f"\nansatz psi0: ranks {list(psi0.R)}, built in {t_ansatz:.3f} s")
print(f"  initial residual ||A x0 - F||/||F||:   ones {res0_ones:.3e}   ansatz {res0_ans:.3e}")
print(f"  initial error ||x0 - psi_1||/||psi_1||: ones {err0_ones:.3e}   ansatz {err0_ans:.3e}")

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
print(f"{'':28s} | {'x0 = ones':>12s} | {'x0 = P1 ansatz':>14s}")
print(f"{'sweeps':28s} | {len(sw_1):12d} | {len(sw_2):14d}")
print(f"{'AMEn wall time [s]':28s} | {wall_1:12.2f} | {wall_2:14.2f}")
print(f"{'  + diffusion/ansatz [s]':28s} | {'':12s} | {t_pre:14.2f}")
print(f"{'GMRES local its (total)':28s} | {sum(its_1):12d} | {sum(its_2):14d}")
print(f"{'max rank during solve':28s} | {max(max(s['ranks']) for s in sw_1):12d} | "
      f"{max(max(s['ranks']) for s in sw_2):14d}")
print(f"{'final ranks':28s} | {str(list(psi_1.R)):>12s} | {str(list(psi_2.R)):>14s}")
print(f"{'true residual':28s} | {res_1:12.2e} | {res_2:14.2e}")
print(f"{'initial residual':28s} | {res0_ones:12.2e} | {res0_ans:14.2e}")
print(f"{'initial error vs psi_1':28s} | {err0_ones:12.2e} | {err0_ans:14.2e}")
print(f"||psi_2 - psi_1|| / ||psi_1|| = {diff_12:.2e}")

with np.errstate(divide='ignore', invalid='ignore'):
    rel_dt = phi_d_c / phi_tr_c - 1
z_c = model.z_mid
idx = [0, N_z // 6, N_z // 3, N_z // 2, 2 * N_z // 3, 5 * N_z // 6, N_z - 1]
print("\ncollided scalar flux, diffusion / transport - 1:")
print("g | integrated | ||.||_2 rel | at z = " + ", ".join(f"{z_c[i]:.1f}" for i in idx))
for g in act:
    print(f"{g} | {phi_d_c[:, g].sum() / phi_tr_c[:, g].sum() - 1:+.4f}    | "
          f"{np.linalg.norm(phi_d_c[:, g] - phi_tr_c[:, g]) / np.linalg.norm(phi_tr_c[:, g]):.4f}      | "
          + " ".join(f"{rel_dt[i, g]:+.3f}" for i in idx))

# total flux (collided + uncollided) vs OpenMC MG
dz = L / N_z
z_lo = np.arange(N_z) * dz
phi_u = (np.exp(-sigma_t[0] * z_lo) - np.exp(-sigma_t[0] * (z_lo + dz))) / (sigma_t[0] * dz)
phi_tr_tot, phi_d_tot = phi_tr_c.copy(), phi_d_c.copy()
phi_tr_tot[:, 0] += phi_u
phi_d_tot[:, 0] += phi_u
a = ref['phi']
print(f"\nvs OpenMC MG, ||.-MG||_F/||MG||_F: transport {np.linalg.norm(phi_tr_tot - a) / np.linalg.norm(a):.4f}, "
      f"diffusion {np.linalg.norm(phi_d_tot - a) / np.linalg.norm(a):.4f}")

np.savez(OUT / 'ansatz_results.npz', z=z_c, z_nodes=diff.z, E_edges=E_edges, act=act, D=D_full,
         sigma_tr=diff.sigma_tr_c[0], phi_transport=phi_tr_c, phi_diffusion=phi_d_c, phi_diffusion_nodes=phi_nodes,
         phi_uncollided=phi_u, phi_mg=a,
         maxres_1=np.array([s['maxres'] for s in sw_1]), maxres_2=np.array([s['maxres'] for s in sw_2]),
         maxrank_1=np.array([max(s['ranks']) for s in sw_1]), maxrank_2=np.array([max(s['ranks']) for s in sw_2]),
         time_1=np.array([s['time'] for s in sw_1]), time_2=np.array([s['time'] for s in sw_2]),
         maxres_d=np.array([s['maxres'] for s in sw_d]),
         wall_1=wall_1, wall_2=wall_2, wall_d=wall_d, t_ansatz=t_ansatz, res_1=res_1, res_2=res_2, res_d=res_d,
         res0_ones=res0_ones, res0_ans=res0_ans, err0_ones=err0_ones, err0_ans=err0_ans, diff_12=diff_12,
         err_d_dense=err_d, ranks_1=np.array(psi_1.R), ranks_2=np.array(psi_2.R), ranks_psi0=np.array(psi0.R),
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
fig.suptitle(f'Graphite P{L_MAX} g{G}, transport AMEn (N_mu={args.N_mu}, N_phi={N_PHI}, p={P_DEG}, eps={args.eps:.0e})')
plt.tight_layout()
plt.savefig(OUT / 'amen_convergence.png', dpi=130)

cols = plt.cm.viridis(np.linspace(0, 0.95, G))
fig, ax = plt.subplots(2, 1, figsize=(10, 9), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
for g in act:
    ax[0].semilogy(z_c, phi_tr_c[:, g], color=cols[g], label=f'g={g}: {E_edges[g + 1]:.1e}-{E_edges[g]:.1e} eV')
    ax[0].semilogy(z_c, phi_d_c[:, g], '--', color=cols[g])
    ax[1].plot(z_c, rel_dt[:, g], color=cols[g])
ax[0].plot([], [], 'k-', label='transport (P7)'); ax[0].plot([], [], 'k--', label='diffusion (Q1)')
ax[0].set_ylabel(r'collided $\phi_g(z)$'); ax[0].grid(True, which='both', alpha=0.4); ax[0].legend(fontsize=7, ncol=2)
ax[0].set_title('Graphite slab, 2 MeV beam: collided scalar flux, transport vs diffusion ansatz')
ax[1].axhline(0, color='k', lw=0.8); ax[1].set_ylabel('diffusion / transport - 1'); ax[1].set_xlabel('z [cm]')
ax[1].grid(alpha=0.4); ax[1].set_ylim(-1, 1)
plt.tight_layout()
plt.savefig(OUT / 'diffusion_vs_transport.png', dpi=130)
print(f"\noutputs in {OUT}")

# Convergence study in space and time for the slab transport model (HG angular scattering +
# moderator model), against the smooth manufactured solution
#   psi_g(z, mu, phi, t) = amp_g (a_1 sin(pi z/L) + a_2 cos(pi z/L))
#                          (b_0 Y_00 + b_1 Y_11 + b_2 Y_2,-1) exp(-t/tau)
# with non-zero BC and IC.  Error: relative Frobenius norm of (psi_h - psi_manuf) at the Gauss points
# z_r of every cell, all (mu, phi, g) ordinates and all time steps t_k = k dt, computed in TT format.
#
# Two sweeps, each with the other variable at its finest value:
#   * space sweep: N_z varies, N_t = max, source with the EXACT BACKWARD DIFFERENCE of exp(-t/tau)
#     (the manufactured solution then satisfies backward Euler exactly, so the time error does not
#     hide the spatial one; expected rate p+1 in h)
#   * time sweep : N_t varies, N_z = max, source with the analytic derivative (the true backward
#     Euler error; expected rate 1 in dt)
# Optional FULL_GRID = True runs every (N_z, N_t) pair (analytic time derivative).

import io
import time
import contextlib
import sys
from math import sqrt, factorial
import numpy as np
import torch as tn
import matplotlib.pyplot as plt
from scipy.special import lpmv

import tt_transport.slab as tt

tn.set_default_dtype(tn.float64)

# ---------------------------------------------------------------------------
# manufactured solution
# ---------------------------------------------------------------------------
a_1 = 1.0
a_2 = 1.5

b_0 = 0.5
b_1 = 1.2
b_2 = 0.8

tau = 0.5
amp_g = np.array([0.1, 1.0])        # group factor (E_g in the write-up); not the group energy

modes = [(0, 0, b_0), (1, 1, b_1), (2, -1, b_2)]     # (l, m, coefficient)

# ---------------------------------------------------------------------------
# fixed parameters (angle, energy, anisotropy)
# ---------------------------------------------------------------------------
N_mu = 8
N_phi = 8
G = len(amp_g)
gamma = 0.5
L_max = 4                           # needs N_phi >= L_max + 3; exact quadrature for the l <= 2 modes
E_min, E_max = 1e-2, 1.0            # group energies -> velocities v_g (class default E_max = 1e6)

# space
L_z = 1.0
N_z_list = [8, 16, 32, 64, 128, 256, 512]
h = L_z / np.array(N_z_list)
print("h = ", h)

# time
T = 1.0
N_t_list = [8, 16, 32, 64, 128, 256, 512]
dt = T / np.array(N_t_list)
print("dt = ", dt)

P_LIST = [1]                        # polynomial degrees of the space sweep (e.g. [1, 2])
FULL_GRID = False

# AMEn settings (the tolerance must stay well below the smallest discretisation error)
EPS_AMEN = 1e-10
NSWP = 15
KICKRANK = 2
MAX_FULL = 15000
ROUND_EPS = 1e-11
VERBOSE = True

psi_z = lambda z: a_1 * np.sin(np.pi * z / L_z) + a_2 * np.cos(np.pi * z / L_z)
dpsi_z = lambda z: np.pi / L_z * (a_1 * np.cos(np.pi * z / L_z) - a_2 * np.sin(np.pi * z / L_z))


def A_lm(l, m, mu):
    """Polar factor N_{l|m|} P_l^{|m|}(mu) of the real spherical harmonic Y_lm."""
    ma = abs(m)
    N = sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - ma) / factorial(l + ma))
    return N * lpmv(ma, l, mu)


def Pi_m(m, phi):
    """Azimuthal factor of Y_lm (contains the sqrt(2) for m != 0), same convention as tt.real_sph."""
    if m > 0:
        return sqrt(2.0) * np.cos(m * phi)
    if m < 0:
        return sqrt(2.0) * np.sin(-m * phi)
    return np.ones_like(phi)


def manufactured_solution(model):
    """Exact solution as a TT on the grids of `model` (rank 3: one term per spherical harmonic)."""
    psi = None
    for (l, m, b) in modes:
        term = b * tt.vector_kron([psi_z(model.z_r), A_lm(l, m, model.mu), Pi_m(m, model.phi),
                                   amp_g, np.exp(-model.t / tau)])
        psi = term if psi is None else psi + term
    return psi


def accumulate(model, setter, attr, terms):
    """The class setters take ONE rank-1 term; the data here are sums of rank-1 terms."""
    total = None
    for core_funcs in terms:
        setter(core_funcs)
        x = getattr(model, attr)
        total = x if total is None else total + x
    setattr(model, attr, total.round(1e-12))


def set_manufactured_data(model, time_source):
    """Source, IC and BC of the manufactured solution (see the derivation in the write-up)."""
    v, dt_, g_ = model.v, model.dt, model.gamma
    e_t = lambda t: np.exp(-t / tau)
    if time_source == 'analytic':
        de_t = lambda t: -np.exp(-t / tau) / tau
    else:  # exact backward difference of exp(-t/tau) on the grid t_k = k dt
        de_t = lambda t: (np.exp(-t / tau) - np.exp(-(t - dt_) / tau)) / dt_

    Camp = model.C @ amp_g                                   # sum_g' C_gg' amp_g'
    src, ic, bcL, bcR = [], [], [], []
    for (l, m, b) in modes:
        ang = lambda mu, l=l, m=m, b=b: b * A_lm(l, m, mu)
        mu_ang = lambda mu, l=l, m=m, b=b: mu * b * A_lm(l, m, mu)
        ang_scat = lambda mu, l=l, m=m, b=b: g_ ** l * b * A_lm(l, m, mu)    # HG: Y_lm -> gamma^l Y_lm
        az = lambda phi, m=m: Pi_m(m, phi)
        src += [
            [psi_z, ang, az, amp_g / v, de_t],                       # (1/v_g) d/dt
            [dpsi_z, mu_ang, az, amp_g, e_t],                        # mu d/dz
            [psi_z, ang, az, model.sigma_t * amp_g, e_t],            # sigma_t
            [psi_z, ang_scat, az, -Camp, e_t],                       # - scattering
        ]
        ic.append([psi_z, ang, az, amp_g])
        bcL.append([ang, az, amp_g * psi_z(0.0), e_t])               # mu > 0 (class applies the mask)
        bcR.append([ang, az, amp_g * psi_z(model.L_z), e_t])         # mu < 0

    accumulate(model, model.set_source, 'q', src)
    accumulate(model, model.set_IC, 'IC', ic)
    accumulate(model, model.set_BC_left, 'BC_L', bcL)
    accumulate(model, model.set_BC_right, 'BC_R', bcR)


def check_angular_kernel(model):
    """The discrete HG kernel must map Y_lm -> gamma^l Y_lm for the modes of the solution."""
    ang = ang_s = None
    for (l, m, b) in modes:
        t1 = b * tt.vector_kron([A_lm(l, m, model.mu), Pi_m(m, model.phi)])
        t2 = b * gamma ** l * tt.vector_kron([A_lm(l, m, model.mu), Pi_m(m, model.phi)])
        ang = t1 if ang is None else ang + t1
        ang_s = t2 if ang_s is None else ang_s + t2
    return ((model.K @ ang - ang_s).norm() / ang_s.norm()).item()


def run_case(N_z, N_t, p, time_source):
    """Build the model, set the manufactured data, solve with AMEn, return the Frobenius error."""
    with contextlib.redirect_stdout(io.StringIO() if not VERBOSE else sys.stdout):
        model = tt.SlabTransportModel(N_z=N_z, p=p, L_z=L_z, N_mu=N_mu, N_phi=N_phi, G=G, N_t=N_t, T=T,
                                      gamma=gamma, L_max=L_max, N_recon=3, E_min=E_min, E_max=E_max, eta=0.25)
        set_manufactured_data(model, time_source)
        model.construct_tt_operator()
        model.construct_tt_rhs()
        t0 = time.time()
        psi = model.run_amen(nswp=NSWP, eps=EPS_AMEN, max_full=MAX_FULL, kickrank=KICKRANK,
                             round_eps=ROUND_EPS, verbose=VERBOSE)
        t_solve = time.time() - t0
    psi_ex = manufactured_solution(model)
    err = ((psi - psi_ex).norm() / psi_ex.norm()).item()
    print(f"  N_z={N_z:4d} N_t={N_t:4d} p={p} [{time_source:8s}]  rel. Frobenius error = {err:.4e}"
          f"   max rank = {max(psi.R)}   AMEn time = {t_solve:.1f} s")
    return err


def observed_orders(errors):
    e = np.asarray(errors)
    return np.concatenate([[np.nan], np.log2(e[:-1] / e[1:])])


if __name__ == "__main__":
    # one-off consistency check of the angular kernel (no spatial/temporal discretisation involved)
    with contextlib.redirect_stdout(io.StringIO()):
        m0 = tt.SlabTransportModel(N_z=4, p=1, L_z=L_z, N_mu=N_mu, N_phi=N_phi, G=G, N_t=4, T=T,
                                   gamma=gamma, L_max=L_max, E_min=E_min, E_max=E_max, eta=0.25)
    print(f"discrete HG kernel vs gamma^l Y_lm on the solution modes: rel. error "
          f"{check_angular_kernel(m0):.2e} (must be ~1e-15)")

    # ---------------- space sweep (finest time step, exact time source) ----------------
    err_space = {}
    for p in P_LIST:
        print(f"space sweep, p = {p}, N_t = {N_t_list[-1]}")
        err_space[p] = [run_case(N_z, N_t_list[-1], p, 'discrete') for N_z in N_z_list]

    # ---------------- time sweep (finest mesh, backward Euler error) ----------------
    print(f"time sweep, p = {P_LIST[0]}, N_z = {N_z_list[-1]}")
    err_time = [run_case(N_z_list[-1], N_t, P_LIST[0], 'analytic') for N_t in N_t_list]

    # ---------------- optional full grid ----------------
    if FULL_GRID:
        print("full (N_z, N_t) grid")
        err_grid = np.array([[run_case(N_z, N_t, P_LIST[0], 'analytic') for N_t in N_t_list]
                             for N_z in N_z_list])

    # ---------------- table ----------------
    print("\n   h        " + "".join(f"  p={p}: err    order" for p in P_LIST) + "     dt        err_time   order")
    ot = observed_orders(err_time)
    os_ = {p: observed_orders(err_space[p]) for p in P_LIST}
    for i in range(len(N_z_list)):
        row = f"{h[i]:9.3e} " + "".join(f"  {err_space[p][i]:.3e} {os_[p][i]:6.2f}" for p in P_LIST)
        row += f"   {dt[i]:9.3e}  {err_time[i]:.3e} {ot[i]:6.2f}"
        print(row)

    np.savez("convergence_study.npz", h=h, dt=dt, N_z=N_z_list, N_t=N_t_list,
             **{f"err_space_p{p}": err_space[p] for p in P_LIST}, err_time=err_time)

    # ---------------- plots ----------------
    ncols = 3 if FULL_GRID else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    for p in P_LIST:
        axes[0].loglog(h, err_space[p], '-o', label=f'p={p}')
        axes[0].loglog(h, err_space[p][0] * (h / h[0]) ** (p + 1), 'k--', lw=0.8)
    axes[0].set_xlabel('h'); axes[0].set_ylabel('relative Frobenius error')
    axes[0].set_title(f'space (N_t={N_t_list[-1]}, exact time source); dashed: $h^{{p+1}}$')
    axes[0].grid(True, which='both')
    axes[0].legend()

    axes[1].loglog(dt, err_time, '-o')
    axes[1].loglog(dt, err_time[0] * (dt / dt[0]), 'k--', lw=0.8, label=r'$\Delta t$')
    axes[1].set_xlabel(r'$\Delta t$'); axes[1].set_ylabel('relative Frobenius error')
    axes[1].set_title(f'time (N_z={N_z_list[-1]}, p={P_LIST[0]}); backward Euler')
    axes[1].grid(True, which='both')
    axes[1].legend()

    if FULL_GRID:
        im = axes[2].pcolormesh(np.log10(dt), np.log10(h), np.log10(err_grid), shading='nearest')
        axes[2].set_xlabel(r'$\log_{10}\Delta t$'); axes[2].set_ylabel(r'$\log_{10} h$')
        axes[2].set_title(r'$\log_{10}$ error, all $(h,\Delta t)$')
        plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig("convergence_study.png", dpi=150)
    plt.show()











# # convergence study in space and time discretisation for the slab transport model (HG angular scattering + moderator model). Error against a smooth manufactured solution, with non-zero BC and IC. Use the class SLabTransportModel

# import numpy as np
# import matplotlib.pyplot as plt

# import tt_neutron_transport as tt

# # coefficients for the manufactured solution
# a_1 = 1.0
# a_2 = 1.5

# b_0 = 0.5
# b_1 = 1.2
# b_2 = 0.8

# tau = 0.5


# def manufactured_solution(z, mu, phi, E_g, t):
#     psi_z = a_1 * np.sin(np.pi * z) + a_2 * np.cos(np.pi * z)
#     psi_ang = b_0 * tt.real_sph(0, 0, mu, phi) + b_1 * tt.real_sph(1, 1, mu, phi) + b_2 * tt.real_sph(2, -1, mu, phi)
#     psi_t = np.exp(-t / tau)
#     return tt.concat_cores(psi_z, psi_ang, E_g, psi_t,)

# def source_term(z, mu, phi, C, gamma, E_g, t):
#     # C, gamma from class... add terms like in the latex
#     pass

# # fixed grid parameters (angle, energy)
# N_mu = 8
# N_phi = 8
# G = 2
# E_g = np.array([0.1, 1.0])

# # space
# L_z = 1.0
# N_z_list = [8, 16, 32, 64, 128, 256, 512]
# h = L_z / np.array(N_z_list)
# print("h = ", h)

# # time
# T = 1.0
# N_t_list = [8, 16, 32, 64, 128, 256, 512]
# dt = T / np.array(N_t_list)
# print("dt = ", dt)

# # iterate through space and time discretisation (use the higher resolution for the other parameter)
# for N_z in N_z_list:
#     for N_t in N_t_list:
#         # call class and run amen, and let's use Frobenius norm error for plotting convergence study
#         pass

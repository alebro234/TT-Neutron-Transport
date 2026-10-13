import tt_transport.slab as tt
import numpy as np
import matplotlib.pyplot as plt


Q0 = 0.5  # constant isotropic source

model = tt.SlabTransportModel(
    N_z = 512,
    p = 1,
    L_z = 4,
    N_mu = 3,
    N_phi = 1,
    G = 1, # transport is diagonal in energy, so only one speed is needed
    E_max = 1,
    E_min = 1, # v = 1
    N_t = 1028,
    T = 4,
    sigma_a_frac = 1.0, # pure absorber, scattering term is zero
    L_max = 0, # no scattering Legendre moments
    N_recon = 1,
)

# Initial condition psi0(z, mu) = |mu| cos(pi z / 2L) and left inflow psi_left(mu, t) = mu * bc_t(t):
# at the left corner (z = 0, t = 0) psi0(0, mu) = mu = psi_left(mu, 0) for mu > 0, so the data are compatible.
# cos(pi z / 2L) also vanishes at z = L, matching the zero right inflow at the right corner.
ic_z = lambda z: np.cos(np.pi * z / (2 * model.L_z))
ic_mu = lambda mu: np.abs(mu)
bc_mu = lambda mu: mu
bc_t = lambda t: 1.0 + 0.0*t
assert np.isclose(ic_z(0.0), bc_t(0.0)), "IC and BC must match at the left corner"

model.set_IC([
    ic_z,
    ic_mu,
    lambda phi: 1.0 + 0.0*phi,
    [1.0],
])
model.set_BC_left([
    bc_mu,
    lambda phi: 1.0 + 0.0*phi,
    [1.0],
    bc_t
])
model.set_source([
    lambda z: Q0 + 0.0*z,
    lambda mu: 1.0 + 0.0*mu,
    lambda phi: 1.0 + 0.0*phi,
    [1.0],
    lambda t: 1.0 + 0.0*t
])


def analytical_solution(z, mu, t, v, sigma_t, L, Q, psi0, psi_left):
    """
    Characteristic solution of the purely absorbing slab with initial condition psi0(z, mu),
    zero right inflow, inflow psi_left(mu, t) at z = 0 and constant isotropic source Q:
        psi = H(tau - t) psi0(z - mu v t, mu) exp(-sigma_t v t)
            + H(t - tau) psi_left(mu, t - tau) exp(-sigma_t v tau)
            + Q / sigma_t (1 - exp(-sigma_t v min(t, tau))),
    with tau the time of flight from the inflow boundary (infinite for mu = 0); the right
    inflow is zero, so the boundary term only contributes for mu > 0.
    Returns an array of shape (len(z), len(mu), len(t)).
    """
    Z, MU, T = np.meshgrid(z, mu, t, indexing='ij')
    with np.errstate(divide='ignore', invalid='ignore'):
        tau = np.where(MU > 0, Z / (MU * v),
              np.where(MU < 0, (L - Z) / (np.abs(MU) * v), np.inf))
    arrived = T > tau  # causality: boundary data reach z only after the time of flight
    tau_fin = np.where(arrived, tau, 0.0)
    z0 = np.clip(Z - MU * v * T, 0.0, L)  # foot of the characteristic at t = 0 (clipped where unused)
    ic = np.where(~arrived, psi0(z0, MU) * np.exp(-sigma_t * v * T), 0.0)
    bc = np.where(arrived & (MU > 0), psi_left(MU, T - tau_fin) * np.exp(-sigma_t * v * tau_fin), 0.0)
    src = Q / sigma_t * (1.0 - np.exp(-sigma_t * v * np.minimum(T, tau)))
    return ic + bc + src


model.run_amen(nswp=30, eps=1e-6, max_full=15, kickrank=2, verbose=True)
tt.save_tt(model.psi, "beam_source_ic_psi_tt.pt")

# numerical solution at (z_r, mu, t) for phi = 0, g = 0
psi_num = model.psi.full().numpy()[:, :, 0, 0, :]
psi_ex = analytical_solution(model.z_r, model.mu, model.t, model.v[0], model.sigma_t[0], model.L_z, Q0,
                             lambda z, mu: ic_z(z) * ic_mu(mu),
                             lambda mu, t: bc_mu(mu) * bc_t(t))

# relative L2 error in (z, mu) at each time, and over the whole space-time domain
w_zmu = model.w_z_r[:, None] * model.w_mu[None, :]
err_t = np.sqrt(np.einsum('ij,ijk->k', w_zmu, (psi_num - psi_ex)**2))
norm_t = np.sqrt(np.einsum('ij,ijk->k', w_zmu, psi_ex**2))
rel_err = np.sqrt(np.sum(err_t**2) / np.sum(norm_t**2))
print(f"relative space-angle-time L2 error = {rel_err:.3e}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
n = 2  # largest mu ordinate
for ti in range(model.N_t//10, model.N_t, model.N_t//5):
    line, = ax1.plot(model.z_r, psi_num[:, n, ti], label=f't = {model.t[ti]:.3f}')
    ax1.plot(model.z_r, psi_ex[:, n, ti], '--', color=line.get_color())
ax1.set_xlabel('z')
ax1.set_ylabel(rf'$\psi(z, \mu={model.mu[n]:.3f}, t)$')
ax1.set_title('TT-AMEn (solid) vs analytical (dashed)')
ax1.legend()

with np.errstate(divide='ignore', invalid='ignore'):
    ax2.semilogy(model.t, np.where(norm_t > 0, err_t / norm_t, np.nan))
ax2.set_xlabel('t')
ax2.set_ylabel(r'relative $L^2(z,\mu)$ error')
ax2.set_title(f'global relative error = {rel_err:.2e}')

plt.tight_layout()
plt.savefig("beam_source_ic.png", dpi=150)
plt.show()

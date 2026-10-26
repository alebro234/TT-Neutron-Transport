# Manufactured-solution test of BoxTransport (docs, sec. "Manufactured solution") with the P7, 10-group
# graphite MG data (graphite_P7_g10_mgxs.npz, copied from ~/openmc/graphite/results_P7_g10):
#   psi_g(x, y, z, Omega, t) = alpha_g f_x(x) f_y(y) f_z(z) (b_0 Y_00 + b_1 Y_11 + b_2 Y_2,-1) exp(-t/tau),
#   f_d(d) = a_d sin(pi d/L_d) + c_d cos(pi d/L_d),
# inflow on all six faces, initial condition and source Q from the exact solution (analytic time derivative).
# Solves A psi = F with AMEn and prints ||psi - psi_manuf||_F / ||psi||_F at the reconstruction points,
# all ordinates, groups and time steps.
# The last data group is empty (sigma_t = 0, no scattering in or out, inv_velocity = 0): it is pure streaming.

import time
from pathlib import Path

import numpy as np
import torch as tn

HERE = Path(__file__).resolve().parent
import tt_transport_box.box as tt

tn.set_default_dtype(tn.float64)

# ---------------------------------------------------------------------------
# cross sections: P7, 10 groups, g = 0 the fastest
# ---------------------------------------------------------------------------
xs = np.load(HERE / "graphite_P7_g10_mgxs.npz")
sigma_t = xs["sigma_t"]                         # (G,) [1/cm]
S = xs["scatter_legendre"]                      # S[l, g, g'] = sigma_s^l(g' -> g), row = receiving group
with np.errstate(divide="ignore"):
    v = 1.0 / xs["inv_velocity"]                # [cm/s]; inf in the empty group -> 1/v = 0
G = len(sigma_t)
L_max = S.shape[0] - 1

# ---------------------------------------------------------------------------
# discretisation
# ---------------------------------------------------------------------------
N_x = N_y = N_z = 100
p = 1
N_mu = N_phi = 64
N_t = 200
L_x = L_y = L_z = 10.0                          # [cm]
T = 1e-5                                        # [s]; tau v_g spans ~0.6 cm (thermal) to 2e4 cm (fast)

# ---------------------------------------------------------------------------
# manufactured solution
# ---------------------------------------------------------------------------
a = dict(x=1.0, y=0.8, z=1.2)
c = dict(x=1.5, y=1.2, z=0.7)                   # f_d(0) = c_d, f_d(L_d) = -c_d: nonzero inflow on every face
L = dict(x=L_x, y=L_y, z=L_z)
tau = T / 2
alpha = np.linspace(1.0, 0.1, G)                # group amplitudes

f = {d: (lambda s, d=d: a[d] * np.sin(np.pi * s / L[d]) + c[d] * np.cos(np.pi * s / L[d])) for d in "xyz"}
df = {d: (lambda s, d=d: np.pi / L[d] * (a[d] * np.cos(np.pi * s / L[d]) - c[d] * np.sin(np.pi * s / L[d])))
      for d in "xyz"}

# modes (l, b, A(mu), Pi(phi)) with Y_lm = A(mu) Pi(phi), Condon-Shortley phase as in scipy lpmv
eta = lambda mu: np.sqrt(1 - mu**2)
modes = [
    (0, 0.5, lambda mu: np.full_like(mu, 1 / np.sqrt(4 * np.pi)), lambda phi: np.ones_like(phi)),
    (1, 1.2, lambda mu: -np.sqrt(3 / (4 * np.pi)) * eta(mu), np.cos),
    (2, 0.8, lambda mu: -np.sqrt(15 / (4 * np.pi)) * mu * eta(mu), np.sin),
]
# direction cosines Omega_d = Omega_d^mu(mu) Omega_d^phi(phi)
Omega = dict(x=(eta, np.cos), y=(eta, np.sin), z=(lambda mu: mu, lambda phi: np.ones_like(phi)))
time_f = lambda t: np.exp(-t / tau)

# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
model = tt.BoxTransport(N_x, N_y, N_z, p, L_x, L_y, L_z, N_mu, N_phi,
                        sigma_t=sigma_t, sigma_s=[S[l].T for l in range(L_max + 1)],   # BoxTransport: row = incoming
                        v=v, E_edges=xs["group_edges_desc"], N_t=N_t, T=T)
Sigma_l = model.Sigma_l                         # = S[l], row = receiving group
V_inv = np.diag(model.V_inv)


def add(acc, term):
    return term if acc is None else acc + term


# source: 12 rank-one terms (docs, eq. (24))
q = None
X = [model._project(f[d], k) for k, d in enumerate("xyz")]
dX = [model._project(df[d], k) for k, d in enumerate("xyz")]
for l, b, A_mu, Pi_phi in modes:
    E_reac = b * ((sigma_t - V_inv / tau) * alpha - Sigma_l[l] @ alpha)
    q = add(q, tt.vector_kron(X + [A_mu(model.mu), Pi_phi(model.phi), E_reac, time_f(model.t)]))
    for k, d in enumerate("xyz"):
        space = [dX[j] if j == k else X[j] for j in range(3)]
        Om_mu, Om_phi = Omega[d]
        q = add(q, tt.vector_kron(space + [Om_mu(model.mu) * A_mu(model.mu), Om_phi(model.phi) * Pi_phi(model.phi),
                                           b * alpha, time_f(model.t)]))
model.q = q.round(1e-13)

# initial condition: 3 rank-one terms
IC = None
for l, b, A_mu, Pi_phi in modes:
    model.set_IC([f["x"], f["y"], f["z"], A_mu, Pi_phi, b * alpha])
    IC = add(IC, model.IC)
model.IC = IC

# inflow: 3 rank-one terms per face, psi on the face = (+-c_d) alpha (transverse f) Y exp(-t/tau)
for face in model.FACES:
    d = face[-1]
    sign = 1.0 if face[0] == "L" else -1.0
    transverse = [f[e] for e in "xyz" if e != d]
    BC = None
    for l, b, A_mu, Pi_phi in modes:
        model.set_BC(face, transverse + [A_mu, Pi_phi, sign * c[d] * b * alpha, time_f])
        BC = add(BC, model.BC[face])
    model.BC[face] = BC

# ---------------------------------------------------------------------------
# solve and error
# ---------------------------------------------------------------------------
t0 = time.time()
model.construct_tt_operator()
print(f"operator ranks (raw) {model.A_ranks_raw}, rounded {list(model.A.R)}, {time.time() - t0:.1f} s")
model.construct_tt_rhs(round_eps=1e-13)
model.run_amen(nswp=20, eps=1e-7, kickrank=4, round_eps=1e-9, verbose=True)
print(f"true residual ||A psi - F|| / ||F|| = {model.residual():.3e}")

psi_manuf = None
for l, b, A_mu, Pi_phi in modes:
    psi_manuf = add(psi_manuf, tt.vector_kron([f["x"](model.x_r), f["y"](model.y_r), f["z"](model.z_r),
                                               A_mu(model.mu), Pi_phi(model.phi), b * alpha, time_f(model.t)]))
diff = (model.psi - psi_manuf).round(1e-12)
err = float(diff.norm() / model.psi.norm())
print(f"psi ranks {list(model.psi.R)}")
print(f"||psi - psi_manuf||_F / ||psi||_F = {err:.4e}")
print(f"||psi - psi_manuf||_F / ||psi_manuf||_F = {float(diff.norm() / psi_manuf.norm()):.4e}")

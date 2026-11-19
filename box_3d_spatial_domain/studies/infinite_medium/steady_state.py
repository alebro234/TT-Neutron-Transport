import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch as tn
from tt_transport_box.box import BoxTransport, save_tt

tn.set_default_dtype(tn.float64)
HERE = Path(__file__).resolve().parent

# scattering ratio
c = 0.5

model = BoxTransport(
     N_x = 200,
     N_y = 200,
     N_z = 200,
     L_x = 10,
     L_y = 10,
     L_z = 10,
     p = 1,
     N_mu = 64,
     N_phi = 64,
     sigma_t = 1,
     sigma_s = c,
     steady_state = True,
     N_recon = 2
        )


# Q(x, y, z) = 1/(4\pi) * e^(-|r - r_0|^2/(2 \alpha^2)) / (2\pi\alpha^2)^(3/2), centred in the box r_0 = (L/2, L/2, L/2):
# the box is [0, L]^3, so the source must be shifted to the middle to mimic the infinite medium (5 mfp to each face)
alpha = 0.5
x0, y0, z0 = model.L_x / 2, model.L_y / 2, model.L_z / 2
model.set_source(
     [
    lambda x: 1 / (4*np.pi * (2*np.pi*alpha**2)**(3/2)) * np.exp(-(x - x0)**2/(2*alpha**2)),
    lambda y: np.exp(-(y - y0)**2/(2*alpha**2)),
    lambda z: np.exp(-(z - z0)**2/(2*alpha**2)),
    lambda mu: 1 + 0*mu,
    lambda phi: 1 + 0*phi,
    [1]
 ]
        )
# BC are zero, default

# 3D fourier transfor, source
# Q_hat = e^{-0.5 \alpha^2 k^2} / (4\pi)
# and scalar flux 
# \Phi_hat = arctan(|k|) / (|k| - c*arctan(|k|))  e^{-0.5\alpha^2*k^2}
def Phi_hat(k, c):
    return np.arctan(k) / (k - c*np.arctan(k)) * np.exp(-0.5*alpha**2 * k**2)

# shows rapid decay, good for numerical inversion
# k = np.linspace(0, 10, 100)
# for c in [0.1, 0.5, 0.9, 0.999]:
#     plt.plot(k, Phi_hat(k, c), label=f'c={c}')
# plt.xlabel('k')
# plt.ylabel(r'$\hat{\Phi}(k)$')
# plt.legend()
# plt.show()

# Phi(r) = 1/(2\pi^2 r) \int_0^\infty k \sin(kr) \hat{\Phi}(k) dk = 
#  = 1/(2\pi^2 r) \int_0^\infty k \sin(kr) arctan(k)/(k - c*arctan(k)) e^{-0.5\alpha^2 k^2} dk

# let's plot the integrand for different r, to see how it behaves
# def integrand(k, r, c):
#     return k * np.sin(k*r) * Phi_hat(k, c) / (2*np.pi**2 * r)
#
# for r in np.logspace(-4, 0, 5):
#     k = np.linspace(0, 10, 1000)
#     plt.plot(k, integrand(k, r, c), label=f'r={r}')
# plt.xlabel('k')
# plt.ylabel(r'Integrand')
# plt.legend()
# plt.title('Integrand of the inverse Fourier transform for different r')
# plt.show()
# seems to decay rapidly, so we can use numerical integration to compute Phi(r)


# The integrand carries the Gaussian e^{-alpha^2 k^2/2} (alpha = 0.5: below 1e-40 for k > 30), not an e^{-k} weight, so
# Gauss-Laguerre is a poor fit; Gauss-Legendre on [0, k_max] resolves the sin(kr) oscillations (r <= 5: < 25 periods)
# to machine precision. Phi_hat is regular at k = 0 (-> 1/(1-c)) and the Gauss nodes avoid k = 0 anyway.
r_bins = np.linspace(0.01, 5, 100)
r_mid = 0.5*(r_bins[1:] + r_bins[:-1])

def Phi_ref(r, c, k_max=30.0, n_k=600):
    xk, wk = np.polynomial.legendre.leggauss(n_k)
    k, wk = k_max/2*(xk + 1), k_max/2*wk
    return (wk * k * np.sin(np.outer(r, k)) * Phi_hat(k, c)).sum(axis=1) / (2*np.pi**2 * r)

Phi_r = Phi_ref(r_mid, c)
assert np.allclose(Phi_r, Phi_ref(r_mid, c, k_max=40.0, n_k=1200), rtol=1e-10, atol=0)
np.savez(HERE / f"Phi_ref_c{c}.npz", r_bins=r_bins, r_mid=r_mid, Phi=Phi_r, c=c, alpha=alpha)

# run amen, compute scalar flux
NSWP, KICKRANK = 40, 2                                     # sweeps of the tight solve, rank increment per sweep
TAG = f"_cont_nswp{NSWP}_kr{KICKRANK}"                      # output suffix (the 20-sweep kickrank-4 run is "_cont")
t0 = time.time()
model.construct_tt_operator()
model.construct_tt_rhs(round_eps=1e-13)
# continuation: a loose solve (eps = 1e-3, 5 sweeps) is the initial guess of the tight one
model.run_amen(nswp=5, eps=1e-3, kickrank=KICKRANK, round_eps=1e-8)
print(f"loose solve: {time.time() - t0:.1f} s, true residual {model.residual():.3e}, ranks {list(model.psi_modal.R)}")
model.run_amen(nswp=NSWP, eps=1e-6, kickrank=KICKRANK, round_eps=1e-8, x0=model.psi_modal)
print(f"total solve time {time.time() - t0:.1f} s, true residual {model.residual():.3e}")
Phi_tt = model.compute_scalar_flux()                      # TT over (x_r, y_r, z_r, g, t)
save_tt(Phi_tt, HERE / f"phi_tt_c{c}{TAG}.pt", vars=[model.x_r, model.y_r, model.z_r])

# Phi_amen has shape (400, 400, 400) since N_recon = 2 and N_x = N_y = N_z = 200. It should be radially dependent:
# bin the points by r = |x - r_0| and average, Phi[i] = mean(Phi(r_bins[i] <= r < r_bins[i+1])).
# With N_recon = 2 the Gauss weights within a cell are equal, so the plain mean is the volume average.
# Done one z-slab at a time (bincount) to keep memory at a single (400, 400, 400) array.
Phi_amen = Phi_tt.full().reshape(len(model.x_r), len(model.y_r), len(model.z_r)).numpy()
rho2 = (model.x_r[:, None] - x0)**2 + (model.y_r[None, :] - y0)**2
sums, counts = np.zeros(len(r_mid)), np.zeros(len(r_mid))
for kz, z in enumerate(model.z_r):
    idx = np.digitize(np.sqrt(rho2 + (z - z0)**2), r_bins) - 1
    inside = (idx >= 0) & (idx < len(r_mid))
    sums += np.bincount(idx[inside], weights=Phi_amen[:, :, kz][inside], minlength=len(r_mid))
    counts += np.bincount(idx[inside], minlength=len(r_mid))
Phi_bin = sums / counts

# compare: the bin average of the reference (fine Gauss-Legendre in r, volume weight r^2) vs the binned TT flux
xr, wr = np.polynomial.legendre.leggauss(8)
Phi_ref_bin = np.array([
    np.sum(wr * rr**2 * Phi_ref(rr, c)) / np.sum(wr * rr**2)
    for rr in (0.5*(r_bins[:-1, None] + r_bins[1:, None]) + 0.5*(r_bins[1:, None] - r_bins[:-1, None])*xr[None, :])])
dr = np.diff(r_bins)
rel_L2 = np.sqrt(np.sum((Phi_bin - Phi_ref_bin)**2 * r_mid**2 * dr) / np.sum(Phi_ref_bin**2 * r_mid**2 * dr))
for r_cut in (2.0, 4.0):
    m = r_mid < r_cut
    e = np.sqrt(np.sum((Phi_bin[m] - Phi_ref_bin[m])**2 * r_mid[m]**2 * dr[m]) / np.sum(Phi_ref_bin[m]**2 * r_mid[m]**2 * dr[m]))
    print(f"relative L2 error (r^2 dr), r < {r_cut}: {e:.3e}")
print(f"relative L2 error (r^2 dr), r < {r_bins[-1]}: {rel_L2:.3e}")
np.savez(HERE / f"Phi_amen_c{c}{TAG}.npz", r_mid=r_mid, Phi_bin=Phi_bin, Phi_ref_bin=Phi_ref_bin, counts=counts)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.semilogy(r_mid, Phi_ref_bin, "k-", label="reference (Fourier inversion)")
ax1.semilogy(r_mid, Phi_bin, "r--", label="TT AMEn, bin average")
ax1.set_xlabel("r [mfp]"); ax1.set_ylabel(r"$\Phi(r)$"); ax1.legend(); ax1.set_title(f"c = {c}, alpha = {alpha}")
ax2.semilogy(r_mid, np.abs(Phi_bin - Phi_ref_bin) / Phi_ref_bin)
ax2.set_xlabel("r [mfp]"); ax2.set_ylabel("relative error")
fig.tight_layout()
fig.savefig(HERE / f"steady_state_c{c}{TAG}.png", dpi=150)

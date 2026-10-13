import torch as tn 
import torchtt as tntt
import numpy as np
import matplotlib.pyplot as plt
from tt_transport.slab import SlabTransportModel, integrate_dim, extract_tensor
from tt_transport.integral_transforms import double_inversion




model = SlabTransportModel(
    N_z = 256,
    p = 1,
    L_z = 10,
    N_mu = 128,
    N_phi = 128,
    G = 1, # one speed
    E_max = 1,
    E_min = 1, # v = 1
    N_t = 512,
    T = 1,
    sigma_a_frac = 0.5,
    gamma = 0.0, # isotropic scattering
    L_max = 0,
    N_recon = 1,
)


# scalar flux in Fourier-Laplace space
alpha = 0.5    # source: exp(-alpha x^2)
lambd = 0.1    # source: exp(-lambd t)
c = model.C[0,0]        # scattering ratio
print(f"c = {c}")

def Phi_hat(k, s):
    T = 4*np.pi / k * np.arctan(k / (s + 1))          # transport kernel: (s + 1), the 1 is sigma_t
    q_0 = np.sqrt(np.pi/alpha) * np.exp(-k**2 / (4*alpha)) / (s + lambd)
    return q_0 * T / (1 - c * T / (4*np.pi))   # isotropic kernel K_s = 1/(4 pi) -> scattering source c*phi/(4 pi)

model.set_source([
    lambda z: np.exp(-alpha*(z - model.L_z/2)**2),  # Gaussian in z
    lambda mu: 1.0 + 0.0*mu,  # isotropic
    lambda phi: 1.0 + 0.0*phi, # isotropic
    [1.0], # one speed
    lambda t: np.exp(-lambd*t)
])
# zero IC
model.set_IC(0)
model.set_BC_left(0)
model.set_BC_right(0)


model.run_amen(nswp=20, eps=1e-6, max_full=15, verbose=True)
PHI_TT = model.compute_scalar_flux() 
print("PHI_TT is a")
print(PHI_TT)


s_right_bound = c - 1.0   # pole of 1/(1 - c T/(4 pi)) at k -> 0

gamma_inv = max(0.5, s_right_bound + 1.5)

print("Rightmost-pole bound:", s_right_bound)
print("Laplace contour:", gamma_inv)


t_idx = np.arange(model.N_t//10, model.N_t, model.N_t//4)   # indices of the plotted times in model.t
t_plot = model.t[t_idx]
# double numerical Laplace-Fourier inversion to get phi(z, t)
print("Computing reference solution by double inversion...")
z_input = model.z_r - model.L_z/2 # shift to center of slab (real function in Fourier space, easier inversion)
Phi_ref = double_inversion(
    Phi_hat,
    z_input,
    t_plot,
    gamma=gamma_inv,
    assume_real=True
)


np.savez('Phi_ref.npz', Phi_ref=Phi_ref, z_input=z_input, t_plot=t_plot)

def L2_error(f, g, z):
    """Compute the L2 error between two functions f and g defined on the same grid z."""
    return np.sqrt(np.trapezoid((f - g)**2, z) / np.trapezoid(f**2, z))

for ti in range(len(t_plot)):
    print(f"t = {t_plot[ti]:.2f}")
    # plt.subplot(2, 2, ti + 1)
    PHI_xt = extract_tensor(PHI_TT, [-1, [0], [t_idx[ti]]])  # (z, g=0, t)
    color = f"C{ti}"
    plt.plot(model.z_r, PHI_xt.squeeze(), color=color, label=f"TT t={t_plot[ti]:.2f}")
    plt.plot(model.z_r, Phi_ref[:, ti], 'o', color=color, label=f"Ref t={t_plot[ti]:.2f}")
    # L2 error
    err = L2_error(PHI_xt.squeeze(), Phi_ref[:, ti], model.z_r)
    print(f"L2 error: {err:.3e}")


plt.xlabel("z")
plt.ylabel("Scalar flux")
plt.legend()
plt.savefig("decaying_gaussian_inf_medium.png", dpi=300)
plt.show()

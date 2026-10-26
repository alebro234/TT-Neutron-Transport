# Decaying Gaussian source in an (effectively) infinite medium: TT solve + reference by double
# Laplace-Fourier inversion. Saves the TT scalar flux (phi_tt.pt) and the reference (Phi_ref.npz);
# plotting is done in scalar_flux_plot.py
import numpy as np
from tt_transport.slab import SlabTransportModel, save_tt
from tt_transport.integral_transforms import double_inversion




model = SlabTransportModel(
    N_z = 512,
    p = 1,
    L_z = 10,
    N_mu = 256,
    N_phi = 256,
    G = 1, # one speed
    E_max = 1,
    E_min = 1, # v = 1
    N_t = 1024,
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


model.run_amen(nswp=20, eps=1e-6, max_full=15000, verbose=True)
print("Psi is a")
print(model.psi)
PHI_TT = model.compute_scalar_flux() 
print("PHI_TT is a")
print(PHI_TT)

save_tt(PHI_TT, "phi_tt.pt", vars=[model.z_r, model.t])   # (z, g, t)

# load ref if commented out
# s_right_bound = c - 1.0   # pole of 1/(1 - c T/(4 pi)) at k -> 0
#
# gamma_inv = max(0.5, s_right_bound + 1.5)
#
# print("Rightmost-pole bound:", s_right_bound)
# print("Laplace contour:", gamma_inv)
#
#
# t_idx = np.arange(model.N_t//10, model.N_t, model.N_t//4)   # indices of the plotted times in model.t
# t_plot = model.t[t_idx]
# # double numerical Laplace-Fourier inversion to get phi(z, t)
# print("Computing reference solution by double inversion...")
# z_input = model.z_r - model.L_z/2 # shift to center of slab (real function in Fourier space, easier inversion)
# Phi_ref = double_inversion(
#     Phi_hat,
#     z_input,
#     t_plot,
#     gamma=gamma_inv,
#     assume_real=True
# )
#
#
# np.savez('Phi_ref.npz', Phi_ref=Phi_ref, z_input=z_input, t_plot=t_plot, t_idx=t_idx)

# Plot the TT scalar flux against the double-inversion reference.
# Run decaying_gaussian_inf_medium.py first (it writes phi_tt.pt and Phi_ref.npz)
import numpy as np
import matplotlib.pyplot as plt
from tt_transport.slab import load_tt, extract_tensor

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "legend.fontsize": 13,
})

PHI_TT, (z_r, t) = load_tt("phi_tt.pt")   # (z, g, t)
print(PHI_TT)
ref = np.load("Phi_ref.npz")
Phi_ref, t_plot, t_idx = ref["Phi_ref"], ref["t_plot"], ref["t_idx"]


def L2_error(f, g, z):
    """Compute the L2 error between two functions f and g defined on the same grid z."""
    return np.sqrt(np.trapezoid((f - g)**2, z) / np.trapezoid(f**2, z))


plt.figure(figsize=(10, 6))
for ti in range(len(t_plot)):
    PHI_xt = extract_tensor(PHI_TT, [-1, [0], [t_idx[ti]]]).squeeze()   # (z, g=0, t)
    color = f"C{ti}"
    plt.plot(z_r, PHI_xt, color=color, label=f"TT t={t_plot[ti]:.2f}")
    plt.plot(z_r, Phi_ref[:, ti], 'o', color=color, label=f"Ref t={t_plot[ti]:.2f}")
    print(f"t = {t_plot[ti]:.2f}   L2 error: {L2_error(PHI_xt, Phi_ref[:, ti], z_r):.3e}")

plt.xlabel("z")
plt.ylabel("Scalar flux")
plt.legend()
plt.tight_layout()
plt.savefig("decaying_gaussian_inf_medium.png", dpi=300)
plt.show()

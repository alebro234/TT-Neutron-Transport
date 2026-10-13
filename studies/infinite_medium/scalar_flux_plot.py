import numpy as np
import matplotlib.pyplot as plt
from tt_transport.integral_transforms import double_inversion

c = 0.5        # scattering ratio sigma_s / sigma_t
alpha = 0.5    # source: exp(-alpha x^2)
lambd = 0.1    # source: exp(-lambd t)

x = np.linspace(-5, 5, 50)
t = np.linspace(0.1, 1, 5)    # for t < ~0.05 increase Nq_t below (about 0.4 / t)


def F(k, s):
    T = 4*np.pi / k * np.arctan(k / (s + 1))          # transport kernel: (s + 1), the 1 is sigma_t
    q_0 = np.sqrt(np.pi/alpha) * np.exp(-k**2 / (4*alpha)) / (s + lambd)
    return q_0 * T / (1 - c * T / (4*np.pi))          # scattering source is (c / 4 pi) * phi


# rightmost singularity in s is s = -lambd (the others are at s <= c - 1), so gamma = 0.5 is valid;
# F is real for real k, s, hence assume_real=True (twice as fast)
f = double_inversion(F, x, t, gamma=0.5, assume_real=True)
print(f.shape)

for i in range(len(t)):
    plt.plot(x, f[:, i], label=f't={t[i]:.3f}')
plt.xlabel('x')
plt.ylabel('f(x, t)')
plt.title('Scalar Flux vs x for Different Times')
plt.legend()
plt.grid()
plt.savefig('scalar_flux_plot.png')   
plt.show()
"""
Rank-1 Kronecker inverse of the BoxTransport operator as a candidate preconditioner.

A (TT-matrix, cores (x, y, z, mu, phi, g, t)) has ranks > 1, so its cores are not matrices. Its rank-1 TT rounding
A_1 = A_1^(1) (x) ... (x) A_1^(d) is a Kronecker product of n_i x n_i matrices, whose inverse is exact and cheap:
    B = (A_1^(1))^-1 (x) ... (x) (A_1^(d))^-1,   B = A_1^-1.
Reports ||A - A_1||_F / ||A||_F, the conditioning of every core, and the deviation of A B from the TT identity,
||A B - I||_F / ||I||_F (with the check ||A_1 B - I||_F / ||I||_F ~ machine precision).
Grid: the infinite-medium study (../infinite_medium/steady_state.py).
"""
import time

import numpy as np
import torch as tn
import torchtt as tntt
from tt_transport_box.box import BoxTransport

tn.set_default_dtype(tn.float64)
tn.set_num_threads(2)                     # leave the CPU to the running AMEn solve

c = 0.5
model = BoxTransport(N_x=200, N_y=200, N_z=200, L_x=10, L_y=10, L_z=10, p=1, N_mu=64, N_phi=64,
                     sigma_t=1, sigma_s=c, steady_state=True, N_recon=2)
t0 = time.time()
model.construct_tt_operator()
A = model.A
print(f"A: ranks {list(A.R)}, mode sizes {list(A.N)}, built in {time.time() - t0:.1f} s")

# rank-1 Kronecker approximation and its core-wise inverse
A_1 = A.round(1e-14, rmax=1)
nA = float(A.norm())
print(f"||A - A_1||_F / ||A||_F = {float((A - A_1).norm()) / nA:.3e}")
names = ["x", "y", "z", "mu", "phi", "g", "t"]
B_cores = []
for name, core in zip(names, A_1.cores):
    M = core[0, :, :, 0].numpy()
    s = np.linalg.svd(M, compute_uv=False)
    print(f"  core {name:3s}: {M.shape[0]:4d} x {M.shape[1]:4d}, cond = {s[0] / s[-1]:.3e}")
    B_cores.append(tn.tensor(np.linalg.inv(M))[None, :, :, None])
B = tntt.TT(B_cores)

I = tntt.eye(list(A.M))
nI = float(I.norm())                      # sqrt(prod n_i)
t0 = time.time()
AB = A @ B
print(f"A B: ranks {list(AB.R)} ({time.time() - t0:.1f} s)")
print(f"||A_1 B - I||_F / ||I||_F = {float((A_1 @ B - I).norm()) / nI:.3e}   (check: B = A_1^-1)")
print(f"||A B - I||_F / ||I||_F   = {float((AB - I).norm()) / nI:.3e}")
print(f"||A B||_F / ||I||_F       = {float(AB.norm()) / nI:.3e}")

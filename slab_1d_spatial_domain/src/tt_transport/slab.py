import torch as tn
import torchtt as tntt
import numpy as np
from scipy.special import legendre, eval_legendre
from numpy.polynomial.legendre import leggauss
from math import sqrt, factorial
from scipy.special import lpmv
import time
import warnings
from numpy.polynomial.legendre import leggauss


def matrix_kron(cores_list):
    """Create a rank-1 TT-matrix from a list of 2D arrays (one core each)."""
    return tntt.TT([tn.Tensor(tn.tensor(np.expand_dims(c,[0,3]))) for c in cores_list],[1]*len(cores_list))
        
def vector_kron(cores_list):
    """Create a rank-1 TT-tensor from a list of 1D arrays (one core each)."""
    return tntt.TT([tn.Tensor(tn.tensor(np.expand_dims(c,[0,2]))) for c in cores_list],[1]*len(cores_list))

def concat_cores(blocks):
    """Kronecker product of TT-matrix blocks: arrays (one core) and tntt.TT objects (their cores)."""
    cores = []
    for b in blocks:
        if isinstance(b, tntt.TT):
            cores += list(b.cores)
        else:
            cores.append(tn.tensor(np.asarray(b, dtype=np.float64)[None, :, :, None]))
    return tntt.TT(cores)


import itertools

def extract_tensor(tt, idxs):
    """
    Extract a tensor from a TT object at specified indices.

    tt  : torchtt.TT
    idxs: list of indices 
    if idxs[k] == -1, then all indices in that dimension are extracted.

    returns: numpy array
    """
    d = len(tt.cores)

    # build full index lists per dimension
    full_idxs = []
    shape = []

    for k in range(d):
        if idxs[k] == -1:
            nk = tt.cores[k].shape[1]
            inds = list(range(nk))
        else:
            inds = list(idxs[k])

        full_idxs.append(inds)
        shape.append(len(inds))

    # allocate result
    result = np.zeros(shape)

    # iterate over all multi-indices
    for out_multi_idx in itertools.product(*[range(s) for s in shape]):
        
        # map to actual TT indices
        tt_multi_idx = tuple(full_idxs[k][i] for k, i in enumerate(out_multi_idx))

        # evaluate TT at that index
        val = None
        for k, i_k in enumerate(tt_multi_idx):
            core = tt.cores[k]
            slice_k = core[:, i_k, :]

            if val is None:
                val = slice_k
            else:
                val = val @ slice_k

        result[out_multi_idx] = val.squeeze().item()

    return result

def extract_elements(tt, idxs):
    # tt   : torchtt.TT tensor
    # idxs : list of tuples, each tuple is a multi-index (i1,...,id)
    #
    # returns: numpy array of scalars

    results = []

    for multi_idx in idxs:
        val = None

        for k, i_k in enumerate(multi_idx):
            core = tt.cores[k]              # (r_{k-1}, n_k, r_k)
            slice_k = core[:, i_k, :]       # (r_{k-1}, r_k)

            if val is None:
                val = slice_k              # (1, r1)
            else:
                val = val @ slice_k        # contract ranks

        results.append(val.squeeze().item())

    return np.array(results)

def integrate_dim(tt, dim, weights):
    """
    Integrate TT tensor along dimension `dim` using weights.

    tt      : torchtt.TT
    dim     : int (dimension to integrate over)
    weights : array of shape (n_dim,)

    returns : new TT with one fewer dimension
    """
    cores = tt.cores.copy()
    w = tn.tensor(weights, dtype=cores[0].dtype)

    # 1. contract core with weights
    core = cores[dim]                          # (r_{k-1}, n_k, r_k)
    contracted = tn.einsum('aib,i->ab', core, w)  # (r_{k-1}, r_k)

    # 2. absorb into left neighbor
    if dim > 0:
        left = cores[dim - 1]                  # (r_{k-2}, n_{k-1}, r_{k-1})
        new_left = tn.einsum('aib,bc->aic', left, contracted)
        cores[dim - 1] = new_left
        del cores[dim]
    else:
        # absorb into right neighbor
        right = cores[dim + 1]                 # (r_k, n_{k+1}, r_{k+1})
        new_right = tn.einsum('ab,bic->aic', contracted, right)
        cores[dim + 1] = new_right
        del cores[dim]

    return tntt.TT(cores)

def extract_angle_slice_2d(tt, k):
    """
    Extract u(x,y,theta_k) from a 3D TT (x,y,theta).
    
    tt: torchtt.TT or similar
    k: angular index
    """
    cores = tt.cores  # list of 3 cores

    Gx, Gy, Gth = cores

    # Fix theta index -> slice middle dimension
    Gth_k = Gth[:, k, :]   # shape (r2, r3)

    # Contract TT: Gx - Gy - Gth_k
    # Gx: (1, nx, r1)
    # Gy: (r1, ny, r2)
    # Gth_k: (r2, 1)

    tmp = np.einsum('a i b, b j c -> a i j c', Gx, Gy)   # (1, nx, ny, r2)
    u_xy = np.einsum('a i j b, b c -> i j', tmp, Gth_k)  # (nx, ny)


def real_sph(l, m, mu, phi):
    """
    Real spherical harmonics Y_l^m (physics convention)

    Parameters:
        l : int (degree)
        m : int (order, can be negative)
        mu : cos(theta) where theta is the polar angle [0, pi]
        phi : azimuthal angle [0, 2*pi]

    Returns:
        tt tensor representing the real spherical harmonic Y_l^m evaluated at (theta, phi)
    """
    m_abs = np.abs(m)

    # Normalization factor
    N = sqrt((2*l + 1)/(4*np.pi) * factorial(l - m_abs)/factorial(l + m_abs))

    # Associated Legendre polynomial
    P = lpmv(m_abs, l, mu)

    if m > 0:
        return sqrt(2) * N * vector_kron([P,  np.cos(m_abs * phi)])
    elif m < 0:
        return sqrt(2) * N * vector_kron([P,  np.sin(m_abs * phi)])    
    else:
        return N * vector_kron([P, np.ones_like(phi)])

def heneey_greenstein(g, Lmax, n_pol, n_az, max_rank=25, round_prec=1e-6):

    """
    Construct the Heneey-Greenstein scattering kernel in TTM format (real sph expansion).

    Parameters:
        g : anisotropy factor (0 for isotropic, close to 1 for highly forward scattering)
        Lmax : maximum spherical harmonic degree to include in the expansion
        n_pol : number of quadrature points for the polar angle (mu = cos(theta))
        n_az : number of quadrature points for the azimuthal angle (phi)
        max_rank : maximum TT rank before rounding
        round_prec : precision for TT rounding
    Returns:
        K : 2 dimensional TTM of size (n_pol, n_pol) x (n_az, n_az)
    """

    # Gauss legendre points and weights for the polar angle (mu = cos(theta)) 
    mu, w_mu = leggauss(n_pol)
    # Fourier points and weights for the azimuthal angle (phi)
    phi = np.linspace(0, 2*np.pi, n_az, endpoint=False)
    w_phi = (2*np.pi) / n_az * np.ones(n_az)

    # Construct the Heneey-Greenstein kernel in TTM format
    K = tntt.zeros([(n_pol, n_pol), (n_az, n_az)])

    for l in range(Lmax + 1):
        for m in range(-l, l + 1):
            # Normalization factor
            N = sqrt((2*l + 1)/(4*np.pi) * factorial(l - abs(m))/factorial(l + abs(m)))
            # Polar matrix for the associated Legendre polynomial
            P = lpmv(np.abs(m), l, mu).reshape(n_pol, 1)
            PP = np.kron(P, P.T * w_mu)
            # Azimuthal matrix
            if m > 0:
                A = np.cos(m * phi).reshape(n_az, 1)
                AA = np.kron(A, A.T * w_phi)
                N = N * sqrt(2)
            elif m < 0:
                A = np.sin(abs(m) * phi).reshape(n_az, 1)
                AA = np.kron(A, A.T * w_phi)
                N = N * sqrt(2)
            else:
                AA = np.ones((n_az, n_az)) * w_phi

            # sum mode contribution to the kernel
            eigenvalue = g**l
            K = K +  eigenvalue * N**2 * matrix_kron([PP, AA])
        print(f"Added modes (l={l}), current TT rank: {max(K.R)}")
        if max(K.R) > max_rank:
            print(f"TT rank exceeded max_rank={max_rank}, rounding...")
            K = K.round(round_prec)
            print(f"After rounding, TT rank: {max(K.R)}")

    return K.round(round_prec)

def save_tt(tt, path, vars=None):
    """
    Save TT tensor and associated independent variables.

    Parameters
    ----------
    tt : TT object
        Tensor train object with attribute `.cores`
    path : str
        Output .pt file
    vars : list
        List of arrays/tensors for independent variables
    """

    if vars is None:
        vars = []

    # processed_vars = []

    # for v in vars:
    #     if isinstance(v, np.ndarray):
    #         processed_vars.append(tn.from_numpy(v))
    #     elif tn.is_tensor(v):
    #         processed_vars.append(v.detach().cpu())
    #     else:
    #         processed_vars.append(v)

    data = {
        "cores": [c.detach().cpu() for c in tt.cores],
        "vars": vars,
    }

    tn.save(data, path)


def load_tt(path, device="cpu"):
    data = tn.load(path, map_location=device, weights_only=False)

    cores = [c.to(device) for c in data["cores"]]
    vars = data["vars"]

    return tntt.TT(cores), vars


# modal Legendre coefficients -> point values at the Gauss points xr of every cell
def legendre_to_physical_space(Gc, p, Ns, xi):
    r0, n, r1 = Gc.shape
    P = tn.tensor(np.stack([eval_legendre(a, xi) for a in range(p + 1)], axis=1))   # (Nr, p+1)
    Gc = Gc.reshape(r0, Ns, p + 1, r1)
    return tn.einsum('iNaj,ra->iNrj', Gc, P).reshape(r0, Ns * len(xi), r1)



# modal Legendre coefficients -> point values at the Gauss points xi of every cell
def legendre_to_physical_space(Gc, p, Ns, xi):
    r0, n, r1 = Gc.shape
    P = tn.tensor(np.stack([eval_legendre(a, xi) for a in range(p + 1)], axis=1), dtype=Gc.dtype)  # (Nr, p+1)
    Gc = Gc.reshape(r0, Ns, p + 1, r1)
    return tn.einsum('iNaj,ra->iNrj', Gc, P).reshape(r0, Ns * len(xi), r1)


class SlabTransportModel:
    """
    Multigroup, 3D-angle slab transport with Henyey-Greenstein angular phase function, DG in z, S_N in (mu, phi),
    backward Euler, all-at-once in TT format.  Tensor ordering: (z-dof, mu, phi, group, time).
    """

    def __init__(self, N_z, p, L_z, N_mu, N_phi, G, N_t, T, gamma=0.5, L_max=20, N_recon=3,
                 E_min=1e-2, E_max=1e6, v0=1.0, sigma_a_frac=0.5, eta=0.0,
                 K_max_rank=40, K_round_prec=1e-8):
        self.N_z = N_z           # number of spatial cells
        self.p = p               # polynomial degree
        self.L_z = L_z           # length of the slab
        self.N_mu = N_mu         # number of quadrature points for polar angle
        self.N_phi = N_phi       # number of quadrature points for azimuthal angle
        self.G = G               # number of energy groups
        self.N_t = N_t           # number of time steps
        self.T = T               # final time
        self.gamma = gamma       # HG anisotropy factor
        self.L_max = L_max       # maximum spherical-harmonic degree of the kernel
        self.N_recon = N_recon   # number of reconstruction points per cell
        if N_mu < L_max + 3 or N_phi < L_max + 3:
            warnings.warn("N_mu and N_phi should be >= L_max + 3 for exact angular quadrature of the kernel")

        # ---------------- grids ----------------
        self.ndof_z = N_z * (p + 1)
        self.h = L_z / N_z
        self.z_mid = (np.arange(N_z) + 0.5) * self.h               # cell midpoints
        self.z = self.z_mid
        self.mu, self.w_mu = leggauss(N_mu)                        # even N_mu -> no mu = 0 ordinate
        self.mu_plus = np.where(self.mu > 0, self.mu, 0.0)
        self.mu_minus = np.where(self.mu < 0, self.mu, 0.0)
        self.phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
        self.w_phi = (2 * np.pi) / N_phi * np.ones(N_phi)
        self.dt = T / N_t
        self.t = self.dt * np.arange(1, N_t + 1)                   # unknowns t_k = k dt, k = 1..N_t

        self.xi, self.w_xi = leggauss(N_recon)                     # local to cell
        self.z_r = (self.z_mid[:, None] + self.h / 2 * self.xi[None, :]).ravel()   # reconstruction points
        self.w_z_r = np.tile(self.w_xi, N_z) * self.h / 2          # quadrature weights on z_r

        self.A = None   # TT operator
        self.F = None   # TT right-hand side
        self.q = None   # source term (TT)
        self.IC = None  # initial condition term (TT)
        self.BC_L = None
        self.BC_R = None
        self.psi_modal = None   # solution, Legendre modal coefficients
        self.psi = None         # solution at the reconstruction points z_r

        # ---------------- DG matrices ----------------
        self.S = np.zeros((p + 1, p + 1))
        for a in range(p + 1):
            for b in range(p + 1):
                if a > b and (a + b) % 2 == 1:
                    self.S[a, b] = 2.0

        self.B_p_R = np.ones((p + 1, p + 1))
        self.B_p_L = np.zeros((p + 1, p + 1))
        for a in range(p + 1):
            self.B_p_L[a, :] = (-1)**a
        self.B_m_R = np.zeros((p + 1, p + 1))
        for b in range(p + 1):
            self.B_m_R[:, b] = (-1)**b
        self.B_m_L = np.zeros((p + 1, p + 1))
        for a in range(p + 1):
            for b in range(p + 1):
                self.B_m_L[a, b] = (-1)**(a + b)

        M_local = np.zeros((p + 1, p + 1))
        for a in range(p + 1):
            M_local[a, a] = self.h / (2 * a + 1)
        self.M = np.zeros((self.ndof_z, self.ndof_z))
        for i in range(N_z):
            I = slice(i * (p + 1), (i + 1) * (p + 1))
            self.M[I, I] = M_local

        # ---------------- energy: group velocities and moderator scattering matrix ----------------
        E_edges = np.logspace(np.log10(E_max), np.log10(E_min), G + 1)   # decreasing energy
        self.E = np.sqrt(E_edges[:-1] * E_edges[1:])                    # group-centered energies
        self.v = v0 * np.sqrt(self.E / E_max)                           # group velocities
        self.sigma_t = np.ones(G)                                       # total cross section normalised to 1

        sigma_a0 = sigma_a_frac * self.v[-1] / v0       # absorption probability sigma_a_frac in the lowest group
        self.C = np.zeros((G, G))                       # C[g, g'] : g' -> g
        for gp in range(G):
            abs_prob = sigma_a0 * v0 / self.v[gp]
            ctot = 1 - abs_prob
            self.C[gp, gp] = (1 - eta) * ctot           # in-group scattering
            if gp < G - 1:
                self.C[gp + 1, gp] = eta * ctot         # down-scatter (the last group has none)

        # ---------------- angular HG kernel: 2-core TTM over (mu, phi) ----------------
        self.K = heneey_greenstein(gamma, L_max, N_mu, N_phi,
                                   max_rank=K_max_rank, round_prec=K_round_prec)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _project_z(self, f_z, n_quad=50):
        """int_{K_i} f(z) P_a dz for every (cell i, mode a), as a vector of length ndof_z."""
        xi, w_xi = leggauss(n_quad)
        out = np.zeros(self.ndof_z)
        for i in range(self.N_z):
            for a in range(self.p + 1):
                out[i * (self.p + 1) + a] = np.sum(
                    w_xi * legendre(a)(xi) * f_z(self.z_mid[i] + self.h / 2 * xi)
                ) * self.h / 2
        return out

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    def set_source(self, core_funcs):
        """core_funcs = [q_z, q_mu, q_phi, q_E, q_t]; q_E is a vector of length G, the others functions."""
        if core_funcs == 0:
            self.q = tntt.zeros([self.ndof_z, self.N_mu, self.N_phi, self.G, self.N_t])
            return
        q_z, q_mu, q_phi, q_E, q_t = core_funcs
        Q_z = self._project_z(q_z)
        self.q = vector_kron([Q_z, q_mu(self.mu), q_phi(self.phi), np.asarray(q_E, float), q_t(self.t)])

    def set_IC(self, core_funcs):
        """core_funcs = [ic_z, ic_mu, ic_phi, ic_E]; enters at the first time step only."""
        if core_funcs == 0:
            self.IC = tntt.zeros([self.ndof_z, self.N_mu, self.N_phi, self.G, self.N_t])
            return
        ic_z, ic_mu, ic_phi, ic_E = core_funcs
        IC_z = self._project_z(ic_z)
        delta_t_0 = np.zeros(self.N_t)
        delta_t_0[0] = 1.0
        self.IC = 1 / self.dt * vector_kron([IC_z, ic_mu(self.mu), ic_phi(self.phi),
                                             np.asarray(ic_E, float) / self.v, delta_t_0])

    def set_BC_left(self, core_funcs):
        """core_funcs = [bc_mu, bc_phi, bc_E, bc_t]; inflow at z = 0, only mu > 0 (weighted by mu)."""
        if core_funcs == 0:
            self.BC_L = tntt.zeros([self.ndof_z, self.N_mu, self.N_phi, self.G, self.N_t])
            return
        bc_mu, bc_phi, bc_E, bc_t = core_funcs
        BC_z = np.zeros(self.ndof_z)
        for a in range(self.p + 1):
            BC_z[a] = (-1)**a
        self.BC_L = vector_kron([BC_z, self.mu_plus * bc_mu(self.mu), bc_phi(self.phi),
                                 np.asarray(bc_E, float), bc_t(self.t)])

    def set_BC_right(self, core_funcs):
        """core_funcs = [bc_mu, bc_phi, bc_E, bc_t]; inflow at z = L, only mu < 0 (weighted by -mu)."""
        if core_funcs == 0:
            self.BC_R = tntt.zeros([self.ndof_z, self.N_mu, self.N_phi, self.G, self.N_t])
            return
        bc_mu, bc_phi, bc_E, bc_t = core_funcs
        BC_z = np.zeros(self.ndof_z)
        for a in range(self.p + 1):
            BC_z[-(a + 1)] = 1.0
        self.BC_R = vector_kron([BC_z, -self.mu_minus * bc_mu(self.mu), bc_phi(self.phi),
                                 np.asarray(bc_E, float), bc_t(self.t)])

    # ------------------------------------------------------------------
    # operator and right-hand side
    # ------------------------------------------------------------------
    def construct_tt_operator(self):
        p, N_z, ndof = self.p, self.N_z, self.ndof_z
        In, Iphi, Ig, It = np.eye(self.N_mu), np.eye(self.N_phi), np.eye(self.G), np.eye(self.N_t)

        # time derivative (backward Euler); energy core 1/v_g
        A = matrix_kron([self.M, In, Iphi, np.diag(1 / self.v),
                         1 / self.dt * np.diag(np.ones(self.N_t))
                         - 1 / self.dt * np.diag(np.ones(self.N_t - 1), k=-1)])
        # mass term; energy core sigma_t,g
        A += matrix_kron([self.M, In, Iphi, np.diag(self.sigma_t), It])

        # streaming term
        space = np.zeros((ndof, ndof))
        for i in range(N_z):
            I = slice(i * (p + 1), (i + 1) * (p + 1))
            space[I, I] += -self.S
        A += matrix_kron([space, np.diag(self.mu), Iphi, Ig, It])

        # face terms: positive mu
        space = np.zeros((ndof, ndof))
        for i in range(N_z):
            I = slice(i * (p + 1), (i + 1) * (p + 1))
            space[I, I] += self.B_p_R
            if i > 0:
                Im = slice((i - 1) * (p + 1), i * (p + 1))
                space[I, Im] -= self.B_p_L
        A += matrix_kron([space, np.diag(self.mu_plus), Iphi, Ig, It])

        # face terms: negative mu
        space = np.zeros((ndof, ndof))
        for i in range(N_z):
            I = slice(i * (p + 1), (i + 1) * (p + 1))
            space[I, I] -= self.B_m_L
            if i < N_z - 1:
                Ip = slice((i + 1) * (p + 1), (i + 2) * (p + 1))
                space[I, Ip] += self.B_m_R
        A += matrix_kron([space, np.diag(self.mu_minus), Iphi, Ig, It])

        # scattering term:  M (x) K_mu (x) K_phi (x) c_{gg'} (x) I_t
        A -= concat_cores([self.M, self.K, self.C, It])

        self.A = A
        print("A is a")
        print(self.A)

    def construct_tt_rhs(self, round_eps=None):
        terms = [x for x in (self.q, self.IC, self.BC_L, self.BC_R) if x is not None]
        if not terms:
            raise ValueError("set at least one of source, IC, BC_L, BC_R before building the right-hand side")
        F = terms[0]
        for x in terms[1:]:
            F = F + x
        self.F = F.round(round_eps) if round_eps else F
        print("F is a")
        print(self.F)

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------
    def run_amen(self, nswp=40, eps=1e-6, max_full=15000, kickrank=4, round_eps=1e-8, verbose=True):
        """AMEn solve; ranks grow by `kickrank` per sweep, so a rank equal to 1 + kickrank*nswp = not converged."""
        if self.A is None:
            self.construct_tt_operator()
        if self.F is None:
            self.construct_tt_rhs()

        time_start = time.time()
        psi = tntt.solvers.amen_solve(self.A, self.F, eps=eps, max_full=max_full,
                                      kickrank=kickrank, nswp=nswp, verbose=verbose)
        print(f"AMEn solver completed in {time.time() - time_start:.2f} seconds.")

        self.psi_modal = psi.round(round_eps) if round_eps else psi
        cores = self.psi_modal.cores
        # Legendre reconstruction at the Gauss points z_r of every cell
        self.psi = tntt.TT([legendre_to_physical_space(cores[0], self.p, self.N_z, self.xi)] + list(cores[1:]))
        print("Psi is a")
        print(self.psi)
        return self.psi

    def compute_scalar_flux(self):
        if self.psi_modal is None:
            raise ValueError("Run the solver first (run_amen stores psi_modal).")
        phi = integrate_dim(self.psi_modal, 2, self.w_phi)     # (z, mu, phi, g, t) -> (z, mu, g, t)
        phi = integrate_dim(phi, 1, self.w_mu)                 # -> (z, g, t)
        cores = list(phi.cores)
        cores[0] = legendre_to_physical_space(cores[0], self.p, self.N_z, self.xi)
        return tntt.TT(cores)


def legendre_scattering_kernels(L, n_pol, n_az, azimuthal_symmetry=False):
    """
    Angular kernels K_l = sum_m Y_lm(Omega_nq) Y_lm(Omega_n'q') w_n' w_q', l = 0..L, as 2-core TTMs over (mu, phi).

    sigma_s(Omega.Omega') = sum_l (2l+1)/(4 pi) sigma_s^l P_l(Omega.Omega') = sum_l sigma_s^l sum_m Y_lm Y_lm,
    so the discrete scattering operator is sum_l K_l (x) Sigma^l.  Each K_l has (mu|phi) rank 2l+1.

    azimuthal_symmetry = True: only the m = 0 terms, which are the only ones acting on a phi-independent psi;
    requires n_az = 1 (single ordinate phi = 0 with weight 2 pi), so K_l = (2l+1)/2 P_l(mu) P_l(mu') w_mu' (x) [1]
    has (mu|phi) rank one and the summed scattering operator has (mu|phi) rank L+1 instead of ~(L+1)(L+2)/2.
    """
    if azimuthal_symmetry and n_az != 1:
        raise ValueError("azimuthal_symmetry requires n_az = 1")
    mu, w_mu = leggauss(n_pol)
    phi = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    w_phi = (2 * np.pi) / n_az * np.ones(n_az)

    kernels = []
    for l in range(L + 1):
        K_l = None
        for m in ([0] if azimuthal_symmetry else range(-l, l + 1)):
            N2 = (2 * l + 1) / (4 * np.pi) * factorial(l - abs(m)) / factorial(l + abs(m))
            P = lpmv(abs(m), l, mu).reshape(n_pol, 1)
            PP = np.kron(P, P.T * w_mu)
            if m > 0:
                A = np.cos(m * phi).reshape(n_az, 1)
                AA, N2 = np.kron(A, A.T * w_phi), 2 * N2
            elif m < 0:
                A = np.sin(abs(m) * phi).reshape(n_az, 1)
                AA, N2 = np.kron(A, A.T * w_phi), 2 * N2
            else:
                AA = np.ones((n_az, n_az)) * w_phi
            term = N2 * matrix_kron([PP, AA])
            K_l = term if K_l is None else K_l + term
        kernels.append(K_l)
    return kernels


class SlabTransport:
    """
    Multigroup slab transport with user-supplied cross sections, homogeneous or piecewise constant in z (materials),
    DG in z, S_N in (mu, phi), backward Euler all-at-once in time (or steady state), TT format.
    Tensor ordering: (z-dof, mu, phi, group, time).

    Cross sections (groups ordered as in the input, conventionally g = 0 the fastest):
      homogeneous
        sigma_t : (G,)           total cross section sigma_{t,g}
        sigma_s : list of (G, G) Legendre moments, sigma_s[l][g, g'] = sigma_{s,l,g -> g'} (row = incoming group),
                                 with sigma_s(mu_0) = sum_l (2l+1)/(4 pi) sigma_{s,l} P_l(mu_0)
      heterogeneous, C materials (constant per cell; material interfaces on cell edges)
        sigma_t    : (C, G)                total cross section of every material, sigma_t^c[g]
        sigma_s    : (C, L_max+1, G, G)    sigma_s[c, l] as above for material c
        region_map : (N_z, C) 0/1 matrix chi, chi[i, c] = 1 if cell i is made of material c; mandatory, exactly one
                     material per cell (no overlaps, no gaps), so sigma_t(z_i, g) = sum_c chi[i, c] sigma_t^c[g]
      common
        v       : (G,)           group velocities (not needed if steady_state)
        E_edges : (G+1,)         group energy boundaries (optional, bookkeeping/plotting only)
    Operator: the reaction and scattering terms carry the cell masks M_c = diag(chi[:, c]) (x) M_local in the z core,
        sum_c M_c (x) [I (x) I (x) diag(sigma_t^c) - sum_l K_l (x) Sigma^{l,c}] (x) I_t,
    so the (z|mu) rank of A is C + 2 (streaming adds 2) after the final rounding (A_round_prec).

    steady_state = True: the time dimension has a single point and the time-derivative term is dropped;
    N_t, T and v are ignored, and the time functions passed to set_source / set_BC_* are not used.

    azimuthal_symmetry = True: for problems whose source, IC and BCs do not depend on phi (e.g. a normal beam),
    psi does not either, and only the m = 0 part of the scattering kernel acts on it.  Requires N_phi = 1: the
    phi grid is the single ordinate phi = 0 with weight 2 pi, and the scattering operator has (mu|phi) rank
    L_max+1 instead of ~(L_max+1)(L_max+2)/2.  phi-dependent data are then silently evaluated at phi = 0 only.

    mode = "adj": adjoint problem (-1/v d_t - mu d_z + sigma_t) psi^dag = S^dag psi^dag + q^dag, solved backward
    from T as the mirrored forward problem chi(z, mu, phi, tau) = psi^dag(z, -mu, phi + pi, T - tau) (notes,
    sec. "Time-dependent adjoint").  sigma_s is the same nuclear data as in "frw"; the group transpose is applied
    internally (adjoint_sigma_s).  All data are given in physical variables (Omega, t), and run_amen returns the
    physical psi^dag (mirrored back):
        set_source   q^dag (e.g. a detector cross section), sampled at t_k like a forward source;
        set_IC       the FINAL condition psi^dag(T) (default 0); for a detector read at T, R = <r, psi(T)>, pass v_g r_g;
        set_BC_*     psi^dag on the adjoint-incoming directions, mu < 0 at z = 0 and mu > 0 at z = L (default 0).
    Assigning model.q directly is also allowed (physical variables; it is mirrored in construct_tt_rhs).
    In time-dependent problems psi^dag[k] approximates psi^dag(t_{k-1}) (staggered levels, attribute t_sol).
    Responses: R = adj.inner(F_frw) = frw.inner(F_adj^q) with inner() below, F_frw the full forward right-hand side.
    N_phi must be 1 or even (phi -> phi + pi is a shift of N_phi/2 ordinates).
    """

    MODES = ("frw", "adj")

    def __init__(self, N_z, p, L_z, N_mu, N_phi, sigma_t, sigma_s, v=None, E_edges=None,
                 N_t=1, T=1.0, steady_state=False, N_recon=3, K_round_prec=1e-12, azimuthal_symmetry=False,
                 mode="frw", region_map=None, A_round_prec=1e-13):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        if mode == "adj" and N_phi > 1 and N_phi % 2:
            raise ValueError("mode = 'adj' needs N_phi = 1 or even (phi -> phi + pi on the ordinate grid)")
        # ---------------- cross sections: region map chi (N_z, C), tables per material ----------------
        sigma_t = np.asarray(sigma_t, dtype=float)
        if sigma_t.ndim == 2:                                     # (C, G) table: heterogeneous, chi mandatory
            if region_map is None:
                raise ValueError("a (C, G) sigma_t table needs region_map, the (N_z, C) 0/1 material matrix chi")
            sigma_t_c = sigma_t
            self.region_map = self.check_region_map(region_map, N_z, sigma_t_c.shape[0])
            self.heterogeneous = True
        elif sigma_t.ndim == 1:                                   # homogeneous (G,)
            if region_map is not None:
                raise ValueError("region_map is only used with a (C, G) sigma_t table")
            sigma_t_c = sigma_t[None, :]
            self.region_map = np.ones((N_z, 1))
            self.heterogeneous = False
        else:
            raise ValueError(f"sigma_t must be (G,) or (C, G), got shape {sigma_t.shape}")
        self.C, self.G = C, G = sigma_t_c.shape
        self.sigma_t_c = sigma_t_c                                # (C, G)
        self.sigma_t = sigma_t_c if self.heterogeneous else sigma_t_c[0]
        if self.heterogeneous:
            sigma_s = np.asarray(sigma_s, dtype=float)
            if sigma_s.ndim != 4 or sigma_s.shape[0] != C or sigma_s.shape[2:] != (G, G):
                raise ValueError(f"heterogeneous sigma_s must be (C, L_max+1, G, G) = ({C}, L+1, {G}, {G}), "
                                 f"got {sigma_s.shape}")
        else:
            if isinstance(sigma_s, np.ndarray) and sigma_s.ndim == 2:
                sigma_s = [sigma_s]                               # P0 only
            sigma_s = np.stack([np.asarray(s, dtype=float).reshape(G, G) for s in sigma_s])[None]
        self.sigma_s_data = sigma_s                               # nuclear data, as given: (C, L_max+1, G, G)
        # matrices used by the operator: the data in "frw", their group transpose in "adj"
        self.sigma_s = self.adjoint_sigma_s(self.sigma_s_data) if mode == "adj" else self.sigma_s_data
        self.L_max = self.sigma_s.shape[1] - 1
        self.steady_state = steady_state
        if not steady_state:
            if v is None:
                raise ValueError("group velocities v are required for time-dependent problems")
            self.v = np.asarray(v, dtype=float).ravel()
            if len(self.v) != G:
                raise ValueError(f"v has length {len(self.v)}, expected G = {G}")
        else:
            self.v = None if v is None else np.asarray(v, dtype=float).ravel()
        self.E_edges = None
        if E_edges is not None:
            self.E_edges = np.asarray(E_edges, dtype=float).ravel()
            if len(self.E_edges) != G + 1:
                raise ValueError(f"E_edges has length {len(self.E_edges)}, expected G + 1 = {G + 1}")
            self.E = np.sqrt(self.E_edges[:-1] * self.E_edges[1:])    # group-centered energies (geometric mean)

        self.N_z = N_z           # number of spatial cells
        self.p = p               # polynomial degree
        self.L_z = L_z           # length of the slab
        self.N_mu = N_mu         # number of quadrature points for polar angle
        self.N_phi = N_phi       # number of quadrature points for azimuthal angle
        self.N_t = 1 if steady_state else N_t     # number of time steps
        self.T = None if steady_state else T      # final time
        self.N_recon = N_recon   # number of reconstruction points per cell
        self.azimuthal_symmetry = azimuthal_symmetry
        if azimuthal_symmetry:
            if N_phi != 1:
                raise ValueError("azimuthal_symmetry requires N_phi = 1")
            if N_mu < self.L_max + 3:
                warnings.warn("N_mu should be >= L_max + 3 for exact angular quadrature of the kernel")
        elif N_mu < self.L_max + 3 or N_phi < self.L_max + 3:
            warnings.warn("N_mu and N_phi should be >= L_max + 3 for exact angular quadrature of the kernel")

        # ---------------- grids ----------------
        self.ndof_z = N_z * (p + 1)
        self.h = L_z / N_z
        self.z_mid = (np.arange(N_z) + 0.5) * self.h               # cell midpoints
        self.z = self.z_mid
        self.mu, self.w_mu = leggauss(N_mu)                        # even N_mu -> no mu = 0 ordinate
        self.mu_plus = np.where(self.mu > 0, self.mu, 0.0)
        self.mu_minus = np.where(self.mu < 0, self.mu, 0.0)
        self.phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)   # azimuthal_symmetry: [0], weight 2 pi
        self.w_phi = (2 * np.pi) / N_phi * np.ones(N_phi)
        if steady_state:
            self.dt = None
            self.t = np.zeros(1)
        else:
            self.dt = T / N_t
            self.t = self.dt * np.arange(1, N_t + 1)               # unknowns t_k = k dt, k = 1..N_t
        # physical times of the solution values: t_k (frw), t_{k-1} (adj, staggered by the transposed time stencil)
        self.t_sol = self.t - self.dt if (mode == "adj" and not steady_state) else self.t

        self.xi, self.w_xi = leggauss(N_recon)                     # local to cell
        self.z_r = (self.z_mid[:, None] + self.h / 2 * self.xi[None, :]).ravel()   # reconstruction points
        self.w_z_r = np.tile(self.w_xi, N_z) * self.h / 2          # quadrature weights on z_r

        self.A = None   # TT operator
        self.F = None   # TT right-hand side
        self.q = None   # source term (TT)
        self.IC = None  # initial condition term (TT)
        self.BC_L = None
        self.BC_R = None
        self.psi_modal = None   # solution, Legendre modal coefficients
        self.psi = None         # solution at the reconstruction points z_r

        # ---------------- DG matrices ----------------
        self.S = np.zeros((p + 1, p + 1))
        for a in range(p + 1):
            for b in range(p + 1):
                if a > b and (a + b) % 2 == 1:
                    self.S[a, b] = 2.0

        self.B_p_R = np.ones((p + 1, p + 1))
        self.B_p_L = np.zeros((p + 1, p + 1))
        for a in range(p + 1):
            self.B_p_L[a, :] = (-1)**a
        self.B_m_R = np.zeros((p + 1, p + 1))
        for b in range(p + 1):
            self.B_m_R[:, b] = (-1)**b
        self.B_m_L = np.zeros((p + 1, p + 1))
        for a in range(p + 1):
            for b in range(p + 1):
                self.B_m_L[a, b] = (-1)**(a + b)

        M_local = np.zeros((p + 1, p + 1))
        for a in range(p + 1):
            M_local[a, a] = self.h / (2 * a + 1)
        self.M = np.zeros((self.ndof_z, self.ndof_z))
        for i in range(N_z):
            I = slice(i * (p + 1), (i + 1) * (p + 1))
            self.M[I, I] = M_local
        self.M_local = M_local
        # cell masks of the materials: M_c = diag(chi[:, c]) (x) M_local, sum_c M_c = M
        self.M_c = [np.kron(np.diag(self.region_map[:, c]), M_local) for c in range(C)]
        self.A_round_prec = A_round_prec

        # ---------------- angular kernels K_l: 2-core TTMs over (mu, phi) ----------------
        self.K = legendre_scattering_kernels(self.L_max, N_mu, N_phi, azimuthal_symmetry=azimuthal_symmetry)
        self.K_round_prec = K_round_prec

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _project_z(self, f_z, n_quad=50):
        """int_{K_i} f(z) P_a dz for every (cell i, mode a), as a vector of length ndof_z."""
        xi, w_xi = leggauss(n_quad)
        out = np.zeros(self.ndof_z)
        for i in range(self.N_z):
            for a in range(self.p + 1):
                out[i * (self.p + 1) + a] = np.sum(
                    w_xi * legendre(a)(xi) * f_z(self.z_mid[i] + self.h / 2 * xi)
                ) * self.h / 2
        return out

    def _time_core(self, f_t):
        """Time core of a data term: f_t(t_k), or [1] in steady state."""
        return np.ones(1) if self.steady_state else f_t(self.t)

    def _zeros(self):
        return tntt.zeros([self.ndof_z, self.N_mu, self.N_phi, self.G, self.N_t])

    @staticmethod
    def adjoint_sigma_s(sigma_s):
        """Group transpose of the Legendre moments: sigma_{s,l,g -> g'} -> sigma_{s,l,g' -> g} (adjoint transfer),
        for every material and order (last two axes)."""
        return np.ascontiguousarray(np.swapaxes(np.asarray(sigma_s, dtype=float), -1, -2))

    @staticmethod
    def check_region_map(region_map, N_z, C):
        """Validate the region map chi (N_z, C): entries 0 or 1 and exactly one material per cell (no overlaps,
        no gaps). Returns chi as a float array. Warns if a material occupies no cell."""
        chi = np.asarray(region_map, dtype=float)
        if chi.shape != (N_z, C):
            raise ValueError(f"region_map must be the (N_z, C) = ({N_z}, {C}) material matrix chi, got {chi.shape}")
        if not np.all((chi == 0) | (chi == 1)):
            raise ValueError("region_map must contain only 0 and 1")
        n_mat = chi.sum(axis=1)
        if np.any(n_mat != 1):
            bad = np.flatnonzero(n_mat != 1)
            raise ValueError(f"region_map: every cell needs exactly one material; cells {bad[:10].tolist()} have "
                             f"{n_mat[bad[:10]].astype(int).tolist()}")
        empty = np.flatnonzero(chi.sum(axis=0) == 0)
        if len(empty):
            warnings.warn(f"region_map: materials {empty.tolist()} occupy no cell")
        return chi

    def mirror(self, T):
        """Angle-time mirror (Omega, t) -> (-Omega, reversed time index) of a TT over (z-dof, mu, phi, g, t):
        reverse the mu index (symmetric Gauss nodes), shift phi by N_phi/2 (phi + pi), reverse the time index.
        An involution; maps the physical adjoint data/solution to the mirrored forward problem and back."""
        cores = list(T.cores)
        cores[1] = cores[1].flip(1)
        if self.N_phi > 1:
            cores[2] = tn.roll(cores[2], shifts=self.N_phi // 2, dims=1)
        cores[4] = cores[4].flip(1)
        return tntt.TT(cores)

    def inner(self, F):
        """<psi, f> = sum over (z, Omega, g) and time of the solution against a right-hand-side-like TT F (load
        vectors in z, values on the ordinates): exact in z, S_N quadrature in angle, rectangle rule dt in time.
        The z integral is the dot of the DG coefficients psi_modal with the load vectors int_{K_i} f P_a dz of F
        (inside tntt.dot); no z weights. F given as modal coefficients would need the mass matrix M instead.
        Both in physical variables, e.g. R = adj.inner(frw.F) = frw.inner(adj.q)."""
        if self.psi_modal is None:
            raise ValueError("Run the solver first (run_amen stores psi_modal).")
        cores = list(F.cores)
        cores[1] = cores[1] * tn.tensor(self.w_mu, dtype=cores[1].dtype)[None, :, None]
        cores[2] = cores[2] * tn.tensor(self.w_phi, dtype=cores[2].dtype)[None, :, None]
        w_t = 1.0 if self.steady_state else self.dt
        return w_t * float(tntt.dot(self.psi_modal, tntt.TT(cores)))

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    def set_source(self, core_funcs):
        """core_funcs = [q_z, q_mu, q_phi, q_E, q_t]; q_E is a vector of length G, the others functions.
        In steady state q_t may be omitted (it is ignored).  "adj": q^dag in physical variables."""
        if core_funcs == 0:
            self.q = self._zeros()
            return
        q_z, q_mu, q_phi, q_E = core_funcs[:4]
        q_t = core_funcs[4] if len(core_funcs) > 4 else None
        Q_z = self._project_z(q_z)
        self.q = vector_kron([Q_z, q_mu(self.mu), q_phi(self.phi), np.asarray(q_E, float), self._time_core(q_t)])

    def set_IC(self, core_funcs):
        """core_funcs = [ic_z, ic_mu, ic_phi, ic_E]; enters at the first time step only.
        "adj": the final condition psi^dag(T), stored at the last time index (the first after the mirror)."""
        if self.steady_state:
            if core_funcs != 0:
                raise ValueError("no initial condition in a steady-state problem")
            self.IC = None
            return
        if core_funcs == 0:
            self.IC = self._zeros()
            return
        ic_z, ic_mu, ic_phi, ic_E = core_funcs
        IC_z = self._project_z(ic_z)
        delta_t_0 = np.zeros(self.N_t)
        delta_t_0[-1 if self.mode == "adj" else 0] = 1.0
        self.IC = 1 / self.dt * vector_kron([IC_z, ic_mu(self.mu), ic_phi(self.phi),
                                             np.asarray(ic_E, float) / self.v, delta_t_0])

    def set_BC_left(self, core_funcs):
        """core_funcs = [bc_mu, bc_phi, bc_E, bc_t]; inflow at z = 0, only mu > 0 (weighted by mu).
        In steady state bc_t may be omitted (it is ignored).  "adj": psi^dag(0, mu < 0) (weighted by |mu|)."""
        if core_funcs == 0:
            self.BC_L = self._zeros()
            return
        bc_mu, bc_phi, bc_E = core_funcs[:3]
        bc_t = core_funcs[3] if len(core_funcs) > 3 else None
        BC_z = np.zeros(self.ndof_z)
        for a in range(self.p + 1):
            BC_z[a] = (-1)**a
        mu_in = -self.mu_minus if self.mode == "adj" else self.mu_plus   # incoming half range at z = 0, |mu|
        self.BC_L = vector_kron([BC_z, mu_in * bc_mu(self.mu), bc_phi(self.phi),
                                 np.asarray(bc_E, float), self._time_core(bc_t)])

    def set_BC_right(self, core_funcs):
        """core_funcs = [bc_mu, bc_phi, bc_E, bc_t]; inflow at z = L, only mu < 0 (weighted by -mu).
        In steady state bc_t may be omitted (it is ignored).  "adj": psi^dag(L, mu > 0) (weighted by mu)."""
        if core_funcs == 0:
            self.BC_R = self._zeros()
            return
        bc_mu, bc_phi, bc_E = core_funcs[:3]
        bc_t = core_funcs[3] if len(core_funcs) > 3 else None
        BC_z = np.zeros(self.ndof_z)
        for a in range(self.p + 1):
            BC_z[-(a + 1)] = 1.0
        mu_in = self.mu_plus if self.mode == "adj" else -self.mu_minus   # incoming half range at z = L, |mu|
        self.BC_R = vector_kron([BC_z, mu_in * bc_mu(self.mu), bc_phi(self.phi),
                                 np.asarray(bc_E, float), self._time_core(bc_t)])

    # ------------------------------------------------------------------
    # operator and right-hand side
    # ------------------------------------------------------------------
    def construct_tt_operator(self):
        p, N_z, ndof = self.p, self.N_z, self.ndof_z
        In, Iphi, Ig, It = np.eye(self.N_mu), np.eye(self.N_phi), np.eye(self.G), np.eye(self.N_t)

        # reaction term; per material: cell mask M_c, energy core sigma_t,g^c
        A = None
        for M_c, sig_t in zip(self.M_c, self.sigma_t_c):
            term = matrix_kron([M_c, In, Iphi, np.diag(sig_t), It])
            A = term if A is None else A + term

        # time derivative (backward Euler); energy core 1/v_g
        if not self.steady_state:
            A += matrix_kron([self.M, In, Iphi, np.diag(1 / self.v),
                              1 / self.dt * np.diag(np.ones(self.N_t))
                              - 1 / self.dt * np.diag(np.ones(self.N_t - 1), k=-1)])

        # streaming term
        space = np.zeros((ndof, ndof))
        for i in range(N_z):
            I = slice(i * (p + 1), (i + 1) * (p + 1))
            space[I, I] += -self.S
        A += matrix_kron([space, np.diag(self.mu), Iphi, Ig, It])

        # face terms: positive mu
        space = np.zeros((ndof, ndof))
        for i in range(N_z):
            I = slice(i * (p + 1), (i + 1) * (p + 1))
            space[I, I] += self.B_p_R
            if i > 0:
                Im = slice((i - 1) * (p + 1), i * (p + 1))
                space[I, Im] -= self.B_p_L
        A += matrix_kron([space, np.diag(self.mu_plus), Iphi, Ig, It])

        # face terms: negative mu
        space = np.zeros((ndof, ndof))
        for i in range(N_z):
            I = slice(i * (p + 1), (i + 1) * (p + 1))
            space[I, I] -= self.B_m_L
            if i < N_z - 1:
                Ip = slice((i + 1) * (p + 1), (i + 2) * (p + 1))
                space[I, Ip] += self.B_m_R
        A += matrix_kron([space, np.diag(self.mu_minus), Iphi, Ig, It])

        # scattering term:  sum_c sum_l  M_c (x) K_l (x) Sigma^{l,c} (x) I_t,  Sigma^{l,c}[g, g'] = sigma^c_{s,l,g' -> g}
        scat = None
        for M_c, sig_c in zip(self.M_c, self.sigma_s):
            for K_l, sig_l in zip(self.K, sig_c):
                if not np.any(sig_l):
                    continue
                term = concat_cores([M_c, K_l, sig_l.T, It])
                scat = term if scat is None else scat + term
        if scat is not None:
            A -= scat.round(self.K_round_prec)

        # term-by-term assembly has (z|mu) rank 2C + 3; rounding merges it to C + 2 (time term: M = sum_c M_c)
        self.A_ranks_raw = list(A.R)
        self.A = A.round(self.A_round_prec) if self.A_round_prec else A
        print("A is a")
        print(self.A)

    def construct_tt_rhs(self, round_eps=None):
        terms = [x for x in (self.q, self.IC, self.BC_L, self.BC_R) if x is not None]
        if not terms:
            raise ValueError("set at least one of source, IC, BC_L, BC_R before building the right-hand side")
        F = terms[0]
        for x in terms[1:]:
            F = F + x
        if self.mode == "adj":
            F = self.mirror(F)        # data are physical; the solver sees the mirrored forward problem
        self.F = F.round(round_eps) if round_eps else F
        print("F is a")
        print(self.F)

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------
    def run_amen(self, nswp=40, eps=1e-6, max_full=15000, kickrank=4, round_eps=1e-8, verbose=True, x0=None):
        """AMEn solve; ranks grow by `kickrank` per sweep, so a rank equal to 1 + kickrank*nswp = not converged.
        x0: initial guess, a TT of DG modal coefficients over (z-dof, mu, phi, g, t) like psi_modal (physical
        variables; mirrored in "adj" mode); None = torchtt default (all ones, rank 1)."""
        if self.A is None:
            self.construct_tt_operator()
        if self.F is None:
            self.construct_tt_rhs()
        if x0 is not None:
            if list(x0.N) != list(self.F.N):
                raise ValueError(f"x0 has mode sizes {list(x0.N)}, expected {list(self.F.N)}")
            if self.mode == "adj":
                x0 = self.mirror(x0)
            x0 = tntt.TT([c.to(self.F.cores[0].dtype).contiguous() for c in x0.cores])

        time_start = time.time()
        psi = tntt.solvers.amen_solve(self.A, self.F, x0=x0, eps=eps, max_full=max_full,
                                      kickrank=kickrank, nswp=nswp, verbose=verbose)
        print(f"AMEn solver completed in {time.time() - time_start:.2f} seconds.")

        self.psi_modal = psi.round(round_eps) if round_eps else psi
        if self.mode == "adj":
            self.psi_modal = self.mirror(self.psi_modal)     # chi -> physical psi^dag
        cores = self.psi_modal.cores
        # Legendre reconstruction at the Gauss points z_r of every cell
        self.psi = tntt.TT([legendre_to_physical_space(cores[0], self.p, self.N_z, self.xi)] + list(cores[1:]))
        print("Psi is a")
        print(self.psi)
        return self.psi

    def residual(self):
        """True relative residual ||A x - F|| / ||F|| of the solved (in "adj" mode: mirrored) system."""
        x = self.mirror(self.psi_modal) if self.mode == "adj" else self.psi_modal
        return float((self.A @ x - self.F).norm() / self.F.norm())

    def compute_scalar_flux(self):
        if self.psi_modal is None:
            raise ValueError("Run the solver first (run_amen stores psi_modal).")
        phi = integrate_dim(self.psi_modal, 2, self.w_phi)     # (z, mu, phi, g, t) -> (z, mu, g, t)
        phi = integrate_dim(phi, 1, self.w_mu)                 # -> (z, g, t)
        cores = list(phi.cores)
        cores[0] = legendre_to_physical_space(cores[0], self.p, self.N_z, self.xi)
        return tntt.TT(cores)


class SlabDiffusion:
    """
    Multigroup slab diffusion with user-supplied cross sections, homogeneous or piecewise constant in z (materials),
    continuous Q1 (hat functions on the nodes) in z, backward Euler all-at-once in time (or steady state), TT format.
    Tensor ordering: (z-node, group, time); the unknown is the nodal scalar flux phi_g(z_i, t_k), i = 0..N_z.

        (1/v_g) d_t phi_g - d_z (D_g d_z phi_g) + sigma_{t,g} phi_g = sum_g' sigma^0_{s,g'->g} phi_g' + Q_g,
        D_g = 1 / (3 sigma_{tr,g}),   sigma_{tr,g} = sigma_{t,g} - sum_g' sigma^1_{s,g'->g}   (in-scatter, J_g'/J_g = 1)

    Cross sections: same conventions as SlabTransport (groups as in the input, conventionally g = 0 the fastest).
      homogeneous
        sigma_t : (G,)           total cross section sigma_{t,g}
        sigma_s : list of (G, G) Legendre moments, sigma_s[l][g, g'] = sigma_{s,l,g -> g'} (row = incoming group),
                                 with sigma_s(mu_0) = sum_l (2l+1)/(4 pi) sigma_{s,l} P_l(mu_0); only l = 0, 1 are used
      heterogeneous, C materials (constant per cell; material interfaces on nodes)
        sigma_t    : (C, G)                total cross section of every material
        sigma_s    : (C, L_max+1, G, G)    sigma_s[c, l] as above for material c
        region_map : (N_z, C) 0/1 matrix chi, exactly one material per cell (as in SlabTransport)
      common
        v       : (G,)           group velocities (not needed if steady_state)
        E_edges : (G+1,)         group energy boundaries (optional, bookkeeping/plotting only)

    Marshak conditions, from psi = phi/(4 pi) + 3 mu J/(4 pi) with full 3-D angle:
        phi/4 + J/2 = J^L_g(t) at z = 0,   phi/4 - J/2 = J^R_g(t) at z = L   (J^L/R incoming partial currents),
    i.e. B = 1/2 diag(1, 0, ..., 0, 1) in the operator and 2 J^L/R e_0/N_z on the right-hand side.

    Operator, per material c with the masked unit stiffness/mass matrices K_c, M_c (sum_c M_c = M):
        A = M (x) V^-1 (x) D_t
          + sum_c [ K_c (x) diag(D^c) (x) I_t + M_c (x) diag(sigma_t^c) (x) I_t - M_c (x) Sigma^{0,c} (x) I_t ]
          + B (x) I_G (x) I_t,
    with Sigma^{0,c}[g, g'] = sigma^c_{s,0,g' -> g} and D_t = (I - E^-)/dt.  TT ranks [2C+1, 2] after rounding
    ([3, 2] homogeneous; [., 1] in steady state).

    lumped_mass = True replaces M by the row-sum lumped mass h diag(1/2, 1, ..., 1, 1/2) everywhere (M-matrix:
    positive flux for every h); False is the consistent mass h/6 (1, 4, 1).

    steady_state = True: the time dimension has a single point and the time-derivative term is dropped;
    N_t, T and v are ignored, and the time functions passed to set_source / set_BC_* are not used.
    """

    def __init__(self, N_z, L_z, sigma_t, sigma_s, v=None, E_edges=None, N_t=1, T=1.0, steady_state=False,
                 region_map=None, lumped_mass=False, A_round_prec=1e-13):
        # ---------------- cross sections: region map chi (N_z, C), tables per material ----------------
        sigma_t = np.asarray(sigma_t, dtype=float)
        if sigma_t.ndim == 2:                                     # (C, G) table: heterogeneous, chi mandatory
            if region_map is None:
                raise ValueError("a (C, G) sigma_t table needs region_map, the (N_z, C) 0/1 material matrix chi")
            sigma_t_c = sigma_t
            self.region_map = SlabTransport.check_region_map(region_map, N_z, sigma_t_c.shape[0])
            self.heterogeneous = True
        elif sigma_t.ndim == 1:                                   # homogeneous (G,)
            if region_map is not None:
                raise ValueError("region_map is only used with a (C, G) sigma_t table")
            sigma_t_c = sigma_t[None, :]
            self.region_map = np.ones((N_z, 1))
            self.heterogeneous = False
        else:
            raise ValueError(f"sigma_t must be (G,) or (C, G), got shape {sigma_t.shape}")
        self.C, self.G = C, G = sigma_t_c.shape
        self.sigma_t_c = sigma_t_c                                # (C, G)
        self.sigma_t = sigma_t_c if self.heterogeneous else sigma_t_c[0]
        if self.heterogeneous:
            sigma_s = np.asarray(sigma_s, dtype=float)
            if sigma_s.ndim != 4 or sigma_s.shape[0] != C or sigma_s.shape[2:] != (G, G):
                raise ValueError(f"heterogeneous sigma_s must be (C, L_max+1, G, G) = ({C}, L+1, {G}, {G}), "
                                 f"got {sigma_s.shape}")
        else:
            if isinstance(sigma_s, np.ndarray) and sigma_s.ndim == 2:
                sigma_s = [sigma_s]                               # P0 only
            sigma_s = np.stack([np.asarray(s, dtype=float).reshape(G, G) for s in sigma_s])[None]
        self.sigma_s = sigma_s                                    # nuclear data, as given: (C, L_max+1, G, G)

        # sigma_tr,g = sigma_t,g - sum_g' sigma_{s,1,g' -> g}: sum over the incoming (row) index
        if sigma_s.shape[1] < 2:
            warnings.warn("sigma_s has no P1 moment: sigma_tr = sigma_t (isotropic scattering)")
            self.sigma_tr_c = sigma_t_c.copy()
        else:
            self.sigma_tr_c = sigma_t_c - sigma_s[:, 1].sum(axis=1)
        if np.any(self.sigma_tr_c <= 0):
            raise ValueError("sigma_tr <= 0 in some group/material: D_g = 1/(3 sigma_tr) undefined")
        self.D_c = 1.0 / (3.0 * self.sigma_tr_c)                  # (C, G)
        self.D = self.D_c if self.heterogeneous else self.D_c[0]

        self.steady_state = steady_state
        if not steady_state:
            if v is None:
                raise ValueError("group velocities v are required for time-dependent problems")
            self.v = np.asarray(v, dtype=float).ravel()
            if len(self.v) != G:
                raise ValueError(f"v has length {len(self.v)}, expected G = {G}")
        else:
            self.v = None if v is None else np.asarray(v, dtype=float).ravel()
        self.E_edges = None
        if E_edges is not None:
            self.E_edges = np.asarray(E_edges, dtype=float).ravel()
            if len(self.E_edges) != G + 1:
                raise ValueError(f"E_edges has length {len(self.E_edges)}, expected G + 1 = {G + 1}")
            self.E = np.sqrt(self.E_edges[:-1] * self.E_edges[1:])    # group-centered energies (geometric mean)

        # ---------------- grids ----------------
        self.N_z = N_z           # number of cells K_i = [z_{i-1}, z_i], i = 1..N_z
        self.L_z = L_z           # length of the slab
        self.ndof_z = N_z + 1    # nodes z_i, i = 0..N_z, boundary values included
        self.h = L_z / N_z
        self.z = np.linspace(0, L_z, N_z + 1)                      # nodes
        self.N_t = 1 if steady_state else N_t     # number of time steps
        self.T = None if steady_state else T      # final time
        if steady_state:
            self.dt = None
            self.t = np.zeros(1)
        else:
            self.dt = T / N_t
            self.t = self.dt * np.arange(1, N_t + 1)               # unknowns t_k = k dt, k = 1..N_t

        self.A = None   # TT operator
        self.F = None   # TT right-hand side
        self.q = None   # source term (TT)
        self.IC = None  # initial condition term (TT)
        self.BC_L = None
        self.BC_R = None
        self.phi = None # solution, nodal scalar flux (z-node, g, t)

        # ---------------- Q1 matrices (unit data), masked per material ----------------
        h = self.h
        K_e = np.array([[1.0, -1.0], [-1.0, 1.0]]) / h
        M_e = np.diag([h / 2, h / 2]) if lumped_mass else np.array([[2.0, 1.0], [1.0, 2.0]]) * h / 6
        self.lumped_mass = lumped_mass
        self.K_c, self.M_c = [], []
        for c in range(C):
            K, M = np.zeros((N_z + 1, N_z + 1)), np.zeros((N_z + 1, N_z + 1))
            for i in np.flatnonzero(self.region_map[:, c]):        # cell i (0-based) joins nodes i, i+1
                K[i:i + 2, i:i + 2] += K_e
                M[i:i + 2, i:i + 2] += M_e
            self.K_c.append(K)
            self.M_c.append(M)
        self.K = sum(self.K_c)
        self.M = sum(self.M_c)
        self.B = np.zeros((N_z + 1, N_z + 1))                      # Marshak: 1/2 phi on both boundary nodes
        self.B[0, 0] = self.B[-1, -1] = 0.5
        self.A_round_prec = A_round_prec

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _project_z(self, f_z, n_quad=10):
        """Load vector int_0^L f(z) varphi_i dz for every node i, Gauss quadrature on every cell."""
        xi, w_xi = leggauss(n_quad)
        zq = (self.z[:-1, None] + self.z[1:, None]) / 2 + self.h / 2 * xi[None, :]     # (N_z, n_quad)
        fw = f_z(zq) * w_xi[None, :] * self.h / 2
        out = np.zeros(self.ndof_z)
        out[:-1] += (fw * (1 - xi) / 2).sum(axis=1)                # left node of each cell
        out[1:] += (fw * (1 + xi) / 2).sum(axis=1)                 # right node of each cell
        return out

    def _time_core(self, f_t):
        """Time core of a data term: f_t(t_k), or [1] in steady state."""
        return np.ones(1) if self.steady_state else f_t(self.t)

    def _zeros(self):
        return tntt.zeros([self.ndof_z, self.G, self.N_t])

    def _boundary_term(self, J, node, round_prec):
        """2 e_node (x) J with J the incoming partial current J_g(t_k), as a TT over (z, g, t):
           [J_E, J_t]  separable, J_E of length G, J_t a function of t (omitted/ignored in steady state), rank 1;
           (G, N_t) array (or (G,) in steady state): truncated SVD J = sum_r a_r b_r^T, rank r."""
        e = np.zeros(self.ndof_z)
        e[node] = 2.0
        if isinstance(J, (list, tuple)):
            J_E = np.asarray(J[0], float)
            J_t = J[1] if len(J) > 1 else None
            return vector_kron([e, J_E, self._time_core(J_t)])
        J = np.asarray(J, dtype=float).reshape(self.G, self.N_t)
        U, s, Vt = np.linalg.svd(J, full_matrices=False)
        r = max(1, int(np.sum(s > round_prec * s[0]))) if s[0] > 0 else 1
        a, b = U[:, :r] * s[:r], Vt[:r]
        cores = [tn.tensor(e[None, :, None]), tn.tensor(a[None, :, :]), tn.tensor(b[:, :, None])]
        return tntt.TT(cores)

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    def set_source(self, core_funcs):
        """core_funcs = [q_z, q_E, q_t]; q_E is a vector of length G, q_z and q_t functions.
        In steady state q_t may be omitted (it is ignored)."""
        if core_funcs == 0:
            self.q = self._zeros()
            return
        q_z, q_E = core_funcs[:2]
        q_t = core_funcs[2] if len(core_funcs) > 2 else None
        self.q = vector_kron([self._project_z(q_z), np.asarray(q_E, float), self._time_core(q_t)])

    def set_IC(self, core_funcs):
        """core_funcs = [ic_z, ic_E]; 1/dt M phi0_z (x) V^-1 phi0_E (x) e_1, phi0_z the nodal values of ic_z."""
        if self.steady_state:
            if core_funcs != 0:
                raise ValueError("no initial condition in a steady-state problem")
            self.IC = None
            return
        if core_funcs == 0:
            self.IC = self._zeros()
            return
        ic_z, ic_E = core_funcs
        delta_t_0 = np.zeros(self.N_t)
        delta_t_0[0] = 1.0
        self.IC = 1 / self.dt * vector_kron([self.M @ ic_z(self.z), np.asarray(ic_E, float) / self.v, delta_t_0])

    def set_BC_left(self, J, round_prec=1e-12):
        """Incoming partial current J^L_g(t) at z = 0 (Marshak); see _boundary_term for the formats. 0 = vacuum."""
        self.BC_L = self._zeros() if (isinstance(J, int) and J == 0) else self._boundary_term(J, 0, round_prec)

    def set_BC_right(self, J, round_prec=1e-12):
        """Incoming partial current J^R_g(t) at z = L (Marshak); see _boundary_term for the formats. 0 = vacuum."""
        self.BC_R = self._zeros() if (isinstance(J, int) and J == 0) else self._boundary_term(J, -1, round_prec)

    # ------------------------------------------------------------------
    # operator and right-hand side
    # ------------------------------------------------------------------
    def construct_tt_operator(self):
        Ig, It = np.eye(self.G), np.eye(self.N_t)

        # Marshak boundary: B (x) I (x) I
        A = matrix_kron([self.B, Ig, It])

        # time derivative (backward Euler): M (x) V^-1 (x) D_t
        if not self.steady_state:
            D_t = 1 / self.dt * (np.eye(self.N_t) - np.diag(np.ones(self.N_t - 1), k=-1))
            A += matrix_kron([self.M, np.diag(1 / self.v), D_t])

        # per material: stiffness K_c (x) diag(D^c), collision M_c (x) diag(sigma_t^c), scattering -M_c (x) Sigma^{0,c}
        for K_c, M_c, D_c, sig_t, sig_s in zip(self.K_c, self.M_c, self.D_c, self.sigma_t_c, self.sigma_s):
            A += matrix_kron([K_c, np.diag(D_c), It])
            A += matrix_kron([M_c, np.diag(sig_t), It])
            if np.any(sig_s[0]):
                A -= matrix_kron([M_c, sig_s[0].T, It])          # Sigma^0[g, g'] = sigma_{s,0,g' -> g}

        self.A_ranks_raw = list(A.R)
        self.A = A.round(self.A_round_prec) if self.A_round_prec else A
        print("A is a")
        print(self.A)

    def construct_tt_rhs(self, round_eps=None):
        terms = [x for x in (self.q, self.IC, self.BC_L, self.BC_R) if x is not None]
        if not terms:
            raise ValueError("set at least one of source, IC, BC_L, BC_R before building the right-hand side")
        F = terms[0]
        for x in terms[1:]:
            F = F + x
        self.F = F.round(round_eps) if round_eps else F
        print("F is a")
        print(self.F)

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------
    def run_amen(self, nswp=40, eps=1e-6, max_full=15000, kickrank=4, round_eps=1e-8, verbose=True, x0=None):
        """AMEn solve; ranks grow by `kickrank` per sweep, so a rank equal to 1 + kickrank*nswp = not converged.
        x0: initial guess, a TT over (z-node, g, t); None = torchtt default (all ones, rank 1)."""
        if self.A is None:
            self.construct_tt_operator()
        if self.F is None:
            self.construct_tt_rhs()

        time_start = time.time()
        phi = tntt.solvers.amen_solve(self.A, self.F, x0=x0, eps=eps, max_full=max_full,
                                      kickrank=kickrank, nswp=nswp, verbose=verbose)
        print(f"AMEn solver completed in {time.time() - time_start:.2f} seconds.")

        self.phi = phi.round(round_eps) if round_eps else phi
        print("Phi is a")
        print(self.phi)
        return self.phi

    def residual(self):
        """True relative residual ||A x - F|| / ||F||."""
        return float((self.A @ self.phi - self.F).norm() / self.F.norm())

    def compute_scalar_flux(self):
        """Nodal scalar flux phi_g(z_i, t_k), TT over (z-node, g, t) (the solution itself)."""
        if self.phi is None:
            raise ValueError("Run the solver first (run_amen stores phi).")
        return self.phi


def p1_angular_flux(phi_zg, D, model, round_eps=1e-12):
    """P1 (diffusion) angular flux as a SlabTransport initial guess, p = 0 ansatz:
        psi(z, mu, phi, g) = phi_g(z) / (4 pi)  -  3/(4 pi) mu D_g d_z phi_g(z),
    i.e. the TT  phi (x) 1_mu/(4 pi)  -  (D_g d_z phi) (x) mu * 3/(4 pi), constant in the azimuth.
      phi_zg : TT over (z-node, g) or (z-node, g, t), the nodal Q1 flux of SlabDiffusion on the same cells (N_z + 1
               nodes) and, if given, the same time levels (N_t); a 2-core phi is taken constant in time
      D      : (G,) diffusion coefficients D_g
      model  : the SlabTransport model whose grids (N_z, p, N_mu, N_phi, N_t) psi is built on
    On cell K_i the Q1 flux is linear: its average (phi_{i-1} + phi_i)/2 is the Legendre mode 0 of phi and its
    slope (phi_i - phi_{i-1})/h is d_z phi.  Both go into mode 0 of the DG coefficients; the modes a >= 1 are 0.
    Returns psi as DG modal coefficients over (z-dof, mu, phi, g, t), the format of psi_modal and of run_amen(x0=...)."""
    N_z, p, h = model.N_z, model.p, model.h
    shape = list(phi_zg.N)
    if shape[:2] != [N_z + 1, model.G] or len(shape) not in (2, 3) or (len(shape) == 3 and shape[2] != model.N_t):
        raise ValueError(f"phi_zg must be a TT over (z-node, g[, t]) of size ({N_z + 1}, {model.G}[, {model.N_t}]), "
                         f"got {shape}")
    c_z, c_g = phi_zg.cores[:2]                                   # (1, N_z+1, r), (r, G, r_t)
    r = c_z.shape[2]
    dtype = c_z.dtype
    avg = np.zeros((N_z, N_z + 1))
    dif = np.zeros((N_z, N_z + 1))
    for i in range(N_z):
        avg[i, i:i + 2] = 0.5
        dif[i, i:i + 2] = (-1.0 / h, 1.0 / h)
    embed = np.zeros((N_z * (p + 1), N_z))                        # cell value -> Legendre mode 0 of cell i
    embed[np.arange(N_z) * (p + 1), np.arange(N_z)] = 1.0
    P_avg = tn.tensor(embed @ avg, dtype=dtype)
    P_dif = tn.tensor(embed @ dif, dtype=dtype)

    # two terms, stacked along the rank between z and mu: [phi | d_z phi]
    core_z = tn.cat([tn.einsum('ij,ajb->aib', P_avg, c_z), tn.einsum('ij,ajb->aib', P_dif, c_z)], dim=2)
    I_r = tn.eye(r, dtype=dtype)
    mu = tn.tensor(model.mu, dtype=dtype)
    core_mu = tn.zeros((2 * r, model.N_mu, 2 * r), dtype=dtype)
    core_mu[:r, :, :r] = I_r[:, None, :] * (tn.ones_like(mu) / (4 * np.pi))[None, :, None]
    core_mu[r:, :, r:] = I_r[:, None, :] * (-3.0 / (4 * np.pi) * mu)[None, :, None]
    core_phi = tn.eye(2 * r, dtype=dtype)[:, None, :] * tn.ones(model.N_phi, dtype=dtype)[None, :, None]
    D_g = tn.tensor(np.asarray(D, dtype=float), dtype=dtype)
    core_g = tn.cat([c_g, c_g * D_g[None, :, None]], dim=0)       # (2r, G, r_t): phi_g and D_g phi_g
    core_t = phi_zg.cores[2] if len(shape) == 3 else tn.ones((1, model.N_t, 1), dtype=dtype)
    psi = tntt.TT([core_z, core_mu, core_phi, core_g, core_t])
    return psi.round(round_eps) if round_eps else psi

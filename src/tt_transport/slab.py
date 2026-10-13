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

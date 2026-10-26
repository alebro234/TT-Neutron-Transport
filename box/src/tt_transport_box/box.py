import torch as tn
import torchtt as tntt
import numpy as np
from scipy.special import legendre, eval_legendre
from numpy.polynomial.legendre import leggauss
from math import sqrt, factorial
from scipy.special import lpmv
import time
import warnings


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

# modal Legendre coefficients -> point values at the Gauss points xi of every cell
def legendre_to_physical_space(Gc, p, Ns, xi):
    r0, n, r1 = Gc.shape
    P = tn.tensor(np.stack([eval_legendre(a, xi) for a in range(p + 1)], axis=1), dtype=Gc.dtype)  # (Nr, p+1)
    Gc = Gc.reshape(r0, Ns, p + 1, r1)
    return tn.einsum('iNaj,ra->iNrj', Gc, P).reshape(r0, Ns * len(xi), r1)


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


def dg_matrices_1d(N, h, p):
    """1D upwind DG matrices on N uniform cells of size h, Legendre modes 0..p, index s = (i, a), mode fastest
    (docs, sec. "Space: tensor-product upwind DG"):
        M     = I_N (x) h M_hat,  M_hat = diag(1/(2a+1))
        S_sf  = I_N (x) S,        S_ab = int P_b P_a' dxi
        U_plus  = I_N (x) p+ p+^T - E^- (x) p- p+^T
        U_minus = E^+ (x) p+ p-^T - I_N (x) p- p-^T
        G_plus/G_minus = U_plus/U_minus - S_sf
    """
    M_hat = np.diag(1.0 / (2 * np.arange(p + 1) + 1))
    S = np.zeros((p + 1, p + 1))
    for a in range(p + 1):
        for b in range(p + 1):
            if a > b and (a + b) % 2 == 1:
                S[a, b] = 2.0
    p_plus = np.ones(p + 1)                          # P_a(+1)
    p_minus = (-1.0) ** np.arange(p + 1)             # P_a(-1)
    I_N = np.eye(N)
    E_minus = np.eye(N, k=-1)                        # (i, i-1): upwind neighbour for Omega_d > 0
    E_plus = np.eye(N, k=1)                          # (i, i+1): upwind neighbour for Omega_d < 0
    M = np.kron(I_N, h * M_hat)
    S_sf = np.kron(I_N, S)
    U_plus = np.kron(I_N, np.outer(p_plus, p_plus)) - np.kron(E_minus, np.outer(p_minus, p_plus))
    U_minus = np.kron(E_plus, np.outer(p_plus, p_minus)) - np.kron(I_N, np.outer(p_minus, p_minus))
    return dict(M_hat=M_hat, S=S, p_plus=p_plus, p_minus=p_minus, M=M, S_sf=S_sf,
                U_plus=U_plus, U_minus=U_minus, G_plus=U_plus - S_sf, G_minus=U_minus - S_sf)


class BoxTransport:
    """
    Multigroup transport in the box [0, L_x] x [0, L_y] x [0, L_z], homogeneous medium, forward problem.
    Tensor-product upwind DG of degree p in x, y, z (uniform cells), S_N product quadrature in (mu, phi),
    backward Euler all-at-once in time (or steady state), TT format (docs/theory_and_docs.tex, sec. 1).
    Tensor ordering: (x-dof, y-dof, z-dof, mu, phi, group, time), x-dof s_x = (i_x, a_x), mode fastest.

    Direction cosines: Omega = (eta cos(phi), eta sin(phi), mu), eta = sqrt(1 - mu^2).

    Cross sections (as in SlabTransport, homogeneous):
        sigma_t : (G,)               total cross section sigma_{t,g}; a scalar means one speed (G = 1)
        sigma_s : list of (G, G)     Legendre moments, sigma_s[l][g, g'] = sigma_{s,l,g -> g'} (row = incoming group),
                                     with sigma_s(mu_0) = sum_l (2l+1)/(4 pi) sigma_{s,l} P_l(mu_0);
                                     the operator uses Sigma_l = sigma_s[l]^T (row = receiving group)
                                     one speed (scalar sigma_t): a scalar sigma_s is isotropic scattering, a list of
                                     scalars the Legendre moments [sigma_s0, sigma_s1, ...]
        v       : (G,)               group velocities (not needed if steady_state)
        E_edges : (G+1,)             group energy boundaries (optional, bookkeeping/plotting only)

    Operator (docs, eq. (16)), M_sf = M_x (x) M_y (x) M_z:
        A =   M_sf (x) I (x) I (x) Sigma_t (x) I                                   collision
            + M_sf (x) I (x) I (x) V^-1 (x) D_t                                    time derivative
            + sum_pm G_x^pm (x) M_y (x) M_z (x) D_eta (x) D_cos^pm (x) I (x) I      streaming x
            + sum_pm M_x (x) G_y^pm (x) M_z (x) D_eta (x) D_sin^pm (x) I (x) I      streaming y
            + sum_pm M_x (x) M_y (x) G_z^pm (x) D_mu^pm (x) I (x) I (x) I           streaming z
            - sum_l  M_sf (x) K_l (x) Sigma_l (x) I                                scattering

    Data (core functions; space and angle as functions, energy as a vector of length G, time as a function):
        set_source([q_x, q_y, q_z, q_mu, q_phi, q_E, q_t])
        set_IC([ic_x, ic_y, ic_z, ic_mu, ic_phi, ic_E])
        set_BC(face, [b_1, b_2, b_mu, b_phi, b_E, b_t]),  face in FACES; b_1, b_2 are the two transverse space
            functions in axis order (x-faces: (y, z), y-faces: (x, z), z-faces: (x, y)).
    In steady state the time functions may be omitted (they are ignored).

    steady_state = True: the time dimension has a single point and the time-derivative term is dropped;
    N_t, T and v are ignored.
    """

    FACES = ("L_x", "R_x", "L_y", "R_y", "L_z", "R_z")

    def __init__(self, N_x, N_y, N_z, p, L_x, L_y, L_z, N_mu, N_phi, sigma_t, sigma_s, v=None, E_edges=None,
                 N_t=1, T=1.0, steady_state=False, N_recon=3, K_round_prec=1e-12, A_round_prec=1e-13):
        # ---------------- cross sections ----------------
        self.sigma_t = np.asarray(sigma_t, dtype=float).ravel()
        self.G = G = len(self.sigma_t)
        if np.ndim(sigma_s) == 0:
            sigma_s = [sigma_s]                                   # scalar: one-speed, isotropic
        elif isinstance(sigma_s, np.ndarray) and sigma_s.ndim == 2:
            sigma_s = [sigma_s]                                   # P0 only
        if all(np.ndim(s) == 0 for s in sigma_s) and G != 1:      # list of scalars: one-speed Legendre moments
            raise ValueError(f"a list of scalar sigma_s means one-speed Legendre moments, but sigma_t has G = {G}")
        self.sigma_s = np.stack([np.asarray(s, dtype=float).reshape(G, G) for s in sigma_s])   # (L_max+1, G, G)
        self.L_max = self.sigma_s.shape[0] - 1
        self.Sigma_t = np.diag(self.sigma_t)
        self.Sigma_l = [s.T for s in self.sigma_s]               # Sigma_l[g, g'] = sigma_{s,l,g' -> g}
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

        self.N_x, self.N_y, self.N_z = N_x, N_y, N_z     # number of cells per direction
        self.p = p                                       # polynomial degree, same in x, y, z
        self.L_x, self.L_y, self.L_z = L_x, L_y, L_z     # box edges
        self.N_mu = N_mu         # number of quadrature points for polar angle
        self.N_phi = N_phi       # number of quadrature points for azimuthal angle
        self.N_t = 1 if steady_state else N_t     # number of time steps
        self.T = None if steady_state else T      # final time
        self.N_recon = N_recon   # number of reconstruction points per cell and direction
        if N_mu < self.L_max + 1 or N_phi < 2 * self.L_max + 1:
            warnings.warn("exact angular quadrature of the kernel needs N_mu >= L_max + 1 and N_phi >= 2 L_max + 1")

        # ---------------- grids ----------------
        self.N_s_x, self.N_s_y, self.N_s_z = N_x * (p + 1), N_y * (p + 1), N_z * (p + 1)
        self.h_x, self.h_y, self.h_z = L_x / N_x, L_y / N_y, L_z / N_z
        self.x_mid = (np.arange(N_x) + 0.5) * self.h_x            # cell midpoints
        self.y_mid = (np.arange(N_y) + 0.5) * self.h_y
        self.z_mid = (np.arange(N_z) + 0.5) * self.h_z
        self.mu, self.w_mu = leggauss(N_mu)                        # even N_mu -> no mu = 0 ordinate
        self.eta = np.sqrt(1 - self.mu**2)                         # > 0 on the Gauss nodes
        self.phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
        self.w_phi = (2 * np.pi) / N_phi * np.ones(N_phi)
        self.cos_phi, self.sin_phi = np.cos(self.phi), np.sin(self.phi)
        if steady_state:
            self.dt = None
            self.t = np.zeros(1)
        else:
            self.dt = T / N_t
            self.t = self.dt * np.arange(1, N_t + 1)               # unknowns t_k = k dt, k = 1..N_t

        self.xi, self.w_xi = leggauss(N_recon)                     # local to cell
        self.x_r = (self.x_mid[:, None] + self.h_x / 2 * self.xi[None, :]).ravel()   # reconstruction points
        self.y_r = (self.y_mid[:, None] + self.h_y / 2 * self.xi[None, :]).ravel()
        self.z_r = (self.z_mid[:, None] + self.h_z / 2 * self.xi[None, :]).ravel()
        self.w_x_r = np.tile(self.w_xi, N_x) * self.h_x / 2        # quadrature weights on x_r, y_r, z_r
        self.w_y_r = np.tile(self.w_xi, N_y) * self.h_y / 2
        self.w_z_r = np.tile(self.w_xi, N_z) * self.h_z / 2

        self.A = None   # TT operator
        self.F = None   # TT right-hand side
        self.q = None   # source term F_Q (TT)
        self.IC = None  # initial condition term F_0 (TT)
        self.BC = {}    # inflow terms F_{L_x}, ..., F_{R_z} (TT), keyed by face
        self.psi_modal = None   # solution, Legendre modal coefficients
        self.psi = None         # solution at the reconstruction points (x_r, y_r, z_r)

        # ---------------- DG matrices per direction ----------------
        dg_x, dg_y, dg_z = dg_matrices_1d(N_x, self.h_x, p), dg_matrices_1d(N_y, self.h_y, p), \
            dg_matrices_1d(N_z, self.h_z, p)
        self.M_hat, self.S = dg_x["M_hat"], dg_x["S"]             # reference matrices, same in every direction
        self.p_plus, self.p_minus = dg_x["p_plus"], dg_x["p_minus"]
        self.M_x, self.M_y, self.M_z = dg_x["M"], dg_y["M"], dg_z["M"]
        self.S_x, self.S_y, self.S_z = dg_x["S_sf"], dg_y["S_sf"], dg_z["S_sf"]
        self.U_x_plus, self.U_x_minus = dg_x["U_plus"], dg_x["U_minus"]
        self.U_y_plus, self.U_y_minus = dg_y["U_plus"], dg_y["U_minus"]
        self.U_z_plus, self.U_z_minus = dg_z["U_plus"], dg_z["U_minus"]
        self.G_x_plus, self.G_x_minus = dg_x["G_plus"], dg_x["G_minus"]
        self.G_y_plus, self.G_y_minus = dg_y["G_plus"], dg_y["G_minus"]
        self.G_z_plus, self.G_z_minus = dg_z["G_plus"], dg_z["G_minus"]

        # ---------------- angular diagonals: direction cosines and their upwind split ----------------
        self.D_mu = np.diag(self.mu)
        self.D_mu_plus, self.D_mu_minus = np.diag(np.maximum(self.mu, 0)), np.diag(np.minimum(self.mu, 0))
        self.D_eta = np.diag(self.eta)
        self.D_cos = np.diag(self.cos_phi)
        self.D_cos_plus, self.D_cos_minus = np.diag(np.maximum(self.cos_phi, 0)), np.diag(np.minimum(self.cos_phi, 0))
        self.D_sin = np.diag(self.sin_phi)
        self.D_sin_plus, self.D_sin_minus = np.diag(np.maximum(self.sin_phi, 0)), np.diag(np.minimum(self.sin_phi, 0))
        self.W_mu, self.W_phi = np.diag(self.w_mu), np.diag(self.w_phi)

        # ---------------- time ----------------
        if not steady_state:
            self.V_inv = np.diag(1 / self.v)
            self.D_t = (np.eye(self.N_t) - np.eye(self.N_t, k=-1)) / self.dt   # backward Euler, D_t = (I - E^-)/dt
        self.A_round_prec = A_round_prec

        # ---------------- angular kernels K_l: 2-core TTMs over (mu, phi), all m ----------------
        self.K = legendre_scattering_kernels(self.L_max, N_mu, N_phi)
        self.K_round_prec = K_round_prec

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _axis(self, d):
        """(N, h, cell midpoints) of axis d in {0: x, 1: y, 2: z}."""
        return [(self.N_x, self.h_x, self.x_mid), (self.N_y, self.h_y, self.y_mid),
                (self.N_z, self.h_z, self.z_mid)][d]

    def _project(self, f, d, n_quad=50):
        """Load vector int_{K_i} f P_a along axis d for every (cell i, mode a), length N_d (p+1)."""
        N, h, mid = self._axis(d)
        xi, w_xi = leggauss(n_quad)
        P = np.stack([legendre(a)(xi) for a in range(self.p + 1)])          # (p+1, n_quad)
        f_q = f(mid[:, None] + h / 2 * xi[None, :])                        # (N, n_quad)
        return (h / 2 * (f_q * w_xi) @ P.T).ravel()

    def _face(self, d, side):
        """e_1 (x) p^- (side "L") or e_N (x) p^+ (side "R") along axis d: the test functions on the face."""
        N = self._axis(d)[0]
        e = np.zeros(N)
        e[0 if side == "L" else -1] = 1.0
        return np.kron(e, self.p_minus if side == "L" else self.p_plus)

    def _time_core(self, f_t):
        """Time core of a data term: f_t(t_k), or [1] in steady state."""
        return np.ones(1) if self.steady_state else f_t(self.t)

    def _shape(self):
        return [self.N_s_x, self.N_s_y, self.N_s_z, self.N_mu, self.N_phi, self.G, self.N_t]

    def _zeros(self):
        return tntt.zeros(self._shape())

    def inner(self, F):
        """<psi, f> of the solution against a right-hand-side-like TT F (load vectors in x, y, z, values on the
        ordinates): exact in space, S_N quadrature in angle, rectangle rule dt in time (docs, eq. (21))."""
        if self.psi_modal is None:
            raise ValueError("Run the solver first (run_amen stores psi_modal).")
        cores = list(F.cores)
        cores[3] = cores[3] * tn.tensor(self.w_mu, dtype=cores[3].dtype)[None, :, None]
        cores[4] = cores[4] * tn.tensor(self.w_phi, dtype=cores[4].dtype)[None, :, None]
        w_t = 1.0 if self.steady_state else self.dt
        return w_t * float(tntt.dot(self.psi_modal, tntt.TT(cores)))

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    def set_source(self, core_funcs):
        """core_funcs = [q_x, q_y, q_z, q_mu, q_phi, q_E, q_t]: F_Q = Q_x (x) Q_y (x) Q_z (x) Q_mu (x) Q_phi (x) Q_E (x) Q_t,
        space factors as load vectors.  In steady state q_t may be omitted."""
        if core_funcs == 0:
            self.q = self._zeros()
            return
        q_x, q_y, q_z, q_mu, q_phi, q_E = core_funcs[:6]
        q_t = core_funcs[6] if len(core_funcs) > 6 else None
        self.q = vector_kron([self._project(q_x, 0), self._project(q_y, 1), self._project(q_z, 2),
                              q_mu(self.mu), q_phi(self.phi), np.asarray(q_E, float), self._time_core(q_t)])

    def set_IC(self, core_funcs):
        """core_funcs = [ic_x, ic_y, ic_z, ic_mu, ic_phi, ic_E]:
        F_0 = dt^-1 Psi0_x (x) Psi0_y (x) Psi0_z (x) Psi0_mu (x) Psi0_phi (x) V^-1 Psi0_E (x) e_1."""
        if self.steady_state:
            if core_funcs != 0:
                raise ValueError("no initial condition in a steady-state problem")
            self.IC = None
            return
        if core_funcs == 0:
            self.IC = self._zeros()
            return
        ic_x, ic_y, ic_z, ic_mu, ic_phi, ic_E = core_funcs
        e_1 = np.zeros(self.N_t)
        e_1[0] = 1.0
        self.IC = 1 / self.dt * vector_kron([self._project(ic_x, 0), self._project(ic_y, 1), self._project(ic_z, 2),
                                             ic_mu(self.mu), ic_phi(self.phi),
                                             self.V_inv @ np.asarray(ic_E, float), e_1])

    def set_BC(self, face, core_funcs):
        """Inflow on one face (docs, eq. (19)), face in FACES = (L_x, R_x, L_y, R_y, L_z, R_z).
        core_funcs = [b_1, b_2, b_mu, b_phi, b_E, b_t], b_1, b_2 the transverse space functions in axis order;
        in steady state b_t may be omitted.  Only the incoming half range of the face acts, weighted by |Omega_d|:
            x faces: D_eta b_mu (x) D_cos^+ b_phi (L),  D_eta b_mu (x) (-D_cos^-) b_phi (R)
            y faces: D_eta b_mu (x) D_sin^+ b_phi (L),  D_eta b_mu (x) (-D_sin^-) b_phi (R)
            z faces: D_mu^+ b_mu (x) b_phi (L),         (-D_mu^-) b_mu (x) b_phi (R)"""
        if face not in self.FACES:
            raise ValueError(f"face must be one of {self.FACES}, got {face!r}")
        if core_funcs == 0:
            self.BC[face] = self._zeros()
            return
        side, d = face[0], "xyz".index(face[-1])
        b_1, b_2, b_mu, b_phi, b_E = core_funcs[:5]
        b_t = core_funcs[5] if len(core_funcs) > 5 else None
        space = [None, None, None]
        space[d] = self._face(d, side)
        transverse = [e for e in range(3) if e != d]
        space[transverse[0]] = self._project(b_1, transverse[0])
        space[transverse[1]] = self._project(b_2, transverse[1])
        D_in = {("L", 0): (self.D_eta, self.D_cos_plus), ("R", 0): (self.D_eta, -self.D_cos_minus),
                ("L", 1): (self.D_eta, self.D_sin_plus), ("R", 1): (self.D_eta, -self.D_sin_minus),
                ("L", 2): (self.D_mu_plus, np.eye(self.N_phi)), ("R", 2): (-self.D_mu_minus, np.eye(self.N_phi))}
        D_in_mu, D_in_phi = D_in[(side, d)]
        self.BC[face] = vector_kron(space + [D_in_mu @ b_mu(self.mu), D_in_phi @ b_phi(self.phi),
                                             np.asarray(b_E, float), self._time_core(b_t)])

    # ------------------------------------------------------------------
    # operator and right-hand side
    # ------------------------------------------------------------------
    def construct_tt_operator(self):
        I_mu, I_phi, I_g, I_t = np.eye(self.N_mu), np.eye(self.N_phi), np.eye(self.G), np.eye(self.N_t)
        M_x, M_y, M_z = self.M_x, self.M_y, self.M_z

        # collision: M_x (x) M_y (x) M_z (x) I (x) I (x) Sigma_t (x) I
        A = matrix_kron([M_x, M_y, M_z, I_mu, I_phi, self.Sigma_t, I_t])

        # time derivative (backward Euler): M_x (x) M_y (x) M_z (x) I (x) I (x) V^-1 (x) D_t
        if not self.steady_state:
            A += matrix_kron([M_x, M_y, M_z, I_mu, I_phi, self.V_inv, self.D_t])

        # streaming x: sum_pm G_x^pm (x) M_y (x) M_z (x) D_eta (x) D_cos^pm (x) I (x) I
        A += matrix_kron([self.G_x_plus, M_y, M_z, self.D_eta, self.D_cos_plus, I_g, I_t])
        A += matrix_kron([self.G_x_minus, M_y, M_z, self.D_eta, self.D_cos_minus, I_g, I_t])

        # streaming y: sum_pm M_x (x) G_y^pm (x) M_z (x) D_eta (x) D_sin^pm (x) I (x) I
        A += matrix_kron([M_x, self.G_y_plus, M_z, self.D_eta, self.D_sin_plus, I_g, I_t])
        A += matrix_kron([M_x, self.G_y_minus, M_z, self.D_eta, self.D_sin_minus, I_g, I_t])

        # streaming z: sum_pm M_x (x) M_y (x) G_z^pm (x) D_mu^pm (x) I (x) I (x) I
        A += matrix_kron([M_x, M_y, self.G_z_plus, self.D_mu_plus, I_phi, I_g, I_t])
        A += matrix_kron([M_x, M_y, self.G_z_minus, self.D_mu_minus, I_phi, I_g, I_t])

        # scattering: sum_l M_x (x) M_y (x) M_z (x) K_l (x) Sigma_l (x) I
        scat = None
        for K_l, Sigma_l in zip(self.K, self.Sigma_l):
            if not np.any(Sigma_l):
                continue
            term = concat_cores([M_x, M_y, M_z, K_l, Sigma_l, I_t])
            scat = term if scat is None else scat + term
        if scat is not None:
            A -= scat.round(self.K_round_prec)

        # term-by-term assembly; rounding brings the ranks to <= [3, 5, 7, r_K + 6, L_max + 3, 2] (docs, eq. (20))
        self.A_ranks_raw = list(A.R)
        self.A = A.round(self.A_round_prec) if self.A_round_prec else A
        print("A is a")
        print(self.A)

    def construct_tt_rhs(self, round_eps=None):
        """F = F_Q + F_0 + sum over faces of F_{L_d} + F_{R_d}."""
        terms = [x for x in (self.q, self.IC, *self.BC.values()) if x is not None]
        if not terms:
            raise ValueError("set at least one of source, IC, BC before building the right-hand side")
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
        """AMEn solve; ranks grow by `kickrank` per sweep, so a rank equal to rank(x0) + kickrank*nswp = not converged.
        x0: initial guess in the modal (Legendre) space, e.g. self.psi_modal of a looser solve (continuation);
        None = AMEn default (ones tensor)."""
        if self.A is None:
            self.construct_tt_operator()
        if self.F is None:
            self.construct_tt_rhs()

        time_start = time.time()
        psi = tntt.solvers.amen_solve(self.A, self.F, eps=eps, max_full=max_full,
                                      kickrank=kickrank, nswp=nswp, verbose=verbose, x0=x0)
        print(f"AMEn solver completed in {time.time() - time_start:.2f} seconds.")

        self.psi_modal = psi.round(round_eps) if round_eps else psi
        self.psi = self._reconstruct(self.psi_modal)
        print("Psi is a")
        print(self.psi)
        return self.psi

    def _reconstruct(self, T):
        """Legendre reconstruction of the three space cores at the Gauss points (x_r, y_r, z_r) of every cell."""
        cores = list(T.cores)
        for d, N in enumerate((self.N_x, self.N_y, self.N_z)):
            cores[d] = legendre_to_physical_space(cores[d], self.p, N, self.xi)
        return tntt.TT(cores)

    def residual(self):
        """True relative residual ||A x - F|| / ||F||."""
        return float((self.A @ self.psi_modal - self.F).norm() / self.F.norm())

    def compute_scalar_flux(self):
        """phi_g = sum_nq w_n w_q psi_g(Omega_nq) at the reconstruction points, TT over (x, y, z, g, t)."""
        if self.psi_modal is None:
            raise ValueError("Run the solver first (run_amen stores psi_modal).")
        phi = integrate_dim(self.psi_modal, 4, self.w_phi)     # (x, y, z, mu, phi, g, t) -> (x, y, z, mu, g, t)
        phi = integrate_dim(phi, 3, self.w_mu)                 # -> (x, y, z, g, t)
        return self._reconstruct(phi)

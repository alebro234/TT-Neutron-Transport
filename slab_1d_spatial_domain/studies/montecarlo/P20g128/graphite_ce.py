"""
Graphite slab, step 1/2: continuous-energy OpenMC run that generates the multigroup cross sections.

Same problem as ../water_slab/water_ce.py with the water replaced by graphite:
Geometry : slab 0 < z < L, vacuum on z = 0 and z = L, REFLECTIVE planes at x, y = +-W
           (=> laterally infinite slab; a point beam is then equivalent to a uniform plane beam)
Source   : monoenergetic 2 MeV, monodirectional along +z (delta(mu - 1)), injected just inside z = 0
Material : graphite, natural C (C0), bound via S(alpha,beta) (c_Graphite, 296 K data), 294 K, 1.7 g/cm^3
Groups   : same 10 groups as the water case (narrow top group around E0)

Two cases (command-line argument):
  P0tr  P0 with the transport correction (openmc.mgxs correction='P0'), IN-SCATTER form (OpenMC tallies the
        P1 scatter with an EnergyoutFilter, i.e. by receiving group):
          sigma_s1,g = sum_g' sigma_s^1(g'->g) phi_g' / phi_g
          sigma_tr,g = sigma_t,g - sigma_s1,g ,   sigma~_s0(g'->g) = sigma_s0(g'->g) - delta_gg' sigma_s1,g
        mgxs.h5 holds sigma_tr as the total and sigma~_s0 as the (isotropic) scattering matrix.
        Also tallied, uncorrected: sigma_t,g, sigma_s^0(g'->g), sigma_s^1(g'->g) (a separate P1 matrix; OpenMC
        moments have no (2l+1) factor), and the out-scatter form sigma_s1,g' = sum_g sigma_s^1(g'->g) for reference.
  P<L>  raw P_L Legendre moments (correction=None), sigma_t,g and sigma_s^l(g'->g), l = 0..L (e.g. P7, P20)

Outputs (in results_<case>/ for 10 groups, results_<case>_g_<G>/ otherwise):
  graphite_ce_flux.npz  CE reference scalar flux on the z mesh per group
  mgxs.h5               OpenMC MGXS library (xsdata 'graphite'), read by graphite_mg.py
  graphite_mgxs.npz     same data in the TT-solver convention (see below) + the problem parameters
  ce_run/               OpenMC input/output files of this run

Group convention (same as the TT solver): g = 0 is the HIGHEST energy group.
Scattering matrix is returned as  S[l, g, g'] = sigma_s^l(g' -> g)  (row = receiving group),
with the expansion  sigma_s(g'->g, mu0) = sum_l (2l+1)/(4 pi) sigma_s^l(g'->g) P_l(mu0).

Run:  python graphite_ce.py P0tr|P<L> [G] [particles per batch]     (defaults: G = 10, 100000)
"""
import re
import sys
from pathlib import Path

import numpy as np
import openmc
import openmc.mgxs

HERE = Path(__file__).resolve().parent
CASE = sys.argv[1] if len(sys.argv) > 1 else 'P7'
assert CASE == 'P0tr' or re.fullmatch(r'P\d+', CASE), CASE
G = int(sys.argv[2]) if len(sys.argv) > 2 else 10
OUT = HERE / (f'results_{CASE}' + (f'_g_{G}' if G != 10 else ''))
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
L = 30.0                    # slab thickness [cm]
W = 50.0                    # lateral half-width [cm] (irrelevant with reflective sides)
DENSITY = 1.7               # reactor-grade graphite [g/cm^3]
TEMPERATURE = 294.0         # [K]; C0 at 294 K, c_Graphite at 296 K (within OpenMC's default tolerance)
SAB = 'c_Graphite'          # thermal scattering table (None = free-gas C)
E0 = 2.0e6                  # source energy [eV]

# energy groups: edges in eV, ASCENDING (OpenMC convention); identical to the water case
E_MIN = 1.0e-5
TOP_LO = 0.95 * E0
TOP_HI = 1.05 * E0
group_edges = np.append(np.logspace(np.log10(E_MIN), np.log10(TOP_LO), G), TOP_HI)

L_MAX = 0 if CASE == 'P0tr' else int(CASE[1:])     # Legendre order of the library scattering matrix

N_z = 200                   # mesh cells in z of the CE reference flux tally (same as graphite_mg.py)

PARTICLES = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000   # per batch
BATCHES = 50

# ---------------------------------------------------------------------------
# material, geometry
# ---------------------------------------------------------------------------
graphite = openmc.Material(name='graphite')
graphite.add_nuclide('C0', 1.0)
graphite.set_density('g/cm3', DENSITY)
graphite.temperature = TEMPERATURE
if SAB is not None:
    graphite.add_s_alpha_beta(SAB)
materials = openmc.Materials([graphite])

z_left = openmc.ZPlane(z0=0.0, boundary_type='vacuum')
z_right = openmc.ZPlane(z0=L, boundary_type='vacuum')
x_lo = openmc.XPlane(x0=-W, boundary_type='reflective')
x_hi = openmc.XPlane(x0=W, boundary_type='reflective')
y_lo = openmc.YPlane(y0=-W, boundary_type='reflective')
y_hi = openmc.YPlane(y0=W, boundary_type='reflective')

slab = openmc.Cell(name='slab', fill=graphite,
                   region=+z_left & -z_right & +x_lo & -x_hi & +y_lo & -y_hi)
geometry = openmc.Geometry([slab])

# ---------------------------------------------------------------------------
# source: normal beam, 2 MeV, at the left face (just inside, to avoid starting on the surface)
# ---------------------------------------------------------------------------
source = openmc.IndependentSource(
    space=openmc.stats.Point((0.0, 0.0, 1.0e-8)),
    angle=openmc.stats.Monodirectional((0.0, 0.0, 1.0)),
    energy=openmc.stats.Discrete([E0], [1.0]),
)

settings = openmc.Settings()
settings.run_mode = 'fixed source'
settings.source = source
settings.particles = PARTICLES
settings.batches = BATCHES

# ---------------------------------------------------------------------------
# MGXS library
# ---------------------------------------------------------------------------
groups = openmc.mgxs.EnergyGroups(group_edges)

mgxs_lib = openmc.mgxs.Library(geometry)
mgxs_lib.energy_groups = groups
# 'consistent nu-scatter matrix' is what create_mg_library() writes as the scattering matrix;
# 'consistent scatter matrix' is needed with it to build the multiplicity (= 1 here)
types = ['total', 'absorption', 'inverse-velocity', 'consistent scatter matrix', 'consistent nu-scatter matrix']
if CASE == 'P0tr':
    types.insert(1, 'transport')       # with correction='P0' the library writes sigma_tr as the total
mgxs_lib.mgxs_types = types
mgxs_lib.scatter_format = 'legendre'
if L_MAX > 10:
    # the Python MGXS API caps the order at 10 (input check only; LegendreFilter and the MG solver have no cap)
    openmc.mgxs.mgxs._MAX_LEGENDRE = L_MAX
    mgxs_lib._legendre_order = L_MAX
else:
    mgxs_lib.legendre_order = L_MAX
mgxs_lib.correction = 'P0' if CASE == 'P0tr' else None
mgxs_lib.domain_type = 'material'
mgxs_lib.domains = [graphite]
mgxs_lib.by_nuclide = False
mgxs_lib.build_library()

# P0tr: uncorrected P1 scattering matrix alongside, for sigma_s^0(g'->g), sigma_s^1(g'->g), sigma_s1,g
raw_p1 = None
if CASE == 'P0tr':
    raw_p1 = openmc.mgxs.ScatterMatrixXS(domain=graphite, domain_type='material', energy_groups=groups)
    raw_p1.formulation = 'consistent'
    raw_p1.scatter_format = 'legendre'
    raw_p1.correction = None
    raw_p1.legendre_order = 1

# ---------------------------------------------------------------------------
# tallies: CE reference scalar flux (mesh in z) x energy groups, same as graphite_mg.py
# ---------------------------------------------------------------------------
mesh = openmc.RegularMesh(name='z mesh')
mesh.lower_left = (-W, -W, 0.0)
mesh.upper_right = (W, W, L)
mesh.dimension = (1, 1, N_z)

flux_tally = openmc.Tally(name='flux_z_E')
flux_tally.filters = [openmc.MeshFilter(mesh), openmc.EnergyFilter(group_edges)]
flux_tally.scores = ['flux']
flux_tally.estimator = 'tracklength'

tallies = openmc.Tallies([flux_tally])
mgxs_lib.add_to_tallies(tallies, merge=True)
if raw_p1 is not None:
    for t in raw_p1.tallies.values():
        tallies.append(t, merge=True)

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
run_dir = OUT / 'ce_run'
run_dir.mkdir(exist_ok=True)
model = openmc.Model(geometry=geometry, materials=materials, settings=settings, tallies=tallies)
sp_path = model.run(cwd=run_dir, output=True)

# ---------------------------------------------------------------------------
# post-processing: CE flux, OpenMC MGXS library for the MG run
# ---------------------------------------------------------------------------
with openmc.StatePoint(sp_path) as sp:
    t = sp.get_tally(name='flux_z_E')
    # filter order (mesh, energy); OpenMC energy bins are ascending -> reverse to g = 0 highest
    flux = t.mean.reshape(N_z, G)[:, ::-1]
    flux_sd = t.std_dev.reshape(N_z, G)[:, ::-1]
    mgxs_lib.load_from_statepoint(sp)
    if raw_p1 is not None:
        raw_p1.load_from_statepoint(sp)

# scalar flux per unit incident beam intensity (transverse area cancels, reflective sides)
dz = L / N_z
z_centres = (np.arange(N_z) + 0.5) * dz
np.savez(OUT / 'graphite_ce_flux.npz',
         z=z_centres, dz=dz, L=L, group_edges_desc=group_edges[::-1],
         phi=flux / dz, phi_sd=flux_sd / dz, E0=E0)

mg_library = mgxs_lib.create_mg_library(xs_type='macro', xsdata_names=['graphite'])
mg_library.export_to_hdf5(str(OUT / 'mgxs.h5'))

# ---------------------------------------------------------------------------
# post-processing: same data in the TT-solver convention
# ---------------------------------------------------------------------------
def vec(name):
    # order_groups='increasing' -> OpenMC group 1 (highest energy) first, i.e. g = 0 highest
    return np.asarray(mgxs_lib.get_mgxs(graphite, name).get_xs(order_groups='increasing')).reshape(G)


def legendre_matrix(m, n_l):
    S_inout = np.asarray(m.get_xs(order_groups='increasing', row_column='inout')).reshape(G, G, n_l)
    return np.transpose(S_inout, (2, 1, 0))   # S[l, g, g'] = sigma_s^l(g' -> g), row = receiving group


sigma_t = vec('total')
sigma_a = vec('absorption')
inv_v = vec('inverse-velocity')
S_lib = legendre_matrix(mgxs_lib.get_mgxs(graphite, 'consistent nu-scatter matrix'), L_MAX + 1)

np.set_printoptions(linewidth=150)
print(f"\ncase {CASE}")
print("group edges (eV, descending):", group_edges[::-1])
print("sigma_t [1/cm]:", sigma_t)
print("sigma_a [1/cm]:", sigma_a)
print("1/v [s/cm]:", inv_v)

save = dict(case=CASE, L=L, W=W, E0=E0, density=DENSITY, temperature=TEMPERATURE, L_max=L_MAX,
            group_edges_desc=group_edges[::-1], sigma_t=sigma_t, sigma_a=sigma_a, inv_velocity=inv_v)

if CASE == 'P0tr':
    sigma_tr = vec('transport')
    S_raw = legendre_matrix(raw_p1, 2)                 # uncorrected sigma_s^0, sigma_s^1 (g' -> g)
    sigma_s1 = sigma_t - sigma_tr                      # in-scatter P1, what OpenMC's correction uses
    sigma_s1_out = S_raw[1].sum(axis=0)                # out-scatter P1 of each incoming group g' (reference)
    S0_tr = S_lib[0]                                   # library (corrected) P0 matrix
    print("sigma_tr [1/cm]:", sigma_tr)
    print("sigma_s1,g = sigma_t - sigma_tr (in-scatter) [1/cm]:", sigma_s1)
    print("check sum_g' sigma_s^1(g'->g) phi_g' / phi_g:", (S_raw[1] @ flux.sum(axis=0)) / flux.sum(axis=0))
    print("out-scatter sum_g sigma_s^1(g'->g) [1/cm]:", sigma_s1_out)
    print("mean cosine per source group (out-scatter / sigma_s0):", sigma_s1_out / S_raw[0].sum(axis=0))
    print("diag(S0_raw - S0_tr) (matrix correction):", np.diag(S_raw[0] - S0_tr))
    print("off-diagonal max |S0_raw - S0_tr|:", np.abs(S_raw[0] - S0_tr - np.diag(np.diag(S_raw[0] - S0_tr))).max())
    print("balance sigma_tr - sigma_a - sum_g S0_tr (should be ~0):", sigma_tr - sigma_a - S0_tr.sum(axis=0))
    print("balance sigma_t - sigma_a - sum_g S0_raw (should be ~0):", sigma_t - sigma_a - S_raw[0].sum(axis=0))
    save.update(sigma_tr=sigma_tr, sigma_s1=sigma_s1, sigma_s1_out=sigma_s1_out,
                scatter_legendre=S0_tr[None],          # what the MG run uses: isotropic, corrected
                scatter_legendre_raw=S_raw)            # uncorrected sigma_s^0, sigma_s^1
else:
    print("balance sigma_t - sigma_a - sum_g S0 (should be ~0):", sigma_t - sigma_a - S_lib[0].sum(axis=0))
    print("mean cosine per source group sum_g sigma_s^1 / sum_g sigma_s^0:", S_lib[1].sum(axis=0) / S_lib[0].sum(axis=0))
    print("||S_l||_F / ||S_0||_F:", " ".join(f"{np.linalg.norm(S_lib[l]) / np.linalg.norm(S_lib[0]):.1e}"
                                              for l in range(L_MAX + 1)))
    save.update(scatter_legendre=S_lib)

np.savez(OUT / 'graphite_mgxs.npz', **save)

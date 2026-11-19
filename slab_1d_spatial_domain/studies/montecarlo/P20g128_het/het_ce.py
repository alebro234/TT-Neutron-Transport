"""
Graphite slab with a He-3 detector gas region, step 1/2: continuous-energy OpenMC run that tallies the reference
flux and generates the multigroup cross sections of BOTH materials (domain_type = 'material').

Same problem as ../graphite/graphite_ce.py, plus a detector region:
Geometry : slab 0 < z < L, vacuum on z = 0 and z = L, REFLECTIVE planes at x, y = +-W (laterally infinite slab)
           graphite | He-3 (Z_DET < z < Z_DET + W_DET) | graphite
Source   : monoenergetic 2 MeV, monodirectional along +z, injected just inside z = 0
Materials: graphite, natural C (C0) + c_Graphite S(alpha,beta), 294 K, 1.7 g/cm^3   (as in graphite_ce.py)
           He-3 gas, pure He3, 294 K, 0.001 g/cm^3 (free gas, no S(alpha,beta))
Groups   : same structure as graphite_ce.py (G log-spaced groups, narrow top group around E0); default P20, G = 128

The flux tally mesh has N_z = 300 cells (h = 0.1 cm), so the detector faces are mesh edges and the TT solver
(het_tt_slab.py) uses the same cells with the material interfaces on cell edges.

Outputs (in results_<case>_g_<G>/):
  het_ce_flux.npz   CE reference scalar flux on the z mesh per group
  mgxs.h5           OpenMC MGXS library, xsdata 'graphite' and 'he3', read by het_mg.py
  het_mgxs.npz      same data in the TT-solver convention, stacked over materials c = 0 (graphite), 1 (He-3):
                    sigma_t, sigma_a, inv_velocity (C, G);  scatter_legendre (C, L+1, G, G),
                    scatter_legendre[c, l, g, g'] = sigma_s^l(g' -> g) (row = receiving group);
                    + region_map (N_z,), the problem parameters and the group edges (descending)
  ce_run/           OpenMC input/output files of this run

The He-3 MGXS are flux weighted over the 2 cm region only: the P_L scattering matrix of He-3 (Sigma_s ~ 1e-3 /cm)
will be statistically poor; it hardly matters for the transport (absorption dominates there).

Group convention (same as the TT solver): g = 0 is the HIGHEST energy group.

Run:  python het_ce.py [P<L>] [G] [particles per batch]     (defaults: P20, 128, 100000)
"""
import re
import sys
from pathlib import Path

import numpy as np
import openmc
import openmc.mgxs

HERE = Path(__file__).resolve().parent
CASE = sys.argv[1] if len(sys.argv) > 1 else 'P20'
assert re.fullmatch(r'P\d+', CASE), CASE
G = int(sys.argv[2]) if len(sys.argv) > 2 else 128
OUT = HERE / f'results_{CASE}_g_{G}'
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
L = 30.0                    # slab thickness [cm]
W = 50.0                    # lateral half-width [cm] (irrelevant with reflective sides)
Z_DET = 14.0                # left face of the He-3 region [cm]
W_DET = 2.0                 # thickness of the He-3 region [cm]
DENSITY = 1.7               # reactor-grade graphite [g/cm^3]
DENSITY_HE3 = 1.0e-3        # He-3 gas [g/cm^3]
TEMPERATURE = 294.0         # [K]
SAB = 'c_Graphite'          # thermal scattering table of graphite (None = free-gas C)
E0 = 2.0e6                  # source energy [eV]

# energy groups: edges in eV, ASCENDING (OpenMC convention); identical to graphite_ce.py
E_MIN = 1.0e-5
TOP_LO = 0.95 * E0
TOP_HI = 1.05 * E0
group_edges = np.append(np.logspace(np.log10(E_MIN), np.log10(TOP_LO), G), TOP_HI)

L_MAX = int(CASE[1:])       # Legendre order of the library scattering matrices

N_z = 300                   # mesh cells in z (h = 0.1 cm: the detector faces are mesh edges)
dz = L / N_z
assert all(abs(zb / dz - round(zb / dz)) < 1e-9 for zb in (Z_DET, Z_DET + W_DET)), "detector faces off the mesh"

PARTICLES = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000   # per batch
BATCHES = 50

# ---------------------------------------------------------------------------
# materials, geometry
# ---------------------------------------------------------------------------
graphite = openmc.Material(name='graphite')
graphite.add_nuclide('C0', 1.0)
graphite.set_density('g/cm3', DENSITY)
graphite.temperature = TEMPERATURE
if SAB is not None:
    graphite.add_s_alpha_beta(SAB)

he3 = openmc.Material(name='he3')
he3.add_nuclide('He3', 1.0)
he3.set_density('g/cm3', DENSITY_HE3)
he3.temperature = TEMPERATURE
materials = openmc.Materials([graphite, he3])
MATERIALS = [graphite, he3]                          # material index c in het_mgxs.npz / the TT region map

z_left = openmc.ZPlane(z0=0.0, boundary_type='vacuum')
z_right = openmc.ZPlane(z0=L, boundary_type='vacuum')
z_d0 = openmc.ZPlane(z0=Z_DET)
z_d1 = openmc.ZPlane(z0=Z_DET + W_DET)
x_lo = openmc.XPlane(x0=-W, boundary_type='reflective')
x_hi = openmc.XPlane(x0=W, boundary_type='reflective')
y_lo = openmc.YPlane(y0=-W, boundary_type='reflective')
y_hi = openmc.YPlane(y0=W, boundary_type='reflective')
lateral = +x_lo & -x_hi & +y_lo & -y_hi

cells = [openmc.Cell(name='graphite left', fill=graphite, region=+z_left & -z_d0 & lateral),
         openmc.Cell(name='he3 detector', fill=he3, region=+z_d0 & -z_d1 & lateral),
         openmc.Cell(name='graphite right', fill=graphite, region=+z_d1 & -z_right & lateral)]
geometry = openmc.Geometry(cells)

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
# MGXS library, one domain per material (both graphite cells share the graphite domain)
# ---------------------------------------------------------------------------
groups = openmc.mgxs.EnergyGroups(group_edges)

mgxs_lib = openmc.mgxs.Library(geometry)
mgxs_lib.energy_groups = groups
# 'consistent nu-scatter matrix' is what create_mg_library() writes as the scattering matrix;
# 'consistent scatter matrix' is needed with it to build the multiplicity (= 1 here)
mgxs_lib.mgxs_types = ['total', 'absorption', 'inverse-velocity', 'consistent scatter matrix',
                       'consistent nu-scatter matrix']
mgxs_lib.scatter_format = 'legendre'
if L_MAX > 10:
    # the Python MGXS API caps the order at 10 (input check only; LegendreFilter and the MG solver have no cap)
    openmc.mgxs.mgxs._MAX_LEGENDRE = L_MAX
    mgxs_lib._legendre_order = L_MAX
else:
    mgxs_lib.legendre_order = L_MAX
mgxs_lib.correction = None
mgxs_lib.domain_type = 'material'
mgxs_lib.domains = MATERIALS
mgxs_lib.by_nuclide = False
mgxs_lib.build_library()

# ---------------------------------------------------------------------------
# tallies: CE reference scalar flux (mesh in z) x energy groups, same as het_mg.py
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

z_centres = (np.arange(N_z) + 0.5) * dz
region_map = ((z_centres > Z_DET) & (z_centres < Z_DET + W_DET)).astype(int)    # 0 graphite, 1 He-3
np.savez(OUT / 'het_ce_flux.npz',
         z=z_centres, dz=dz, L=L, group_edges_desc=group_edges[::-1],
         phi=flux / dz, phi_sd=flux_sd / dz, E0=E0, region_map=region_map)

mg_library = mgxs_lib.create_mg_library(xs_type='macro', xsdata_names=[m.name for m in MATERIALS])
mg_library.export_to_hdf5(str(OUT / 'mgxs.h5'))

# ---------------------------------------------------------------------------
# post-processing: same data in the TT-solver convention, stacked over materials
# ---------------------------------------------------------------------------
def vec(mat, name):
    # order_groups='increasing' -> OpenMC group 1 (highest energy) first, i.e. g = 0 highest
    return np.asarray(mgxs_lib.get_mgxs(mat, name).get_xs(order_groups='increasing')).reshape(G)


def legendre_matrix(m, n_l):
    S_inout = np.asarray(m.get_xs(order_groups='increasing', row_column='inout')).reshape(G, G, n_l)
    return np.transpose(S_inout, (2, 1, 0))   # S[l, g, g'] = sigma_s^l(g' -> g), row = receiving group


sigma_t = np.stack([vec(m, 'total') for m in MATERIALS])
sigma_a = np.stack([vec(m, 'absorption') for m in MATERIALS])
inv_v = np.stack([vec(m, 'inverse-velocity') for m in MATERIALS])
S_lib = np.stack([legendre_matrix(mgxs_lib.get_mgxs(m, 'consistent nu-scatter matrix'), L_MAX + 1)
                  for m in MATERIALS])

np.set_printoptions(linewidth=150, precision=4)
print(f"\ncase {CASE}, G = {G}, materials {[m.name for m in MATERIALS]}")
for c, m in enumerate(MATERIALS):
    print(f"{m.name}: sigma_t [1/cm] g=0 {sigma_t[c, 0]:.4e}, g=G-1 {sigma_t[c, -1]:.4e};  "
          f"sigma_a g=G-1 {sigma_a[c, -1]:.4e}")
    print(f"  balance max |sigma_t - sigma_a - sum_g S0| / sigma_t:",
          np.nanmax(np.abs(sigma_t[c] - sigma_a[c] - S_lib[c, 0].sum(axis=0)) / sigma_t[c]))

np.savez(OUT / 'het_mgxs.npz', case=CASE, materials=[m.name for m in MATERIALS], L=L, W=W, E0=E0,
         z_det=Z_DET, w_det=W_DET, density=[DENSITY, DENSITY_HE3], temperature=TEMPERATURE, L_max=L_MAX,
         group_edges_desc=group_edges[::-1], sigma_t=sigma_t, sigma_a=sigma_a, inv_velocity=inv_v,
         scatter_legendre=S_lib, region_map=region_map, N_z=N_z)

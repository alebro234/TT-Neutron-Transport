# TT-Neutron-Transport

Tensor-train (TT) solvers for the time-dependent neutron transport equation, built on
[`torchtt`](https://github.com/ion-g-ion/torchTT). The full phase space
(space × polar angle × azimuthal angle × energy group × time) is discretised and solved as a single
linear system in TT format with AMEn.

## Repository layout

```
src/tt_transport/            Python package
    slab.py                  Slab-geometry model (SlabTransportModel), DG in space, backward Euler in time,
                             HG angular scattering, multigroup moderator model, TT I/O helpers
    integral_transforms.py   Numerical inverse Laplace/Fourier transforms (reference solutions)
studies/                     Numerical experiments (each script writes its outputs next to itself)
    convergence_study/       Space/time convergence against a manufactured solution
    pure_absorber/           Beam front and beam with source + initial condition in a pure absorber
    infinite_medium/         Decaying Gaussian source in an infinite medium vs. transform-based reference
notes/                       LaTeX notes (tt_neutron_transport.tex/.pdf), references and figures
```

## Setup

Requires Python ≥ 3.10.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

`pip install -e .` installs `tt_transport` in editable mode, so changes to `src/` are picked up
immediately and the study scripts can be run from any directory.

## Running the studies

Run each script from its own folder, so outputs (`.png`, `.npz`, `.pt`) land next to it:

```bash
cd studies/convergence_study && python manuf_sol.py
cd studies/pure_absorber     && python beam_front.py
cd studies/pure_absorber     && python beam_source_ic.py
cd studies/infinite_medium   && python decaying_gaussian_inf_medium.py
cd studies/infinite_medium   && python scalar_flux_plot.py
```

Saved TT solutions (`*.pt`) can be reloaded with `tt_transport.slab.load_tt(path)`.

## Notes

```bash
cd notes
latexmk -pdf tt_neutron_transport.tex
```

Figures used by the notes live in `notes/images/`. If you regenerate a figure in `studies/`, copy it
there.

# Particle-neighbour reference experiment

**Implemented:** offline 3-D SPH-inspired operators and a deterministic fixture.
**Observed on fixture:** dense-oracle agreement and structural controls below.
**Added 2026-09-13:** [10-scene numerical validation](NUMERICAL_VALIDATION.md),
independent dense RK4, frozen splits and reproducible reference snapshots.
**Box-contact follow-up:** [analytic impacts and coupled ledgers](BOUNDARY_VALIDATION.md),
with 9 validation cases and 33 tests in the combined suite.
**Planned:** physically validated fluid dynamics, learned prediction, ShardJEPA comparison and runtime adapters.

This module is outside the installed `cogniarc` packages and is not imported by
the agent or existing simulator. It uses only Python's standard library.
Run from the repository root with Python 3.12 or later:

```bash
python -m unittest discover -s tests -p test_particle_graph_reference.py -v
python -m experiments.particle_graph.benchmark --output experiments/particle_graph/evidence/2026-09-10.json
```

## Model and limits

Each particle has a 3-D position, velocity and positive mass. A spatial hash with
cell width `h` rebuilds undirected edges for distance strictly below `h`.
Self-density is included; self-edges are omitted; coincident particles have zero
pressure direction. The poly6 density, spiky pressure gradient and viscosity
Laplacian use the 3-D constants from
[Müller et al. (2003)](https://matthias-research.github.io/pages/publications/sca03.pdf).

Pressure is `max(0, stiffness * (density - rest_density))`: the zero floor is an
explicit non-tensile simplification. Symmetric pair forces are divided by each
particle's mass. Semi-implicit Euler uses a fixed time step and does not guarantee
stability for arbitrary inputs.

The dimensionless fixture uses 27 particles, 50 steps and `dt=0.002`. It has no
boundaries, gravity, surface tension, incompressibility constraint or calibration
to water. The separate gravity operator has a unit test. Visual resemblance and
passing tests are insufficient physical validation.

## Verification and reproduction

On 2026-09-10, **11 focused tests passed**. Neighbours are compared to all-pairs
Euclidean distances; operators to independently written directed dense sums.
Controls cover unequal masses, cutoff, negative coordinates, coincident
particles, force balance, instantaneous viscous dissipation, translation,
rotation, permutation, replay, momentum, graph rebuilding, gravity and invalid inputs.

The [raw report](evidence/2026-09-10.json) records seed, exact initial state,
parameters, source hashes, Python/OS/architecture, mass, momentum drift,
internal-force residual and final-state hash. Original code and synthetic data
use the repository MIT license; no external data are included.

Prediction RMSE is per coordinate against the generated trajectory at horizons
1, 10 and 50. Both baselines start at t=0: persistence holds initial position;
constant velocity extrapolates initial velocity without future observations.
These are reference errors, not learned-model scores or water-accuracy measures.
Exact output hashes are environment-specific; mathematical controls use tolerances.
No speed, memory, GPU or homelab benchmark was performed.

The contribution guide mentions `scripts/run_tests.py`, absent at the base
commit. The command above is the focused experiment gate; broader suite results
or blockers are recorded in the PR.

The original 2026-09-10 source and evidence remain unchanged. The current
combined suite passes 33 tests. The [numerical follow-up](NUMERICAL_VALIDATION.md)
and [box-contact follow-up](BOUNDARY_VALIDATION.md) record their protocols, raw
outputs and remaining R1 gates.

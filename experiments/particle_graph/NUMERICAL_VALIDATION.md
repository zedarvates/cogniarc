# Numerical validation and split snapshots — 2026-09-13

**Implemented:** a frozen scene manifest, an independent dense RK4 reference,
equal-time refinement comparisons and sparse snapshot export.
**Observed fixture:** all 10 synthetic scenes pass the numerical checks below;
the combined particle-graph suite passes 21 tests.
**Planned:** boundaries/contact, calibrated fluid validation, longer rollouts,
trained predictors and runtime integration. R1 is still partial.

## Reproduce

From the repository root, with Python 3.12 or later and no third-party packages:

```bash
python -m unittest discover -s tests -p 'test_particle_graph*.py' -v
python -m experiments.particle_graph.validation --output-dir experiments/particle_graph/evidence/2026-09-13
```

The second command exits nonzero if a scene fails its numerical gates and keeps
the failed gate values in the report. Invalid manifests or non-finite numerical
states raise an error; such runs do not produce a new validated artifact.

Source: [candidate](reference.py), [independent dense reference](dense_reference.py),
[protocol/runner](validation.py), [frozen manifest](scene_manifest.json),
[tests](../../tests/test_particle_graph_validation.py).
Raw artifacts: [validation.json](evidence/2026-09-13/validation.json) and
[snapshots.jsonl](evidence/2026-09-13/snapshots.jsonl).

The report records Python/OS/architecture, all parameters and scene recipes,
source/manifest SHA-256 values and the exact snapshot-file SHA-256. This execution
used CPython 3.12.14 on Linux x86_64. No hardware-performance claim is made.
Original code and synthetic data use the repository MIT license; no external
data, model weights or proprietary runtime source are included.

## What the comparison checks

The candidate uses semi-implicit Euler at dt = 0.002, 0.001 and 0.0005, with
50, 100 and 200 steps respectively. Every endpoint is at the same dimensionless
time T = 0.1. The reference uses independently written directed all-pairs sums
and classical RK4 at dt = 0.0005 and 0.00025. It does not call the candidate's
graph, force accumulation or integrator. An analytic ballistic-motion test
checks RK4 independently of the particle-force comparison.

The finer RK4 result supplies the numerical comparison target. The difference
between the two RK4 resolutions estimates reference sensitivity; it is not a
rigorous error bound. Both references solve the same simplified, clamped-pressure
equations as the candidate, so this is not an independent validation of water.

The fixed gates apply separately to endpoint position and velocity RMSE:

1. Errors must be finite, nonnegative and non-increasing with refinement, within
   relative tolerance 1e-10 and absolute tolerance 1e-12.
2. The finest error must be at most 75% of the coarsest error, or below 1e-12.
3. Reference-resolution disagreement must be at most 5% of the finest candidate
   error, or below 1e-10.
4. Mass error must be at most 1e-12; momentum error, accounting for the applied
   external gravity impulse, at most 1e-10.

These numerical tolerances were fixed before the 10-scene run. They are not
water-accuracy thresholds or model-selection criteria.

## Observed position error

RMSE is per coordinate at T = 0.1, relative to the finer dense RK4 reference.
All quantities are dimensionless. The raw report also contains velocity errors,
reference discrepancies, conservation residuals and each individual gate.

| Scene | Split | Particles | dt 0.002 | dt 0.001 | dt 0.0005 |
| --- | --- | --- | --- | --- | --- |
| train-01 | train | 8 | 1.623667e-05 | 8.117772e-06 | 4.058745e-06 |
| train-02 | train | 8 | 2.360246e-05 | 1.180070e-05 | 5.900221e-06 |
| train-03 | train | 12 | 1.593963e-05 | 7.969208e-06 | 3.984453e-06 |
| validation-01 | validation | 8 | 1.589789e-05 | 7.948190e-06 | 3.973907e-06 |
| validation-02 | validation | 12 | 2.357921e-05 | 1.178923e-05 | 5.894525e-06 |
| test-seed | test | 8 | 1.642510e-05 | 8.212253e-06 | 4.106053e-06 |
| test-count-18 | test | 18 | 2.292112e-05 | 1.146019e-05 | 5.730002e-06 |
| test-count-27 | test | 27 | 1.431194e-05 | 7.155715e-06 | 3.577794e-06 |
| test-viscosity | test | 12 | 2.393752e-05 | 1.196727e-05 | 5.983263e-06 |
| test-stiffness | test | 8 | 6.350318e-05 | 3.174806e-05 | 1.587316e-05 |

On these cases the position error falls by approximately two when dt is halved,
consistent with first-order convergence of the candidate. This does not prove
stability for larger time steps, different scenes or longer durations.

## Snapshot schema and reserved scenes

The corpus contains 10 UTF-8 JSON lines, one complete scene per line. Its size is
68,595 bytes; the report supplies the file SHA-256. Training has 3 scenes,
validation 2 and test 5. Seeds and IDs are unique; generated initial states are
checked for duplication. Particle counts 18 and 27 occur only in test, along
with reserved viscosity and stiffness configurations. These are reserved
conditions for future learning, not evidence of learned generalisation.

Each line has schema `cogniarc.particle-graph-snapshots.v1` and these fields:

| Field | Meaning |
| --- | --- |
| scene_id, split, condition, seed | Stable scene identity, declared partition, condition and generator seed |
| config, gravity, units | All four SPH parameters, constant external acceleration and dimensionless units |
| particle_ids, masses | Stable per-scene ordering and mass for each particle |
| integrator, internal_dt | `independent_dense_rk4` and 0.00025 for this manifest |
| snapshots | Four states at coarse steps 0, 1, 10 and 50; each has step, time, positions and velocities |

Positions and velocities are arrays of N three-component vectors, aligned with
particle_ids and masses. Step zero is the exact generated initial state. These
are sparse snapshots, not all intermediate integration steps. The manifest is
validated by `validate_manifest`; the small end-to-end test checks snapshot
identity, ordering, time alignment and conservation.

Persistence and constant-velocity baselines use only step zero and are scored
at steps 1/10/50 against the dense reference. No model has been trained or tuned
on any split. Future affine/model fitting must use training scenes only;
configuration selection belongs to validation, with test reported separately.

The equations still omit boundaries, contact, surface tension and
incompressibility. No runtime speedup, water accuracy, production suitability
or neural capability follows from these results. The underlying SPH inspiration
and primary-source attribution remain in the [experiment overview](README.md).

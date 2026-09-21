# Paired material baseline — 2026-09-13

**Implemented:** a second, grouped synthetic corpus and a ridge model with
material–state interactions, compared with an otherwise identical material-blind
refit and the unchanged v1 model. **Measured on synthetic fixtures:** lower mean
error, but regressions on some reserved material/particle-count combinations.
**Planned:** autoregressive error growth, stronger geometry representations and
any ShardJEPA/runtime adaptation. This is offline CogniARC code.

At step 50, the material model lowers mean scene position RMSE by **20.29%**
relative to the blind v2 refit and **14.75%** relative to the frozen v1 model.
However, it increases mean position RMSE by **10.89% on 27-particle scenes** and
**31.57% for the joint-interpolation material**, relative to the blind v2 refit.
The average gain does not establish improvement across all held-out conditions.

## Protocol committed before data generation and test scoring

The [manifest](material_manifest.json) and [protocol](material_protocol.json) were
committed at [0751eb36ca1b01d7a1cc2f1484f49cd563d900e9](https://github.com/zedarvates/cogniarc/commit/0751eb36ca1b01d7a1cc2f1484f49cd563d900e9)
before implementation, generation, model selection or test scoring. The protocol
fixes material values, initial-condition seeds, splits, numerical tolerances,
feature maps, regularization candidates, comparators and paired-change metrics.
No feature, candidate grid or model was changed after the reserved scores.

The [v1 experiment](AFFINE_BASELINE.md), its implementation, manifest, model and
raw evidence remain unchanged. Its model is evaluated on the new corpus as a
fixed comparator; its earlier 41.11% result belongs to the earlier corpus and
must not be compared directly with a percentage from this new test set.

## Reproduction and controls

From the repository root with CPython 3.12+ and no third-party dependencies:

```bash
python -m unittest discover -s tests -p 'test_particle_graph*.py' -v
python -m experiments.particle_graph.materials generate --output-dir experiments/particle_graph/evidence/2026-09-13-materials
python -m experiments.particle_graph.materials fit --data-dir experiments/particle_graph/evidence/2026-09-13-materials --output experiments/particle_graph/evidence/2026-09-13-materials/models.json
python -m experiments.particle_graph.materials evaluate --data-dir experiments/particle_graph/evidence/2026-09-13-materials --model experiments/particle_graph/evidence/2026-09-13-materials/models.json --output experiments/particle_graph/evidence/2026-09-13-materials/evaluation.json
```

The recorded run used CPython 3.12.14 on Linux x86_64. **57 focused tests passed**,
including 12 added controls for grouped partitions, paired initial states,
material labels, analytic acceleration recovery, numerical checks, reader
integrity, validation isolation and evaluation without fitting. A regression
control verifies that the blind refit is exactly the unchanged v1 fit algorithm.
The full CogniARC repository suite is not claimed to pass.

Sources: [generator and reader](material_data.py), [fitter and evaluator](materials.py),
[new controls](../../tests/test_particle_graph_materials.py), and the unchanged
[affine solver](affine.py), [dense reference](dense_reference.py) and
[numerical helpers](validation.py). Generator failures retain their numerical
report; the fitter/evaluator reject failed numerical gates or inconsistent
source/input hashes. The fit command reads train and validation snapshot files
only. Both selected models are saved before the separate test command runs.

## Grouped scenes and controlled material changes

Each initial condition is replayed with four materials. Position, velocity,
mass and gravity at t=0 are identical within that group; only stiffness and/or
viscosity change. All variants of a group stay in the same partition. Seeds are
unique between groups and absent from v1; identical initial states cannot cross
group boundaries. The files contain 36 scene records but only nine initial
conditions, of which three are reserved for test.

| Partition | Initial groups | Material variants per group | Scene records | Particle instances |
| --- | --- | --- | --- | --- |
| Train | 4 | 4 | 16 | 160 |
| Validation | 2 | 4 | 8 | 80 |
| Test | 3 | 4 | 12 | 212 |

Training and validation use 8 or 12 particles. Test initial conditions have 8,
18 and 27 particles; counts 18 and 27 do not occur in development. All scenes
use the same jittered lattice family, mass 0.25, smoothing radius 0.85 and rest
density 1.0. Gravity is either zero or (0, -0.3, 0). There are no walls.

| Partition / material condition | Stiffness | Viscosity |
| --- | --- | --- |
| Train, full 2 × 2 grid | 0.075 or 0.30 | 0.02 or 0.08 |
| Validation, full 2 × 2 grid | 0.10 or 0.20 | 0.03 or 0.06 |
| Test base | 0.15 | 0.04 |
| Test joint interpolation | 0.225 | 0.05 |
| Test stiffness extrapolation | 0.60 | 0.04 |
| Test viscosity extrapolation | 0.15 | 0.20 |

Validation and interpolation test materials lie within the training parameter
ranges. Stiffness/viscosity extrapolation changes only that parameter from the
test base material, keeping the other fixed. Material configurations are
distinct between partitions. The repeated variants and horizons are correlated;
12 test scenes or 36 scene/horizon pairs are not independent trials.

## Numerical targets

The unchanged independent dense RK4 implementation runs at dt = 0.0005 and
0.00025, taking 200 and 400 steps to reach T = 0.1. Both are compared at coarse
steps 1, 10 and 50; the finer trajectory supplies sparse targets. These checks
measure resolution sensitivity of the same simplified SPH equations, not a
new validation against physical water or another fluid model.

All 36 scenes pass the frozen checks at every horizon:

| Check | Maximum observed | Allowed maximum |
| --- | --- | --- |
| Position difference between RK4 resolutions, per-coordinate RMSE | 6.454515e-14 | 1e-9 |
| Velocity difference between RK4 resolutions, per-coordinate RMSE | 1.982196e-12 | 1e-9 |
| Mass error | 0 | 1e-12 |
| Momentum error after accounting for gravity | 1.190405e-16 | 1e-10 |

Raw values and per-horizon gates are in
[numerics.json](evidence/2026-09-13-materials/numerics.json). No Euler refinement
or wall-contact accuracy result is added by this experiment; earlier proofs
remain separate.

## Models and selection

The blind v2 model uses the original nine features: scene-centred initial
position, scene-centred initial velocity, and known gravity. The material model
adds six products: stiffness times each centred position coordinate, and
viscosity times each centred velocity coordinate. It is affine in **15 expanded
features**, and bilinear in the raw material/state inputs. It does not observe
neighbours, density, simulator forces, future states, IDs, split or seed.

Both models predict the same six targets at each horizon:
`(x_t-x_0-t*v_0)/t^2` and `(v_t-v_0)/t`. The residuals are added to constant-velocity
extrapolation. Every prediction starts at t=0; these are direct horizon maps,
not learned integration steps or autoregressive rollouts. There are 60 affine
coefficients/biases per horizon for the blind model and 96 for the material model.

Both fits use the same 16 training scenes and the same five-alpha selection
budget. Training-only weighted means and standard deviations normalize features.
Each scene has equal weight, and each particle within it equal weight; four
variants per group consequently give groups equal weight. The intercept is
unpenalized. The v1 ridge solver and its constant-column handling are reused.

Selection uses the v1 score: the square root of the mean, over validation scenes
and horizons, of `((position_RMSE/t^2)^2 + (velocity_RMSE/t)^2)/2`. One alpha is
selected per model across the three horizons; exact ties prefer larger alpha.
There is no refit on validation and no comparison-based automatic activation.

| Alpha | Blind v2 validation score | Material v2 validation score |
| --- | --- | --- |
| 0.000001 | 0.0601572089 | **0.0281397864** |
| 0.0001 | 0.0601472200 | 0.0281398733 |
| 0.01 | **0.0591800435** | 0.0282032630 |
| 1 | 0.0759156956 | 0.0573200032 |
| 100 | 0.1482980265 | 0.1471919482 |

The two smallest material-model scores are very close. Alpha 1e-6 is the winner
of this fixed grid, not evidence of a generally optimal setting; the grid was
not expanded after looking at test scores. The saved artifact includes all
candidate validation errors, coefficient hashes and the two selected models.

## Reserved prediction errors

The primary comparison is material v2 against blind v2, which shares its data
and selection budget. Frozen v1 is an additional unchanged reference. Persistence,
constant velocity and known-gravity ballistic prediction use the definitions
from [v1](AFFINE_BASELINE.md). RMSE is per coordinate; reported aggregates are
arithmetic means of scene RMSEs, not pooled particle errors. Lower is better.

| Step | Frozen v1 position RMSE | Blind v2 position RMSE | Material v2 position RMSE | Material v2 velocity RMSE |
| --- | --- | --- | --- | --- |
| 1 | 3.180154e-07 | 3.399776e-07 | 2.727318e-07 | 2.726922e-04 |
| 10 | 3.175659e-05 | 3.395365e-05 | 2.719952e-05 | 2.715735e-03 |
| 50 | 7.863229e-04 | 8.409680e-04 | 6.703571e-04 | 1.327258e-02 |

At step 50, blind v2 velocity RMSE is 1.669094e-2 and frozen v1 velocity RMSE is
1.560581e-2. Material v2 lowers those means by 20.48% and 14.95%, respectively.
Known-gravity ballistic position/velocity RMSE is 1.406582e-3 / 2.792610e-2.
All six methods at all three horizons are recorded in
[evaluation.json](evidence/2026-09-13-materials/evaluation.json).

Material v2 beats blind v2 on 27 of 36 scene/horizon pairs, frozen v1 on 24 of
36, and ballistic prediction on all 36, for both position and velocity. The
blind v2 refit itself has worse mean error than frozen v1 on this test set.
Extra training data alone did not improve this baseline here.

### Regressions retained

The following means use step-50 position RMSE. Percentage change is relative to
blind v2; negative denotes lower error and positive denotes regression.

| Material condition | Blind v2 | Material v2 | Change |
| --- | --- | --- | --- |
| Base interpolation | 5.439363e-04 | 3.616124e-04 | -33.52% |
| Joint interpolation | 4.076910e-04 | 5.363871e-04 | **+31.57%** |
| Stiffness extrapolation | 1.834984e-03 | 1.422789e-03 | -22.46% |
| Viscosity extrapolation | 5.772613e-04 | 3.606398e-04 | -37.53% |

| Particle count | Blind v2 | Material v2 | Change |
| --- | --- | --- | --- |
| 8 | 7.623969e-04 | 2.878436e-04 | -62.24% |
| 18 | 8.812854e-04 | 7.482375e-04 | -15.10% |
| 27 | 8.792218e-04 | 9.749901e-04 | **+10.89%** |

At step 50, the individual regressions against blind v2 are the joint-interpolation
material with 18 or 27 particles, and high stiffness with 27 particles. The raw
report retains these cases and the corresponding velocity errors. Neither
interpolation within parameter ranges nor a lower overall mean guarantees an
improvement when geometry/count also changes.

### Paired material response

For each test geometry, the evaluator subtracts the base-material trajectory
from each of the three other material trajectories, for targets and predictions
separately. It measures RMSE between those changes and records the true change's
magnitude. This isolates the model's response to material inputs at fixed initial
state. Blind models and physical extrapolation baselines give exactly zero
response, so their change error equals the true change's RMS magnitude.

Material v2 has lower change error in all 27 paired contrasts, for position and
velocity. This narrower result can coexist with worse absolute prediction in
some scenes: improved response to a parameter change does not remove errors in
the base trajectory. These are correlated contrasts over only three geometries,
not proof of general physical parameter identification.

## Artifact contract and provenance

Original code and generated data use the repository MIT license; no external
dataset, pretrained weights or ShardJEPA runtime are included. The intended use
is offline comparison on a small, dimensionless, synthetic SPH corpus.

| Artifact | Bytes | SHA-256 |
| --- | --- | --- |
| [train.jsonl](evidence/2026-09-13-materials/train.jsonl) | 95,580 | `80882080552ff731f21d20f2217ee7dc54c0b1a0dbb560a5f6e8ee6802466519` |
| [validation.jsonl](evidence/2026-09-13-materials/validation.jsonl) | 47,926 | `00c7cdf2301afd7b58859ccd2a28afd8318253f3a937c05bce1982f0a31fdec2` |
| [test.jsonl](evidence/2026-09-13-materials/test.jsonl) | 120,239 | `0fba680c2082a87353b1d24ba7e29f61cc61725e9d9edb1c82fbfff4e84dc988` |
| [numerics.json](evidence/2026-09-13-materials/numerics.json) | 65,121 | `f4a704bd2639234da3400e05589df5865a0244c289be4d3ce306cb1696c6202b` |
| [models.json](evidence/2026-09-13-materials/models.json) | 85,016 | `698d24c09f77eb31625e0defda1896b5d8a2ae60c6c7332fc8f0c445cc8bf67d` |
| [evaluation.json](evidence/2026-09-13-materials/evaluation.json) | 94,461 | `7f9efa0e8a96668cacc76d509d38c9bd85bfc9dfbe7ddf87b3b58845b8b5ea56` |

The snapshot schema is `cogniarc.particle-material-snapshots.v2`. Each UTF-8 JSON
line is one complete scene with the following fields, matching `read_split`:

| Fields | Meaning |
| --- | --- |
| scene_id, group_id, material_id, split, seed | Variant identity and the group retained within one partition |
| geometry_condition, material_condition | Declared held-out conditions; these are metadata, not model inputs |
| config, gravity, units | Four SPH parameters, constant acceleration, dimensionless units |
| particle_ids, masses, initial_state_sha256 | Per-scene particle ordering, masses and exact initial state identity |
| integrator, internal_dt | Independent dense RK4, internal dt 0.00025 |
| snapshots | Steps 0/1/10/50 with time and N × 3 position/velocity arrays |

`read_split` checks bytes, SHA-256, record hash, scene order, group/material
metadata against the manifest, generated initial state, IDs, masses, vector
shape, finite values and time alignment. The numerical report uses schema
`cogniarc.particle-material-numerics.v2` with per-scene/horizon resolution and
conservation checks, record/file hashes and provenance. The model schema is
`cogniarc.particle-material-model.v2`, with both variants, all selection scores,
coefficients and development group/seed/initial-state identities. Evaluation
schema `cogniarc.particle-material-evaluation.v2` contains every prediction
error, grouped means and paired changes. Model/evaluation provenance binds the
numerical report, corpus, frozen v1 model, source files, protocol and manifest.
Exact hashes are execution-environment specific; numerical controls use tolerances.

## Limits and next bounded experiment

Only short direct predictions and one lattice family have been evaluated. No
walls, incompressibility, physical calibration, actions, neural/JEPA model,
uncertainty calibration, conservation guarantee, speed or memory advantage is
established. The six interaction features are a small hand-designed basis and
do not encode local density or neighbour geometry.

Keep v1 and v2 frozen. Next, predeclare a longer autoregressive evaluation that
feeds predicted states back into the step-1 maps, alongside the numerical solver
and physical baselines. Report error growth, conservation and failures separately
by material and particle count, including the 27-particle regressions. Any change
to geometry features needs its own future protocol and fresh evaluation scenes;
the current test results must not become a tuning set.

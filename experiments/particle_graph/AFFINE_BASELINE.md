# Direct affine baseline — 2026-09-13

**Implemented:** standard-library ridge regression with training-only fitting,
validation-only regularization selection, a saved model and a separate test
evaluator. **Measured on synthetic fixtures:** lower position and velocity RMSE
than persistence, constant velocity and known-gravity ballistic prediction in
all five reserved scenes at each of three horizons. **Planned:** material-aware
models, autoregressive rollouts, neural/JEPA comparison and runtime integration.

This is an external offline CogniARC baseline, not the ShardJEPA task-local
runtime. The test scenes contain no walls. No physical water accuracy, broad
generalisation, statistical significance or compute advantage is established.

## Frozen protocol and reproduction

The [protocol](affine_protocol.json) was committed at
[5e39eb67db3526eeea63d525ba7b9aeafc4b0677](https://github.com/zedarvates/cogniarc/commit/5e39eb67db3526eeea63d525ba7b9aeafc4b0677)
before the predictor implementation and test scoring. It fixes features,
normalization, targets, five regularization candidates, selection score,
baselines and aggregation. The existing manifest and snapshot bytes are unchanged.
Their numerical results were already public: the test set is reserved from
fitting and selection, not a blind benchmark.

From the repository root, with CPython 3.12+ and its standard library:

```bash
python -m unittest discover -s tests -p 'test_particle_graph*.py' -v
python -m experiments.particle_graph.affine fit --output experiments/particle_graph/evidence/2026-09-13-affine/model.json
python -m experiments.particle_graph.affine evaluate --model experiments/particle_graph/evidence/2026-09-13-affine/model.json --output experiments/particle_graph/evidence/2026-09-13-affine/evaluation.json
```

The recorded run used CPython 3.12.14 on Linux x86_64. The fitting command saved
the selected coefficients before the evaluation command ran. No test-dependent
refitting, feature changes or candidate-grid changes followed the result.
The test evaluator does not call the fitter. A worse score would still be
reported: there is no tuned success threshold or omitted failing scene.

Source and reader: [affine.py](affine.py). Controls:
[test_particle_graph_affine.py](../../tests/test_particle_graph_affine.py).
The combined particle-graph suite passed **45 tests**, including 12 new controls
for analytic ridge recovery, collinearity, scene weighting, partition rejection,
validation isolation, no-refit evaluation, checksum failures, translation,
velocity boosts and permutation. These are focused checks; the complete
CogniARC repository suite is not claimed to pass.

## Observations, targets and fitting

Only the initial position, initial velocity and declared constant gravity are
available to prediction. Each particle supplies nine features: its three
position coordinates relative to the scene mean, three velocity coordinates
relative to the scene mean, and three gravity components. Means use the observed
initial scene only. IDs, split, seed, material parameters, neighbours, density,
simulator forces and future states are excluded from model inputs.

For each horizon at time t, six regression targets are formed on training:

```text
position residual = (x(t) - x(0) - t*v(0)) / t^2
velocity residual = (v(t) - v(0)) / t
```

A separate affine map predicts these residuals at each of steps 1, 10 and 50
(times 0.002, 0.02 and 0.1). Adding them back to constant-velocity extrapolation
gives position and velocity. Each horizon starts again at t=0; this is direct
horizon regression, not a sequence of learned integration steps.

Three training scenes contain 28 particles. Training feature means and population
standard deviations use row weight `1 / (3 * particles_in_scene)`. Constant
columns use scale 1. Weighted squared error plus `alpha * squared_coefficient_norm`
is minimized independently per output; the intercept is unpenalized. The small
positive-definite systems use a Cholesky solve. There are 60 fitted affine
parameters per horizon, including coefficients of constant columns that remain
zero. The maps are not constrained to conserve momentum or to rotate equivariantly.

Two validation scenes contain 20 particles. One alpha is chosen for all horizons
by minimizing the square root of the mean, over validation scenes and horizons,
of `((position_RMSE/t^2)^2 + (velocity_RMSE/t)^2) / 2`. RMSE is per coordinate.
This prevents large scenes or long horizons from dominating selection. Exact
ties prefer the larger alpha. The chosen model is **not** refitted on validation.

| Alpha | Validation selection score |
| --- | --- |
| 0.000001 | 0.0255944119 |
| **0.0001** | **0.0255936787** |
| 0.01 | 0.0256090073 |
| 1 | 0.0799995194 |
| 100 | 0.1436631132 |

The first three scores are close; this small validation set does not justify
claiming a generally optimal regularization setting. The saved artifact includes
every candidate's validation errors and fitted-coefficient hash.

## Test results

Five reserved test scenes contain 73 particles. All methods start from the same
initial observation. Persistence holds position and velocity; constant velocity
moves position linearly and holds velocity; the additional known-gravity
baseline uses `x=x0+t*v0+0.5*t^2*g`, `v=v0+t*g`. This stronger comparator prevents
attributing a gain from gravity alone to learned particle interactions.

The table is the arithmetic mean of five per-scene, per-coordinate RMSEs, not
a pooled particle RMSE. Lower is better; all quantities are dimensionless.

| Step | Method | Position RMSE | Velocity RMSE |
| --- | --- | --- | --- |
| 1 | Persistence | 6.119233e-05 | 5.675918e-04 |
| 1 | Constant velocity | 5.676234e-07 | 5.675918e-04 |
| 1 | Known-gravity ballistic | 5.064145e-07 | 5.063764e-04 |
| 1 | Affine | 2.982931e-07 | 2.982745e-04 |
| 10 | Persistence | 6.107571e-04 | 5.666310e-03 |
| 10 | Constant velocity | 5.670040e-05 | 5.666310e-03 |
| 10 | Known-gravity ballistic | 5.056751e-05 | 5.052347e-03 |
| 10 | Affine | 2.979112e-05 | 2.976737e-03 |
| 50 | Persistence | 3.380669e-03 | 2.799186e-02 |
| 50 | Constant velocity | 1.407472e-03 | 2.799186e-02 |
| 50 | Known-gravity ballistic | 1.252737e-03 | 2.487859e-02 |
| 50 | Affine | 7.376996e-04 | 1.463711e-02 |

At step 50, mean position RMSE is 41.11% lower and mean velocity RMSE is 41.17%
lower than known-gravity ballistic prediction. Both errors are lower than all
three baselines in all 15 scene/horizon pairs. Those pairs are correlated
measurements of only five scenes, not 15 independent trials.

| Test scene at step 50 | Particles | Ballistic position RMSE | Affine position RMSE | Affine velocity RMSE |
| --- | --- | --- | --- | --- |
| Unseen seed | 8 | 8.205520e-04 | 1.148083e-04 | 2.302582e-03 |
| Reserved count 18 | 18 | 7.464424e-04 | 4.081602e-04 | 8.116867e-03 |
| Reserved count 27 | 27 | 7.142336e-04 | 5.398866e-04 | 1.071487e-02 |
| Reserved viscosity | 12 | 8.068310e-04 | 2.198859e-04 | 4.385830e-03 |
| Reserved stiffness | 8 | 3.175628e-03 | 2.405757e-03 | 4.766538e-02 |

The stiffness case has the largest remaining error. All development scenes use
one material configuration and the affine model has no material inputs. It
cannot identify the effect of changing those parameters; outperforming weak
extrapolation here is not evidence of learning their physical dependence.

## Artifacts, reader and provenance

All original code and synthetic data use the repository MIT license. No external
data, model weights or ShardJEPA runtime are included. The intended use is a small
offline baseline comparison against the existing simplified SPH reference.

| Artifact | Bytes | SHA-256 |
| --- | --- | --- |
| [model.json](evidence/2026-09-13-affine/model.json) | 15,431 | `1d179137c38de6d032a552e3973fc4b0fe923062dcc2ada15743ad710a8c7ccd` |
| [evaluation.json](evidence/2026-09-13-affine/evaluation.json) | 14,047 | `0fb4907e324af72a71cc76cd032e067d23c66dd7a6053e38bcb92fc60f894ba1` |
| [snapshots.jsonl](evidence/2026-09-13/snapshots.jsonl) | 68,595 | `27b525fdec468839dd1bbdf1fdb715d944015732caf67c5158a43845ceb1d204` |

The model schema is `cogniarc.particle-affine-model.v1`: selected alpha, feature
normalizers, six output coefficient rows and biases per horizon, candidate
validation scores/errors, fit hashes, development scene IDs and provenance.
The evaluator checks source/protocol/corpus hashes and that the saved model
matches the validation selection. The evaluation schema is
`cogniarc.particle-affine-evaluation.v1`: all test scene/horizon metrics, macro
means and model/input/source hashes with command and environment.

`read_corpus` verifies exact input bytes, manifest identity, split metadata,
particle IDs, masses, physical configuration, vector shape, finite values and
snapshot time alignment; the fitter receives development records only. The
deterministic target generator, numerical validation, complete snapshot schema
and prior license/provenance are in [NUMERICAL_VALIDATION.md](NUMERICAL_VALIDATION.md).
Historical source hashes still match. Numerical agreement between the two
simplified solvers is not calibrated water accuracy.

## Next experiment

Keep this protocol and result frozen. Before changing the feature set, freeze a
new corpus version with material variation in training/validation and new test
seeds; compare a material-conditioned affine model with this baseline. Then
measure autoregressive error growth and conservation on longer trajectories.
No learned rollout stability, wall prediction, physical calibration, latency,
memory saving or production-readiness claim follows from the current experiment.

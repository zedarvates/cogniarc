# Frozen-model autoregressive evaluation — 2026-09-13

**Implemented:** feedback of frozen step-1 predictions over 500 steps, independent
reference-resolution checks and explicit failure/conservation records.
**Measured on synthetic fixtures:** results below distinguish finite completion,
prediction error and momentum conservation. **Planned:** any corrected predictor,
physical water validation and ShardJEPA/runtime adaptation.

## Frozen experiment and scope

The [protocol](rollout_protocol.json) was committed at
[3737d3e256acc654954b0d0a5359b20096cc0c8b](https://github.com/zedarvates/cogniarc/commit/3737d3e256acc654954b0d0a5359b20096cc0c8b)
before the runner, measurements and results. It pins the v1/v2 model files,
material manifest, prior numerical report and test corpus by SHA-256, and fixes
time steps, checkpoints, reference tolerances, diagnostics and failure handling.
All earlier model implementations, coefficients, manifests and raw evidence
remain unchanged. No training, model selection, feature changes, correction,
clipping or parameter adjustment is performed in this experiment.

The 12 existing v2 material variants come from only three initial-condition
groups with 8, 18 or 27 particles. Each group is tested with the base,
joint-interpolation, high-stiffness and high-viscosity material. These previously
inspected fixtures are a temporal extension, not a new blind benchmark. Material
variants and checkpoints are correlated observations, not independent trials.
All quantities are dimensionless, and the corpus contains no walls.

Time advances in increments of 0.002 for 500 steps to T = 1.0, ten times the
maximum duration of the direct-horizon experiments. Checkpoints are 1, 10, 50,
100, 250 and 500 (times 0.002, 0.02, 0.1, 0.2, 0.5 and 1.0).

## Reproduce and inspect

From the repository root with CPython 3.12+ and its standard library:

```bash
python -m unittest discover -s tests -p 'test_particle_graph*.py' -v
python -m experiments.particle_graph.rollouts --output-dir experiments/particle_graph/evidence/2026-09-13-rollouts
```

The run uses CPython 3.12.14 on Linux x86_64. No timing, memory, GPU or cross-machine
performance result is claimed. The combined particle-graph suite passes **66
focused tests**. The nine added controls cover analytic acceleration over 500
feedback steps, immutable snapshots, step-1-only prediction, exclusion of future
observations, unequal-mass ledgers, first-failure handling, particle loss,
reference qualification, input hashes and aggregation without survivor-only
means. The full CogniARC repository suite is not claimed to pass.

Source: [rollouts.py](rollouts.py). Controls:
[test_particle_graph_rollouts.py](../../tests/test_particle_graph_rollouts.py).
The runner reuses the unchanged [prediction functions](materials.py),
[affine solver/model reader](affine.py), [SPH Euler solver](reference.py),
[dense RK4](dense_reference.py) and [v2 corpus reader](material_data.py).

The command writes the raw evaluation and reference snapshots. It returns
nonzero if a trajectory aborts or a numerical reference is unqualified. Momentum
diagnostic failures do not change the exit status: some declared baselines omit
gravity by definition, and a completed measurement must retain those failures.
A zero exit status does not mean that a predictor is accurate or conserves
momentum.

## Methods

| Method | What advances through time |
| --- | --- |
| Persistence | Exact fixed initial position and velocity |
| Constant velocity | Exact initial-state linear extrapolation, without gravity |
| Known-gravity ballistic | Exact constant-gravity formula from the initial state |
| SPH Euler | The unchanged known-equation semi-implicit Euler solver at dt 0.002 |
| Frozen v1 | The saved v1 step-1 affine map, called 500 times |
| Blind v2 | The saved material-blind v2 step-1 map, called 500 times |
| Material v2 | The saved v2 map with material–state interactions, called 500 times |

The three learned maps receive only their own current predicted positions and
velocities, plus fixed gravity/material observations. Scene centring is
recomputed from those predictions. The learned step-10 and step-50 maps are
never used. There is no future reference state input or reset. Masses and the
particle ordering remain fixed inputs, not predicted outputs. The SPH comparator
knows the simplified equations; its score is not a learned-model score or a
measurement of the relative cost of simulation and prediction.

## Reference qualification

The independent dense RK4 implementation takes 2,000 steps at dt 0.0005 and
4,000 at dt 0.00025. The finer states provide numerical targets. At each
checkpoint, both resolutions must be present, satisfy the mass/momentum checks,
and agree within the predeclared tolerance. At steps 1/10/50, the finer states
must also reproduce the previously committed v2 snapshots exactly.

For position and velocity separately, the allowed per-coordinate RMSE between
resolutions is `max(1e-12, min(absolute_cap, 0.01 * ballistic_RMSE))`, with absolute
caps 1e-7 for position and 1e-6 for velocity. Both resolutions must have mass
error at most 1e-12 and gravity-accounted momentum error at most 1e-10. These
longer-horizon tolerances were fixed before measurement; raw gaps and actual
tolerances are retained at every checkpoint. Resolution disagreement estimates
sensitivity of the same simplified equations, not physical water accuracy or a
rigorous error bound.

## Diagnostics and failed trajectories

The evaluator records per-coordinate position/velocity RMSE, state hashes,
completion status and these diagnostics at every checkpoint:

- Total momentum compared with `P0 + M*g*t`; the diagnostic threshold is 1e-8.
- Centre of mass compared with `c0 + mean_velocity*t + 0.5*g*t^2`.
- Mass error, with a threshold of 1e-12. The maps carry the observed masses;
  zero mass error by construction is not a learned conservation result.
- Kinetic energy and its difference from the numerical target. Kinetic energy
  is not expected to be conserved under gravity, pressure and viscosity.

A non-finite state, changed particle count/vector shape, or position/velocity
component above the fixed absolute bound of 1e6 stops that trajectory at its
first invalid step. The report retains the reason, failed step and earlier
checkpoints. This is a numerical guard, not a physical stability threshold.
There is no repair, restart or silently substituted prediction.

Mean errors weight each scene equally. Every aggregate also shows the number of
expected scenes, available predictions, qualified references and passed mass/
momentum diagnostics. If any member has failed or has an unqualified reference,
the aggregate mean prediction errors are null. The report does not hide failures
behind means computed only on surviving scenes. Conservation and completion are
reported independently of prediction accuracy; there is no tuned accuracy gate.

## Measured results and retained failures

All **84 comparison trajectories** (12 scenes × 7 methods) finish 500 steps
without hitting the numerical abort guard. All 72 scene/checkpoint references
qualify, and the 36 available short-prefix checks match v2 exactly. Maximum
RK4 resolution disagreement is 6.426019e-10 for position and 1.848105e-9 for
velocity; maximum reference momentum error is 2.012570e-15.

The mean position RMSE below weights each of the 12 scene variants equally:

| Step | Time | Frozen v1 | Blind v2 | Material v2 | SPH Euler |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.002 | 3.180154e-07 | 3.399776e-07 | 2.727318e-07 | 6.087695e-07 |
| 10 | 0.02 | 3.175395e-05 | 3.394636e-05 | 2.716637e-05 | 6.086428e-06 |
| 50 | 0.1 | 7.865167e-04 | 8.412488e-04 | 6.707915e-04 | 3.027268e-05 |
| 100 | 0.2 | 3.091198e-03 | 3.313014e-03 | 2.678069e-03 | 5.954623e-05 |
| 250 | 0.5 | 1.763503e-02 | 1.920164e-02 | 1.874927e-02 | 1.323408e-04 |
| 500 | 1.0 | 5.814767e-02 | 6.796472e-02 | 1.188473e-01 | 1.913961e-04 |

The material model's early average advantage does not persist. At step 500 it
has **74.87% higher position error than blind v2**, 104.39% higher than frozen v1,
and 19.28% higher than the known-gravity ballistic baseline. Frozen v1 has the
lowest final mean error among these three learned maps. The known-equation SPH
Euler solver is much more accurate on these targets; this is not a comparison
of their computation costs.

| Method at step 500 | Mean position RMSE | Mean velocity RMSE | Momentum checks passed / 12 | Maximum momentum error |
| --- | --- | --- | --- | --- |
| Persistence | 1.184975e-01 | 1.740330e-01 | 8 | 1.350000e+00 |
| Constant velocity | 1.113446e-01 | 1.740330e-01 | 8 | 1.350000e+00 |
| Known-gravity ballistic | 9.963695e-02 | 1.470485e-01 | 12 | 0 |
| SPH Euler | 1.913961e-04 | 2.582669e-05 | 12 | 3.686696e-16 |
| Frozen v1 | 5.814767e-02 | 1.110850e-01 | **0** | 8.999100e-05 |
| Blind v2 | 6.796472e-02 | 1.472324e-01 | **0** | 1.002475e-02 |
| Material v2 | 1.188473e-01 | 3.572946e-01 | **0** | 1.012499e-06 |

All methods keep mass fixed by construction. None of the three learned maps
meets the predeclared 1e-8 momentum tolerance on any final scene. Material v2 has
smaller momentum drift than the other learned maps, but still exceeds that
tolerance. It passes all 12 momentum checks at step 1, four at step 10 and none
at step 50 or later. Finite completion therefore does not establish either
accuracy or conservation. Persistence and constant velocity fail only the four
gravity scenes, as expected from their definitions.

SPH Euler's maximum centre-of-mass error at T=1 is 3e-4, despite its very small
momentum residual. Its force-kick/drift integration of gravity is first order;
momentum conservation alone does not imply an exact position trajectory. Kinetic
energy and all centre-of-mass errors are retained in the raw report and are not
presented as total-energy conservation tests.

### Conditions that remain difficult

The following tables give step-500 mean position RMSE. Percentage change is
material v2 relative to blind v2; positive is a regression.

| Material condition | Blind v2 | Material v2 | Change |
| --- | --- | --- | --- |
| Base interpolation | 6.819260e-02 | 4.524465e-02 | -33.65% |
| Joint interpolation | 4.853656e-02 | 7.471694e-02 | +53.94% |
| Stiffness extrapolation | 7.652814e-02 | 3.249556e-01 | **+324.62%** |
| Viscosity extrapolation | 7.860159e-02 | 3.047214e-02 | -61.23% |

| Particle count | Blind v2 | Material v2 | Change |
| --- | --- | --- | --- |
| 8 | 4.991379e-02 | 7.730550e-02 | +54.88% |
| 18 | 7.161232e-02 | 1.221741e-01 | +70.60% |
| 27 | 8.236806e-02 | 1.570624e-01 | **+90.68%** |

Some material conditions still improve, while high stiffness dominates the
regression. Grouped means do not identify the physical cause by themselves.
The largest particle count remains problematic. The complete scene-level
errors, including velocity, avoid turning an aggregate result into a claim
about every individual scene.

### Raw artifacts

| Artifact | Bytes | SHA-256 |
| --- | --- | --- |
| [evaluation.json](evidence/2026-09-13-rollouts/evaluation.json) | 633,067 | `b5705a45e13d17d97cd12139edbdb81796b532e3ae6b513438a2ff39f686382f` |
| [reference.jsonl](evidence/2026-09-13-rollouts/reference.jsonl) | 201,831 | `5e6c61c812a48c2c76b162c5851326a4f01722bd16d996b01b998860fd14efdb` |

## Artifact contract and limitations

The evaluation schema is `cogniarc.particle-rollout-evaluation.v1`. It contains
the protocol, each scene's reference checks, each method's completion/failure
record and checkpoint errors/diagnostics, plus means grouped by horizon,
material condition and particle count. Failed checkpoints retain null errors
or provisional errors when only the reference is unqualified; aggregation
follows the strict rule above. Every available state has a SHA-256 of the
canonical positions/velocities dictionary.

Each UTF-8 line of the reference file has schema
`cogniarc.particle-rollout-reference.v1`: scene/group/material identity, seed,
config, gravity, units, particle IDs, masses and initial-state hash; integrator,
internal dt, completion/failure; and available snapshots with coarse step,
time and N × 3 positions/velocities. It records the finer reference only; coarse
reference gaps, state hashes and qualifications are in the evaluation. Exact
generation and the actual reader for initial observations are supplied in code.

Provenance records the command, environment, source hashes, protocol revision,
frozen input hashes and reference-file size/hash. The coefficients and initial
recipes make predictions reproducible; all intermediate prediction arrays need
not be stored to reconstruct them. Hashes are environment-specific, while
numerical controls use tolerances. Original code and synthetic data use the
repository MIT license; no external data or model weights are included.

This experiment does not establish calibrated water behaviour, walls,
incompressibility, long-term stability beyond the measured interval,
uncertainty calibration, action-conditioned learning, neural/JEPA capability,
runtime integration or resource savings. The earlier direct-horizon results
remain separate evidence and must not be read as autoregressive validation.

## Consequence and next experiment

These results do not justify replacing the physical solver with the current
learned maps. Keep this measurement and the earlier direct-prediction evidence
frozen; no automatic activation or runtime integration follows.

Before another predictor revision, prepare a separate protocol that handles
known gravity explicitly and tests zero net internal momentum change, using
fresh reserved scenes. Diagnose the high-stiffness error separately, including
whether the state representation accounts for changing neighbour support.
Improving conservation alone would not establish lower trajectory error.
Any new candidate must face both the frozen baselines and the numerical solver
over longer trajectories, with material/count regressions retained.

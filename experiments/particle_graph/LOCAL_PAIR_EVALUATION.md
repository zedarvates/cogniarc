# Compact local pair-force approximation — 2026-09-13

**Implemented:** a four-coefficient local force model, opposite force accumulation
per pair, a shared midpoint integrator and a separate reserved evaluator.
**Measured:** development-only fitting and new synthetic rollouts below.
**Planned:** further physical approximation, uncertainty/resource measurements,
calibrated fluid validation and ShardJEPA/runtime adaptation.

## Two freezes before test measurements

The [protocol](local_pair_protocol.json) and [scene manifest](local_pair_manifest.json)
were published at [c03ab1bcb3f56527f4b34cb49856f03226ab8f99](https://github.com/zedarvates/cogniarc/commit/c03ab1bcb3f56527f4b34cb49856f03226ab8f99)
before implementation, fitting and reserved evaluation. They fix features,
nonnegativity, regularization candidates, data partitions, integration,
new scenes, controls, tolerances and failure reporting.

The implementation, tests, generated development labels, complete fit candidates
and selected model were then published at
[7cdd4d199fe8fcbfc499473d25171f95478f1c05](https://github.com/zedarvates/cogniarc/commit/7cdd4d199fe8fcbfc499473d25171f95478f1c05)
before generating the new test observations or trajectories. Seven exact local
file blobs were checked against that remote tree before the measured command.
The evaluator verifies the saved model checksum, source/protocol/label hashes
and consistency with validation selection. It cannot fit or select a model.
The CLI records the supplied full model commit; the publication check supplies
the independent connection between those source bytes and that revision.

Every earlier v1/v2/projection source, model and raw result remains frozen.
Only their existing readers, graph, numerical operators and evaluation helpers
are reused. No prior test result is a target for this new fit.

## Model and structural constraints

At every force evaluation, the existing spatial hash rebuilds undirected pairs
whose current separation r is strictly below the configured radius h. Define
q = 1 - r/h, the direction n = (x_i - x_j)/r (zero at coincidence), and relative
velocity dv = v_j - v_i. For observed masses, stiffness k and viscosity mu,
the force on particle i is a nonnegative weighted sum of four vector features:

| Feature | Force on i before its learned coefficient |
| --- | --- |
| pressure_q2 | m_i * m_j * k * q^2 * n |
| pressure_q3 | m_i * m_j * k * q^3 * n |
| viscosity_q1 | m_i * m_j * mu * q * dv |
| viscosity_q2 | m_i * m_j * mu * q^2 * dv |

The opposite force is added to j. Accumulated forces are divided by each
particle's own mass to obtain internal accelerations. There is no force outside
support, self-edge, constant bias, global mean projection or reference-state
input. Positions, velocities, masses and material/radius observations suffice.

This is a proposed dimensionless polynomial approximation with four fitted
scalars, not a graph neural network or a JEPA model. Unlike the numerical SPH
reference, it does not compute density or a pressure threshold. In particular,
it can repel neighbouring particles where the reference clamps pressure to
zero. Its shape, physical constraints and material factors are supplied by the
experiment; the four coefficients are fitted. It does not discover arbitrary
physical laws or establish validity after changing units or radius.

Opposite force accumulation supplies net internal momentum conservation.
The nonnegative viscosity coefficients give nonpositive instantaneous kinetic
power from the viscous component. Pressure is radial; the viscosity component
need not be, so no general angular-momentum guarantee is claimed. Neither
instantaneous viscous dissipation nor a momentum ledger proves finite-step
energy behaviour, unconditional stability or accurate trajectories.

## Development-only fitting

The existing verified v2 corpus contributes 16 training scenes and eight
validation scenes, from only four/two initial-condition groups. Each contributes
the observed state at steps 0/1/10/50: 64/32 snapshots and 1,920/960 scalar
coordinate rows. Original observations and checksums remain unchanged.

Labels are internal accelerations generated from each development observation
by the unchanged independent dense SPH implementation at zero gravity. They
are exported and actually reloaded through a reader checking scene/split/step,
observation identity and exact teacher values. These are synthetic labels from
known simplified equations, not measured water or external data.

Feature columns are scaled by their training-only weighted RMS, without
centering or an intercept. The objective is weighted squared acceleration
error plus alpha times squared scaled coefficients, constrained nonnegative.
Weights give each scene, its snapshots, particles and coordinates equal
respective shares. All 16 active subsets of four features are considered;
the positive ridge makes each active linear solve positive definite. Negative
solutions are rejected, and the feasible minimum is retained.

The five predeclared alphas are 1e-8, 1e-6, 1e-4, 0.01 and 1. Selection uses only
the analogous validation acceleration RMSE; exact ties favour the larger alpha.
No refit uses validation observations. Training/selection do not access any
new test scene, trajectory or test score. The full candidate list, scaling,
objectives and numerical optimality residuals remain in the saved model.

Validation selects **alpha = 1e-08**. All four coefficients are positive. The selected training acceleration RMSE is 5.068124e-03; validation acceleration RMSE is 4.124214e-03.

| Feature | Frozen coefficient |
| --- | --- |
| pressure_q2 | 1.88956411405759 |
| pressure_q3 | 9.42331297954681 |
| viscosity_q1 | 2.89855796222442 |
| viscosity_q2 | 3.45672961492166 |

| Alpha | Train acceleration RMSE | Validation acceleration RMSE | Active coefficients |
| --- | --- | --- | --- |
| 1e-08 | 5.068124e-03 | 4.124214e-03 | 4 |
| 1e-06 | 5.068125e-03 | 4.124553e-03 | 4 |
| 0.0001 | 5.075991e-03 | 4.154743e-03 | 4 |
| 0.01 | 5.375157e-03 | 4.386877e-03 | 4 |
| 1 | 7.775592e-02 | 5.674717e-02 | 2 |

The maximum recorded KKT optimality residual is 2.775558e-17. These development errors measure approximation to the synthetic force teacher; they do not qualify trajectory accuracy. The selected model SHA-256 is `ecfa41b6e2aae7e5da3d1fe64fe60c74b50264ced1b630bcb4e692b7550ac6e0`.

## Reserved scenes and independent controls

Fresh seeds 5001/5002/5003 generate 8/18/27-particle groups. Every group is crossed
with base/high stiffness (0.15/0.6, viscosity 0.04) and equal/unequal mass, under
known oblique gravity (0.12, -0.24, 0.09). The unequal condition repeats factors
0.5/1.5 in particle-ID order and rescales them to keep total mass equal to
N times the default mass 0.25. Thus material and mass variants share initial
positions/velocities, while total mass stays fixed. The radius is 0.85.

These are **12 main variants from only three geometry groups**, all from the
same lattice family. Unequal masses were absent from fitting. Groups, variants
and checkpoints are correlated; the test is not a broad independent benchmark.
Its aggregate cannot be directly compared with earlier aggregate values as
though the corpora were identical. Seed and actual initial-state overlap with
all v1/v2 and conservation observations is rejected.

Three explicitly specified controls use seeds 5101/5102/5103 only as identities:
an isolated particle, an already-separated unequal-mass pair at distance 1.25h
moving apart, and a separating unequal-mass pair starting at 0.9h. They use the
base material and oblique gravity. The first two have exact ballistic targets;
the third can initially interact and uses the numerical reference. Controls
are reported individually and are excluded from the main means by declaration.

## Integrators and diagnostics

All methods advance 500 steps of 0.002 to T=1.0. Checkpoints are
1/10/50/100/250/500. Comparators are known-gravity ballistic, unchanged SPH
Euler, SPH with explicit midpoint, the three unchanged projected affine maps,
and the new local pair model.

The new pair model and the SPH midpoint comparator share exactly the same
integrator: evaluate internal acceleration at the current state, construct the
half-step state, rebuild/evaluate forces there, then advance with midpoint
velocity and acceleration. Constant uniform gravity is added explicitly at
both derivative evaluations. This uses two force evaluations per step, versus
one for SPH Euler. No computational-cost comparison is inferred from accuracy.

Each method feeds back its own state. The local model has no projection,
reference reset, clipping, adaptive step or correction from targets. Invalid
shape/count, nonfinite values or a component above 1e6 stop the trajectory and
retain the first failed step/reason and earlier snapshots. The shared midpoint
also validates its intermediate state. This bound is a numerical guard rather
than a physical stability criterion.

The unchanged independent dense RK4 reference runs at dt/4 and dt/8. At each
checkpoint, position/velocity resolution gaps must be below
`max(1e-12, min(absolute_cap, 0.01 * ballistic_RMSE))`, with caps 1e-7/1e-6.
Both references must pass mass 1e-12 and gravity-accounted momentum 1e-10 gates.
For the two ballistic controls, the finer reference must also agree with the
analytic state within 1e-10 RMSE. These checks concern the same simplified
equations; they are not physical calibration or rigorous error bounds.

Method diagnostics retain global per-coordinate RMSE, mass-centred RMSE,
momentum against P0+M*g*t (tolerance 1e-8), centre of mass against exact known
gravity (tolerance 1e-10), kinetic energy/difference, and neighbour/isolated
counts. Masses are fixed inputs, not predicted quantities. Centred RMSE
subtracts each state's own mass-weighted mean but averages coordinate errors
without mass weighting; both global and centred definitions are retained.

Passive force diagnostics on saved local/physical-midpoint states record
internal acceleration RMS, net internal force norm, pair count and maximum
acceleration on isolated particles. They do not feed back into integration or
add integration steps. Every method receives its own analytic-control scores.

Main means weight scenes equally. If any aggregate member failed or has an
unqualified reference, its global/centred means are null. The evaluator keeps
all failures; it never reports survivor-only mean errors or tunes an accuracy
threshold. The command returns nonzero for incomplete trajectories, unqualified
references, failed local conservation ledgers or failed local ballistic
controls. A zero exit code is not an accuracy pass for the main corpus.

## Measured results

All **105 comparison trajectories** complete 500 steps: 12 main variants plus three controls, each with seven methods. All **90 numerical reference checkpoints** qualify. All 90 local-model mass/momentum/centre-of-mass checkpoint ledgers and all 12 local analytic-control checks pass.

The maximum local momentum error across main scenes and controls is 4.055242e-15; maximum centre-of-mass error is 3.556053e-15. The maximum recorded net internal force norm is 2.834676e-16 and maximum acceleration on isolated particles is exactly zero in all saved local-force diagnostics.

Maximum reference position/velocity resolution gaps are 1.739068e-09 / 3.682207e-09. Maximum reference momentum error is 4.611541e-14. Execution used CPython 3.12.14 on Linux x86_64, standard library only; no timing, memory, GPU or cross-machine performance result is claimed.

### Main-corpus errors at T=1

| Method | Mean position RMSE | Mean velocity RMSE | Momentum pass / 12 | COM pass / 12 |
| --- | --- | --- | --- | --- |
| Known-gravity ballistic | 1.327542e-01 | 1.888939e-01 | 12 | 12 |
| SPH Euler | 2.646174e-04 | 2.861949e-05 | 12 | 0 |
| SPH midpoint | 3.210079e-07 | 1.994487e-07 | 12 | 12 |
| Projected v1 | 7.754538e-02 | 1.141675e-01 | 12 | 12 |
| Projected blind v2 | 7.850053e-02 | 1.364719e-01 | 12 | 12 |
| Projected material v2 | 1.881451e-01 | 5.844519e-01 | 12 | 12 |
| Local pair model | 6.556763e-03 | 1.688988e-02 | 12 | 12 |

Against the projected affine models on these identical new scenes, the local model reduces mean final position RMSE by **91.54% (v1), 91.65% (blind v2), and 96.52% (material v2)**. Its final position error is lower than each of the three projected models on every one of the 12 main variants. Velocity mean errors fall 85.21%, 87.62%, 97.11%, respectively.

The local model also improves centred errors: the gain concerns internal trajectories, rather than only global translation. This result compares complete predictors with different force representations, training objectives and integration paths; it does not isolate the causal effect of each individual change.

**Retained limitation:** the known-equation SPH solvers remain substantially more accurate. The local model position RMSE is 24.78 times the SPH Euler value. The same-midpoint physical comparator reaches 3.210079e-07, showing that this midpoint implementation can be accurate with the reference force law. No resource advantage or replacement of the physical solver is established. Euler passes momentum but fails the tight COM tolerance under nonzero gravity, with maximum error 2.830194e-04.

### Error growth over the measured interval

| Step | Local pair position RMSE | Projected v1 | Projected blind v2 | Projected material v2 |
| --- | --- | --- | --- | --- |
| 1 | 6.731147e-08 | 5.118865e-07 | 5.068856e-07 | 3.774850e-07 |
| 10 | 6.697863e-06 | 5.113342e-05 | 5.062522e-05 | 3.768396e-05 |
| 50 | 1.615858e-04 | 1.267178e-03 | 1.253810e-03 | 9.401077e-04 |
| 100 | 5.988707e-04 | 4.966849e-03 | 4.912506e-03 | 3.802007e-03 |
| 250 | 2.584608e-03 | 2.738930e-02 | 2.712545e-02 | 2.772971e-02 |
| 500 | 6.556763e-03 | 7.754538e-02 | 7.850053e-02 | 1.881451e-01 |

### Material conditions

| Condition | Local pair position RMSE | Local pair velocity RMSE | Projected blind v2 position RMSE |
| --- | --- | --- | --- |
| interpolation_base | 3.608189e-03 | 6.096898e-03 | 6.934892e-02 |
| stiffness_extrapolation | 9.505336e-03 | 2.768286e-02 | 8.765214e-02 |

### Mass conditions

| Condition | Local pair position RMSE | Local pair velocity RMSE | Projected blind v2 position RMSE |
| --- | --- | --- | --- |
| equal | 5.330785e-03 | 1.574995e-02 | 7.494795e-02 |
| unequal | 7.782740e-03 | 1.802981e-02 | 8.205311e-02 |

### Particle counts

| Condition | Local pair position RMSE | Local pair velocity RMSE | Projected blind v2 position RMSE |
| --- | --- | --- | --- |
| 8 | 6.629399e-03 | 2.568643e-02 | 6.899723e-02 |
| 18 | 5.319475e-03 | 1.317927e-02 | 8.294252e-02 |
| 27 | 7.721414e-03 | 1.180394e-02 | 8.356184e-02 |

The unequal-mass final local position mean is higher than the equal-mass mean, and high stiffness remains more difficult than the base material. These retained residuals prevent a claim of exact or general physical behaviour. At T=1 no main local-model particle is isolated; neighbourhoods can still differ from the reference (for example 86 versus 79 pairs on the 27-particle high-stiffness unequal-mass variant). Counts alone do not establish a causal explanation.

### Controls retained separately

| Control | Local pair position RMSE vs RK4 | SPH midpoint position RMSE vs RK4 | Projected v1 position RMSE vs RK4 |
| --- | --- | --- | --- |
| isolated | 6.867958e-15 | 6.867958e-15 | 7.358020e-15 |
| separated | 6.774299e-15 | 6.774299e-15 | 1.143934e-01 |
| separating | 2.712250e-03 | 1.678651e-07 | 8.735489e-02 |

The local model, SPH midpoint and ballistic baseline pass every analytic checkpoint on the isolated and already-separated controls. The projected affine maps pass the isolated control but fail the already-separated pair control: zero net momentum alone did not eliminate their artificial relative motion. SPH Euler misses the tight position tolerance on both ballistic controls because of its first-order gravity drift.

The separating control is not analytic free flight from t=0. The local model ends with zero force after separation, but retains a final position error of 2.712250e-03 versus 2.773407e-03 for ballistic prediction. This small improvement and the much lower physical-midpoint error retain a concrete weakness in approximating initially interacting, low-density particles; no density-related cause has yet been isolated.

## Reproduction and source

Run from the repository root with CPython 3.12+ and its standard library:

```bash
python -m unittest discover -s tests -p 'test_particle_graph*.py' -v
python -m experiments.particle_graph.local_pairs --output-dir experiments/particle_graph/evidence/2026-09-13-local-pairs
python -m experiments.particle_graph.local_pair_evaluation --model-dir experiments/particle_graph/evidence/2026-09-13-local-pairs --model-revision 7cdd4d199fe8fcbfc499473d25171f95478f1c05 --output-dir experiments/particle_graph/evidence/2026-09-13-local-pairs
```

The fit command reproduces development-only coefficients; it was run and its
outputs committed before the final command. For evaluation alone, use the
committed model directory. The evaluator does not train.

The combined suite passes **91 focused tests**, including 14 new controls:
unequal-mass momentum, strict cutoff and graph rebuilding, coincident particles,
viscous power, rotation/translation/boost/permutation, analytic midpoint motion,
invalid output, an independently solvable constrained regression, label/model
tampering, partition isolation, mass variants and separate end-to-end controls.
Disposable tests use seed 9101 and control identities 9201/9202/9203, separate
from the reserved evaluation. The retained [test output](evidence/2026-09-13-local-pairs/tests.txt)
records the successful suite. The full repository suite is not claimed to pass;
the `scripts/run_tests.py` named by CONTRIBUTING.md remains absent.

Source: [local_pairs.py](local_pairs.py),
[local_pair_evaluation.py](local_pair_evaluation.py) and
[test_particle_graph_local_pairs.py](../../tests/test_particle_graph_local_pairs.py).
Earlier [conservation controls](conservation.py), [rollout helpers](rollouts.py),
[SPH graph/forces](reference.py), [dense reference](dense_reference.py),
[affine models](affine.py) and [v2 readers](material_data.py) remain unchanged.

## Artifact contracts

| Artifact | Bytes | SHA-256 |
| --- | --- | --- |
| [force_targets.jsonl](evidence/2026-09-13-local-pairs/force_targets.jsonl) | 86,320 | `f40e8407f04dd16732527e1769b368844af2bda84c4e838dde9f0eabe003c45a` |
| [model.json](evidence/2026-09-13-local-pairs/model.json) | 9,853 | `ecfa41b6e2aae7e5da3d1fe64fe60c74b50264ced1b630bcb4e692b7550ac6e0` |
| [model_lock.json](evidence/2026-09-13-local-pairs/model_lock.json) | 138 | `da2d9c067640b202ea4f99b5540e0928e7cb93d0c134e6a400721c082ad74cc4` |
| [tests.txt](evidence/2026-09-13-local-pairs/tests.txt) | 15,056 | `7cbd79ab3a82d674a576482030e32c13fc17efec1562c2ca82a5344ae6f5993d` |
| [initial.jsonl](evidence/2026-09-13-local-pairs/initial.jsonl) | 40,109 | `7e7557c0b34597d8eb7eb7447f9fd8efbe7fc5973f655aa549861fd2eb902fcd` |
| [reference.jsonl](evidence/2026-09-13-local-pairs/reference.jsonl) | 215,077 | `d67a2035f5773062933dfadd73d6ecd6b93b23e80aabb931c3803f43cba538b4` |
| [evaluation.json](evidence/2026-09-13-local-pairs/evaluation.json) | 1,329,079 | `3da4606eb642cb7e3d6d2c4cd0520456830fa32aad71d574548b8b73d907e4d5` |

- `force_targets.jsonl`: 96 records with schema
  `cogniarc.local-pair-force-target.v1`, scene/group/split/step, hash of the
  observed state/masses/config and N-by-3 internal accelerations. `label_records`
  generates these; `read_labels` actually reloads and verifies them before fit.
- `model.json`: `cogniarc.local-pair-model.v1`, selected alpha/coefficients,
  every candidate and training scale/objective/optimality residual, validation
  scores, development IDs/counts, no-refit flags and source/input/label provenance.
  `model_lock.json` records the model-file SHA-256. `read_model` verifies the
  saved file and its source/label/selection consistency before evaluation.
- `initial.jsonl`: `cogniarc.local-pair-initial.v1`, one record per main/control
  scene, role, mass/material/gravity/group metadata, seed, units, configuration,
  masses/particle IDs, state hash and a single initial snapshot. `make_scenes`
  generates it; `read_scenes` checks exact recipe equality, missing/duplicate
  records and prior-corpus isolation before use.
- `reference.jsonl`: `cogniarc.local-pair-reference.v1`, finer RK4 snapshots at
  0/1/10/50/100/250/500, times, positions/velocities, support counts, scene/mass
  metadata and completion/failure. Coarse hashes/gaps/gates remain in evaluation.
- `evaluation.json`: `cogniarc.local-pair-evaluation.v1`, protocol, all per-scene
  references and method checkpoints, global/centred errors, physical/force
  diagnostics, analytic controls, main-only group means, and exact command,
  model revision, input/source hashes and artifact sizes/checksums.

Exact state hashes may depend on the environment; numerical tests use explicit
tolerances. Prediction arrays can be reproduced from fixed coefficients,
observations and source; checkpoint hashes identify measured states. The source
and synthetic data use the repository MIT license. No external data/model
weights, paid services or GPU training are introduced.

## Limits and next bounded step

Freeze this four-coefficient model and the full positive/negative evidence. The next bounded step is a predeclared acceleration-error diagnostic across neighbour density, separation and mass ratio, with explicit low-density pressure/viscosity controls. Use that diagnosis to define any density-aware feature change and reserve new evaluation scenes before fitting it. Keep the existing same-midpoint physical comparison and conservation/compact-support controls. No feature change or retuning was performed after the present test scores.

The current evidence covers one dimensionless synthetic lattice family and three main initial-condition groups, known constant gravity and a short fixed simulated duration. It includes neither walls, calibrated water, incompressibility, learned uncertainty, general graph/neural capability, a ShardJEPA implementation, resource savings nor runtime integration. Publication gates and the remaining fluid-validation work stay open. No merge or automatic activation follows from these gains.

# Explicit gravity and global conservation — 2026-09-13

**Implemented:** deterministic mass-weighted projection of each frozen step-1
proposal onto the known constant-gravity centre-of-mass and momentum update.
**Measured:** new reserved synthetic trajectories, with paired raw/corrected
errors and passive neighbour counts below. **Planned:** accurate local learned
interactions, calibrated fluids, ShardJEPA comparison and runtime integration.

## Predeclared experiment

The [protocol](conservation_protocol.json) and [new scene recipes](conservation_manifest.json)
were published at [e254b3acab69dcdd4b59f63af51216df9bfd0548](https://github.com/zedarvates/cogniarc/commit/e254b3acab69dcdd4b59f63af51216df9bfd0548)
before implementation, tests or measurements. The protocol pins previous model,
manifest, data and rollout inputs by SHA-256. The models retain all earlier
coefficients and normalizers; there is no training, selection, hyperparameter,
refitting or adjustment after inspecting these new scenes.

Three fresh seeds, 4001/4002/4003, generate groups of 8/18/27 particles. Every
group is crossed with two materials (stiffness 0.15 or 0.6, viscosity 0.04) and
two gravities: zero and (0.12, -0.24, 0.09). The oblique gravity has x/z components
absent from training. Each group shares exactly the same initial positions,
velocities and masses across its four variants. The reader checks both seed and
actual initial-state isolation against every prior v1/v2 partition.

There are 12 variants but only three independent initial-condition groups,
all from the same lattice family. Material/gravity variants and time checkpoints
are correlated. This new reserved evaluation extends the evidence; it is not a
broad blind benchmark. Its means must not be directly compared with the earlier
12-scene rollout means as if the corpus were identical.

All eight methods take 500 steps of 0.002 to T=1.0. Checkpoints are
1/10/50/100/250/500. Comparators are known-gravity ballistic prediction, the
unchanged SPH Euler solver, frozen v1, blind v2, material v2, and a projected
version of each of the three learned maps. Only the saved step-1 maps are used.

## Correction and its exact scope

For fixed positive masses, let M be total mass, c the current mass-weighted
position mean and w the current mass-weighted velocity mean. The unchanged
learned map proposes positions q and velocities u. With known constant uniform
gravity g and step dt, the required means are:

```text
c_next = c + dt*w + 0.5*dt^2*g
w_next = w + dt*g
q_corrected[i] = q[i] + (c_next - weighted_mean(q))
u_corrected[i] = u[i] + (w_next - weighted_mean(u))
```

The wrapper adds the same position translation and velocity increment to every
particle. It therefore leaves every pairwise position/velocity difference
unchanged within floating-point error for that proposal. It enforces zero net
internal momentum change globally. It does not construct equal-and-opposite
local pair forces, constrain angular momentum or energy, or repair inaccurate
relative dynamics. The known gravity is supplied by the experiment, not inferred.
Positions and velocities remain separate learned output channels.

Each method feeds back only its own current predicted state. The projection
uses that state's means, fixed observed masses, known gravity and dt; it never
reads a future target or resets to the initial/reference trajectory. Masses are
carried as fixed inputs rather than learned. Neither raw nor corrected maps
read the neighbour diagnostics or numerical reference.

Raw proposals are checked for finite values, shape/count and the fixed absolute
component bound of 1e6 **before** correction. A uniform invalid or excessive
proposal cannot be concealed by the projection. Corrected states are checked
again. First-failure step/reason and earlier checkpoints remain in the report;
there is no clipping, repair, restart or survivor-only aggregate.

## Numerical qualification and diagnostic definitions

The unchanged independent dense RK4 integrator generates references at dt/4 and
dt/8 (2,000 and 4,000 steps). Finer snapshots are exported. There are no prior
short-prefix targets for these new scenes. At each checkpoint, reference
position/velocity RMSE gaps must satisfy
`max(1e-12, min(absolute_cap, 0.01 * ballistic_RMSE))`, with caps 1e-7 and 1e-6.
Both reference resolutions require mass error <=1e-12 and gravity-accounted
momentum error <=1e-10. These are resolution checks of the same simplified
equations, not physical water validation or rigorous error bounds.

Candidate diagnostics check momentum against `P0 + M*g*t` (tolerance 1e-8), and
centre of mass against `c0 + w0*t + 0.5*g*t^2` (tolerance 1e-10). Kinetic energy
and its difference from the reference are retained without assuming conservation
under gravity, pressure and viscosity.

Position/velocity errors are per-coordinate RMSE; reported means weight every
scene equally. Centred errors first subtract each state's own mass-weighted
position and velocity means, isolating relative deformation/motion from global
translation. The benchmark masses are equal; unequal masses are covered by
analytic projection tests, not by a learned generalisation claim. Raw/corrected
centred states are also compared directly with a fixed 1e-10 RMSE diagnostic.

Neighbour counts use undirected pairs at strict distance below the configured
kernel radius, exclude self-pairs and count particles with no neighbours. These
passive checkpoint counts describe support along a trajectory; they neither
modify the predictor nor, on their own, establish why an error occurs.

If any member of an aggregate has failed or has an unqualified reference, both
global and centred mean prediction errors are null. Completion, conservation
and accuracy are separate outcomes. No accuracy acceptance threshold is tuned.

## Reproduce and inspect

Run from the repository root with CPython 3.12+ and the standard library:

```bash
python -m unittest discover -s tests -p 'test_particle_graph*.py' -v
python -m experiments.particle_graph.conservation --output-dir experiments/particle_graph/evidence/2026-09-13-conservation
```

The combined suite passes **77 focused tests**, including 11 new conservation,
data-isolation, reader and failure controls. They include unequal-mass analytic
ledgers; 500 biased single-particle steps compared with ballistic motion;
relative-state preservation; rejection of invalid raw proposals before
projection; exclusion of future targets and unused horizon maps; split/seed/
state isolation; strict-radius neighbour counts; and failure-aware centred
aggregation. Disposable control recipes use seed 9001, separate from the new
reserved benchmark seeds. The full repository suite is not claimed to pass;
`scripts/run_tests.py` in the contribution guide remains absent.

Source: [conservation.py](conservation.py). Tests:
[test_particle_graph_conservation.py](../../tests/test_particle_graph_conservation.py).
The runner reuses the unchanged [rollout controls](rollouts.py),
[model readers and prediction](materials.py), [affine implementation](affine.py),
[scene generator](validation.py), [SPH operators](reference.py) and
[dense RK4 reference](dense_reference.py).

The command records per-scene progress and returns nonzero for incomplete
trajectories, unqualified references or failed corrected mass/momentum/centre
ledgers. Raw-model conservation failures remain visible and do not prevent a
completed measurement. A zero exit code does not certify prediction accuracy.

## Measurements

All **96 comparison trajectories** complete their 500 steps. All **72 reference checkpoints** qualify. All **216 projected checkpoint ledgers** pass their mass, momentum and centre-of-mass tolerances. The maximum projected momentum error across checkpoints is 3.362876e-14; maximum centre-of-mass error is 8.353765e-15.

The maximum reference gaps are 4.376206e-10 for position and 1.441977e-09 for velocity; maximum reference momentum error is 1.661470e-15. The environment is CPython 3.12.14 on Linux x86_64, standard library only. No timing, memory or hardware comparison is claimed.

### Position error before and after correction at T=1

| Frozen map | Raw mean RMSE | Projected mean RMSE | Change |
| --- | --- | --- | --- |
| Frozen v1 | 7.935132e-02 | 7.261308e-02 | -8.49% |
| Blind v2 | 8.028628e-02 | 7.414801e-02 | -7.65% |
| Material v2 | 1.942550e-01 | 1.891015e-01 | -2.65% |

These paired gains come from correcting the global motion. All 216 raw/corrected centred-state comparisons pass the fixed 1e-10 diagnostic; their maximum position/velocity RMSE difference is 7.463106e-15. Thus, on this measured interval, projection leaves the internal learned trajectories unchanged to roundoff. This is conservation supplied by the wrapper, not conservation learned by the models.

### Full final-horizon results

| Method | Position RMSE | Velocity RMSE | Momentum pass / 12 | COM pass / 12 | Max momentum error |
| --- | --- | --- | --- | --- | --- |
| Known-gravity ballistic | 1.282326e-01 | 1.818413e-01 | 12 | 12 | 2.482534e-16 |
| SPH Euler | 2.289353e-04 | 2.720203e-05 | 12 | 6 | 1.359740e-15 |
| Frozen v1 | 7.935132e-02 | 1.219506e-01 | 0 | 0 | 1.012500e+00 |
| Blind v2 | 8.028628e-02 | 1.467544e-01 | 0 | 0 | 1.012518e+00 |
| Material v2 | 1.942550e-01 | 5.934516e-01 | 0 | 0 | 1.012500e+00 |
| Projected v1 | 7.261308e-02 | 1.058631e-01 | 12 | 12 | 1.597641e-14 |
| Projected blind v2 | 7.414801e-02 | 1.326175e-01 | 12 | 12 | 1.431062e-14 |
| Projected material v2 | 1.891015e-01 | 5.853907e-01 | 12 | 12 | 3.362876e-14 |

The three raw learned maps still fail both final conservation ledgers on all twelve variants. Oblique gravity exposes an unlearned x/z response; zero-gravity variants retain smaller learned bias. The numerical SPH Euler comparator preserves momentum but fails the tight COM gate on the six nonzero-gravity scenes: its final maximum COM error is 2.830194e-04. Its force-kick/drift gravity update is first order. The ballistic formula passes both ledgers yet has substantial internal prediction error, another example of why conservation does not establish accuracy.

### Retained deformation failures

Projected material v2 still has **155.03% higher** final position RMSE than projected blind v2. The known-equation SPH Euler comparator remains much more accurate; the experiment does not compare their computational costs.

| Material condition | Projected v1 | Projected blind v2 | Projected material v2 | Material vs blind |
| --- | --- | --- | --- | --- |
| interpolation_base | 4.864400e-02 | 7.055302e-02 | 4.820401e-02 | -31.68% |
| stiffness_extrapolation | 9.658217e-02 | 7.774300e-02 | 3.299990e-01 | +324.47% |

The base material improves relative to the blind model; the high-stiffness condition regresses strongly. These condition means do not establish improvement or failure on every individual trajectory.

| Particles | Projected v1 | Projected blind v2 | Projected material v2 |
| --- | --- | --- | --- |
| 8 | 6.405866e-02 | 6.402064e-02 | 1.251282e-01 |
| 18 | 7.632932e-02 | 7.719148e-02 | 1.949355e-01 |
| 27 | 7.745126e-02 | 8.123193e-02 | 2.472409e-01 |

| Gravity | Raw material v2 position RMSE | Projected material v2 position RMSE |
| --- | --- | --- |
| zero | 1.891015e-01 | 1.891015e-01 |
| oblique | 1.994085e-01 | 1.891015e-01 |

The projected means are the same across the paired gravities to the reported precision; the raw models carry a global-motion error on oblique gravity. Internal dynamics remain problematic even when the external acceleration is supplied exactly.

### Passive neighbour-support observations

For the high-stiffness material at T=1, the two gravity variants have identical support counts. The table shows one row per initial-condition group:

| Particles | Reference pairs | Projected material pairs | Reference isolated | Projected material isolated |
| --- | --- | --- | --- | --- |
| 8 | 12 | 2 | 0 | 4 |
| 18 | 42 | 5 | 0 | 8 |
| 27 | 78 | 7 | 0 | 14 |

These counts show different neighbourhoods along the learned and numerical trajectories. They are descriptive observations, not a causal proof that neighbourhood loss alone explains the drift. The current affine features use centred position/velocity and material interactions; they do not consume the neighbour graph.

### Raw artifacts

| Artifact | Bytes | SHA-256 |
| --- | --- | --- |
| [initial.jsonl](evidence/2026-09-13-conservation/initial.jsonl) | 36,220 | `5e57f12c641daca5bbdf852ef0ad2d2e6e1b83155291f6fc27432af02594b39f` |
| [reference.jsonl](evidence/2026-09-13-conservation/reference.jsonl) | 206,365 | `77cc96d90b938b7df0c5531ae669aa7127b66f50e0966556c8653ec36035fadc` |
| [evaluation.json](evidence/2026-09-13-conservation/evaluation.json) | 1,039,264 | `c827ad0d7162afece526ed1d97b2eaf692a532d64ff53ed252f4995ade6de0de` |

## Artifact contract

`initial.jsonl` has one `cogniarc.particle-conservation-initial.v1` record per
variant: scene/group/material/gravity identity, test split, seed, units, config,
particle IDs, masses, initial-state checksum and a single step-0 snapshot with
N-by-3 positions/velocities. `make_scenes` deterministically generates it;
`read_scenes` actually reads the file and rejects missing, duplicate or altered
records by exact comparison with the pinned recipes and prior-corpus isolation.

`reference.jsonl` contains `cogniarc.particle-conservation-reference.v1`
records with matching identity/config/mass metadata, integrator/internal dt,
completion/failure and finer snapshots at 0/1/10/50/100/250/500. Each snapshot
includes time, positions, velocities and passive support counts. Coarse states
are represented by hashes/gaps/qualifications in the evaluation.

`evaluation.json`, schema `cogniarc.particle-conservation-evaluation.v1`, records
the protocol, per-scene reference qualifications, all method failure/checkpoint
records, global/centred errors, ledgers, support counts and raw/corrected centred
comparisons. Aggregates group by time, material, gravity and particle count.
It includes exact command, date, environment, protocol/source/input hashes and
data artifact sizes/checksums. Prediction arrays are reproducible from the fixed
coefficients, generator and source; checkpoint hashes identify the measured
states. Exact hashes can depend on the environment; checks use tolerances.

Original code and synthetic data use the repository MIT license. No external
dataset, model weights, paid service or neural training is introduced. The
experiment remains dimensionless, without walls, calibrated water,
incompressibility, an uncertainty model, action-conditioned learning,
ShardJEPA/runtime integration or measured resource savings.

## Next bounded experiment

Keep this global correction and every raw/corrected result frozen. Predeclare a separate local-interaction experiment with fresh reserved scenes: a predictor that uses current neighbour geometry and material observations, has zero interaction outside the prescribed support, and constructs opposite forces for each pair before mass conversion. Compare it with the projected affine maps and known-equation solver, retaining deformation error, changing support, unequal-mass controls and long-horizon failures. First test isolated and separating particles explicitly. This is planned work; the current support observations do not establish a successful architecture or justify runtime activation. No merge or automatic promotion follows from passing conservation ledgers.

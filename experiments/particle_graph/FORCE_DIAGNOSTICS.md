# Frozen pair-force diagnostic — 2026-09-13

**Implemented:** offline same-state force decomposition, exact replay checks,
static controls, actual evidence readers and stratified summaries.
**Measured:** the four-coefficient approximation misses the pressure threshold
and the density normalization of viscosity in controlled cases. These structural
limitations coexist with the previously measured improvement over affine maps.
**Planned:** a separate density-aware feature ablation with fresh reserved scenes.
No new model was fitted in this diagnostic.

## Frozen scope and provenance

The [diagnostic protocol](force_diagnostic_protocol.json) was published at
[1d6d8fa0038d6cd6dd2ed7982729e861345512f6](https://github.com/zedarvates/cogniarc/commit/1d6d8fa0038d6cd6dd2ed7982729e861345512f6)
before the new diagnostic implementation and measurements. It fixes the bins,
81 static pair recipes, four contextual recipes, tolerances and failure rules.
Its SHA-256 is `bc94d98d8b4ff40e7fe1646e127f6bcb9fe46bf884cacd146bd82f5633a67704`.

The [module](force_diagnostics.py), [15 new focused tests](../../tests/test_particle_graph_force_diagnostics.py)
and complete 106-test log were then published at
[8f714ff3dab42827ae0dd870bfd1c5cd6059972a](https://github.com/zedarvates/cogniarc/commit/8f714ff3dab42827ae0dd870bfd1c5cd6059972a).
The three source/test/log blobs and protocol blob were verified exactly against
that remote tree before running the diagnostic. The CLI records this revision;
the independent blob check binds it to the actual measured source bytes.

The model remains the one frozen at
[7cdd4d199fe8fcbfc499473d25171f95478f1c05](https://github.com/zedarvates/cogniarc/commit/7cdd4d199fe8fcbfc499473d25171f95478f1c05):
coefficients `[1.889564114057587, 9.423312979546806, 2.89855796222442, 3.4567296149216564]`. Twenty prior source/model/evidence
files are checksum-locked by the protocol. Existing model readers also verify
selection, labels and source hashes. All earlier sources, coefficients, datasets
and raw evaluations remain unchanged.

We reuse the [published local-pair evaluation](LOCAL_PAIR_EVALUATION.md) at
[645ec7c6994981079ed5e7e946986d5af1f7c444](https://github.com/zedarvates/cogniarc/commit/645ec7c6994981079ed5e7e946986d5af1f7c444).
Those scenes are now diagnostic data, not a new blind test. The main corpus has
12 variants from only three geometry groups, with 8/18/27 particles, equal/unequal
masses and two stiffnesses. The three old controls remain separate from means.

## What is measured

At steps 0/1/10/50/100/250/500, each state is supplied unchanged to both the
frozen local model and the independent dense physical force calculation, with
zero gravity for internal-force evaluation. Pressure-only calls set viscosity
to zero; viscosity-only calls set stiffness to zero; combined calls retain both.
These diagnostic calls never advance or reset a trajectory.

Two observation paths are kept separate: the saved finer RK4 reference states,
and an independently replayed local-model trajectory. All 15 replays complete;
all 90 noninitial local state digests match the earlier publication exactly.
All 90 reference checkpoints match their published qualified digests, and all
15 initial references match the initial records. Reference convergence is
inherited from the earlier evaluation, not re-measured here.

The diagnostic separately reconstructs dense distances, self/neighbor densities
and both implementations' pair forces. Accumulated pair forces divided by each
particle's own mass must reproduce each existing pressure, viscosity and total
acceleration function. Checks also cover decomposition, net force, isolated and
coincident cases, reference zero-pressure cases and instantaneous viscous power.
They do not assert that the approximate force is accurate.

**Results:** 295 diagnostic states (105 reference, 105 replayed, 85 static),
**7,376 numerical/structural checks passed**. Maximum acceleration reconstruction
difference is 1.554312e-15; maximum internal net-force component is 3.782781e-16;
maximum viscous power is zero. These are repeated checks, not 7,376 independent
experiments. **106 focused unit tests pass** (91 existing + 15 new).

All JSONL is written and actually reloaded: identity, complete expected records,
state/config/masses, hashes and every recomputed diagnostic must agree. Changed,
missing or duplicate records abort. Any numerical check failure would retain
raw diagnostics, suppress all aggregate scores and return a nonzero status.
No failure or score was discarded; successful execution is not an accuracy gate.

## Aggregation and main force errors

An acceleration MSE averages the three coordinates equally, then particles
within one state. The reported RMSE is the square root of the equal-state mean
of those MSEs. Thus each of the 12 main scenes and seven selected checkpoints has
equal weight in unconditional means. This is not a continuous-time average or
the earlier mean of per-scene **position** RMSEs. Relative error divides by
reference RMS with identical weights, and is null when that RMS is at most 1e-12.

| Observation path | States / particle observations | Pressure RMSE | Viscosity RMSE | Total RMSE | Total / reference RMS |
| --- | --- | --- | --- | --- | --- |
| reference | 84 / 1484 | 3.826241e-02 | 1.547891e-02 | 4.124474e-02 | 10.77% |
| pair_local_replay | 84 / 1484 | 3.794929e-02 | 1.706489e-02 | 4.185701e-02 | 10.95% |

On reference states, pressure relative error is **9.84%**
and viscosity relative error is **74.53%**.
Pressure still has the larger absolute error across these sampled times.
Pressure and viscosity error vectors can cancel: their RMSEs must not be added.
The replay path evaluates both laws on the same replayed state; it is not a
subtraction between forces evaluated at two different states.

| Step | Reference-state total RMSE | Reference-state total / reference RMS | Replayed-state total RMSE |
| --- | --- | --- | --- |
| 0 | 4.248574e-02 | 9.24% | 4.248574e-02 |
| 1 | 4.245505e-02 | 9.24% | 4.244768e-02 |
| 10 | 4.198461e-02 | 9.18% | 4.188565e-02 |
| 50 | 3.657533e-02 | 8.31% | 3.569230e-02 |
| 100 | 2.967945e-02 | 7.44% | 2.832046e-02 |
| 250 | 4.279871e-02 | 21.76% | 4.290315e-02 |
| 500 | 4.987356e-02 | 104.94% | 5.464563e-02 |

At the final checkpoint the instantaneous total force-law error exceeds the
reference acceleration RMS on both observation paths. This does not replace or
revise the earlier final position errors: no new predictor or rollout was
introduced. Same-state residuals isolate force approximation at those states,
not the causal share of every mechanism in the accumulated position error.

## Density, neighbour, separation and mass strata

Within a bin, states with at least one member receive equal weight, then their
member particles/pairs receive equal weight. Counts make this conditioning
explicit: bin means cannot be added or treated as independent samples. Empty
bins have null scores. Density includes each particle's own mass contribution;
neighbour-only density and neighbour count are also saved.

| Particle density / rest density | Populated states / particles | Pressure RMSE | Viscosity RMSE | Total RMSE | Total / reference RMS |
| --- | --- | --- | --- | --- | --- |
| <1 | 10 / 90 | 4.852346e-02 | 4.421993e-02 | 7.528231e-02 | 105.92% |
| [1,2) | 32 / 197 | 3.914495e-02 | 1.893897e-02 | 3.678003e-02 | 10.63% |
| [2,4) | 74 / 1032 | 3.303473e-02 | 4.998194e-03 | 3.112614e-02 | 7.48% |
| >=4 | 42 / 165 | 5.766255e-02 | 6.689107e-03 | 5.826846e-02 | 21.00% |

| Neighbour count | Populated states / particles | Total acceleration RMSE | Total / reference RMS |
| --- | --- | --- | --- |
| <1 | 0 / 0 | — | — |
| [1,4) | 5 / 26 | 7.415178e-02 | 212.22% |
| [4,8) | 37 / 299 | 4.023153e-02 | 11.23% |
| [8,16) | 35 / 421 | 3.496725e-02 | 9.52% |
| >=16 | 52 / 738 | 4.117037e-02 | 11.18% |

The low-density and low-neighbour strata have large relative residuals, but
these are correlated with time, mass, stiffness and geometry. A particle below
rest density can still receive pressure from a neighbour whose pressure is
positive. Only a pair with **both** pressures zero is a zero-pressure control.

The following are **force-on-one-particle per unordered pair**, before division
by mass; their units and weighting differ from the acceleration tables. There
are 10,761 main reference pair observations and 10,746 replay pair observations.
The separation overflow bin means 0.9 <= r/h < 1, since only active pairs enter.
Mass-ratio boundaries use the declared 1e-12 margin for ratios 1/3/9.


| separation_over_radius | Populated states / pairs | Pressure force RMSE | Viscosity force RMSE | Total force RMSE | Total / reference RMS |
| --- | --- | --- | --- | --- | --- |
| <0.25 | 0 / 0 | — | — | — | — |
| [0.25,0.5) | 75 / 2301 | 7.866171e-03 | 2.897645e-04 | 7.840952e-03 | 22.91% |
| [0.5,0.75) | 83 / 4484 | 2.665581e-03 | 1.425118e-03 | 3.500798e-03 | 22.06% |
| [0.75,0.9) | 66 / 1313 | 1.768687e-03 | 1.588966e-03 | 2.662173e-03 | 71.95% |
| >=0.9 | 61 / 2663 | 2.926490e-04 | 5.645765e-04 | 6.952553e-04 | 93.45% |


| mass_ratio | Populated states / pairs | Pressure force RMSE | Viscosity force RMSE | Total force RMSE | Total / reference RMS |
| --- | --- | --- | --- | --- | --- |
| <1.000000000001 | 84 / 7778 | 5.154025e-03 | 1.347100e-03 | 5.474433e-03 | 21.46% |
| [1.000000000001,3.000000000001) | 42 / 2983 | 2.751838e-03 | 1.210105e-03 | 3.062484e-03 | 23.44% |
| [3.000000000001,9.000000000001) | 0 / 0 | — | — | — | — |
| >=9.000000000001 | 0 / 0 | — | — | — | — |


| pressure_zero | Populated states / pairs | Pressure force RMSE | Viscosity force RMSE | Total force RMSE | Total / reference RMS |
| --- | --- | --- | --- | --- | --- |
| both_zero | 8 / 163 | 4.622440e-03 | 3.644091e-03 | 7.706580e-03 | 196.70% |
| active | 81 / 10598 | 4.236019e-03 | 1.066038e-03 | 4.424763e-03 | 20.87% |


The reference trajectory contains 163 pair observations in eight states with
both endpoint pressures zero. Their local pressure-force RMSE is 4.622440e-3;
the reference pressure force is exactly zero. Near the support boundary, small
absolute forces can have large relative errors; both values are retained.

| Condition | States | Total acceleration RMSE |
| --- | --- | --- |
| equal | 42 | 3.634317e-02 |
| unequal | 42 | 4.562271e-02 |
| interpolation_base | 42 | 1.462666e-02 |
| stiffness_extrapolation | 42 | 5.646520e-02 |
| 8 | 28 | 3.628471e-02 |
| 18 | 28 | 3.987829e-02 |
| 27 | 28 | 4.686714e-02 |

Unequal-mass and high-stiffness variants retain larger absolute force-law errors.
These group summaries do not establish a causal mass or stiffness effect alone.
The raw report includes the analogous strata for replayed states and all old
controls, plus pair minimum-density bins omitted from this readable selection.

## Controlled counterexamples

The predeclared static grid crosses total masses 0.1/0.5/2, ratios 1/3/9 and
r/h = 0/0.2/0.5/0.8/0.9/0.99/1/1.01/1.25. It fixes h=0.85, rest density=1,
k=0.15, mu=0.04 and velocities (-0.2,0.1,0)/(0.2,-0.1,0). These 81 deterministic
states are diagnostic interventions, not independently sampled test scenes.
Each has pressure-only, viscosity-only and combined calls.

**Pressure threshold:** all **22 interacting, noncoincident pairs with both
densities below the threshold** have zero physical pressure force but nonzero
local pressure force. The largest local pressure acceleration RMS among those
controls is 3.573015e-2 at total mass 0.5, ratio 1, r/h=0.5. The model's positive
pressure polynomial cannot express this zero region using separation alone.
All 27 at/outside-support cases have exactly zero internal acceleration in both
implementations. Coincident pairs have zero pressure direction, while viscosity
can remain active. These passing support checks do not remove the interior error.

**Mass scaling and viscosity normalization:** at a fixed equal-mass pair geometry
r/h=0.5, the measured component RMS values are:

| Total mass | Density at each particle | Dense pressure RMS | Local pressure RMS | Dense viscosity RMS | Local viscosity RMS |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 1.813658e-01 | 0.000000e+00 | 7.146031e-03 | 2.534030e-01 | 1.194666e-03 |
| 0.5 | 9.068289e-01 | 0.000000e+00 | 3.573015e-02 | 5.068059e-02 | 5.973332e-03 |
| 2.0 | 3.627316e+00 | 1.186313e-01 | 1.429206e-01 | 1.267015e-02 | 2.389333e-02 |

Multiplying both masses by 20 increases the local viscous acceleration by 20,
whereas the physical viscous acceleration falls by 20. This scaling follows
directly from the implemented equations: under mass scaling s, density scales
by s; physical pair viscosity contains m_i*m_j/(rho_i*rho_j), while the local
law contains m_i*m_j. After division by own mass, the respective accelerations
scale as 1/s and s. This is a structural mismatch with this fixed-viscosity
reference, not a statement about all possible fluid discretizations.

Changing mass ratio at fixed total mass also retains substantial viscous error:

| Mass ratio (fixed total 0.5, r/h=0.9) | Dense viscosity RMS | Local viscosity RMS | Viscosity relative error | Total acceleration RMSE |
| --- | --- | --- | --- | --- |
| 1 | 2.021423e-02 | 8.376568e-04 | 95.86% | 1.992685e-02 |
| 3 | 2.986418e-02 | 9.365288e-04 | 96.86% | 2.954237e-02 |
| 9 | 6.860702e-02 | 1.072724e-03 | 98.44% | 6.499377e-02 |

**Neighbour context:** keep pair 0–1's positions, velocities, masses and materials
identical, then add 0/2/4/6 neighbours from the exact recipe. The physical force
on that pair responds to the changed endpoint densities:

| Added neighbours | Density i | Density j | Dense pair pressure Fx on i | Dense pair viscosity Fx on i |
| --- | --- | --- | --- | --- |
| 0 | 7.224736e-01 | 1.091184e+00 | -2.789518e-03 | 1.535604e-02 |
| 2 | 1.228946e+00 | 1.597657e+00 | -1.015337e-02 | 6.165707e-03 |
| 4 | 1.735419e+00 | 2.104129e+00 | -1.214965e-02 | 3.315298e-03 |
| 6 | 2.059496e+00 | 2.428207e+00 | -1.199724e-02 | 2.420765e-03 |

For all four states the local target-pair pressure Fx is exactly
-1.160371e-2 and its viscosity Fx is exactly 1.735096e-3. The target-pair local
contribution cannot distinguish these contexts. Other edges and full particle
forces also change, so this does not claim that every net force is identical or
assign a causal percentage of rollout error to the missing context.

**Previously separating pair:** at its initial state the physical pressure
acceleration is exactly zero, but local pressure RMS is 6.854927e-04.
Physical viscous acceleration RMS is 5.342268e-02 versus
1.675314e-03 locally: a **96.86%**
underestimate of the initial viscous response. Both implementations return zero
force on both saved paths from step 100 onward after separation. The earlier
position residual persists because earlier motion differs; late zero force
cannot undo it. This identifies the initial force discrepancy without claiming
a new quantitative attribution of the final position error.

## Reproduction and evidence contract

From the repository root, using Python 3.12 or later:

```bash
python -m unittest discover -s tests -p 'test_particle_graph*.py' -v
python -m experiments.particle_graph.force_diagnostics --source-revision 8f714ff3dab42827ae0dd870bfd1c5cd6059972a --output-dir experiments/particle_graph/evidence/2026-09-13-force-diagnostics
```

The measured environment was CPython 3.12.14,
Linux x86_64, standard library only.
`load_observations` consumes the prior verified records and replays the frozen
model; `make_controls` is the static recipe generator. `diagnose` produces the
records and `read_records` is their actual verifying consumer before `summarize`.

| File / schema | Contents and interpretation |
| --- | --- |
| force_diagnostic_protocol.json / cogniarc.force-diagnostic-protocol.v1 | Fixed recipes, weights, bins, tolerances, source/input hashes and limits |
| Three JSONL files / cogniarc.force-diagnostic-observation.v1 | Exact state, IDs, path, time, config, masses, density/count descriptors, component accelerations, per-state pair-bin sufficient statistics, target-pair forces for context controls and all checks |
| diagnostic.json / cogniarc.force-diagnostic-summary.v1 | Complete stratified aggregates, all static cases, separate trajectory controls, replay checks, provenance, artifact sizes/hashes and false fit/selection flags |
| tests.txt | Raw output of the 106-test focused suite |

The observation state plus frozen coefficients and equations reconstruct every
pair contribution; per-state pair sufficient statistics store coordinate MSE,
reference mean square, local mean square and item count for actual aggregation.
Gravity is deliberately excluded from internal-force calls. All quantities use
the existing dimensionless convention; no physical unit conversion is asserted.
The data and new original code use the repository MIT license; no external data
or third-party implementation was imported.

| Artifact | Bytes | SHA-256 |
| --- | --- | --- |
| [diagnostic.json](evidence/2026-09-13-force-diagnostics/diagnostic.json) | 574910 | `ded89a8c769ecc25b7fdfe264464cf14a085b7fb90dc5b6e64cbd8e63ac9e73e` |
| [local_observations.jsonl](evidence/2026-09-13-force-diagnostics/local_observations.jsonl) | 1407838 | `d4a040cb3541fd54869d92ff761e60137fffd3d65c470f19e8e1c876c3e1f223` |
| [reference_observations.jsonl](evidence/2026-09-13-force-diagnostics/reference_observations.jsonl) | 1407280 | `ec4b8532e56d0d2e6f04a8da31514505c5f1a927db85aa39b1dbd4aa33bf847b` |
| [static_controls.jsonl](evidence/2026-09-13-force-diagnostics/static_controls.jsonl) | 363666 | `4dbdce6dbef7d4826d313971adb9150fd5a5319ca6d0e4a0246301396c354035` |
| [tests.txt](evidence/2026-09-13-force-diagnostics/tests.txt) | 17663 | `79c86d933b506d7344a6ad93f7e2f49da9306fd8c6d37d815cb7a58c93613e21` |

An independent arithmetic check recomputed all 48 main acceleration aggregates
(two paths, all times plus seven checkpoints, three components) from the saved
raw vectors. All agreed within 1e-12 relative / 1e-15 absolute tolerance. The 20
protected inputs, 14 measured source hashes, 47 local documentation links and
`git diff --check` also passed verification.

Exact JSONL and replay hashes are environment-sensitive; numerical comparisons
use the declared absolute/relative tolerances. The contribution guide's
`scripts/run_tests.py` remains absent. Only the 106-test focused particle-graph
suite is claimed passed; the unrelated full repository suite is not re-certified.

## Decision and remaining work

The next experiment should predeclare an ablation of the **supplied pressure
threshold** and **inverse-density normalization**, separately and jointly, on
identical training/validation observations and acceleration labels. Compute
density from current positions/masses only. Preserve opposite pair forces,
compact support, nonnegative viscous response and the common midpoint integrator.
Keep this four-coefficient model and the physical SPH law as frozen comparators.
An exact physical feature basis would supply known equations, not discover them.

Reserve new independent geometry groups and new mass-scale/ratio conditions
before any fitting; freeze all fitted candidates/selection before generating
and scoring those future evaluation observations. Report instantaneous errors,
500-step position/velocity and conservation, controls, and actual computation
cost separately. Retain negative results and declare any accuracy/cost decision
before measurement. No new fit, neural/GNN/JEPA model, ShardJEPA runtime,
water calibration, resource gain, activation or merge occurred in this tranche.

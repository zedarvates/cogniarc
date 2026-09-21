# Scientific transfer: physical graphs, drawing and experimentation

Date: 2026-09-10. Updated 2026-09-13: offline reference, direct affine, paired-material,
raw/projected and local-pair rollout comparisons implemented; residual diagnostics and consumer integration planned.
This complements the world-model, organic-writing and Socratic-evaluation plans.

## Research questions

1. Can action-conditioned prediction beat simple extrapolation on held-out scenes and longer horizons?
2. Does selecting a discriminating intervention reduce uncertainty more reliably than arbitrary actions at the same budget?
3. Can a drawing critic improve unseen compositions through targeted practice, independently of access to an exact curve renderer?

The water idea is provisionally interpreted as a particle-neighbour graph. The earlier
formulation was not recovered; this does not identify the user's exact algorithm or
claim novelty. SPH evaluates physical kernels over neighbours; a learned graph
simulator learns message functions.

## Sequence and exit criteria

| Stage | Deliverable | Acceptance / stopping rule | Status |
| --- | --- | --- | --- |
| R0 | Dependency-free synthetic particle operators | Dense-search/operator agreement; finite outputs; mass/momentum controls; raw fixture | Implemented; see evidence |
| R1 | Trajectories and numerical validation | Split by scene/seed before fitting; dt refinement, particle counts, gravity, velocities, boundaries; units, hashes, provenance | Partial: 10 split scenes, sparse snapshots, unbounded numerical convergence and fixed-box contact controls implemented; coupled boundary accuracy and physical water validation remain planned |
| R2 | Offline physical hypothesis adapter | Selected/random/fixed interventions at equal budget on held-out observations; abstain when all hypotheses fail | Planned |
| R3 | Learned rollout comparison | Persistence, constant velocity and affine baselines; horizons 1/10/50; unseen scenes, longer rollouts and resource use | Partial: direct, raw/projected and local-pair rollouts plus frozen force diagnostics measured; pressure-threshold and viscosity-density failures retained. Density-aware ablation and resource use remain planned |
| D0 | Stroke-transfer protocol | Separate raster fit from motor trajectories, pressure and order; hold out compositions | Planned |
| D1 | Socratic correction experiment | Targeted/no/random correction, equal budget; closure, intersections, perspective and rubric quality | Planned |
| C0 | Authoring interchange | Versioned units/frames/IDs; deterministic replay; incompatible-input rejection; ordinary runtime fallback | Planned |

R0 tests implementation properties on small synthetic inputs. It does not validate
water behaviour, arbitrary-dt stability, incompressibility, speedup, learning or
long-horizon generalisation. Failed R1/R3 results remain in the report; an
inconclusive experiment does not qualify for production integration.

## Reuse existing components

- `cogniarc/active_experiment.py` already scores hypotheses, selects experiments and updates beliefs; add an adapter and comparison, not a duplicate engine.
- `cogniarc/world_model.py` provides an existing predictor path. Do not relabel nearest-neighbour replay as a newly trained physical JEPA.
- `cogniarc/world_model_physics/physics/simulator/physics.py` already applies buoyancy and drag in fluid zones. The particle experiment is a separate reference.
- `human_skills/render_svg.py` and `human_skills/organic.py` already render smooth strokes with variable width. The new question is transfer, not merely attractive splines.

## Evidence and next implementation

The [experiment README](../../experiments/particle_graph/README.md) contains commands
and a raw report with exact initial particles, parameters, baselines, environment
and source SHA-256 values. The [2026-09-13 numerical follow-up](../../experiments/particle_graph/NUMERICAL_VALIDATION.md)
adds a frozen scene manifest, independent dense RK4, equal-time refinement and
snapshots reserved by scene. The [box-contact follow-up](../../experiments/particle_graph/BOUNDARY_VALIDATION.md)
adds analytic collision controls and coupled containment/ledger checks.
The [affine follow-up](../../experiments/particle_graph/AFFINE_BASELINE.md) adds
training-only ridge fitting, validation-only selection and test errors at three
horizons against persistence, constant velocity and known-gravity ballistic
prediction. Its protocol was committed before test scoring; 45 focused tests
passed at that stage. The [paired-material follow-up](../../experiments/particle_graph/MATERIAL_BASELINE.md)
freezes v2 before generation/scoring, replays each initial condition across four
materials, and compares a material-conditioned ridge model with a blind refit
and the frozen v1 model. All 36 numerical scenes and 57 focused tests passed at that stage.
The lower average prediction error coexists with regressions at 27 particles and
on an interpolated material; the detailed errors remain in the report.
The [rollout follow-up](../../experiments/particle_graph/ROLLOUT_EVALUATION.md) freezes
its protocol before evaluating the step-1 maps over 500 feedback steps. All 84
comparison trajectories finish and all 72 reference checkpoints qualify; 66
focused tests pass. Material v2 loses its average advantage at the final horizon,
and all three learned maps fail the final momentum tolerance on every scene.
The [conservation follow-up](../../experiments/particle_graph/CONSERVATION_EVALUATION.md)
predeclares deterministic projection onto known-gravity global motion, then
compares all three raw/corrected maps on 12 fresh reserved variants. All 96
trajectories complete, 72 references qualify and 77 focused tests pass. All 216
corrected checkpoint ledgers pass; centred internal trajectories remain unchanged
to roundoff. At T=1, corrected material v2 remains 155.03% worse in mean position
error than corrected blind v2. Passive support counts retain the high-stiffness
neighbourhood discrepancy. The [local-pair follow-up](../../experiments/particle_graph/LOCAL_PAIR_EVALUATION.md)
now fits four nonnegative coefficients on development observations only, freezes
the model before reserved evaluation, and compares compact opposite pair forces
with projected affine maps and same-midpoint SPH. All 105 trajectories complete,
90 references qualify and 91 tests pass. Final position RMSE falls 91.65% against
projected blind v2 and 96.52% against projected material v2 on the new main corpus;
local conservation and ballistic controls pass. The physical solvers remain much
more accurate, and low-density/separating and unequal-mass residuals are retained.
The [frozen force diagnostic](../../experiments/particle_graph/FORCE_DIAGNOSTICS.md)
now reuses those states and exact local replays, plus 81 static pair and four
neighbour-context controls. Its protocol and implementation were published before
measurement. All 295 records pass 7,376 checks; 90 replayed state hashes match and
106 focused tests pass. These are diagnostic observations, not a new blind test.
The local law applies pressure on 22 below-threshold interacting static pairs
whose reference pressure is zero, and its mass scaling gives the opposite
viscous-acceleration scaling from the reference. The separating pair's initial
viscous response is underestimated by 96.86%; contextual physical pair forces
change while the local pair contribution remains fixed. Old coefficients and
evidence stay unchanged; bins alone do not establish causality for rollout error.
Next, predeclare separate and joint pressure-threshold/inverse-density ablations,
using current observations only and identical train/validation targets. Reserve
new geometry and mass-scale/ratio test conditions before fitting, freeze selection
before their generation, and retain the current local model and same-midpoint
physical comparator. Separate force accuracy, 500-step prediction/conservation
and actual cost. Exact physical features supply known equations, not discovery.
Do not infer general accuracy or runtime readiness from these bounded gains.
Extend physical validation to coupled contact accuracy and fluid wall
treatment. Keep generated targets out of policy observations.
R1 remains partial: numerical convergence and confinement do not establish
calibrated water behaviour.

## Primary sources

- Müller, Charypar and Gross (2003), [Particle-Based Fluid Simulation for Interactive Applications](https://matthias-research.github.io/pages/publications/sca03.pdf): 3-D kernels and physical reference operators.
- Sanchez-Gonzalez et al. (2020), [Learning to Simulate Complex Physics with Graph Networks](https://arxiv.org/abs/2002.09405): learned message passing and rollout error.
- Huang et al. (2019), [Learning to Paint](https://arxiv.org/abs/1903.04411): sequential stroke actions, not proof of our skills.
- Li et al. (2020), [Differentiable Vector Graphics Rasterization](https://people.csail.mit.edu/tzumao/diffvg/): curve fitting and raster losses as reference evaluators.
- Zsolnai et al. (2011), [Procedural Brush Synthesis](https://users.cg.tuwien.ac.at/zsolnai/gfx/procedural-brush-synthesis-paper/): controllable brush generation.

The [Two Minute Papers video](https://www.youtube.com/watch?v=mOvtumfyjCs) was the
discovery entry point. The papers above are individually selected for this plan;
they are not all attributed to the presenter or asserted to appear in that video.
No third-party code, model weights or datasets are vendored.

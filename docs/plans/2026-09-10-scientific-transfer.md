# Scientific transfer: physical graphs, drawing and experimentation

Date: 2026-09-10. First offline reference implemented; learning and consumer integration planned.
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
| R3 | Learned rollout comparison | Persistence, constant velocity and affine baselines; horizons 1/10/50; unseen scenes, longer rollouts and resource use | Planned; no trained model |
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
Next, fit the affine baseline on training scenes only, with validation-only
selection; extend physical validation to coupled contact accuracy, fluid wall
treatment and longer rollouts. Keep generated targets out of policy observations.
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

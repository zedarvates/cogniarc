# CogniARC research roadmap

Updated: 2026-09-13. This scientific-transfer track complements the existing
[implementation plans](docs/plans/). Checkboxes describe artifacts, not broad capabilities.

## First: a verifiable physical reference

- [x] Isolated 3-D particle-neighbour experiment: density, symmetric pressure and viscosity forces, deterministic stepping.
- [x] Independent dense-neighbour/operator comparisons; mass, momentum, viscous dissipation and transformation controls.
- [x] Reproducible 27-particle, 50-step fixture with persistence and constant-velocity prediction baselines. See [source and evidence](experiments/particle_graph/README.md).
- [x] Freeze 10 synthetic scenes into 3 train / 2 validation / 5 test scenes, including reserved particle counts and parameters; export sparse reference snapshots with hashes and split-leakage controls.
- [x] Compare three semi-implicit Euler resolutions at identical physical times against independently coded dense RK4, with a second RK4 resolution checking reference error. All ten scenes pass the numerical gates; see [numerical evidence and scope](experiments/particle_graph/NUMERICAL_VALIDATION.md).
- [x] Add fixed frictionless box contacts with swept straight-line drift, restitution, a wall-contact radius and impulse/energy ledgers. All 9 boundary cases passed; the suite had 33 tests at that stage. See [boundary evidence](experiments/particle_graph/BOUNDARY_VALIDATION.md).
- [ ] Validate accelerated contacts and time-step refinement with walls, fluid-specific wall-density/no-slip treatment, longer rollouts and comparison with a calibrated fluid reference before describing this as a water simulator. Dense RK4 and geometric containment do not establish physical water accuracy.

## Next: hypotheses that can fail

- [ ] Adapt fixture observations to the existing [hypothesis and experiment selector](cogniarc/active_experiment.py). Distinguish friction, collision and gravity hypotheses through predicted observable outcomes.
- [ ] Compare selected, random and fixed interventions at equal budgets. Record held-out prediction error, calibration, actions used and cases where no hypothesis fits. Keep hidden simulator state out of policy inputs.
- [ ] Evaluate the drawing critic on stroke order, closure, junctions and perspective, reusing the existing organic-writing and Socratic-evaluation plans.
- [ ] Compare targeted correction with random correction and unchanged output at equal budgets; retain failed transfers and prerequisite regressions. Separate reference rendering from practiced motor skill.

## Later: learned prediction and consumers

- [x] Freeze the initial scene-level training/validation/test manifest before fitting any predictor. The published targets are reserved from fitting/selection; they are not a blind benchmark.
- [x] Fit a direct affine baseline on three training scenes; select alpha on two validation scenes, save the model, then score five test scenes against persistence, constant velocity and known-gravity ballistic prediction. Protocol committed before test scoring; 45 focused tests passed at that stage. See [affine evidence](experiments/particle_graph/AFFINE_BASELINE.md).
- [x] Publish errors at steps 1/10/50 by scene and held-out count/material condition. On these small synthetic scenes, the affine baseline lowers mean step-50 position RMSE by 41.11% against ballistic prediction; the stiffness case retains the largest error. These direct predictions do not establish learned rollout stability or broad physical generalisation.
- [x] Freeze and generate v2 with four material variants per initial-condition group: 16 train / 8 validation / 12 test scenes from 4/2/3 groups, with seeds absent from v1. All 36 scenes passed their numerical checks; the suite had 57 tests at that stage. See [material evidence](experiments/particle_graph/MATERIAL_BASELINE.md).
- [x] Retain v2 regressions: mean step-50 position error falls 20.29% versus the blind refit, but rises 10.89% at 27 particles and 31.57% on the joint-interpolation material. A lower average does not establish improvement across all conditions; v1 sources/models/raw evidence remain unchanged.
- [x] Predeclare and execute 500-step autoregressive evaluation of the frozen step-1 maps, physical baselines and SPH Euler. All 84 comparison trajectories finish; 72 reference checkpoints qualify and 66 focused tests pass. See [rollout evidence](experiments/particle_graph/ROLLOUT_EVALUATION.md).
- [x] Retain the long-rollout failures: at step 500, material v2 mean position error is 74.87% higher than blind v2, and all three learned maps fail the 1e-8 momentum tolerance on all 12 final scenes. Finite completion and short-horizon gains do not justify replacing the physical solver.
- [x] Predeclare a fixed global gravity/momentum projection and evaluate raw/corrected maps on 12 fresh reserved variants (three groups). All 96 trajectories and 72 numerical reference checkpoints complete/qualify; 77 focused tests pass. All 216 corrected checkpoint ledgers pass. See [conservation evidence](experiments/particle_graph/CONSERVATION_EVALUATION.md).
- [x] Retain the correction limit: final position RMSE falls 8.49% / 7.65% / 2.65% for v1 / blind v2 / material v2, but centred internal trajectories remain unchanged to roundoff. Corrected material v2 still has 155.03% higher final position error than corrected blind v2; high-stiffness support differs strongly from the reference. All earlier predictors/evidence remain frozen.
- [x] Freeze and fit a four-coefficient nonnegative local pair-force model on v2 training observations, select on validation, and publish the model before reserved evaluation. All 91 focused tests pass; 105 trajectories (12 main variants plus three controls, seven methods) complete and 90 references qualify. See [local-pair evidence](experiments/particle_graph/LOCAL_PAIR_EVALUATION.md).
- [x] Retain measured gains and limits: final mean position RMSE falls 91.65% versus projected blind v2 and 96.52% versus projected material v2, with lower error than all three projected maps on each main scene. All local conservation/ballistic controls pass. Local position RMSE remains 24.78 times the SPH Euler value, and same-midpoint SPH is more accurate again; no speedup or physical-water claim.
- [x] Predeclare and execute frozen acceleration-error diagnostics across density, neighbour count, separation and mass ratio. All 295 observation/control states pass 7,376 numerical checks; 90 replayed local checkpoints exactly match published states and 106 focused tests pass. See [force diagnostics](experiments/particle_graph/FORCE_DIAGNOSTICS.md). Old test scenes are reused for diagnosis, not presented as a new blind test.
- [x] Retain structural failures: 22 low-density interacting static pairs have nonzero local pressure where the reference has zero; mass scaling produces the opposite viscous-acceleration scaling from the reference. The separating pair's initial viscous response is underestimated by 96.86%. Context controls change physical target-pair forces while the density-free pair contribution stays fixed. These diagnose force-law limitations without assigning a causal percentage of final rollout error.
- [ ] Predeclare separate/joint ablations of supplied pressure-threshold and inverse-density factors on identical training/validation observations, with density computed only from current positions/masses. Reserve fresh geometry groups and mass-scale/ratio conditions before fitting; freeze candidates/selection before new test generation. Keep the four-coefficient model and same-midpoint SPH as comparators; report force errors, 500-step accuracy/conservation, controls and actual cost separately. Known physical features are supplied equations, not discovered laws. Keep runtime adaptation/activation separate.
- [ ] Measure uncertainty and resource cost before a graph/JEPA predictor. A graph structure alone is not a graph neural network or a JEPA result.
- [ ] Export versioned observations/constraints only after reproducibility, units, coordinate frames, identities and fallback checks pass.

[Detailed sequence, sources and acceptance criteria](docs/plans/2026-09-10-scientific-transfer.md).
The existing fluid-zone buoyancy/drag implementation and agent defaults are not used
by this first experiment. Runtime integration remains planned.

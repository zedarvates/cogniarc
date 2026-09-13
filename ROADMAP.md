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
- [x] Fit a direct affine baseline on three training scenes; select alpha on two validation scenes, save the model, then score five test scenes against persistence, constant velocity and known-gravity ballistic prediction. Protocol committed before test scoring; 45 focused tests pass. See [affine evidence](experiments/particle_graph/AFFINE_BASELINE.md).
- [x] Publish errors at steps 1/10/50 by scene and held-out count/material condition. On these small synthetic scenes, the affine baseline lowers mean step-50 position RMSE by 41.11% against ballistic prediction; the stiffness case retains the largest error. These direct predictions do not establish learned rollout stability or broad physical generalisation.
- [ ] Freeze a v2 corpus with material variation in development and new test seeds before changing features; compare a material-conditioned affine baseline while preserving the v1 result.
- [ ] Measure longer autoregressive rollouts, conservation, uncertainty and resource cost before a graph/JEPA predictor. A graph structure alone is not a graph neural network or a JEPA result.
- [ ] Export versioned observations/constraints only after reproducibility, units, coordinate frames, identities and fallback checks pass.

[Detailed sequence, sources and acceptance criteria](docs/plans/2026-09-10-scientific-transfer.md).
The existing fluid-zone buoyancy/drag implementation and agent defaults are not used
by this first experiment. Runtime integration remains planned.

# Box-contact validation — 2026-09-13

**Implemented:** fixed, axis-aligned box walls, a shared contact radius,
frictionless normal restitution and explicit collision impulse/energy ledgers.
**Observed fixture:** 33 particle-graph tests passed at this stage; all 9 cases
below pass their stated gates. The later [affine comparison](AFFINE_BASELINE.md)
brings the suite to 45 tests. **Planned:** fluid-specific wall treatment,
physically validated water dynamics, learned wall prediction and runtime integration.

## Reproduce

From the repository root with Python 3.12+ and its standard library:

```bash
python -m unittest discover -s tests -p 'test_particle_graph*.py' -v
python -m experiments.particle_graph.boundary_validation --output experiments/particle_graph/evidence/2026-09-13-boundaries.json
```

The validation command exits nonzero for failed cases and records their gates or
numerical error. Sources: [box contacts](boundaries.py),
[fixture generator and measurement](boundary_validation.py),
[tests](../../tests/test_particle_graph_boundaries.py).
The [raw report](evidence/2026-09-13-boundaries.json) includes full initial/final
states, box/configuration, time steps, seeds where applicable, ledgers, source
hashes and environment. Its size is 67,313 bytes and SHA-256 is
`2585020f7fa062b8246db09d18e5846cbdadb13bb155c84891652059fcfb52d1`.

The reported run used CPython 3.12.14 on Linux x86_64. All quantities are
dimensionless. Code and synthetic fixtures use the repository MIT license;
no external assets or datasets are used. These nine cases are verification
fixtures; the earlier frozen learning-scene manifest remains a separate artifact.

## Contact model and ledger meaning

`step_in_box` first applies the existing SPH acceleration and gravity to velocity,
then resolves straight-line drift through the box. Each axis computes the next
wall-hit time and consumes the remaining drift time after the bounce. Repeated
crossings are processed; hitting exactly at the endpoint returns the post-impact
velocity. A particle initially at a wall and moving outward receives an
immediate impulse. Edge/corner impacts are counted once per contacted axis.

Normal velocity changes by `v_after = -restitution * v_before`; tangential
velocity is unchanged. Restitution must lie in [0, 1]. `Box.radius` offsets the
accessible centre interval inward from each wall; it does not add hard
particle-particle collisions and is separate from the SPH kernel radius.

`wall_impulses` records impulses applied **to the particles** by the walls. The
box receives the opposite impulse. Particle momentum can therefore change in a
closed box: the validation compares that change with gravity's impulse plus the
recorded wall impulses. It does not assume particle momentum alone is conserved.

`dissipated_energy` counts only collision losses. The runner independently
computes the pre-contact force kick with dense directed SPH sums and checks that
post-contact kinetic energy plus wall loss matches pre-contact kinetic energy.
Gravity and pressure may increase kinetic energy during the force stage; this
test is not a total mechanical-energy conservation claim for the coupled solver.

The event calculation is exact for straight flights between fixed walls within
floating-point error. Accelerated curved flight is approximated by the force
kick/drift split. Large dt can still give poor SPH or gravity trajectories.
The default budget is 10,000 impacts per particle axis per call; exhausting it,
invalid geometry or starting outside the box raises without returning a partial
state. Input particles are immutable. A final endpoint clamp handles floating-
point roundoff only after the hit-time calculation bounds the segment.

## Controls and results

The 12 added tests cover all six faces, tangential motion, endpoint contact,
outward/inward starts, zero restitution, simultaneous edge/corner impacts,
contact radius, multiple hits, invalid inputs and impact-budget failure.
Forty seeded elastic flights are compared with a closed-form mirrored-interval
solution, independent of the event loop. Other analytic tests use explicit
single- and double-impact flight times. The prior 21 tests remain included.

| Case | Particles | Axis impacts | Comparison |
| --- | --- | --- | --- |
| elastic-fast | 1 | 29 | Analytic mirrored flight |
| inelastic-two-hits | 1 | 2 | Analytic two-impact trajectory |
| inelastic-corner | 1 | 3 | Simultaneous three-axis impact |
| initial-wall-stick | 1 | 1 | Zero normal restitution, free tangential motion |
| elastic-contact-radius | 1 | 66 | Analytic mirrored flight with inset walls |
| sph-gravity-8 | 8 | 4 | Coupled containment and ledgers |
| sph-gravity-27 | 27 | 10 | Coupled containment and ledgers |
| sph-elastic-27 | 27 | 23 | Coupled containment and ledgers |
| sph-narrow-18 | 18 | 26 | Coupled containment and ledgers |

All cases exercised contact and had zero reported wall penetration. The five
analytic cases had position RMSE at most 1.55e-15 and zero velocity RMSE.
Across all cases, maximum momentum-ledger residual was 1.34e-15 and maximum
collision-energy-ledger residual was 1.12e-16. Gates are 1e-12 for containment
and mass, 1e-10 for analytic error and ledger residuals, and -1e-12 as the lower
roundoff tolerance on passive collision loss. Values and individual gates are
recorded in the linked raw report.

## Remaining scope

The four coupled cases check containment and bookkeeping, not trajectory
accuracy against a validated fluid solver. Wall-density support, no-slip
conditions, surface tension, incompressibility and interparticle hard contacts
are absent. Particles near walls can have biased density estimates. No water
calibration, general stability guarantee, performance gain or neural ability is
established by these results.

The subsequent [affine comparison](AFFINE_BASELINE.md) uses the previously frozen
training/validation/test scenes, without adding these wall fixtures. Physical
validation separately needs accelerated-impact and
time-step refinement controls for coupled contacts, fluid-specific wall
treatment and an independent fluid benchmark. The numerical-reference evidence
remains available in [NUMERICAL_VALIDATION.md](NUMERICAL_VALIDATION.md).

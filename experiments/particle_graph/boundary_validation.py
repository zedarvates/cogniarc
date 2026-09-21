"""Reproduce analytic flights and coupled SPH contact-ledger checks in a box."""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import platform
import random

from .boundaries import Box, step_in_box
from .dense_reference import accelerations
from .reference import Config, Particle, Vec3


@dataclass(frozen=True)
class Case:
    name: str
    category: str
    state: tuple[Particle, ...]
    config: Config
    box: Box
    dt: float
    steps: int
    gravity: Vec3 = (0.0, 0.0, 0.0)
    seed: int | None = None
    expected: tuple[Particle, ...] | None = None


def mirror_solution(state, box, time):
    """Closed-form elastic flight through a periodically mirrored interval."""
    result = []
    for p in state:
        positions, velocities = [], []
        for x, v, lower, upper in zip(p.position, p.velocity, box.lower, box.upper):
            lower += box.radius; upper -= box.radius
            length = upper-lower
            phase = (x-lower+time*v) % (2*length)
            positions.append(lower + min(phase, 2*length-phase))
            if phase == 0:
                velocities.append(abs(v))
            elif phase == length:
                velocities.append(-abs(v))
            else:
                velocities.append(v if phase < length else -v)
        result.append(Particle(tuple(positions), tuple(velocities), p.mass))
    return tuple(result)


def cluster(shape, seed, speed):
    rng = random.Random(seed)
    return tuple(Particle(
        tuple(0.35*(cell[k]-(shape[k]-1)/2)+rng.uniform(-0.01, 0.01) for k in range(3)),
        tuple(rng.uniform(-speed, speed) for _ in range(3)), 0.25)
        for cell in product(*(range(n) for n in shape)))


def cases():
    free = Config(stiffness=0, viscosity=0)
    unit = Box((0, 0, 0), (1, 1, 1))
    fast = (Particle((0.25, 0.3, 0.4), (13, -9, 7), 2),)
    radius_box = Box((0, 0, 0), (1, 1, 1), radius=0.125)
    radius_state = (Particle((0.5, 0.5, 0.5), (5, 2, -3)),)
    return (
        Case("elastic-fast", "analytic", fast, free, unit, 1, 1,
             expected=mirror_solution(fast, unit, 1)),
        Case("inelastic-two-hits", "analytic", (Particle((0.25, 0.5, 0.5), (3, 0, 0)),),
             free, Box((0, 0, 0), (1, 1, 1), restitution=0.5), 2, 1,
             expected=(Particle((0.8125, 0.5, 0.5), (0.75, 0, 0)),)),
        Case("inelastic-corner", "analytic", (Particle((0, 0, 0), (2, 2, 2)),), free,
             Box((-1, -1, -1), (1, 1, 1), restitution=0.5), 0.75, 1,
             expected=(Particle((0.75, 0.75, 0.75), (-1, -1, -1)),)),
        Case("initial-wall-stick", "analytic", (Particle((0, 0.5, 0.5), (-2, 0.3, 0)),), free,
             Box((0, 0, 0), (1, 1, 1), restitution=0), 0.1, 10,
             expected=(Particle((0, 0.8, 0.5), (0, 0.3, 0)),)),
        Case("elastic-contact-radius", "analytic", radius_state, free, radius_box, 0.5, 10,
             expected=mirror_solution(radius_state, radius_box, 5)),
        Case("sph-gravity-8", "coupled_synthetic", cluster((2, 2, 2), 901, 0.1),
             Config(radius=0.7, stiffness=0.15, viscosity=0.04),
             Box((-0.4, -0.4, -0.4), (0.4, 0.4, 0.4), radius=0.03, restitution=0.3),
             0.005, 160, (0, -1, 0), seed=901),
        Case("sph-gravity-27", "coupled_synthetic", cluster((3, 3, 3), 902, 0.1),
             Config(radius=0.7, stiffness=0.15, viscosity=0.04),
             Box((-0.55, -0.55, -0.55), (0.55, 0.55, 0.55), radius=0.03, restitution=0.3),
             0.005, 160, (0, -1, 0), seed=902),
        Case("sph-elastic-27", "coupled_synthetic", cluster((3, 3, 3), 903, 0.45),
             Config(radius=0.7, stiffness=0.15, viscosity=0.04),
             Box((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), radius=0.03, restitution=1),
             0.005, 160, seed=903),
        Case("sph-narrow-18", "coupled_synthetic", cluster((3, 3, 2), 904, 0.1),
             Config(radius=0.7, stiffness=0.15, viscosity=0.04),
             Box((-0.45, -0.45, -0.3), (0.45, 0.45, 0.3), radius=0.025, restitution=0.5),
             0.005, 160, (0, -1, 0.2), seed=904),
    )


def momentum(state):
    return tuple(sum(p.mass*p.velocity[k] for p in state) for k in range(3))


def kinetic(state):
    return sum(0.5*p.mass*sum(v*v for v in p.velocity) for p in state)


def component_error(a, b, attribute):
    return math.sqrt(sum((x-y)**2 for p, q in zip(a, b)
                         for x, y in zip(getattr(p, attribute), getattr(q, attribute))) / (3*len(a)))


def run_case(case):
    state = case.state
    total_mass = sum(p.mass for p in state)
    initial_momentum = momentum(state)
    impulse = [0.0, 0.0, 0.0]
    hits, loss = 0, 0.0
    penetration, mass_error, momentum_error, energy_error, minimum_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    for tick in range(1, case.steps+1):
        # Independent dense force sums check the pre-contact energy ledger.
        dense = accelerations(state, case.config, case.gravity)
        kicked = tuple(Particle(p.position, tuple(v+case.dt*a for v, a in zip(p.velocity, force)), p.mass)
                       for p, force in zip(state, dense))
        result = step_in_box(state, case.config, case.box, case.dt, case.gravity)
        energy_error = max(energy_error, abs(kinetic(kicked)-kinetic(result.particles)-result.dissipated_energy))
        for axis in range(3):
            impulse[axis] += sum(j[axis] for j in result.wall_impulses)
        state = result.particles
        expected_momentum = tuple(p+total_mass*g*case.dt*tick+j
                                  for p, g, j in zip(initial_momentum, case.gravity, impulse))
        momentum_error = max(momentum_error, math.dist(momentum(state), expected_momentum))
        mass_error = max(mass_error, abs(sum(p.mass for p in state)-total_mass))
        for p in state:
            for x, lower, upper in zip(p.position, case.box.lower, case.box.upper):
                penetration = max(penetration, lower+case.box.radius-x, x-(upper-case.box.radius))
        hits += result.impacts
        loss += result.dissipated_energy
        minimum_loss = min(minimum_loss, result.dissipated_energy)
    gates = {"contact_exercised": hits > 0, "contained": penetration <= 1e-12,
             "mass_conservation": mass_error <= 1e-12, "momentum_ledger": momentum_error <= 1e-10,
             "collision_energy_ledger": energy_error <= 1e-10, "passive_contacts": minimum_loss >= -1e-12}
    analytic = None
    if case.expected is not None:
        analytic = {"position_rmse": component_error(state, case.expected, "position"),
                    "velocity_rmse": component_error(state, case.expected, "velocity")}
        gates["analytic_trajectory"] = max(analytic.values()) <= 1e-10
    return {
        "name": case.name, "category": case.category, "seed": case.seed,
        "dt": case.dt, "steps": case.steps, "duration": case.dt*case.steps,
        "config": asdict(case.config), "box": asdict(case.box), "gravity": case.gravity,
        "initial": [asdict(p) for p in case.state], "final": [asdict(p) for p in state],
        "wall_impulse": impulse, "wall_dissipated_energy": loss, "impacts": hits,
        "max_penetration": penetration, "mass_error": mass_error,
        "max_momentum_ledger_error": momentum_error, "max_collision_energy_ledger_error": energy_error,
        "kinetic_energy_initial": kinetic(case.state), "kinetic_energy_final": kinetic(state),
        "analytic_comparison": analytic, "gates": gates, "passed": all(gates.values()),
    }


def run():
    rows = []
    for case in cases():
        try:
            rows.append(run_case(case))
        except (ValueError, OverflowError) as exc:
            rows.append({"name": case.name, "passed": False, "error": str(exc), "case": asdict(case)})
    root = Path(__file__).resolve().parents[2]
    sources = ("experiments/particle_graph/boundaries.py", "experiments/particle_graph/boundary_validation.py",
               "experiments/particle_graph/reference.py", "experiments/particle_graph/dense_reference.py",
               "tests/test_particle_graph_boundaries.py")
    return {
        "schema": "cogniarc.particle-graph-box-validation.v1",
        "claim_class": "observed_on_synthetic_fixtures", "units": "dimensionless",
        "passed": all(r["passed"] for r in rows), "cases": rows,
        "provenance": {"license": "MIT (repository license)", "source_sha256": {
            name: sha256((root/name).read_bytes()).hexdigest() for name in sources}},
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "machine": platform.machine(), "dependencies": "Python standard library only"},
        "limitations": [
            "Straight-line drift contacts after a first-order force kick; continuous accelerated impact times are not solved.",
            "Fixed frictionless box, single shared wall-contact radius, normal restitution; axis impacts are counted separately.",
            "No SPH wall-density correction, no-slip condition, surface tension, incompressibility or interparticle hard contacts.",
            "Wall loss covers collision dissipation only; gravity and pressure may increase particle kinetic energy.",
            "Analytic comparisons cover free flight; coupled scenes check ledgers/containment, not calibrated fluid accuracy.",
            "No learned predictor, runtime integration, stability guarantee or performance claim.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"cases": len(report["cases"]), "passed": report["passed"],
                      "failed_cases": [r["name"] for r in report["cases"] if not r["passed"]]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

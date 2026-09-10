"""Generate a reproducible, dimensionless fixture report; never train a model."""

import argparse
from dataclasses import asdict
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import platform
import random

from .reference import Config, Particle, evaluate, step


def fixture(seed=20260910):
    rng = random.Random(seed)
    return tuple(Particle(
        tuple(0.45 * x for x in cell),
        tuple(rng.uniform(-0.04, 0.04) for _ in range(3)),
        0.25) for cell in product((-1, 0, 1), repeat=3))


def momentum(state):
    return tuple(sum(p.mass * p.velocity[k] for p in state) for k in range(3))


def position_rmse(truth, prediction):
    return math.sqrt(sum((p.position[k]-q[k])**2 for p, q in zip(truth, prediction)
                         for k in range(3)) / (3 * len(truth)))


def run():
    seed, dt, horizons = 20260910, 0.002, (1, 10, 50)
    config = Config(radius=0.85, rest_density=1.0, stiffness=0.05, viscosity=0.02)
    initial = fixture(seed)
    state = initial
    initial_momentum = momentum(initial)
    observations = []
    max_internal_force = 0.0
    edge_counts = []
    for tick in range(max(horizons) + 1):
        operators = evaluate(state, config)
        total_force = tuple(sum(p.mass*a[k] for p, a in zip(state, operators.accelerations))
                            for k in range(3))
        max_internal_force = max(max_internal_force, math.sqrt(sum(x*x for x in total_force)))
        edge_counts.append(len(operators.pairs))
        if tick in horizons:
            persistence = tuple(p.position for p in initial)
            constant_velocity = tuple(tuple(x + tick*dt*v for x, v in zip(p.position, p.velocity))
                                      for p in initial)
            observations.append({
                "horizon_steps": tick,
                "persistence_position_rmse": position_rmse(state, persistence),
                "constant_velocity_position_rmse": position_rmse(state, constant_velocity),
            })
        if tick < max(horizons):
            state = step(state, config, dt)
    root = Path(__file__).resolve().parents[2]
    sources = ("experiments/particle_graph/reference.py", "experiments/particle_graph/benchmark.py",
               "tests/test_particle_graph_reference.py")
    state_json = json.dumps([asdict(p) for p in state], sort_keys=True, separators=(",", ":"))
    return {
        "schema": "cogniarc.particle-graph-fixture-report.v1",
        "claim_class": "observed_on_synthetic_fixture",
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "machine": platform.machine(), "dependencies": "Python standard library only"},
        "provenance": {"generator": "experiments.particle_graph.benchmark.fixture", "seed": seed,
                       "license": "MIT (repository license)", "units": "dimensionless",
                       "source_sha256": {name: sha256((root/name).read_bytes()).hexdigest() for name in sources}},
        "config": asdict(config), "dt": dt, "steps": max(horizons), "gravity": [0.0, 0.0, 0.0],
        "initial_particles": [asdict(p) for p in initial],
        "observations": observations,
        "checks": {
            "particle_count": len(state),
            "total_mass_start": sum(p.mass for p in initial),
            "total_mass_end": sum(p.mass for p in state),
            "momentum_drift_norm": math.dist(initial_momentum, momentum(state)),
            "max_internal_force_norm": max_internal_force,
            "neighbor_edges_min": min(edge_counts), "neighbor_edges_max": max(edge_counts),
            "final_state_sha256": sha256(state_json.encode()).hexdigest(),
        },
        "limitations": [
            "One synthetic 3-D fixture; no physical water calibration or scalability measurement.",
            "Clamped non-tensile pressure, no boundaries, surface tension or incompressibility.",
            "Semi-implicit Euler with a fixed dt; no guarantee for other fixtures or time steps.",
            "Baselines start at t=0 and use no future observations; lower error is not learned skill.",
            "No learned predictor, neural graph network, ShardJEPA runtime or production integration.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(run(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()

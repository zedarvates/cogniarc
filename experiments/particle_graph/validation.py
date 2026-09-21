"""Validate small synthetic scenes at fixed physical times; export split snapshots."""

import argparse
from dataclasses import asdict
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import platform
import random

from .dense_reference import rk4_step
from .reference import Config, Particle, step

MANIFEST = Path(__file__).with_name("scene_manifest.json")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


def positive_number(value, name, allow_zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (not allow_zero and value == 0)):
        raise ValueError(f"invalid {name}")


def validate_manifest(data):
    """Reject split leakage, mislabeled holdouts and unbounded/invalid recipes."""
    required = {"schema", "units", "dt", "steps", "horizons", "refinement_factors",
                "reference_factors", "defaults", "scenes"}
    if set(data) != required or data["schema"] != "cogniarc.particle-graph-scenes.v1":
        raise ValueError("unsupported scene manifest")
    if data["units"] != "dimensionless":
        raise ValueError("this fixture protocol is dimensionless")
    positive_number(data["dt"], "dt")
    if type(data["steps"]) is not int or not 1 <= data["steps"] <= 100:
        raise ValueError("steps must be an integer in [1, 100]")
    if (not data["horizons"] or any(type(h) is not int or not 1 <= h <= data["steps"]
                                   for h in data["horizons"])
            or sorted(set(data["horizons"])) != data["horizons"]
            or data["horizons"][-1] != data["steps"]):
        raise ValueError("horizons must be unique, ordered and end at steps")
    if data["refinement_factors"] != [1, 2, 4] or data["reference_factors"] != [4, 8]:
        raise ValueError("v1 fixes candidate factors 1/2/4 and reference factors 4/8")
    defaults = data["defaults"]
    if set(defaults) != {"mass", "spacing", "jitter", "velocity_scale", "config"}:
        raise ValueError("unknown or missing defaults")
    for name in ("mass", "spacing", "jitter", "velocity_scale"):
        positive_number(defaults[name], name, name in ("jitter", "velocity_scale"))
    config_names = {"radius", "rest_density", "stiffness", "viscosity"}
    if set(defaults["config"]) != config_names:
        raise ValueError("incomplete base physical configuration")
    if not 3 <= len(data["scenes"]) <= 32:
        raise ValueError("expected 3 to 32 scenes")
    ids, seeds, recipes, splits = set(), set(), set(), set()
    development_counts, development_configs = set(), set()
    for scene in data["scenes"]:
        if set(scene) != {"id", "split", "condition", "seed", "shape", "gravity", "config"}:
            raise ValueError("unknown or missing scene fields")
        if not isinstance(scene["id"], str) or not scene["id"] or scene["id"] in ids:
            raise ValueError("scene IDs must be unique nonempty strings")
        if type(scene["seed"]) is not int or scene["seed"] in seeds:
            raise ValueError("seeds must be unique integers across splits")
        if scene["split"] not in {"train", "validation", "test"}:
            raise ValueError("unknown split")
        if (len(scene["shape"]) != 3 or any(type(n) is not int or n <= 0 for n in scene["shape"])
                or math.prod(scene["shape"]) > 64):
            raise ValueError("shape must contain 1 to 64 particles in 3-D")
        if len(scene["gravity"]) != 3:
            raise ValueError("gravity must be a finite 3-D vector")
        for g in scene["gravity"]:
            if isinstance(g, bool) or not isinstance(g, (float, int)) or not math.isfinite(g):
                raise ValueError("gravity must be a finite 3-D vector")
        if not set(scene["config"]) <= config_names:
            raise ValueError("unknown physical parameter")
        config = defaults["config"] | scene["config"]
        for name, value in config.items():
            positive_number(value, name, name in ("stiffness", "viscosity"))
        recipe = digest({k: v for k, v in scene.items() if k not in {"id", "split", "condition"}})
        if recipe in recipes:
            raise ValueError("duplicate scene recipe")
        ids.add(scene["id"]); seeds.add(scene["seed"]); recipes.add(recipe); splits.add(scene["split"])
        if scene["split"] != "test":
            if scene["condition"] != "development":
                raise ValueError("holdout conditions belong only to test scenes")
            development_counts.add(math.prod(scene["shape"]))
            development_configs.add(canonical(config))
    if splits != {"train", "validation", "test"}:
        raise ValueError("all three splits must be present")
    for scene in data["scenes"]:
        if scene["split"] != "test":
            continue
        condition = scene["condition"]
        if condition == "held_out_particle_count":
            if math.prod(scene["shape"]) in development_counts:
                raise ValueError("held-out particle count appears in development scenes")
        elif condition == "held_out_parameters":
            if canonical(defaults["config"] | scene["config"]) in development_configs:
                raise ValueError("held-out parameters appear in development scenes")
        elif condition != "unseen_seed":
            raise ValueError("unknown test condition")


def make_scene(data, scene):
    defaults = data["defaults"]
    rng = random.Random(scene["seed"])
    shape = scene["shape"]
    particles = tuple(Particle(
        tuple(defaults["spacing"] * (cell[k]-(shape[k]-1)/2)
              + rng.uniform(-defaults["jitter"], defaults["jitter"]) for k in range(3)),
        tuple(rng.uniform(-defaults["velocity_scale"], defaults["velocity_scale"]) for _ in range(3)),
        defaults["mass"]) for cell in product(*(range(n) for n in shape)))
    return particles, Config(**(defaults["config"] | scene["config"])), tuple(scene["gravity"])


def integrate(initial, config, gravity, dt, steps, factor, integrator, horizons):
    """Snapshot indices refer to coarse ticks, so every resolution ends at T."""
    state = initial
    snapshots = {0: state}
    ticks = {h*factor: h for h in horizons}
    for tick in range(1, steps*factor+1):
        state = integrator(state, config, dt/factor, gravity)
        if tick in ticks:
            snapshots[ticks[tick]] = state
    return snapshots


def rmse(a, b, attribute="position"):
    if not a or len(a) != len(b):
        raise ValueError("RMSE requires nonempty matching particle sets")
    return math.sqrt(sum((x-y)**2 for p, q in zip(a, b)
                         for x, y in zip(getattr(p, attribute), getattr(q, attribute))) / (3*len(a)))


def convergence_gate(errors, reference_gap):
    """Fixed numerical tolerances; no physical-accuracy or learned-model gate."""
    if (len(errors) != 3 or not all(math.isfinite(x) and x >= 0 for x in errors)
            or not math.isfinite(reference_gap) or reference_gap < 0):
        return False
    return (all(b <= a*(1+1e-10)+1e-12 for a, b in zip(errors, errors[1:]))
            and errors[-1] <= max(1e-12, 0.75*errors[0])
            and reference_gap <= max(1e-10, 0.05*errors[-1]))


def momentum(state):
    return tuple(sum(p.mass*p.velocity[k] for p in state) for k in range(3))


def run(data):
    validate_manifest(data)
    rows, corpus, initial_hashes = [], [], set()
    for scene in data["scenes"]:
        initial, config, gravity = make_scene(data, scene)
        initial_hash = digest([asdict(p) for p in initial])
        if initial_hash in initial_hashes:
            raise ValueError("identical initial states cross scene boundaries")
        initial_hashes.add(initial_hash)
        references = [integrate(initial, config, gravity, data["dt"], data["steps"], factor,
                                rk4_step, data["horizons"]) for factor in data["reference_factors"]]
        candidates = [integrate(initial, config, gravity, data["dt"], data["steps"], factor,
                                step, data["horizons"]) for factor in data["refinement_factors"]]
        end = data["steps"]
        truth = references[-1][end]
        errors = [rmse(s[end], truth) for s in candidates]
        velocity_errors = [rmse(s[end], truth, "velocity") for s in candidates]
        reference_gap = rmse(references[0][end], truth)
        reference_velocity_gap = rmse(references[0][end], truth, "velocity")
        mass = sum(p.mass for p in initial)
        expected_momentum = tuple(p + mass*g*data["dt"]*end for p, g in zip(momentum(initial), gravity))
        momentum_error = max(math.dist(momentum(s[end]), expected_momentum)
                             for s in candidates + references)
        mass_error = max(abs(sum(p.mass for p in s[end])-mass) for s in candidates + references)
        baselines = []
        for horizon in data["horizons"]:
            ballistic = tuple(Particle(tuple(x + v*horizon*data["dt"] for x, v in
                                             zip(p.position, p.velocity)), p.velocity, p.mass)
                              for p in initial)
            target = references[-1][horizon]
            baselines.append({"horizon_steps": horizon, "time": horizon*data["dt"],
                              "persistence_position_rmse": rmse(initial, target),
                              "constant_velocity_position_rmse": rmse(ballistic, target)})
        gates = {
            "position_convergence": convergence_gate(errors, reference_gap),
            "velocity_convergence": convergence_gate(velocity_errors, reference_velocity_gap),
            "mass_conservation": mass_error <= 1e-12,
            "momentum_with_external_gravity": momentum_error <= 1e-10,
        }
        rows.append({"scene_id": scene["id"], "split": scene["split"], "condition": scene["condition"],
                     "particles": len(initial), "initial_state_sha256": initial_hash,
                     "position_rmse_by_refinement": errors, "velocity_rmse_by_refinement": velocity_errors,
                     "reference_position_gap": reference_gap, "reference_velocity_gap": reference_velocity_gap,
                     "mass_error": mass_error, "momentum_error": momentum_error,
                     "baselines": baselines, "gates": gates, "passed": all(gates.values())})
        corpus.append({"schema": "cogniarc.particle-graph-snapshots.v1", "scene_id": scene["id"],
                       "split": scene["split"], "condition": scene["condition"], "seed": scene["seed"],
                       "config": asdict(config), "gravity": gravity, "units": data["units"],
                       "particle_ids": list(range(len(initial))), "masses": [p.mass for p in initial],
                       "integrator": "independent_dense_rk4", "internal_dt": data["dt"]/8,
                       "snapshots": [{"step": h, "time": h*data["dt"],
                                      "positions": [p.position for p in state],
                                      "velocities": [p.velocity for p in state]}
                                     for h, state in sorted(references[-1].items())]})
    return {"schema": "cogniarc.particle-graph-numerical-validation.v1",
            "claim_class": "observed_on_synthetic_scenes", "passed": all(row["passed"] for row in rows),
            "protocol": data, "scenes": rows}, corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    report, corpus = run(data)
    payload = "".join(canonical(record)+"\n" for record in corpus)
    root = Path(__file__).resolve().parents[2]
    sources = ["experiments/particle_graph/reference.py", "experiments/particle_graph/dense_reference.py",
               "experiments/particle_graph/validation.py", "tests/test_particle_graph_validation.py"]
    report["provenance"] = {
        "license": "MIT (repository license)",
        "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
        "snapshots_sha256": sha256(payload.encode()).hexdigest(),
        "source_sha256": {name: sha256((root/name).read_bytes()).hexdigest() for name in sources},
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "machine": platform.machine(), "dependencies": "Python standard library only"},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"snapshots.jsonl").write_text(payload, encoding="utf-8")
    (args.output_dir/"validation.json").write_text(json.dumps(report, indent=2, sort_keys=True,
                                                          allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"scenes": len(report["scenes"]), "passed": report["passed"],
                      "failed_scenes": [r["scene_id"] for r in report["scenes"] if not r["passed"]]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

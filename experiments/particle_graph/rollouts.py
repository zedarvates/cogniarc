"""Autoregressive evaluation of frozen step-1 maps with explicit failure records."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import platform

from . import affine, material_data as data, materials
from .dense_reference import rk4_step
from .reference import Config, Particle, step
from .validation import canonical, digest

PROTOCOL = Path(__file__).with_name("rollout_protocol.json")
PROTOCOL_REVISION = "3737d3e256acc654954b0d0a5359b20096cc0c8b"
DATA_DIR = affine.ROOT / "experiments/particle_graph/evidence/2026-09-13-materials"
NEW_SOURCES = ("experiments/particle_graph/rollouts.py", "tests/test_particle_graph_rollouts.py")


def read_protocol():
    protocol = json.loads(PROTOCOL.read_text())
    if (protocol["schema"] != "cogniarc.particle-rollout-protocol.v1"
            or protocol["dt"] != 0.002 or protocol["steps"] != 500
            or protocol["checkpoints"] != [1, 10, 50, 100, 250, 500]
            or protocol["reference_factors"] != [4, 8]):
        raise ValueError("unsupported rollout protocol")
    return protocol


def load_inputs(protocol):
    for name, wanted in protocol["frozen_inputs_sha256"].items():
        if affine.file_digest(affine.ROOT / name) != wanted:
            raise ValueError("frozen rollout input checksum mismatch: " + name)
    manifest, material_protocol = data.read_config()
    numerics = data.read_report(DATA_DIR, manifest, material_protocol)
    models = json.loads((DATA_DIR / "models.json").read_text())
    materials.verify_model(models, numerics, DATA_DIR, material_protocol)
    frozen = materials.load_frozen_v1(material_protocol)
    scenes = data.read_split(DATA_DIR, "test", manifest, numerics)
    if any(s["group_id"] in models["development_group_ids"] for s in scenes):
        raise ValueError("rollout scene overlaps development")
    return scenes, models, frozen


def copy_state(state):
    return {key: [list(v) for v in state[key]] for key in ("positions", "velocities")}


def check_state(state, count, limit):
    if set(state) != {"positions", "velocities"}:
        raise ValueError("invalid state fields")
    for key in ("positions", "velocities"):
        if len(state[key]) != count or any(len(v) != 3 for v in state[key]):
            raise ValueError("particle identity/count or vector shape changed")
        if any(not affine.finite(v) for v in state[key]):
            raise ValueError("non-finite state")
        if any(abs(x) > limit for v in state[key] for x in v):
            raise ValueError("absolute component abort bound exceeded")


def trace(initial, masses, dt, steps, checkpoints, advance, limit):
    if (not masses or not affine.finite(masses) or min(masses) <= 0
            or not affine.finite([dt, limit]) or min(dt, limit) <= 0
            or type(steps) is not int or steps < 1
            or not checkpoints or sorted(set(checkpoints)) != list(checkpoints)
            or any(type(h) is not int or not 1 <= h <= steps for h in checkpoints)):
        raise ValueError("invalid rollout controls")
    state = copy_state(initial)
    check_state(state, len(masses), limit)
    result = {"completed": False, "last_step": 0, "failure": None, "snapshots": {0: copy_state(state)}}
    for tick in range(1, steps + 1):
        try:
            proposed = advance(state, dt)
            check_state(proposed, len(masses), limit)
        except (ValueError, ArithmeticError) as error:
            result["failure"] = {"step": tick, "time": tick * dt,
                                 "reason": type(error).__name__ + ": " + str(error)}
            return result
        state = proposed
        result["last_step"] = tick
        if tick in checkpoints:
            result["snapshots"][tick] = copy_state(state)
    result["completed"] = True
    return result


def numerical_stepper(scene, integrator):
    masses, gravity, config = scene["masses"], tuple(scene["gravity"]), Config(**scene["config"])
    def advance(state, dt):
        particles = tuple(Particle(tuple(p), tuple(v), m) for p, v, m in
                          zip(state["positions"], state["velocities"], masses))
        result = integrator(particles, config, dt, gravity)
        if [p.mass for p in result] != masses:
            raise ValueError("numerical step changed particle mass/order")
        return {"positions": [list(p.position) for p in result], "velocities": [list(p.velocity) for p in result]}
    return advance


def method_stepper(scene, method, models, frozen):
    if method == "sph_euler":
        return numerical_stepper(scene, step)
    if method in materials.BASELINES:
        initial, ticks = copy_state(scene["snapshots"][0]), 0
        def analytic(_state, dt):
            nonlocal ticks
            ticks += 1
            return affine.predict(initial, scene["gravity"], ticks * dt, method)
        return analytic
    if method == "frozen_v1":
        model = frozen["horizon_models"]["1"]
    elif method in materials.VARIANTS:
        model = models["variants"][method]["horizon_models"]["1"]
    else:
        raise ValueError("unknown rollout method")
    return lambda state, dt: materials.predict(state, scene["gravity"], scene["config"], dt, method, model)


def weighted_vector(vectors, masses):
    return [math.fsum(m * v[k] for m, v in zip(masses, vectors)) for k in range(3)]


def diagnostics(state, initial, masses, gravity, time, target, protocol):
    mass = math.fsum(masses)
    momentum = weighted_vector(state["velocities"], masses)
    initial_momentum = weighted_vector(initial["velocities"], masses)
    centre = [x / mass for x in weighted_vector(state["positions"], masses)]
    initial_centre = [x / mass for x in weighted_vector(initial["positions"], masses)]
    expected_momentum = [p + mass * g * time for p, g in zip(initial_momentum, gravity)]
    expected_centre = [x + time * p / mass + 0.5 * g * time ** 2
                       for x, p, g in zip(initial_centre, initial_momentum, gravity)]
    def energy(value):
        return 0.5 * math.fsum(m * math.fsum(x * x for x in v) for m, v in zip(masses, value["velocities"]))
    kinetic = energy(state)
    # Masses are fixed observations, not outputs of a learned map.
    result = {"mass_error": 0.0, "momentum_error": math.dist(momentum, expected_momentum),
              "centre_of_mass_error": math.dist(centre, expected_centre), "kinetic_energy": kinetic,
              "kinetic_energy_difference": None if target is None else kinetic - energy(target)}
    result["mass_pass"] = result["mass_error"] <= protocol["diagnostic_gates"]["mass_error_max"]
    result["momentum_pass"] = result["momentum_error"] <= protocol["diagnostic_gates"]["momentum_with_gravity_error_max"]
    return result


def reference_check(coarse, fine, initial, scene, time, old_snapshot, protocol):
    if coarse is None or fine is None:
        return {"qualified": False, "status": "unavailable", "gates": {"both_resolutions_available": False}}
    bounds = protocol["reference_gates"]
    gap = affine.errors(coarse, fine)
    ballistic = affine.errors(affine.predict(initial, scene["gravity"], time, "known_gravity_ballistic"), fine)
    tolerances = {key: max(bounds["roundoff_floor"], min(bounds[absolute], bounds["relative_to_ballistic_error_max"] * ballistic[key]))
                  for key, absolute in (("position_rmse", "position_gap_absolute_max"),
                                        ("velocity_rmse", "velocity_gap_absolute_max"))}
    ledgers = [diagnostics(s, initial, scene["masses"], scene["gravity"], time, fine, protocol) for s in (coarse, fine)]
    gates = {"position_resolution": gap["position_rmse"] <= tolerances["position_rmse"],
             "velocity_resolution": gap["velocity_rmse"] <= tolerances["velocity_rmse"],
             "mass": max(d["mass_error"] for d in ledgers) <= bounds["mass_error_max"],
             "momentum": max(d["momentum_error"] for d in ledgers) <= bounds["momentum_with_gravity_error_max"],
             "short_prefix_replay": old_snapshot is None or copy_state(old_snapshot) == fine}
    return {"qualified": all(gates.values()), "status": "available", "gates": gates,
            "resolution_gap": gap, "tolerances": tolerances,
            "coarse_state_sha256": digest(coarse), "fine_state_sha256": digest(fine),
            "prefix_checked": old_snapshot is not None,
            "max_reference_momentum_error": max(d["momentum_error"] for d in ledgers)}


def run_scene(scene, models, frozen, protocol):
    initial = copy_state(scene["snapshots"][0])
    masses, dt, checkpoints = scene["masses"], protocol["dt"], protocol["checkpoints"]
    limit = protocol["abort_absolute_component_max"]
    references = {factor: trace(initial, masses, dt / factor, protocol["steps"] * factor,
                                [h * factor for h in checkpoints], numerical_stepper(scene, rk4_step), limit)
                  for factor in protocol["reference_factors"]}
    low, high = protocol["reference_factors"]
    old_targets = {s["step"]: s for s in scene["snapshots"]}
    checks = {h: reference_check(references[low]["snapshots"].get(h * low),
                                 references[high]["snapshots"].get(h * high), initial,
                                 scene, h * dt, old_targets.get(h), protocol) for h in checkpoints}
    result = {key: scene[key] for key in ("scene_id", "group_id", "material_id", "material_condition", "geometry_condition")}
    result.update(particles=len(masses), reference_checks={str(h): check for h, check in checks.items()},
                  reference_runs={str(f): {k: v for k, v in r.items() if k != "snapshots"} for f, r in references.items()}, methods={})
    for method in protocol["methods"]:
        rollout = trace(initial, masses, dt, protocol["steps"], checkpoints,
                        method_stepper(scene, method, models, frozen), limit)
        rows = []
        for h in checkpoints:
            state, target = rollout["snapshots"].get(h), references[high]["snapshots"].get(h * high)
            rows.append({"step": h, "time": h * dt, "available": state is not None,
                         "reference_qualified": checks[h]["qualified"],
                         "state_sha256": None if state is None else digest(state),
                         "errors": None if state is None or target is None else affine.errors(state, target),
                         "diagnostics": None if state is None else diagnostics(state, initial, masses, scene["gravity"], h * dt, target, protocol)})
        result["methods"][method] = {k: v for k, v in rollout.items() if k != "snapshots"} | {"checkpoints": rows}
    reference = {key: scene[key] for key in ("scene_id", "group_id", "material_id", "seed", "config", "gravity", "units",
                                            "particle_ids", "masses", "initial_state_sha256")}
    reference.update(schema="cogniarc.particle-rollout-reference.v1", integrator="independent_dense_rk4",
                     internal_dt=dt / high, completed=references[high]["completed"], failure=references[high]["failure"],
                     snapshots=[{"step": h // high, "time": h * dt / high, **state}
                                for h, state in references[high]["snapshots"].items()])
    return result, reference


def aggregate(rows):
    count = len(rows)
    available = sum(r["available"] for r in rows)
    qualified = sum(r["reference_qualified"] for r in rows)
    usable = count > 0 and available == qualified == count and all(r["errors"] is not None for r in rows)
    diagnostics_complete = available == count and count > 0
    return {"scenes": count, "predictions_available": available, "references_qualified": qualified,
            "mass_pass": sum(r["diagnostics"] is not None and r["diagnostics"]["mass_pass"] for r in rows),
            "momentum_pass": sum(r["diagnostics"] is not None and r["diagnostics"]["momentum_pass"] for r in rows),
            "mean_position_rmse": math.fsum(r["errors"]["position_rmse"] for r in rows) / count if usable else None,
            "mean_velocity_rmse": math.fsum(r["errors"]["velocity_rmse"] for r in rows) / count if usable else None,
            "max_momentum_error": max(r["diagnostics"]["momentum_error"] for r in rows) if diagnostics_complete else None,
            "max_centre_of_mass_error": max(r["diagnostics"]["centre_of_mass_error"] for r in rows) if diagnostics_complete else None}


def grouped_summary(scenes, protocol):
    return {str(h): {method: aggregate([next(r for r in scene["methods"][method]["checkpoints"] if r["step"] == h)
                                       for scene in scenes]) for method in protocol["methods"]} for h in protocol["checkpoints"]}


def run(scenes, models, frozen, protocol):
    rows, references = [], []
    for scene in scenes:
        row, reference = run_scene(scene, models, frozen, protocol)
        rows.append(row)
        references.append(reference)
    report = {"schema": "cogniarc.particle-rollout-evaluation.v1", "protocol": protocol, "scenes": rows,
              "all_references_qualified": all(c["qualified"] for s in rows for c in s["reference_checks"].values()),
              "all_trajectories_completed": all(r["completed"] for s in rows for r in s["methods"].values()),
              "model_selection_changed": False, "macro_mean_scene_errors": grouped_summary(rows, protocol)}
    for key in ("material_condition", "particles"):
        report["by_" + key] = {str(value): grouped_summary([s for s in rows if s[key] == value], protocol)
                                for value in sorted({s[key] for s in rows})}
    return report, references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = read_protocol()
    scenes, models, frozen = load_inputs(protocol)
    report, references = run(scenes, models, frozen, protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = "".join(canonical(r) + "\n" for r in references).encode()
    (args.output_dir / "reference.jsonl").write_bytes(payload)
    report["provenance"] = {"date": "2026-09-13", "protocol_revision": PROTOCOL_REVISION,
                            "protocol_sha256": affine.file_digest(PROTOCOL),
                            "frozen_inputs_sha256": protocol["frozen_inputs_sha256"],
                            "source_sha256": {p: affine.file_digest(affine.ROOT / p) for p in (*data.SOURCE_PATHS, *NEW_SOURCES)},
                            "reference_sha256": sha256(payload).hexdigest(), "reference_bytes": len(payload),
                            "environment": {"python": platform.python_version(), "system": platform.system(),
                                            "machine": platform.machine(), "dependencies": "Python standard library only"},
                            "license": "MIT (repository license)",
                            "command": "python -m experiments.particle_graph.rollouts --output-dir " + str(args.output_dir)}
    affine.write_json(args.output_dir / "evaluation.json", report)
    print(canonical({"scenes": len(scenes), "all_references_qualified": report["all_references_qualified"],
                     "all_trajectories_completed": report["all_trajectories_completed"],
                     "final": report["macro_mean_scene_errors"][str(protocol["steps"])]}))
    return 0 if report["all_references_qualified"] and report["all_trajectories_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

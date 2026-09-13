"""Reserved local-force rollouts, equal-integrator comparisons and controls."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from . import affine, conservation as cons, local_pairs as local, rollouts
from .dense_reference import rk4_step
from .reference import Config, Particle, evaluate, neighbor_pairs
from .validation import canonical, digest


def previous_records():
    old = cons.prior_records()
    path = cons.MANIFEST.parent / "evidence/2026-09-13-conservation/initial.jsonl"
    return old + cons.read_scenes(path, json.loads(cons.MANIFEST.read_text()), old)


def make_scenes(manifest, previous):
    if manifest["schema"] != "cogniarc.local-pair-scenes.v1" or manifest["units"] != "dimensionless":
        raise ValueError("unsupported local-pair scene manifest")
    adapted = {"schema": "cogniarc.particle-conservation-scenes.v1", "units": manifest["units"],
               "defaults": manifest["defaults"], "groups": manifest["groups"], "materials": manifest["materials"],
               "gravities": [{"id": "oblique", "value": manifest["gravity"]}]}
    if not manifest["masses"] or len({m["id"] for m in manifest["masses"]}) != len(manifest["masses"]):
        raise ValueError("duplicate or missing mass conditions")
    for condition in manifest["masses"]:
        if not condition["cycle"] or not affine.finite(condition["cycle"]) or min(condition["cycle"]) <= 0:
            raise ValueError("invalid mass cycle")
    records = []
    for base in cons.make_scenes(adapted, previous):
        n = len(base["masses"])
        for condition in manifest["masses"]:
            cycle = condition["cycle"]
            factors = [cycle[i % len(cycle)] for i in range(n)]
            scale = n * manifest["defaults"]["mass"] / math.fsum(factors)
            masses = [f * scale for f in factors]
            record = base | {"schema": "cogniarc.local-pair-initial.v1", "role": "main",
                             "scene_id": base["scene_id"] + "/" + condition["id"],
                             "mass_condition": condition["id"], "masses": masses, "analytic_ballistic": False}
            record["initial_state_sha256"] = digest([asdict(p) for p in local.particles(record["snapshots"][0], masses)])
            records.append(record)
    defaults = manifest["defaults"]
    for control in manifest["controls"]:
        masses = [defaults["mass"] * m for m in control["mass_multipliers"]]
        initial = {"positions": [[defaults["config"]["radius"] * x for x in p] for p in control["positions_in_radius_units"]],
                   "velocities": control["velocities"]}
        values = local.particles(initial, masses)
        records.append({"schema": "cogniarc.local-pair-initial.v1", "role": "control", "split": "test",
                        "scene_id": "control/" + control["id"], "group_id": "control/" + control["id"],
                        "seed": control["seed"], "geometry_condition": control["id"], "gravity_condition": "oblique",
                        "material_id": "base-material", "material_condition": "interpolation_base", "mass_condition": "control",
                        "config": defaults["config"], "gravity": manifest["gravity"], "units": manifest["units"],
                        "masses": masses, "particle_ids": list(range(len(values))), "initial_state_sha256": digest([asdict(p) for p in values]),
                        "snapshots": [{"step": 0, "time": 0.0, **initial}], "analytic_ballistic": control["analytic_ballistic"]})
    def identity(r):
        return digest({"state": rollouts.copy_state(r["snapshots"][0]), "masses": r["masses"]})
    old_seeds, old_states = {r["seed"] for r in previous}, {identity(r) for r in previous}
    seen_states, seeds = {}, {}
    for row in records:
        if row["seed"] in old_seeds or identity(row) in old_states:
            raise ValueError("new scene overlaps a prior seed or initial state")
        if seeds.setdefault(row["seed"], row["group_id"]) != row["group_id"]:
            raise ValueError("seed shared across local groups")
        if seen_states.setdefault(identity(row), row["group_id"]) != row["group_id"]:
            raise ValueError("initial state shared across local groups")
    if len({r["scene_id"] for r in records}) != len(records):
        raise ValueError("duplicate local scene IDs")
    return records


def read_scenes(path, manifest, previous):
    expected = {r["scene_id"]: r for r in make_scenes(manifest, previous)}
    rows, seen = [], set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = row["scene_id"]
        if key in seen or row != expected.get(key):
            raise ValueError("local initial record differs from its recipe")
        seen.add(key)
        rows.append(row)
    if seen != set(expected):
        raise ValueError("missing local scenes")
    return rows


def physical_force(values, config):
    return evaluate(values, config).accelerations


def force_for(method, model):
    return (lambda values, config: local.accelerations(values, config, model["coefficients"])) if method == "pair_local" else physical_force


def method_stepper(scene, method, models, frozen, pair_model, protocol):
    if method not in ("pair_local", "sph_midpoint"):
        return cons.method_stepper(scene, method, models, frozen, protocol)
    force = force_for(method, pair_model)
    return rollouts.numerical_stepper(scene, lambda values, config, dt, gravity: local.midpoint(values, config, dt, gravity, force))


def force_diagnostics(state, scene, force):
    values, config = local.particles(state, scene["masses"]), Config(**scene["config"])
    acceleration = force(values, config)
    pairs = neighbor_pairs(values, config.radius)
    active = {i for pair in pairs for i in pair}
    isolated = [i for i in range(len(values)) if i not in active]
    net = rollouts.weighted_vector(acceleration, scene["masses"])
    return {"internal_acceleration_rms": math.sqrt(math.fsum(x * x for a in acceleration for x in a) / (3 * len(values))),
            "net_internal_force_norm": math.sqrt(math.fsum(x * x for x in net)), "pairs": len(pairs),
            "isolated_particles": len(isolated),
            "max_isolated_acceleration": max((math.sqrt(math.fsum(x * x for x in acceleration[i])) for i in isolated), default=0.0)}


def run_scene(scene, models, frozen, pair_model, protocol):
    initial = rollouts.copy_state(scene["snapshots"][0])
    masses, dt, checkpoints = scene["masses"], protocol["dt"], protocol["checkpoints"]
    limit, factors = protocol["abort_absolute_component_max"], protocol["reference_factors"]
    references = {f: rollouts.trace(initial, masses, dt / f, protocol["steps"] * f, [h * f for h in checkpoints],
                                    rollouts.numerical_stepper(scene, rk4_step), limit) for f in factors}
    low, high = factors
    checks = {}
    for h in checkpoints:
        fine = references[high]["snapshots"].get(h * high)
        check = rollouts.reference_check(references[low]["snapshots"].get(h * low), fine, initial, scene, h * dt, None, protocol)
        if scene["analytic_ballistic"]:
            error = None if fine is None else affine.errors(fine, affine.predict(initial, scene["gravity"], h * dt, "known_gravity_ballistic"))
            check["analytic_errors"] = error
            check["gates"]["analytic_ballistic"] = error is not None and max(error.values()) <= protocol["analytic_rmse_max"]
            check["qualified"] = all(check["gates"].values())
        checks[h] = check
    result = {k: scene[k] for k in ("scene_id", "group_id", "role", "material_id", "material_condition", "mass_condition",
                                    "gravity_condition", "geometry_condition", "analytic_ballistic")}
    result.update(particles=len(masses), reference_checks={str(h): c for h, c in checks.items()},
                  reference_runs={str(f): {k: v for k, v in t.items() if k != "snapshots"} for f, t in references.items()}, methods={})
    for method in protocol["methods"]:
        trace = rollouts.trace(initial, masses, dt, protocol["steps"], checkpoints,
                                method_stepper(scene, method, models, frozen, pair_model, protocol), limit)
        rows = []
        for h in checkpoints:
            state, target = trace["snapshots"].get(h), references[high]["snapshots"].get(h * high)
            ledger = None if state is None else rollouts.diagnostics(state, initial, masses, scene["gravity"], h * dt, target, protocol)
            if ledger is not None:
                ledger["centre_of_mass_pass"] = ledger["centre_of_mass_error"] <= protocol["centre_of_mass_error_max"]
            analytic_error = None if not scene["analytic_ballistic"] or state is None else affine.errors(state, affine.predict(initial, scene["gravity"], h * dt, "known_gravity_ballistic"))
            rows.append({"step": h, "time": h * dt, "available": state is not None, "reference_qualified": checks[h]["qualified"],
                         "state_sha256": None if state is None else digest(state),
                         "errors": None if state is None or target is None else affine.errors(state, target),
                         "centred_errors": None if state is None or target is None else affine.errors(cons.centred(state, masses), cons.centred(target, masses)),
                         "diagnostics": ledger, "support": None if state is None else cons.support_counts(state, scene["config"]["radius"]),
                         "force_diagnostics": None if state is None or method not in ("pair_local", "sph_midpoint") else force_diagnostics(state, scene, force_for(method, pair_model)),
                         "analytic_errors": analytic_error,
                         "analytic_pass": None if not scene["analytic_ballistic"] else analytic_error is not None and max(analytic_error.values()) <= protocol["analytic_rmse_max"]})
        result["methods"][method] = {k: v for k, v in trace.items() if k != "snapshots"} | {"checkpoints": rows}
    reference = {k: scene[k] for k in ("scene_id", "group_id", "role", "material_id", "mass_condition", "seed", "config", "gravity", "units", "particle_ids", "masses", "initial_state_sha256")}
    reference.update(schema="cogniarc.local-pair-reference.v1", integrator="independent_dense_rk4", internal_dt=dt / high,
                     completed=references[high]["completed"], failure=references[high]["failure"],
                     snapshots=[{"step": h // high, "time": (h // high) * dt, **state,
                                 "support": cons.support_counts(state, scene["config"]["radius"])} for h, state in references[high]["snapshots"].items()])
    return result, reference


def run(scenes, models, frozen, pair_model, protocol, progress=None):
    if any(s["group_id"] in pair_model.get("development_group_ids", []) for s in scenes):
        raise ValueError("evaluation group overlaps model development")
    rows, references = [], []
    for scene in scenes:
        row, reference = run_scene(scene, models, frozen, pair_model, protocol)
        rows.append(row); references.append(reference)
        if progress:
            progress(len(rows), scene["scene_id"])
    main = [s for s in rows if s["role"] == "main"]
    controls = [s for s in rows if s["role"] == "control"]
    local_rows = [r for s in rows for r in s["methods"]["pair_local"]["checkpoints"]]
    report = {"schema": "cogniarc.local-pair-evaluation.v1", "protocol": protocol, "scenes": rows,
              "all_references_qualified": all(c["qualified"] for s in rows for c in s["reference_checks"].values()),
              "all_trajectories_completed": all(t["completed"] for s in rows for t in s["methods"].values()),
              "all_local_ledgers_pass": all(r["diagnostics"] is not None and all(r["diagnostics"][k] for k in ("mass_pass", "momentum_pass", "centre_of_mass_pass")) for r in local_rows),
              "all_local_analytic_controls_pass": all(r["analytic_pass"] for s in controls if s["analytic_ballistic"] for r in s["methods"]["pair_local"]["checkpoints"]),
              "model_selection_changed": False, "main_mean_scene_errors": cons.summary(main, protocol),
              "controls": {s["scene_id"]: s["methods"] for s in controls}}
    for key in ("material_condition", "mass_condition", "particles"):
        report["by_" + key] = {str(v): cons.summary([s for s in main if s[key] == v], protocol) for v in sorted({s[key] for s in main})}
    return report, references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.model_revision) != 40 or any(c not in "0123456789abcdef" for c in args.model_revision):
        raise ValueError("a full model commit SHA is required")
    protocol = local.read_protocol()
    pair_model = local.read_model(args.model_dir, protocol)
    _, models, frozen = rollouts.load_inputs(rollouts.read_protocol())
    manifest, previous = json.loads(local.MANIFEST.read_text()), previous_records()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial_path = args.output_dir / "initial.jsonl"
    initial_artifact = cons.write_jsonl(initial_path, make_scenes(manifest, previous))
    scenes = read_scenes(initial_path, manifest, previous)
    report, references = run(scenes, models, frozen, pair_model, protocol,
                             lambda n, name: print(canonical({"scenes_completed": n, "scene": name}), flush=True))
    report["artifacts"] = {"initial": initial_artifact, "reference": cons.write_jsonl(args.output_dir / "reference.jsonl", references)}
    report["provenance"] = pair_model["provenance"] | {
        "model_revision": args.model_revision, "model_sha256": affine.file_digest(args.model_dir / "model.json"),
        "command": "python -m experiments.particle_graph.local_pair_evaluation --model-dir " + str(args.model_dir)
                   + " --model-revision " + args.model_revision + " --output-dir " + str(args.output_dir)}
    affine.write_json(args.output_dir / "evaluation.json", report)
    print(canonical({k: report[k] for k in ("all_references_qualified", "all_trajectories_completed", "all_local_ledgers_pass", "all_local_analytic_controls_pass")}))
    print(canonical(report["main_mean_scene_errors"][str(protocol["steps"])]))
    return 0 if all(report[k] for k in ("all_references_qualified", "all_trajectories_completed", "all_local_ledgers_pass", "all_local_analytic_controls_pass")) else 1


if __name__ == "__main__":
    raise SystemExit(main())

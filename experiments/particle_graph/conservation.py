"""Fixed gravity/momentum projection on new reserved synthetic rollouts."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform

from . import affine, material_data as data, materials, rollouts
from .dense_reference import rk4_step
from .reference import Config
from .validation import canonical, digest, make_scene

MANIFEST = Path(__file__).with_name("conservation_manifest.json")
PROTOCOL = Path(__file__).with_name("conservation_protocol.json")
PROTOCOL_REVISION = "e254b3acab69dcdd4b59f63af51216df9bfd0548"
PROTOCOL_SHA256 = "7a280ceffe47732cf9b479d0f0ba000c243eee2b0b34ba6f199744857fa604bf"
NEW_SOURCES = ("experiments/particle_graph/conservation.py",
               "tests/test_particle_graph_conservation.py")


def read_protocol():
    if affine.file_digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("conservation protocol checksum mismatch")
    protocol = json.loads(PROTOCOL.read_text())
    for name, expected in protocol["frozen_inputs_sha256"].items():
        if affine.file_digest(affine.ROOT / name) != expected:
            raise ValueError("frozen conservation input checksum mismatch: " + name)
    return protocol


def prior_records():
    """Read all verified prior partitions solely to check reserved-scene isolation."""
    records = affine.read_corpus(affine.read_protocol(), {"train", "validation", "test"})
    manifest, protocol = data.read_config()
    report = data.read_report(rollouts.DATA_DIR, manifest, protocol)
    for split in data.SPLITS:
        records.extend(data.read_split(rollouts.DATA_DIR, split, manifest, report))
    return records


def validate_manifest(manifest, previous):
    if (manifest["schema"] != "cogniarc.particle-conservation-scenes.v1"
            or manifest["units"] != "dimensionless"):
        raise ValueError("unsupported conservation scene manifest")
    defaults = manifest["defaults"]
    if (not affine.finite([defaults[k] for k in ("mass", "spacing", "jitter", "velocity_scale")])
            or min(defaults["mass"], defaults["spacing"]) <= 0
            or min(defaults["jitter"], defaults["velocity_scale"]) < 0):
        raise ValueError("invalid scene defaults")
    old_seeds = {r["seed"] for r in previous}
    groups = manifest["groups"]
    if not groups or len({g["id"] for g in groups}) != len(groups) or len({g["seed"] for g in groups}) != len(groups):
        raise ValueError("duplicate or missing groups/seeds")
    for group in groups:
        if type(group["seed"]) is not int or group["seed"] in old_seeds:
            raise ValueError("reserved seed overlaps prior corpus")
        if (len(group["shape"]) != 3 or any(type(n) is not int or n < 1 for n in group["shape"])
                or math.prod(group["shape"]) > 64):
            raise ValueError("invalid reserved shape")
    for key in ("gravities", "materials"):
        if not manifest[key] or len({r["id"] for r in manifest[key]}) != len(manifest[key]):
            raise ValueError("duplicate or missing conditions")
    for gravity in manifest["gravities"]:
        if len(gravity["value"]) != 3 or not affine.finite(gravity["value"]):
            raise ValueError("invalid gravity")
    for material in manifest["materials"]:
        config = defaults["config"] | {k: material[k] for k in ("stiffness", "viscosity")}
        if not affine.finite(list(config.values())) or min(config.values()) <= 0:
            raise ValueError("invalid material")
        Config(**config)


def make_scenes(manifest, previous):
    validate_manifest(manifest, previous)
    # Compare actual initial x/v/m values as well as seeds, independently of IDs.
    def initial_hash(record):
        return digest({"state": rollouts.copy_state(record["snapshots"][0]), "masses": record["masses"]})
    old_hashes = {initial_hash(r) for r in previous}
    seen_groups, result = {}, []
    for group in manifest["groups"]:
        for gravity in manifest["gravities"]:
            for material in manifest["materials"]:
                recipe = group | {"gravity": gravity["value"],
                                  "config": {k: material[k] for k in ("stiffness", "viscosity")}}
                particles, config, acceleration = make_scene(manifest, recipe)
                record = {"schema": "cogniarc.particle-conservation-initial.v1",
                          "scene_id": "/".join((group["id"], gravity["id"], material["id"])),
                          "group_id": group["id"], "seed": group["seed"], "split": "test",
                          "material_id": material["id"], "material_condition": material["condition"],
                          "gravity_condition": gravity["id"], "geometry_condition": "new_reserved_seed",
                          "config": asdict(config), "gravity": list(acceleration), "units": manifest["units"],
                          "particle_ids": list(range(len(particles))), "masses": [p.mass for p in particles],
                          "initial_state_sha256": digest([asdict(p) for p in particles]),
                          "snapshots": [{"step": 0, "time": 0.0,
                                         "positions": [list(p.position) for p in particles],
                                         "velocities": [list(p.velocity) for p in particles]}]}
                identity = initial_hash(record)
                if identity in old_hashes:
                    raise ValueError("reserved initial state overlaps prior corpus")
                if seen_groups.setdefault(identity, group["id"]) != group["id"]:
                    raise ValueError("initial state shared across reserved groups")
                result.append(record)
    return result


def read_scenes(path, manifest, previous):
    expected = {r["scene_id"]: r for r in make_scenes(manifest, previous)}
    rows, seen = [], set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        identifier = row["scene_id"]
        if identifier in seen or row != expected.get(identifier):
            raise ValueError("initial record differs from its reserved recipe")
        seen.add(identifier)
        rows.append(row)
    if seen != set(expected):
        raise ValueError("missing reserved scenes")
    return rows


def mean_vector(vectors, masses):
    return [x / math.fsum(masses) for x in rollouts.weighted_vector(vectors, masses)]


def project_proposal(state, proposed, masses, gravity, dt, limit=1e6):
    """Translate the proposal to the exact constant-gravity mean x/v for this step."""
    if (not masses or not affine.finite(masses) or min(masses) <= 0
            or len(gravity) != 3 or not affine.finite(gravity)
            or not affine.finite([dt, limit]) or min(dt, limit) <= 0):
        raise ValueError("invalid projection controls")
    rollouts.check_state(state, len(masses), limit)
    # Validate BEFORE projection: a uniform invalid impulse must not be hidden.
    rollouts.check_state(proposed, len(masses), limit)
    current_position = mean_vector(state["positions"], masses)
    current_velocity = mean_vector(state["velocities"], masses)
    desired = {"positions": [x + dt * v + 0.5 * dt ** 2 * g
                             for x, v, g in zip(current_position, current_velocity, gravity)],
               "velocities": [v + dt * g for v, g in zip(current_velocity, gravity)]}
    result = {}
    for key in ("positions", "velocities"):
        proposed_mean = mean_vector(proposed[key], masses)
        shift = [wanted - actual for wanted, actual in zip(desired[key], proposed_mean)]
        result[key] = [[x + correction for x, correction in zip(vector, shift)] for vector in proposed[key]]
    rollouts.check_state(result, len(masses), limit)
    return result


def method_stepper(scene, method, models, frozen, protocol):
    raw_method = protocol["projected_methods"].get(method)
    if raw_method is None:
        return rollouts.method_stepper(scene, method, models, frozen)
    advance = rollouts.method_stepper(scene, raw_method, models, frozen)
    return lambda state, dt: project_proposal(state, advance(state, dt), scene["masses"],
                                               scene["gravity"], dt, protocol["abort_absolute_component_max"])


def centred(state, masses):
    result = {}
    for key in ("positions", "velocities"):
        mean = mean_vector(state[key], masses)
        result[key] = [[x - m for x, m in zip(vector, mean)] for vector in state[key]]
    return result


def support_counts(state, radius):
    positions = state["positions"]
    degrees = [0] * len(positions)
    pairs = 0
    for i, p in enumerate(positions):
        for j in range(i + 1, len(positions)):
            if math.dist(p, positions[j]) < radius:
                pairs += 1
                degrees[i] += 1
                degrees[j] += 1
    return {"neighbour_pairs": pairs, "isolated_particles": degrees.count(0)}


def run_scene(scene, models, frozen, protocol):
    initial = rollouts.copy_state(scene["snapshots"][0])
    masses, dt, checkpoints = scene["masses"], protocol["dt"], protocol["checkpoints"]
    limit, factors = protocol["abort_absolute_component_max"], protocol["reference_factors"]
    references = {f: rollouts.trace(initial, masses, dt / f, protocol["steps"] * f,
                                    [h * f for h in checkpoints], rollouts.numerical_stepper(scene, rk4_step), limit)
                  for f in factors}
    low, high = factors
    checks = {h: rollouts.reference_check(references[low]["snapshots"].get(h * low),
                                          references[high]["snapshots"].get(h * high), initial,
                                          scene, h * dt, None, protocol) for h in checkpoints}
    traces = {method: rollouts.trace(initial, masses, dt, protocol["steps"], checkpoints,
                                     method_stepper(scene, method, models, frozen, protocol), limit)
              for method in protocol["methods"]}
    result = {key: scene[key] for key in ("scene_id", "group_id", "material_id", "material_condition",
                                         "gravity_condition", "geometry_condition")}
    result.update(particles=len(masses), reference_checks={str(h): check for h, check in checks.items()},
                  reference_runs={str(f): {k: v for k, v in r.items() if k != "snapshots"} for f, r in references.items()},
                  methods={}, paired_centred_states={})
    for method, trace in traces.items():
        rows = []
        for h in checkpoints:
            state, target = trace["snapshots"].get(h), references[high]["snapshots"].get(h * high)
            ledger = None if state is None else rollouts.diagnostics(state, initial, masses, scene["gravity"], h * dt, target, protocol)
            if ledger is not None:
                ledger["centre_of_mass_pass"] = ledger["centre_of_mass_error"] <= protocol["centre_of_mass_error_max"]
            rows.append({"step": h, "time": h * dt, "available": state is not None,
                         "reference_qualified": checks[h]["qualified"],
                         "state_sha256": None if state is None else digest(state),
                         "errors": None if state is None or target is None else affine.errors(state, target),
                         "centred_errors": None if state is None or target is None else affine.errors(centred(state, masses), centred(target, masses)),
                         "diagnostics": ledger,
                         "support": None if state is None else support_counts(state, scene["config"]["radius"])})
        result["methods"][method] = {k: v for k, v in trace.items() if k != "snapshots"} | {"checkpoints": rows}
    for corrected, raw in protocol["projected_methods"].items():
        rows = []
        for h in checkpoints:
            a, b = traces[corrected]["snapshots"].get(h), traces[raw]["snapshots"].get(h)
            error = None if a is None or b is None else affine.errors(centred(a, masses), centred(b, masses))
            rows.append({"step": h, "available": error is not None, "centred_difference": error,
                         "within_roundoff_gate": error is not None and max(error.values()) <= protocol["centred_pair_state_rmse_max"]})
        result["paired_centred_states"][corrected] = {"raw_method": raw, "checkpoints": rows}
    reference = {key: scene[key] for key in ("scene_id", "group_id", "material_id", "gravity_condition", "seed",
                                            "config", "gravity", "units", "particle_ids", "masses", "initial_state_sha256")}
    reference.update(schema="cogniarc.particle-conservation-reference.v1", integrator="independent_dense_rk4",
                     internal_dt=dt / high, completed=references[high]["completed"], failure=references[high]["failure"],
                     snapshots=[{"step": h // high, "time": (h // high) * dt, **state,
                                 "support": support_counts(state, scene["config"]["radius"])}
                                for h, state in references[high]["snapshots"].items()])
    return result, reference


def aggregate(rows):
    result = rollouts.aggregate(rows)
    usable = result["mean_position_rmse"] is not None and all(r["centred_errors"] is not None for r in rows)
    for key in ("position_rmse", "velocity_rmse"):
        result["mean_centred_" + key] = math.fsum(r["centred_errors"][key] for r in rows) / len(rows) if usable else None
    result["centre_of_mass_pass"] = sum(r["diagnostics"] is not None and r["diagnostics"]["centre_of_mass_pass"] for r in rows)
    return result


def summary(scenes, protocol):
    return {str(h): {method: aggregate([next(r for r in scene["methods"][method]["checkpoints"] if r["step"] == h)
                                       for scene in scenes]) for method in protocol["methods"]} for h in protocol["checkpoints"]}


def run(scenes, models, frozen, protocol, progress=None):
    rows, references = [], []
    for scene in scenes:
        row, reference = run_scene(scene, models, frozen, protocol)
        rows.append(row)
        references.append(reference)
        if progress:
            progress(len(rows), scene["scene_id"])
    projected_rows = [r for s in rows for m in protocol["projected_methods"] for r in s["methods"][m]["checkpoints"]]
    report = {"schema": "cogniarc.particle-conservation-evaluation.v1", "protocol": protocol, "scenes": rows,
              "all_references_qualified": all(c["qualified"] for s in rows for c in s["reference_checks"].values()),
              "all_trajectories_completed": all(r["completed"] for s in rows for r in s["methods"].values()),
              "all_projected_ledgers_pass": all(r["diagnostics"] is not None and all(r["diagnostics"][k] for k in
                                                ("mass_pass", "momentum_pass", "centre_of_mass_pass")) for r in projected_rows),
              "all_paired_centred_states_match": all(r["within_roundoff_gate"] for s in rows
                                                     for p in s["paired_centred_states"].values() for r in p["checkpoints"]),
              "model_selection_changed": False, "macro_mean_scene_errors": summary(rows, protocol)}
    for key in ("material_condition", "gravity_condition", "particles"):
        report["by_" + key] = {str(value): summary([s for s in rows if s[key] == value], protocol)
                                for value in sorted({s[key] for s in rows})}
    return report, references


def write_jsonl(path, rows):
    path.write_text("".join(canonical(row) + "\n" for row in rows))
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": affine.file_digest(path), "records": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = read_protocol()
    manifest = json.loads(MANIFEST.read_text())
    _, models, frozen = rollouts.load_inputs(rollouts.read_protocol())
    previous = prior_records()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial_path = args.output_dir / "initial.jsonl"
    initial_artifact = write_jsonl(initial_path, make_scenes(manifest, previous))
    scenes = read_scenes(initial_path, manifest, previous)
    report, references = run(scenes, models, frozen, protocol,
                             lambda count, scene: print(canonical({"scenes_completed": count, "scene": scene}), flush=True))
    report["artifacts"] = {"initial": initial_artifact,
                            "reference": write_jsonl(args.output_dir / "reference.jsonl", references)}
    paths = (*data.SOURCE_PATHS, "experiments/particle_graph/rollouts.py", *NEW_SOURCES)
    report["provenance"] = {"date": "2026-09-13", "protocol_revision": PROTOCOL_REVISION,
                            "protocol_sha256": affine.file_digest(PROTOCOL),
                            "frozen_inputs_sha256": protocol["frozen_inputs_sha256"],
                            "source_sha256": {p: affine.file_digest(affine.ROOT / p) for p in paths},
                            "environment": {"python": platform.python_version(), "system": platform.system(),
                                            "machine": platform.machine(), "dependencies": "Python standard library only"},
                            "license": "MIT (repository license)",
                            "command": "python -m experiments.particle_graph.conservation --output-dir " + str(args.output_dir)}
    affine.write_json(args.output_dir / "evaluation.json", report)
    print(canonical({k: report[k] for k in ("all_references_qualified", "all_trajectories_completed",
                                           "all_projected_ledgers_pass", "all_paired_centred_states_match")}))
    print(canonical(report["macro_mean_scene_errors"][str(protocol["steps"])]))
    return 0 if report["all_references_qualified"] and report["all_trajectories_completed"] and report["all_projected_ledgers_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

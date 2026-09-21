"""Offline, source-locked diagnostics of the frozen local pair-force model."""

import argparse
from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import platform
import re

from . import affine, conservation, local_pair_evaluation as evaluation
from . import local_pairs as local, rollouts
from .dense_reference import accelerations as dense_accelerations
from .reference import Config, evaluate
from .validation import digest

PROTOCOL = Path(__file__).with_name("force_diagnostic_protocol.json")
PROTOCOL_SHA256 = "bc94d98d8b4ff40e7fe1646e127f6bcb9fe46bf884cacd146bd82f5633a67704"
PROTOCOL_REVISION = "1d6d8fa0038d6cd6dd2ed7982729e861345512f6"
MODES = ("pressure", "viscosity", "total")
IMPLEMENTATIONS = ("dense", "local")
NEW_SOURCES = ("experiments/particle_graph/force_diagnostics.py",
               "tests/test_particle_graph_force_diagnostics.py")


def read_protocol():
    if affine.file_digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("force diagnostic protocol checksum mismatch")
    result = json.loads(PROTOCOL.read_text())
    for name, expected in result["frozen_inputs_sha256"].items():
        if affine.file_digest(affine.ROOT / name) != expected:
            raise ValueError("frozen diagnostic input changed: " + name)
    return result


def unique_index(rows, field):
    result = {}
    for row in rows:
        key = row[field]
        if key in result:
            raise ValueError("duplicate record identity: " + str(key))
        result[key] = row
    return result


def observation(scene, path, snapshot):
    metadata = {k: scene[k] for k in ("scene_id", "group_id", "role", "config", "masses",
                                      "particle_ids", "units", "material_condition", "mass_condition")}
    return metadata | {"path": path, "step": snapshot["step"], "time": snapshot["time"],
                       "state": rollouts.copy_state(snapshot)}


def load_observations(model, protocol):
    """Reuse qualified references and require exact replay of published states."""
    old_protocol = local.read_protocol()
    scenes = evaluation.read_scenes(local.DIRECTORY / "initial.jsonl",
                                   json.loads(local.MANIFEST.read_text()), evaluation.previous_records())
    references = unique_index([json.loads(line) for line in
                               (local.DIRECTORY / "reference.jsonl").read_text().splitlines()], "scene_id")
    report = json.loads((local.DIRECTORY / "evaluation.json").read_text())
    scores = unique_index(report["scenes"], "scene_id")
    expected_ids = {s["scene_id"] for s in scenes}
    if (set(references) != expected_ids or set(scores) != expected_ids or len(scenes) != 15
            or report["schema"] != "cogniarc.local-pair-evaluation.v1"
            or not report["all_references_qualified"] or not report["all_trajectories_completed"]):
        raise ValueError("incomplete or unqualified prior evaluation")
    observations, matches = [], {"reference_checkpoints": 0, "replayed_checkpoints": 0,
                                "reference_initials": 0, "completed_replays": 0}
    for scene in scenes:
        ref, score = references[scene["scene_id"]], scores[scene["scene_id"]]
        if ref["schema"] != "cogniarc.local-pair-reference.v1" or not ref["completed"] or ref["failure"] is not None:
            raise ValueError("invalid prior reference")
        for name in ("group_id", "role", "config", "masses", "particle_ids", "units", "initial_state_sha256"):
            if ref[name] != scene[name]:
                raise ValueError("reference metadata mismatch: " + name)
        snapshots = unique_index(ref["snapshots"], "step")
        if sorted(snapshots) != protocol["steps"]:
            raise ValueError("missing or unexpected reference step")
        initial = rollouts.copy_state(scene["snapshots"][0])
        if rollouts.copy_state(snapshots[0]) != initial:
            raise ValueError("reference initial state mismatch")
        matches["reference_initials"] += 1
        force = evaluation.force_for("pair_local", model)
        advance = rollouts.numerical_stepper(scene, lambda ps, c, dt, g: local.midpoint(ps, c, dt, g, force))
        replay = rollouts.trace(initial, scene["masses"], old_protocol["dt"], old_protocol["steps"],
                                old_protocol["checkpoints"], advance, old_protocol["abort_absolute_component_max"])
        if not replay["completed"] or sorted(replay["snapshots"]) != protocol["steps"]:
            raise ValueError("frozen local replay did not complete")
        matches["completed_replays"] += 1
        published = unique_index(score["methods"]["pair_local"]["checkpoints"], "step")
        if sorted(published) != old_protocol["checkpoints"]:
            raise ValueError("missing published local checkpoint")
        for step in protocol["steps"]:
            state = rollouts.copy_state(snapshots[step])
            rollouts.check_state(state, len(scene["masses"]), old_protocol["abort_absolute_component_max"])
            if snapshots[step]["time"] != step * old_protocol["dt"]:
                raise ValueError("reference observation time mismatch")
            if step:
                check, saved = score["reference_checks"][str(step)], published[step]
                if not check["qualified"] or digest(state) != check["fine_state_sha256"]:
                    raise ValueError("reference qualification or state digest mismatch")
                if (not saved["available"] or not saved["reference_qualified"]
                        or digest(replay["snapshots"][step]) != saved["state_sha256"]):
                    raise ValueError("local replay differs from frozen published state")
                matches["reference_checkpoints"] += 1
                matches["replayed_checkpoints"] += 1
            observations.append(observation(scene, "reference", snapshots[step]))
            observations.append(observation(scene, "pair_local_replay", {
                "step": step, "time": step * old_protocol["dt"], **replay["snapshots"][step]}))
    return observations, matches


def make_controls(protocol):
    """Deterministic static diagnostic recipes; these are not training samples."""
    result = []
    grid = protocol["controlled_pair_grid"]
    def record(name, path, config, positions, velocities, masses, parameters):
        state = {"positions": positions, "velocities": velocities}
        local.particles(state, masses)
        return {"scene_id": name, "group_id": path, "role": "static_control", "path": path,
                "step": 0, "time": 0.0, "units": "dimensionless", "state": state,
                "config": asdict(config), "masses": masses, "particle_ids": list(range(len(masses))),
                "material_condition": "base", "mass_condition": "controlled", "parameters": parameters}
    config = Config(**{k: grid[k] for k in ("radius", "rest_density", "stiffness", "viscosity")})
    for total in grid["total_masses"]:
        for ratio in grid["mass_ratios"]:
            masses = [total / (1 + ratio), total - total / (1 + ratio)]
            for distance in grid["separations_over_radius"]:
                radius = distance * config.radius
                result.append(record(f"pair/M{total:g}/R{ratio:g}/d{distance:g}", "static_pair", config,
                                     [[-radius / 2, 0, 0], [radius / 2, 0, 0]],
                                     [[-0.2, 0.1, 0], [0.2, -0.1, 0]], masses,
                                     {"total_mass": total, "mass_ratio": ratio, "separation_over_radius": distance}))
    if len(result) != grid["expected_states"]:
        raise ValueError("unexpected static pair recipe count")
    context = protocol["context_controls"]
    config = Config(**{k: context[k] for k in ("radius", "rest_density", "stiffness", "viscosity")})
    for count in context["extra_counts"]:
        positions = context["target_positions_in_radius_units"] + context["extra_positions_in_radius_units"][:count]
        result.append(record(f"context/neighbours{count}", "context", config,
                             [[config.radius * x for x in p] for p in positions],
                             context["target_velocities"] + [list(context["extra_velocity"]) for _ in range(count)],
                             context["target_masses"] + [context["extra_mass"]] * count, {"extra_neighbours": count}))
    if len(result) != grid["expected_states"] + context["expected_states"]:
        raise ValueError("unexpected context recipe count")
    unique_index(result, "scene_id")
    return result


def bucket(value, edges):
    index = bisect_right(edges, value)
    if index == 0:
        return "<" + format(edges[0], ".13g")
    if index == len(edges):
        return ">=" + format(edges[-1], ".13g")
    return f"[{edges[index - 1]:.13g},{edges[index]:.13g})"


def labels(edges):
    return [bucket(-math.inf, edges)] + [bucket(edge, edges) for edge in edges]


def density_graph(values, config):
    """Independent dense distances, including self density, for descriptors."""
    h = config.radius
    distances = [[math.dist(a.position, b.position) for b in values] for a in values]
    kernel = 315 / (64 * math.pi * h ** 9)
    densities = [math.fsum(b.mass * kernel * max(0.0, h * h - distance * distance) ** 3
                           for b, distance in zip(values, row)) for row in distances]
    pairs = [(i, j, distances[i][j]) for i in range(len(values)) for j in range(i + 1, len(values))
             if distances[i][j] < h]
    counts = [0] * len(values)
    for i, j, _ in pairs:
        counts[i] += 1
        counts[j] += 1
    neighbour_densities = [math.fsum(b.mass * kernel * max(0.0, h * h - distances[i][j] ** 2) ** 3
                                     for j, b in enumerate(values) if j != i) for i in range(len(values))]
    return densities, neighbour_densities, counts, pairs


def pair_forces(values, config, coefficients, densities, pair):
    """Diagnostic force on i; its negative acts on j. Never used for stepping."""
    i, j, distance = pair
    a, b = values[i], values[j]
    direction = [(x - y) / distance if distance else 0.0 for x, y in zip(a.position, b.position)]
    relative_velocity = [v - u for u, v in zip(a.velocity, b.velocity)]
    q, common = 1 - distance / config.radius, a.mass * b.mass
    pressures = [max(0.0, config.stiffness * (densities[k] - config.rest_density)) for k in (i, j)]
    spiky = 45 / (math.pi * config.radius ** 6)
    physical = common / (densities[i] * densities[j])
    dense_pressure = physical * sum(pressures) * 0.5 * spiky * (config.radius - distance) ** 2
    dense_viscosity = config.viscosity * physical * spiky * (config.radius - distance)
    local_pressure = common * config.stiffness * (coefficients[0] * q ** 2 + coefficients[1] * q ** 3)
    local_viscosity = common * config.viscosity * (coefficients[2] * q + coefficients[3] * q ** 2)
    result = {}
    for name, pressure, viscosity in (("dense", dense_pressure, dense_viscosity),
                                      ("local", local_pressure, local_viscosity)):
        result[name] = {"pressure": [pressure * n for n in direction],
                        "viscosity": [viscosity * v for v in relative_velocity]}
        result[name]["total"] = [p + v for p, v in zip(result[name]["pressure"], result[name]["viscosity"])]
    return result


def vector_stats(dense, predicted):
    count = len(dense)
    if not count or len(predicted) != count:
        raise ValueError("nonempty paired vectors required")
    return {"mse": math.fsum((a - b) ** 2 for x, y in zip(dense, predicted) for a, b in zip(x, y)) / (3 * count),
            "reference_ms": math.fsum(a * a for x in dense for a in x) / (3 * count),
            "local_ms": math.fsum(a * a for x in predicted for a in x) / (3 * count)}


def force_stats(edges):
    return {"items": len(edges), "components": {
        mode: vector_stats([e["dense"][mode] for e in edges], [e["local"][mode] for e in edges]) for mode in MODES}}


def close_vectors(a, b, protocol):
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        return False
    atol, rtol = (protocol["checks"][k] for k in ("comparison_absolute_tolerance", "comparison_relative_tolerance"))
    return all(math.isfinite(x) and math.isfinite(y) and abs(x - y) <= atol + rtol * max(abs(x), abs(y))
               for v, w in zip(a, b) for x, y in zip(v, w))


def diagnose(observed, coefficients, protocol):
    values, config = local.particles(observed["state"], observed["masses"]), Config(**observed["config"])
    configurations = {"pressure": replace(config, viscosity=0), "viscosity": replace(config, stiffness=0), "total": config}
    acceleration = {
        name: {mode: [list(a) for a in force(values, c)] for mode, c in configurations.items()}
        for name, force in (("dense", dense_accelerations),
                            ("local", lambda ps, c: local.accelerations(ps, c, coefficients)))}
    densities, neighbour_densities, counts, pairs = density_graph(values, config)
    pressures = [max(0.0, config.stiffness * (rho - config.rest_density)) for rho in densities]
    reconstructed = {name: {mode: [[0.0] * 3 for _ in values] for mode in MODES} for name in IMPLEMENTATIONS}
    groups = {name: defaultdict(list) for name in ("all", "density_ratio", "separation_over_radius", "mass_ratio", "pressure_zero")}
    target_pair, coincident_pressure_zero = None, True
    for pair in pairs:
        i, j, distance = pair
        force = pair_forces(values, config, coefficients, densities, pair)
        for name in IMPLEMENTATIONS:
            for mode in MODES:
                for axis, value in enumerate(force[name][mode]):
                    reconstructed[name][mode][i][axis] += value
                    reconstructed[name][mode][j][axis] -= value
            if distance == 0:
                coincident_pressure_zero &= all(x == 0 for x in force[name]["pressure"])
        descriptors = {"density_ratio": min(densities[i], densities[j]) / config.rest_density,
                       "separation_over_radius": distance / config.radius,
                       "mass_ratio": max(values[i].mass, values[j].mass) / min(values[i].mass, values[j].mass)}
        groups["all"]["all"].append(force)
        groups["pressure_zero"]["both_zero" if pressures[i] == pressures[j] == 0 else "active"].append(force)
        for name, value in descriptors.items():
            groups[name][bucket(value, protocol["bins"][name]["edges"])].append(force)
        if observed["path"] == "context" and (i, j) == (0, 1):
            target_pair = {"indices": [i, j], **descriptors, "densities": [densities[i], densities[j]],
                           "pressures": [pressures[i], pressures[j]], "forces": force}
    checks = {"coincident_pair_pressure_zero": coincident_pressure_zero,
              "density_matches_existing_operator": close_vectors([densities], [evaluate(values, config).densities], protocol)}
    details = {}
    atol, rtol = (protocol["checks"][k] for k in ("comparison_absolute_tolerance", "comparison_relative_tolerance"))
    for name in IMPLEMENTATIONS:
        combined = [[p + v for p, v in zip(pv, vv)] for pv, vv in
                    zip(acceleration[name]["pressure"], acceleration[name]["viscosity"])]
        checks[name + "_components_sum"] = close_vectors(combined, acceleration[name]["total"], protocol)
        for mode in MODES:
            rebuilt = [[x / particle.mass for x in force] for particle, force in zip(values, reconstructed[name][mode])]
            actual = acceleration[name][mode]
            checks[name + "_" + mode + "_reconstruction"] = close_vectors(rebuilt, actual, protocol)
            details[name + "_" + mode + "_reconstruction_max"] = max(abs(x - y) for v, w in zip(rebuilt, actual) for x, y in zip(v, w))
            net = [math.fsum(p.mass * a[k] for p, a in zip(values, actual)) for k in range(3)]
            scale = [math.fsum(abs(p.mass * a[k]) for p, a in zip(values, actual)) for k in range(3)]
            checks[name + "_" + mode + "_net_force"] = all(abs(x) <= atol + rtol * s for x, s in zip(net, scale))
            checks[name + "_" + mode + "_isolated_zero"] = all(
                all(x == 0 for x in actual[i]) for i, count in enumerate(counts) if count == 0)
            details[name + "_" + mode + "_net_force_max"] = max(map(abs, net))
        terms = [p.mass * v * a for p, vector in zip(values, acceleration[name]["viscosity"])
                 for v, a in zip(p.velocity, vector)]
        details[name + "_viscous_power"] = math.fsum(terms)
        checks[name + "_viscous_dissipation"] = math.fsum(terms) <= atol + rtol * math.fsum(map(abs, terms))
    checks["dense_zero_pressure_below_threshold"] = (any(rho > config.rest_density for rho in densities)
                                                     or all(x == 0 for a in acceleration["dense"]["pressure"] for x in a))
    result = {"schema": "cogniarc.force-diagnostic-observation.v1", "observation": observed,
              "observation_sha256": digest(observed), "densities": densities,
              "neighbour_densities": neighbour_densities, "neighbour_counts": counts, "pressures": pressures,
              "accelerations": acceleration,
              "pair_statistics": {kind: {label: force_stats(edges) for label, edges in bins.items()}
                                  for kind, bins in groups.items()},
              "target_pair": target_pair, "checks": checks, "check_details": details}
    # Also reject nonfinite descriptors, statistics and diagnostics before JSON writing.
    json.dumps(result, allow_nan=False)
    return result


def read_records(path, observations, coefficients, protocol):
    """The actual consumer verifies recipes/identities and recomputes diagnostics."""
    def identity(observed):
        return observed["path"], observed["scene_id"], observed["step"]
    expected = {identity(o): o for o in observations}
    if len(expected) != len(observations):
        raise ValueError("duplicate expected observation")
    result, seen = [], set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = identity(row["observation"])
        if key in seen or key not in expected or row != diagnose(expected[key], coefficients, protocol):
            raise ValueError("diagnostic record differs from frozen observation/computation")
        seen.add(key)
        result.append(row)
    if seen != set(expected):
        raise ValueError("missing diagnostic observations")
    return result


def particle_stats(row, indices=None):
    indices = list(range(len(row["densities"]))) if indices is None else list(indices)
    return {"items": len(indices), "components": {
        mode: vector_stats([row["accelerations"]["dense"][mode][i] for i in indices],
                           [row["accelerations"]["local"][mode][i] for i in indices]) for mode in MODES}}


def combine(entries):
    """Equal weight per populated state, then per item/coordinate in that state."""
    result = {"states": len(entries), "scenes": len({r["observation"]["scene_id"] for r, _ in entries}),
              "groups": len({r["observation"]["group_id"] for r, _ in entries}),
              "items": sum(s["items"] for _, s in entries), "components": {}}
    for mode in MODES:
        stats = [s["components"][mode] for _, s in entries]
        means = {key: math.sqrt(math.fsum(s[key] for s in stats) / len(stats)) if stats else None
                 for key in ("mse", "reference_ms", "local_ms")}
        result["components"][mode] = {"rmse": means["mse"], "reference_rms": means["reference_ms"],
                                      "local_rms": means["local_ms"],
                                      "relative_rmse": means["mse"] / means["reference_ms"]
                                      if means["reference_ms"] is not None and means["reference_ms"] > 1e-12 else None}
    return result


def summarize_path(rows, protocol):
    result = {"all_particles": combine([(r, particle_stats(r)) for r in rows])}
    for field in ("step", "material_condition", "mass_condition", "particles"):
        groups = defaultdict(list)
        for row in rows:
            value = len(row["densities"]) if field == "particles" else row["observation"][field]
            groups[str(value)].append((row, particle_stats(row)))
        result["by_" + field] = {key: combine(entries) for key, entries in groups.items()}
    for name in ("density_ratio", "neighbours"):
        edges = protocol["bins"][name]["edges"]
        groups = {label: [] for label in labels(edges)}
        for row in rows:
            values = ([rho / row["observation"]["config"]["rest_density"] for rho in row["densities"]]
                      if name == "density_ratio" else row["neighbour_counts"])
            for label in groups:
                members = [i for i, value in enumerate(values) if bucket(value, edges) == label]
                if members:
                    groups[label].append((row, particle_stats(row, members)))
        result["particle_by_" + name] = {label: combine(entries) for label, entries in groups.items()}
    result["pairs"] = {}
    for name in ("all", "density_ratio", "separation_over_radius", "mass_ratio", "pressure_zero"):
        bin_names = ["all"] if name == "all" else (["both_zero", "active"] if name == "pressure_zero"
                                                   else labels(protocol["bins"][name]["edges"]))
        result["pairs"][name] = {label: combine([(r, r["pair_statistics"][name][label]) for r in rows
                                                if label in r["pair_statistics"][name]]) for label in bin_names}
    return result


def summarize(rows, protocol):
    failed = [{"path": r["observation"]["path"], "scene_id": r["observation"]["scene_id"],
               "step": r["observation"]["step"], "checks": [k for k, passed in r["checks"].items() if not passed]}
              for r in rows if not all(r["checks"].values())]
    contexts = [r for r in rows if r["observation"]["path"] == "context"]
    context_equal = (len(contexts) == protocol["context_controls"]["expected_states"]
                     and all(r["target_pair"] is not None for r in contexts)
                     and all(r["target_pair"]["forces"]["local"] == contexts[0]["target_pair"]["forces"]["local"] for r in contexts))
    if not context_equal:
        failed.append({"path": "context", "scene_id": "all", "step": 0, "checks": ["context_local_pair_invariant"]})
    result = {"schema": "cogniarc.force-diagnostic-summary.v1", "all_checks_pass": not failed,
              "observations": len(rows), "numerical_checks": sum(len(r["checks"]) for r in rows) + 1,
              "failures": failed, "main": None, "trajectory_controls": None, "static_controls": None,
              "structural_controls": None, "fit_performed": False, "model_selection_changed": False}
    if failed:
        return result
    result["main"] = {path: summarize_path([r for r in rows if r["observation"]["path"] == path
                                           and r["observation"]["role"] == "main"], protocol)
                      for path in protocol["observation_paths"]}
    result["trajectory_controls"] = {
        path: {name: summarize_path([r for r in rows if r["observation"]["path"] == path
                                    and r["observation"]["scene_id"] == name], protocol)
               for name in sorted({r["observation"]["scene_id"] for r in rows if r["observation"]["role"] == "control"})}
        for path in protocol["observation_paths"]}
    static = [r for r in rows if r["observation"]["role"] == "static_control"]
    result["static_controls"] = {r["observation"]["scene_id"]: {
        "parameters": r["observation"]["parameters"], "densities": r["densities"],
        "particles": combine([(r, particle_stats(r))]), "pairs": r["pair_statistics"], "target_pair": r["target_pair"]}
        for r in static}
    below = [r for r in static if r["observation"]["path"] == "static_pair"
             and 0 < r["observation"]["parameters"]["separation_over_radius"] < 1
             and max(r["densities"]) <= r["observation"]["config"]["rest_density"]]
    outside = [r for r in static if r["observation"]["path"] == "static_pair"
               and r["observation"]["parameters"]["separation_over_radius"] >= 1]
    result["structural_controls"] = {
        "below_threshold_interacting_pairs": len(below),
        "below_threshold_nonzero_local_pressure": sum(particle_stats(r)["components"]["pressure"]["local_ms"] > 0 for r in below),
        "below_threshold_scene_ids": [r["observation"]["scene_id"] for r in below],
        "below_threshold_pressure_error": combine([(r, particle_stats(r)) for r in below]),
        "at_or_outside_support_states": len(outside),
        "outside_all_accelerations_zero": all(x == 0 for r in outside for name in IMPLEMENTATIONS for mode in MODES
                                              for a in r["accelerations"][name][mode] for x in a),
        "context_local_pair_invariant": context_equal,
        "context_target_pairs": {r["observation"]["scene_id"]: r["target_pair"] for r in contexts}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.source_revision) is None:
        parser.error("--source-revision requires a full lowercase commit SHA")
    protocol = read_protocol()
    model = local.read_model(local.DIRECTORY, local.read_protocol())
    observations, replay_checks = load_observations(model, protocol)
    controls = make_controls(protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, artifacts = [], {}
    selections = {"reference_observations.jsonl": [o for o in observations if o["path"] == "reference"],
                  "local_observations.jsonl": [o for o in observations if o["path"] == "pair_local_replay"],
                  "static_controls.jsonl": controls}
    for name, selected in selections.items():
        path = args.output_dir / name
        artifacts[name] = conservation.write_jsonl(path, [diagnose(o, model["coefficients"], protocol) for o in selected])
        rows.extend(read_records(path, selected, model["coefficients"], protocol))
        print(f"Verified {len(selected)} records: {name}", flush=True)
    report = summarize(rows, protocol)
    report["replay_checks"] = replay_checks
    report["artifacts"] = artifacts
    report["protocol"] = protocol
    report["provenance"] = {
        "date": protocol["date"], "protocol_revision": PROTOCOL_REVISION, "protocol_sha256": PROTOCOL_SHA256,
        "diagnostic_revision": args.source_revision,
        "model_revision": protocol["frozen_model_revision"], "previous_result_revision": protocol["previous_result_revision"],
        "coefficients": model["coefficients"], "source_sha256": {
            **local.source_hashes(), **{name: affine.file_digest(affine.ROOT / name) for name in NEW_SOURCES}},
        "frozen_inputs_sha256": protocol["frozen_inputs_sha256"],
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "machine": platform.machine(), "dependencies": "Python standard library only"},
        "license": "MIT (repository license)",
        "command": "python -m experiments.particle_graph.force_diagnostics --source-revision " + args.source_revision
                   + " --output-dir " + str(args.output_dir)}
    read_protocol()
    local.read_model(local.DIRECTORY, local.read_protocol())
    affine.write_json(args.output_dir / "diagnostic.json", report)
    print(json.dumps({"observations": report["observations"], "checks": report["numerical_checks"],
                      "all_checks_pass": report["all_checks_pass"], "replay": replay_checks}), flush=True)
    raise SystemExit(0 if report["all_checks_pass"] else 1)


if __name__ == "__main__":
    main()

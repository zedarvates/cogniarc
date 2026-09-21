"""Four-coefficient, nonnegative, compact-support pair-force approximation."""

import argparse
from itertools import combinations
import json
import math
from pathlib import Path
import platform

from . import affine, conservation, material_data as data, rollouts
from .dense_reference import accelerations as dense_accelerations
from .reference import Config, Particle, neighbor_pairs
from .validation import canonical, digest

PROTOCOL = Path(__file__).with_name("local_pair_protocol.json")
MANIFEST = Path(__file__).with_name("local_pair_manifest.json")
PROTOCOL_REVISION = "c03ab1bcb3f56527f4b34cb49856f03226ab8f99"
PROTOCOL_SHA256 = "3010a267fc99af035985efdd16434b709fb251035c3b8c894478717f56f780e0"
DIRECTORY = Path(__file__).parent / "evidence/2026-09-13-local-pairs"
SOURCE_PATHS = (*data.SOURCE_PATHS, "experiments/particle_graph/rollouts.py",
                "experiments/particle_graph/conservation.py",
                "experiments/particle_graph/local_pairs.py",
                "experiments/particle_graph/local_pair_evaluation.py",
                "tests/test_particle_graph_local_pairs.py")


def read_protocol():
    if affine.file_digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("local-pair protocol checksum mismatch")
    protocol = json.loads(PROTOCOL.read_text())
    for path, expected in protocol["frozen_inputs_sha256"].items():
        if affine.file_digest(affine.ROOT / path) != expected:
            raise ValueError("frozen local-pair input checksum mismatch: " + path)
    return protocol


def particles(state, masses):
    rollouts.check_state(rollouts.copy_state(state), len(masses), 1e6)
    if not masses or not affine.finite(masses) or min(masses) <= 0:
        raise ValueError("positive finite masses required")
    return tuple(Particle(tuple(p), tuple(v), m) for p, v, m in zip(state["positions"], state["velocities"], masses))


def state_of(values):
    return {"positions": [list(p.position) for p in values], "velocities": [list(p.velocity) for p in values]}


def basis_accelerations(values, config):
    """Sum opposite pair forces, then divide by each particle's own mass."""
    parameters = [config.radius, config.rest_density, config.stiffness, config.viscosity]
    if not affine.finite(parameters) or min(parameters[:2]) <= 0 or min(parameters[2:]) < 0:
        raise ValueError("invalid pair configuration")
    pairs = neighbor_pairs(values, config.radius)
    forces = [[[0.0] * 3 for _ in range(4)] for _ in values]
    for i, j in pairs:
        a, b = values[i], values[j]
        delta = [x - y for x, y in zip(a.position, b.position)]
        distance = math.sqrt(math.fsum(x * x for x in delta))
        q = max(0.0, 1 - distance / config.radius)
        direction = [x / distance if distance else 0.0 for x in delta]
        relative_velocity = [v - u for u, v in zip(a.velocity, b.velocity)]
        common = a.mass * b.mass
        basis = [[common * config.stiffness * q ** power * n for n in direction] for power in (2, 3)]
        basis += [[common * config.viscosity * q ** power * v for v in relative_velocity] for power in (1, 2)]
        for feature in range(4):
            for axis in range(3):
                forces[i][feature][axis] += basis[feature][axis]
                forces[j][feature][axis] -= basis[feature][axis]
    result = [[[x / p.mass for x in force] for force in basis] for p, basis in zip(values, forces)]
    if any(not affine.finite(vector) for basis in result for vector in basis):
        raise ValueError("non-finite pair basis")
    return result


def accelerations(values, config, coefficients):
    if len(coefficients) != 4 or not affine.finite(coefficients) or min(coefficients) < 0:
        raise ValueError("four finite nonnegative coefficients required")
    basis = basis_accelerations(values, config)
    result = tuple(tuple(math.fsum(c * feature[k] for c, feature in zip(coefficients, row)) for k in range(3)) for row in basis)
    if any(not affine.finite(vector) for vector in result):
        raise ValueError("non-finite pair acceleration")
    return result


def midpoint(values, config, dt, gravity, force):
    """Shared explicit midpoint for the approximate and known-equation forces."""
    if not affine.finite([dt]) or dt <= 0 or len(gravity) != 3 or not affine.finite(gravity):
        raise ValueError("invalid midpoint controls")
    def checked(current):
        rollouts.check_state(state_of(current), len(values), 1e6)
        answer = force(current, config)
        if len(answer) != len(values) or any(len(v) != 3 or not affine.finite(v) for v in answer):
            raise ValueError("invalid internal acceleration")
        return answer
    start = checked(values)
    middle = tuple(Particle(tuple(x + dt * v / 2 for x, v in zip(p.position, p.velocity)),
                            tuple(v + dt * (a + g) / 2 for v, a, g in zip(p.velocity, acceleration, gravity)), p.mass)
                   for p, acceleration in zip(values, start))
    halfway = checked(middle)
    result = tuple(Particle(tuple(x + dt * v for x, v in zip(p.position, m.velocity)),
                            tuple(v + dt * (a + g) for v, a, g in zip(p.velocity, acceleration, gravity)), p.mass)
                   for p, m, acceleration in zip(values, middle, halfway))
    rollouts.check_state(state_of(result), len(values), 1e6)
    return result


def label_records(scenes, protocol):
    labels = []
    for scene in scenes:
        if scene["split"] not in ("train", "validation"):
            raise ValueError("force labels require development partitions")
        if [s["step"] for s in scene["snapshots"]] != protocol["training_steps"]:
            raise ValueError("unexpected development snapshots")
        for snapshot in scene["snapshots"]:
            values = particles(snapshot, scene["masses"])
            labels.append({"schema": "cogniarc.local-pair-force-target.v1", "scene_id": scene["scene_id"],
                           "group_id": scene["group_id"], "split": scene["split"], "step": snapshot["step"],
                           "observation_sha256": digest({"state": state_of(values), "masses": scene["masses"], "config": scene["config"]}),
                           "internal_accelerations": [list(a) for a in dense_accelerations(values, Config(**scene["config"]))]})
    return labels


def read_labels(path, scenes, protocol):
    # Verify the actual teacher labels as well as split, identity and shape.
    expected = {(r["scene_id"], r["step"]): r for r in label_records(scenes, protocol)}
    result, seen = [], set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = (row["scene_id"], row["step"])
        if key in seen or row != expected.get(key):
            raise ValueError("force target does not match the frozen teacher/observation")
        seen.add(key)
        result.append(row)
    if seen != set(expected):
        raise ValueError("missing force targets")
    return result


def design(scenes, labels, partition):
    affine.require_partition(scenes, partition)
    wanted = {s["scene_id"] for s in scenes}
    targets = {(r["scene_id"], r["step"]): r for r in labels if r["split"] == partition and r["scene_id"] in wanted}
    rows, outputs, weights = [], [], []
    for scene in scenes:
        for snapshot in scene["snapshots"]:
            values = particles(snapshot, scene["masses"])
            basis = basis_accelerations(values, Config(**scene["config"]))
            target = targets[(scene["scene_id"], snapshot["step"])]["internal_accelerations"]
            if len(target) != len(values) or any(len(v) != 3 or not affine.finite(v) for v in target):
                raise ValueError("invalid training target shape")
            for features, acceleration in zip(basis, target):
                for k in range(3):
                    rows.append([f[k] for f in features])
                    outputs.append(acceleration[k])
                    weights.append(1 / (len(scenes) * len(scene["snapshots"]) * len(values) * 3))
    return rows, outputs, weights


def nonnegative_ridge(rows, targets, weights, alpha):
    """Solve the four-feature convex problem by all active subsets, with no bias."""
    if (not rows or len(rows) != len(targets) or len(rows) != len(weights)
            or any(len(r) != 4 or not affine.finite(r) for r in rows)
            or not affine.finite(targets + weights + [alpha]) or min(weights) <= 0 or alpha <= 0):
        raise ValueError("invalid nonnegative ridge problem")
    scales = [math.sqrt(math.fsum(w * row[k] ** 2 for w, row in zip(weights, rows))) or 1.0 for k in range(4)]
    normalized = [[x / scale for x, scale in zip(row, scales)] for row in rows]
    gram = [[math.fsum(w * row[i] * row[j] for w, row in zip(weights, normalized)) + (alpha if i == j else 0)
             for j in range(4)] for i in range(4)]
    rhs = [math.fsum(w * row[i] * y for w, row, y in zip(weights, normalized, targets)) for i in range(4)]
    candidates = []
    for count in range(5):
        for active in combinations(range(4), count):
            solution = affine.solve_positive([[gram[i][j] for j in active] for i in active], [rhs[i] for i in active]) if active else []
            if any(x < 0 for x in solution):
                continue
            beta = [0.0] * 4
            for k, value in zip(active, solution):
                beta[k] = value
            residual = math.fsum(w * (math.fsum(x * c for x, c in zip(row, beta)) - y) ** 2
                                 for w, row, y in zip(weights, normalized, targets))
            objective = residual + alpha * math.fsum(c * c for c in beta)
            candidates.append((objective, count, active, beta, residual))
    objective, _, active, beta, residual = min(candidates)
    gradient = [math.fsum(gram[i][j] * beta[j] for j in range(4)) - rhs[i] for i in range(4)]
    violation = max([abs(gradient[i]) if beta[i] > 0 else max(0.0, -gradient[i]) for i in range(4)])
    return {"alpha": alpha, "feature_rms": scales, "scaled_coefficients": beta,
            "coefficients": [x / scale for x, scale in zip(beta, scales)], "active": list(active),
            "objective": objective, "train_acceleration_rmse": math.sqrt(residual), "kkt_violation": violation}


def acceleration_score(design_values, coefficients):
    rows, targets, weights = design_values
    return math.sqrt(math.fsum(w * (math.fsum(x * c for x, c in zip(row, coefficients)) - y) ** 2
                               for w, row, y in zip(weights, rows, targets)))


def fit_model(train, validation, labels, protocol):
    from .materials import check_development
    check_development(train, validation)
    train_design = design(train, labels, "train")
    validation_design = design(validation, labels, "validation")
    candidates = []
    for alpha in protocol["ridge_alphas"]:
        fit = nonnegative_ridge(*train_design, alpha)
        fit["validation_acceleration_rmse"] = acceleration_score(validation_design, fit["coefficients"])
        candidates.append(fit)
    chosen = min(candidates, key=lambda r: (r["validation_acceleration_rmse"], -r["alpha"]))
    return {"schema": "cogniarc.local-pair-model.v1", "feature_names": protocol["feature_names"],
            "selected_alpha": chosen["alpha"], "coefficients": list(chosen["coefficients"]), "candidates": candidates,
            "train_scene_ids": [s["scene_id"] for s in train], "validation_scene_ids": [s["scene_id"] for s in validation],
            "train_snapshots": sum(len(s["snapshots"]) for s in train),
            "validation_snapshots": sum(len(s["snapshots"]) for s in validation),
            "train_coordinate_rows": len(train_design[0]), "validation_coordinate_rows": len(validation_design[0]),
            "development_group_ids": sorted({s["group_id"] for s in train + validation}),
            "refit_on_validation": False, "test_used_for_selection": False}


def source_hashes():
    return {p: affine.file_digest(affine.ROOT / p) for p in SOURCE_PATHS}


def read_model(directory, protocol):
    lock = json.loads((directory / "model_lock.json").read_text())
    if affine.file_digest(directory / "model.json") != lock["model_sha256"]:
        raise ValueError("frozen pair model checksum mismatch")
    model = json.loads((directory / "model.json").read_text())
    if (model["schema"] != "cogniarc.local-pair-model.v1" or model["refit_on_validation"]
            or model["test_used_for_selection"] or model["feature_names"] != protocol["feature_names"]):
        raise ValueError("invalid pair model contract")
    if (model["provenance"]["source_sha256"] != source_hashes()
            or model["provenance"]["protocol_sha256"] != PROTOCOL_SHA256
            or affine.file_digest(directory / "force_targets.jsonl") != model["provenance"]["force_targets"]["sha256"]):
        raise ValueError("pair model source/protocol/labels changed")
    if [r["alpha"] for r in model["candidates"]] != protocol["ridge_alphas"]:
        raise ValueError("pair candidate grid changed")
    chosen = min(model["candidates"], key=lambda r: (r["validation_acceleration_rmse"], -r["alpha"]))
    if chosen["alpha"] != model["selected_alpha"] or chosen["coefficients"] != model["coefficients"]:
        raise ValueError("pair model differs from validation selection")
    if len(model["coefficients"]) != 4 or not affine.finite(model["coefficients"]) or min(model["coefficients"]) < 0:
        raise ValueError("invalid saved pair coefficients")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = read_protocol()
    manifest, old_protocol = data.read_config()
    report = data.read_report(rollouts.DATA_DIR, manifest, old_protocol)
    train = data.read_split(rollouts.DATA_DIR, "train", manifest, report)
    validation = data.read_split(rollouts.DATA_DIR, "validation", manifest, report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_path = args.output_dir / "force_targets.jsonl"
    label_artifact = conservation.write_jsonl(label_path, label_records(train + validation, protocol))
    model = fit_model(train, validation, read_labels(label_path, train + validation, protocol), protocol)
    model["provenance"] = {"date": "2026-09-13", "protocol_revision": PROTOCOL_REVISION,
                            "protocol_sha256": PROTOCOL_SHA256, "source_sha256": source_hashes(),
                            "frozen_inputs_sha256": protocol["frozen_inputs_sha256"], "force_targets": label_artifact,
                            "environment": {"python": platform.python_version(), "system": platform.system(),
                                            "machine": platform.machine(), "dependencies": "Python standard library only"},
                            "license": "MIT (repository license)",
                            "command": "python -m experiments.particle_graph.local_pairs --output-dir " + str(args.output_dir)}
    affine.write_json(args.output_dir / "model.json", model)
    affine.write_json(args.output_dir / "model_lock.json", {"schema": "cogniarc.local-pair-model-lock.v1",
                                                          "model_sha256": affine.file_digest(args.output_dir / "model.json")})
    print(canonical({k: model[k] for k in ("selected_alpha", "coefficients", "train_coordinate_rows", "validation_coordinate_rows")}))
    print(canonical({"candidates": model["candidates"], "model_sha256": affine.file_digest(args.output_dir / "model.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

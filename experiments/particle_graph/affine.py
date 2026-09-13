"""Train-only ridge baselines and a separate evaluator for frozen SPH snapshots."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import platform

from .validation import MANIFEST, canonical, digest, validate_manifest

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("affine_protocol.json")
PROTOCOL_REVISION = "5e39eb67db3526eeea63d525ba7b9aeafc4b0677"
SOURCES = ("experiments/particle_graph/affine.py",
           "experiments/particle_graph/validation.py",
           "tests/test_particle_graph_affine.py")


def file_digest(path):
    return sha256(path.read_bytes()).hexdigest()


def finite(values):
    return all(type(x) in (int, float) and math.isfinite(x) for x in values)


def read_protocol():
    return json.loads(PROTOCOL.read_text())


def read_corpus(protocol, splits, corpus_path=None):
    """Verify the frozen bytes; return only the requested partitions to callers."""
    path = corpus_path or ROOT / protocol["corpus"]
    raw = path.read_bytes()
    if len(raw) != protocol["corpus_bytes"] or sha256(raw).hexdigest() != protocol["corpus_sha256"]:
        raise ValueError("corpus checksum or size mismatch")
    if file_digest(MANIFEST) != protocol["manifest_sha256"]:
        raise ValueError("manifest checksum mismatch")
    manifest = json.loads(MANIFEST.read_text())
    validate_manifest(manifest)
    expected = {s["id"]: s for s in manifest["scenes"]}
    records, seen = [], set()
    for line in raw.decode("utf-8").splitlines():
        record = json.loads(line)
        recipe = expected.get(record["scene_id"])
        if recipe is None or record["scene_id"] in seen:
            raise ValueError("unexpected or duplicate scene")
        seen.add(record["scene_id"])
        if any(record[k] != recipe[k] for k in ("split", "seed", "condition", "gravity")):
            raise ValueError("scene metadata does not match the manifest")
        if record["split"] not in splits:
            continue
        n = math.prod(recipe["shape"])
        config = manifest["defaults"]["config"] | recipe["config"]
        if (record["schema"] != "cogniarc.particle-graph-snapshots.v1"
                or record["units"] != "dimensionless"
                or record["config"] != config
                or record["particle_ids"] != list(range(n))
                or record["masses"] != [manifest["defaults"]["mass"]] * n
                or record["integrator"] != "independent_dense_rk4"
                or record["internal_dt"] != manifest["dt"] / 8):
            raise ValueError("unsupported snapshot contract")
        if [s["step"] for s in record["snapshots"]] != [0] + protocol["horizons"]:
            raise ValueError("missing or unordered horizons")
        for snapshot in record["snapshots"]:
            if snapshot["time"] != snapshot["step"] * manifest["dt"]:
                raise ValueError("snapshot time mismatch")
            for key in ("positions", "velocities"):
                vectors = snapshot[key]
                if len(vectors) != n or any(len(v) != 3 or not finite(v) for v in vectors):
                    raise ValueError("invalid particle vectors")
        records.append(record)
    if seen != set(expected) or {r["split"] for r in records} != set(splits):
        raise ValueError("missing scenes or partitions")
    return records


def features(initial, gravity):
    """Observable initial geometry/motion only; independent of IDs and targets."""
    positions, velocities = initial["positions"], initial["velocities"]
    n = len(positions)
    if (n == 0 or len(velocities) != n or len(gravity) != 3 or not finite(gravity)
            or any(len(v) != 3 or not finite(v) for v in positions + velocities)):
        raise ValueError("invalid initial observation")
    means = [[math.fsum(v[k] for v in vectors) / n for k in range(3)]
             for vectors in (positions, velocities)]
    return [[p[k] - means[0][k] for k in range(3)]
            + [v[k] - means[1][k] for k in range(3)] + list(gravity)
            for p, v in zip(positions, velocities)]


def solve_positive(matrix, rhs):
    """Cholesky solve; positive ridge makes even collinear feature sets usable."""
    n = len(rhs)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            value = matrix[i][j] - math.fsum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("ridge system is not positive definite")
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value / lower[j][j]
    forward = []
    for i in range(n):
        forward.append((rhs[i] - math.fsum(lower[i][j] * forward[j] for j in range(i))) / lower[i][i])
    result = [0.0] * n
    for i in reversed(range(n)):
        result[i] = (forward[i] - math.fsum(lower[j][i] * result[j] for j in range(i + 1, n))) / lower[i][i]
    if not finite(result):
        raise ValueError("non-finite ridge solution")
    return result


def ridge_fit(rows, targets, weights, alpha):
    """Weighted feature normalization and an unpenalized affine intercept."""
    if (not rows or len(rows) != len(targets) or len(rows) != len(weights)
            or not finite([alpha]) or alpha <= 0 or not finite(weights)
            or any(w <= 0 for w in weights)):
        raise ValueError("invalid regression data or regularization")
    d, outputs = len(rows[0]), len(targets[0])
    if not d or not outputs or any(len(r) != d or not finite(r) for r in rows) or any(
            len(y) != outputs or not finite(y) for y in targets):
        raise ValueError("invalid regression shape or values")
    total = math.fsum(weights)
    weights = [w / total for w in weights]
    mean = [math.fsum(w * r[k] for w, r in zip(weights, rows)) for k in range(d)]
    scales = [math.sqrt(math.fsum(w * (r[k] - mean[k]) ** 2
                                 for w, r in zip(weights, rows))) or 1.0 for k in range(d)]
    normalized = [[(r[k] - mean[k]) / scales[k] for k in range(d)] for r in rows]
    bias = [math.fsum(w * y[k] for w, y in zip(weights, targets)) for k in range(outputs)]
    gram = [[math.fsum(w * r[i] * r[j] for w, r in zip(weights, normalized))
             + (alpha if i == j else 0) for j in range(d)] for i in range(d)]
    coefficients = []
    for k in range(outputs):
        rhs = [math.fsum(w * r[j] * (y[k] - bias[k])
                        for w, r, y in zip(weights, normalized, targets)) for j in range(d)]
        coefficients.append(solve_positive(gram, rhs))
    return {"feature_mean": mean, "feature_scale": scales, "bias": bias,
            "coefficients": coefficients}


def affine_values(model, rows):
    return [[b + math.fsum(c * (x - m) / scale for c, x, m, scale in
                          zip(coef, row, model["feature_mean"], model["feature_scale"]))
             for b, coef in zip(model["bias"], model["coefficients"])] for row in rows]


def require_partition(records, name):
    if not records or any(r["split"] != name for r in records):
        raise ValueError(f"expected {name} scenes only")
    if len({r["scene_id"] for r in records}) != len(records):
        raise ValueError("duplicate scene IDs")


def fit_candidate(train, horizons, alpha):
    require_partition(train, "train")
    rows, weights = [], []
    targets = {h: [] for h in horizons}
    for scene in train:
        initial = scene["snapshots"][0]
        observed = features(initial, scene["gravity"])
        rows.extend(observed)
        weights.extend([1 / (len(train) * len(observed))] * len(observed))
        snapshots = {s["step"]: s for s in scene["snapshots"]}
        for h in horizons:
            target = snapshots[h]
            t = target["time"]
            if t <= 0:
                raise ValueError("prediction horizon must be positive")
            for p, v, q, u in zip(initial["positions"], initial["velocities"],
                                  target["positions"], target["velocities"]):
                targets[h].append([(q[k] - p[k] - t * v[k]) / t ** 2 for k in range(3)]
                                  + [(u[k] - v[k]) / t for k in range(3)])
    return {str(h): ridge_fit(rows, targets[h], weights, alpha) for h in horizons}


def predict(initial, gravity, time, method, model=None):
    if not finite([time]) or time <= 0:
        raise ValueError("prediction time must be positive")
    rows = features(initial, gravity)
    if method == "affine":
        residuals = affine_values(model, rows)
    elif method not in ("persistence", "constant_velocity", "known_gravity_ballistic"):
        raise ValueError("unknown prediction method")
    positions, velocities = [], []
    for i, (p, v) in enumerate(zip(initial["positions"], initial["velocities"])):
        if method == "affine":
            dx, dv = residuals[i][:3], residuals[i][3:]
        elif method == "known_gravity_ballistic":
            dx, dv = [0.5 * g for g in gravity], gravity
        else:
            dx, dv = [0.0] * 3, [0.0] * 3
        positions.append([p[k] + (0 if method == "persistence" else time * v[k])
                          + time ** 2 * dx[k] for k in range(3)])
        velocities.append([v[k] + time * dv[k] for k in range(3)])
    if any(not finite(v) for v in positions + velocities):
        raise ValueError("non-finite prediction")
    return {"positions": positions, "velocities": velocities}


def errors(prediction, target):
    result = {}
    for key, name in (("positions", "position_rmse"), ("velocities", "velocity_rmse")):
        differences = [(a - b) ** 2 for p, q in zip(prediction[key], target[key]) for a, b in zip(p, q)]
        result[name] = math.sqrt(math.fsum(differences) / len(differences))
    return result


def select_model(train, validation, protocol):
    """No test argument; candidate fits and normalizers receive training only."""
    require_partition(train, "train")
    require_partition(validation, "validation")
    if {s["scene_id"] for s in train} & {s["scene_id"] for s in validation}:
        raise ValueError("scene overlap across partitions")
    candidates, best = [], None
    for alpha in protocol["ridge_alphas"]:
        models = fit_candidate(train, protocol["horizons"], alpha)
        terms, scene_errors = [], []
        for scene in validation:
            for target in scene["snapshots"][1:]:
                h, t = target["step"], target["time"]
                prediction = predict(scene["snapshots"][0], scene["gravity"], t, "affine", models[str(h)])
                error = errors(prediction, target)
                terms.append(((error["position_rmse"] / t ** 2) ** 2
                              + (error["velocity_rmse"] / t) ** 2) / 2)
                scene_errors.append({"scene_id": scene["scene_id"], "horizon": h, **error})
        score = math.sqrt(math.fsum(terms) / len(terms))
        candidates.append({"alpha": alpha, "validation_score": score,
                           "fit_sha256": digest(models), "validation_errors": scene_errors})
        if best is None or (score, -alpha) < (best[0], -best[1]):
            best = (score, alpha, models)
    return {"schema": "cogniarc.particle-affine-model.v1", "selected_alpha": best[1],
            "horizon_models": best[2], "candidates": candidates,
            "train_scene_ids": [s["scene_id"] for s in train],
            "validation_scene_ids": [s["scene_id"] for s in validation],
            "refit_on_validation": False, "test_used_for_selection": False}


def evaluate_test(model, test, protocol):
    """Pure scoring of a previously selected model; never calls a fit function."""
    require_partition(test, "test")
    used_ids = set(model["train_scene_ids"] + model["validation_scene_ids"])
    if used_ids & {s["scene_id"] for s in test}:
        raise ValueError("test scene overlaps model development")
    methods = protocol["baselines"] + ["affine"]
    rows = []
    for scene in test:
        for target in scene["snapshots"][1:]:
            h, t = target["step"], target["time"]
            metrics = {method: errors(predict(scene["snapshots"][0], scene["gravity"], t,
                                              method, model["horizon_models"][str(h)]), target)
                       for method in methods}
            rows.append({"scene_id": scene["scene_id"], "condition": scene["condition"],
                         "particles": len(scene["particle_ids"]), "horizon": h,
                         "time": t, "metrics": metrics})
    aggregates = {}
    for h in protocol["horizons"]:
        group = [r for r in rows if r["horizon"] == h]
        aggregates[str(h)] = {method: {metric: math.fsum(r["metrics"][method][metric]
                                                       for r in group) / len(group)
                                      for metric in ("position_rmse", "velocity_rmse")}
                             for method in methods}
    return {"schema": "cogniarc.particle-affine-evaluation.v1", "split": "test",
            "selected_alpha": model["selected_alpha"], "scenes": rows,
            "macro_mean_scene_rmse": aggregates,
            "selection_changed_during_evaluation": False}


def provenance(protocol):
    return {"date": "2026-09-13", "protocol_revision": PROTOCOL_REVISION,
            "protocol_sha256": file_digest(PROTOCOL),
            "corpus_sha256": protocol["corpus_sha256"],
            "manifest_sha256": protocol["manifest_sha256"],
            "source_sha256": {p: file_digest(ROOT / p) for p in SOURCES},
            "environment": {"python": platform.python_version(), "system": platform.system(),
                            "machine": platform.machine(), "dependencies": "Python standard library only"},
            "license": "MIT (repository license)"}


def verify_model(model, protocol):
    if model["schema"] != "cogniarc.particle-affine-model.v1":
        raise ValueError("unsupported model schema")
    expected = provenance(protocol)
    for key in ("protocol_revision", "protocol_sha256", "corpus_sha256", "manifest_sha256", "source_sha256"):
        if model["provenance"][key] != expected[key]:
            raise ValueError(f"model {key} mismatch")
    selected = min(model["candidates"], key=lambda c: (c["validation_score"], -c["alpha"]))
    if (model["selected_alpha"] != selected["alpha"]
            or digest(model["horizon_models"]) != selected["fit_sha256"]
            or model["refit_on_validation"] or model["test_used_for_selection"]):
        raise ValueError("model no longer matches the validation selection")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--output", type=Path, required=True)
    score_parser = commands.add_parser("evaluate")
    score_parser.add_argument("--model", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = read_protocol()
    if args.command == "fit":
        development = read_corpus(protocol, {"train", "validation"})
        artifact = select_model([r for r in development if r["split"] == "train"],
                                [r for r in development if r["split"] == "validation"], protocol)
        artifact["provenance"] = provenance(protocol)
        artifact["provenance"]["command"] = "python -m experiments.particle_graph.affine fit --output " + str(args.output)
        write_json(args.output, artifact)
        print(canonical({"selected_alpha": artifact["selected_alpha"],
                         "validation_scores": [{k: c[k] for k in ("alpha", "validation_score")}
                                               for c in artifact["candidates"]],
                         "model_sha256": file_digest(args.output)}))
    else:
        model = json.loads(args.model.read_text())
        verify_model(model, protocol)
        artifact = evaluate_test(model, read_corpus(protocol, {"test"}), protocol)
        artifact["provenance"] = provenance(protocol)
        artifact["provenance"]["command"] = ("python -m experiments.particle_graph.affine evaluate --model "
                                              + str(args.model) + " --output " + str(args.output))
        artifact["provenance"]["model_sha256"] = file_digest(args.model)
        write_json(args.output, artifact)
        print(canonical(artifact["macro_mean_scene_rmse"]))


if __name__ == "__main__":
    main()

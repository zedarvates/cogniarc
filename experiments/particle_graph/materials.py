"""Frozen-protocol comparison of material-blind and material-conditioned ridge."""

import argparse
import json
import math
from pathlib import Path

from . import affine
from . import material_data as data
from .validation import canonical, digest

VARIANTS = ("blind_v2", "material_v2")
BASELINES = ("persistence", "constant_velocity", "known_gravity_ballistic")


def features(initial, gravity, config, variant):
    rows = affine.features(initial, gravity)
    if variant == "blind_v2":
        return rows
    if variant != "material_v2":
        raise ValueError("unknown material feature variant")
    stiffness, viscosity = config["stiffness"], config["viscosity"]
    if not affine.finite([stiffness, viscosity]) or min(stiffness, viscosity) <= 0:
        raise ValueError("material inputs must be finite and positive")
    return [row + [stiffness * x for x in row[:3]] + [viscosity * v for v in row[3:6]] for row in rows]


def fit_candidate(train, horizons, alpha, variant):
    affine.require_partition(train, "train")
    rows, weights = [], []
    targets = {h: [] for h in horizons}
    for scene in train:
        initial = scene["snapshots"][0]
        observed = features(initial, scene["gravity"], scene["config"], variant)
        rows.extend(observed)
        weights.extend([1 / (len(train) * len(observed))] * len(observed))
        snapshots = {s["step"]: s for s in scene["snapshots"]}
        for h in horizons:
            target, t = snapshots[h], snapshots[h]["time"]
            if t <= 0:
                raise ValueError("positive prediction horizon required")
            for p, v, q, u in zip(initial["positions"], initial["velocities"],
                                  target["positions"], target["velocities"]):
                targets[h].append([(q[k] - p[k] - t * v[k]) / t ** 2 for k in range(3)]
                                  + [(u[k] - v[k]) / t for k in range(3)])
    return {str(h): affine.ridge_fit(rows, targets[h], weights, alpha) for h in horizons}


def predict(initial, gravity, config, time, method, horizon_model=None):
    if method in BASELINES or method == "frozen_v1":
        return affine.predict(initial, gravity, time, "affine" if method == "frozen_v1" else method, horizon_model)
    if not affine.finite([time]) or time <= 0:
        raise ValueError("positive prediction time required")
    residuals = affine.affine_values(horizon_model, features(initial, gravity, config, method))
    positions, velocities = [], []
    for p, v, residual in zip(initial["positions"], initial["velocities"], residuals):
        positions.append([p[k] + time * v[k] + time ** 2 * residual[k] for k in range(3)])
        velocities.append([v[k] + time * residual[3 + k] for k in range(3)])
    if any(not affine.finite(v) for v in positions + velocities):
        raise ValueError("non-finite material prediction")
    return {"positions": positions, "velocities": velocities}


def check_development(train, validation):
    affine.require_partition(train, "train")
    affine.require_partition(validation, "validation")
    for key in ("scene_id", "group_id", "seed", "initial_state_sha256"):
        if {r[key] for r in train} & {r[key] for r in validation}:
            raise ValueError(f"development partition overlap: {key}")


def select_models(train, validation, manifest, protocol):
    check_development(train, validation)
    result = {"schema": "cogniarc.particle-material-model.v2", "variants": {},
              "train_scene_ids": [r["scene_id"] for r in train],
              "validation_scene_ids": [r["scene_id"] for r in validation],
              "development_group_ids": sorted({r["group_id"] for r in train + validation}),
              "development_seeds": sorted({r["seed"] for r in train + validation}),
              "development_initial_hashes": sorted({r["initial_state_sha256"] for r in train + validation}),
              "refit_on_validation": False, "test_used_for_selection": False}
    for variant in VARIANTS:
        candidates, best = [], None
        for alpha in protocol["ridge_alphas"]:
            models = fit_candidate(train, manifest["horizons"], alpha, variant)
            terms, errors = [], []
            for scene in validation:
                for target in scene["snapshots"][1:]:
                    h, t = target["step"], target["time"]
                    prediction = predict(scene["snapshots"][0], scene["gravity"], scene["config"],
                                         t, variant, models[str(h)])
                    error = affine.errors(prediction, target)
                    terms.append(((error["position_rmse"] / t ** 2) ** 2 + (error["velocity_rmse"] / t) ** 2) / 2)
                    errors.append({"scene_id": scene["scene_id"], "horizon": h, **error})
            score = math.sqrt(math.fsum(terms) / len(terms))
            candidates.append({"alpha": alpha, "validation_score": score, "fit_sha256": digest(models),
                               "validation_errors": errors})
            if best is None or (score, -alpha) < (best[0], -best[1]):
                best = (score, alpha, models)
        result["variants"][variant] = {"selected_alpha": best[1], "horizon_models": best[2], "candidates": candidates}
    return result


def verify_model(model, report, directory, protocol):
    if model["schema"] != "cogniarc.particle-material-model.v2" or set(model["variants"]) != set(VARIANTS):
        raise ValueError("unsupported material model")
    data.verify_provenance(model["provenance"])
    if (model["provenance"]["numerics_sha256"] != affine.file_digest(directory / "numerics.json")
            or model["provenance"]["corpus_sha256"] != {k: v["sha256"] for k, v in report["artifacts"].items()}
            or model["provenance"]["prior_model_sha256"] != protocol["prior_model_sha256"]):
        raise ValueError("model input evidence mismatch")
    if model["refit_on_validation"] or model["test_used_for_selection"]:
        raise ValueError("model violates partition isolation")
    for variant in model["variants"].values():
        if [c["alpha"] for c in variant["candidates"]] != protocol["ridge_alphas"]:
            raise ValueError("candidate grid changed")
        selected = min(variant["candidates"], key=lambda c: (c["validation_score"], -c["alpha"]))
        if variant["selected_alpha"] != selected["alpha"] or digest(variant["horizon_models"]) != selected["fit_sha256"]:
            raise ValueError("coefficients differ from validation selection")


def load_frozen_v1(protocol):
    path = affine.ROOT / protocol["prior_model"]
    if affine.file_digest(path) != protocol["prior_model_sha256"]:
        raise ValueError("frozen v1 model checksum mismatch")
    model = json.loads(path.read_text())
    affine.verify_model(model, affine.read_protocol())
    return model


def macro_errors(rows, methods):
    if not rows:
        raise ValueError("cannot aggregate an empty group")
    return {method: {metric: math.fsum(r["metrics"][method][metric] for r in rows) / len(rows)
                     for metric in ("position_rmse", "velocity_rmse")} for method in methods}


def delta(a, b):
    return {key: [[x - y for x, y in zip(p, q)] for p, q in zip(a[key], b[key])]
            for key in ("positions", "velocities")}


def evaluate_test(model, frozen_v1, test, manifest, protocol):
    affine.require_partition(test, "test")
    for key, stored in (("group_id", "development_group_ids"), ("seed", "development_seeds"),
                        ("initial_state_sha256", "development_initial_hashes")):
        if {r[key] for r in test} & set(model[stored]):
            raise ValueError(f"test overlaps development: {key}")
    methods, rows, predictions = protocol["comparators"], [], {}
    for scene in test:
        for target in scene["snapshots"][1:]:
            h, t = target["step"], target["time"]
            states = {}
            for method in methods:
                horizon_model = (frozen_v1["horizon_models"][str(h)] if method == "frozen_v1" else
                                 model["variants"][method]["horizon_models"][str(h)] if method in VARIANTS else None)
                states[method] = predict(scene["snapshots"][0], scene["gravity"], scene["config"], t, method, horizon_model)
            predictions[(scene["scene_id"], h)] = states
            rows.append({"scene_id": scene["scene_id"], "group_id": scene["group_id"],
                         "geometry_condition": scene["geometry_condition"], "material_condition": scene["material_condition"],
                         "material_id": scene["material_id"], "particles": len(scene["particle_ids"]),
                         "horizon": h, "time": t,
                         "metrics": {method: affine.errors(state, target) for method, state in states.items()}})
    report = {"schema": "cogniarc.particle-material-evaluation.v2", "split": "test", "scenes": rows,
              "selected_alphas": {k: v["selected_alpha"] for k, v in model["variants"].items()},
              "selection_changed_during_evaluation": False,
              "macro_mean_scene_rmse": {str(h): macro_errors([r for r in rows if r["horizon"] == h], methods)
                                        for h in manifest["horizons"]}}
    for key in ("material_condition", "particles"):
        report["by_" + key] = {str(value): {str(h): macro_errors([r for r in rows if r[key] == value and r["horizon"] == h], methods)
                                          for h in manifest["horizons"]} for value in sorted({r[key] for r in rows})}
    baseline_by_group = {r["group_id"]: r for r in test if r["material_id"] == "base-material"}
    pairs = []
    for scene in test:
        if scene["material_id"] == "base-material":
            continue
        base = baseline_by_group[scene["group_id"]]
        if scene["snapshots"][0] != base["snapshots"][0]:
            raise ValueError("paired materials have different initial states")
        base_targets = {s["step"]: s for s in base["snapshots"]}
        for target in scene["snapshots"][1:]:
            h = target["step"]
            change = delta(target, base_targets[h])
            zero = {k: [[0.0] * 3 for _ in v] for k, v in change.items()}
            metrics = {method: affine.errors(delta(predictions[(scene["scene_id"], h)][method],
                                                   predictions[(base["scene_id"], h)][method]), change)
                       for method in methods}
            pairs.append({"group_id": scene["group_id"], "material_condition": scene["material_condition"],
                          "horizon": h, "target_change_rms": affine.errors(change, zero), "metrics": metrics})
    report["paired_material_changes"] = pairs
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generator = commands.add_parser("generate")
    generator.add_argument("--output-dir", type=Path, required=True)
    fitter = commands.add_parser("fit")
    fitter.add_argument("--data-dir", type=Path, required=True)
    fitter.add_argument("--output", type=Path, required=True)
    evaluator = commands.add_parser("evaluate")
    evaluator.add_argument("--data-dir", type=Path, required=True)
    evaluator.add_argument("--model", type=Path, required=True)
    evaluator.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, protocol = data.read_config()
    if args.command == "generate":
        report = data.generate(manifest, protocol, args.output_dir)
        report["provenance"]["command"] = "python -m experiments.particle_graph.materials generate --output-dir " + str(args.output_dir)
        affine.write_json(args.output_dir / "numerics.json", report)
        print(canonical({"scenes": len(report["scenes"]), "passed": report["passed"], "artifacts": report["artifacts"]}))
        return 0 if report["passed"] else 1
    report = data.read_report(args.data_dir, manifest, protocol)
    if args.command == "fit":
        model = select_models(data.read_split(args.data_dir, "train", manifest, report),
                              data.read_split(args.data_dir, "validation", manifest, report), manifest, protocol)
        model["provenance"] = data.provenance() | {
            "numerics_sha256": affine.file_digest(args.data_dir / "numerics.json"),
            "corpus_sha256": {k: v["sha256"] for k, v in report["artifacts"].items()},
            "prior_model_sha256": protocol["prior_model_sha256"],
            "command": "python -m experiments.particle_graph.materials fit --data-dir " + str(args.data_dir) + " --output " + str(args.output)}
        affine.write_json(args.output, model)
        print(canonical({variant: {"selected_alpha": v["selected_alpha"],
                                   "validation_scores": [{k: c[k] for k in ("alpha", "validation_score")} for c in v["candidates"]]}
                         for variant, v in model["variants"].items()}))
    else:
        model = json.loads(args.model.read_text())
        verify_model(model, report, args.data_dir, protocol)
        result = evaluate_test(model, load_frozen_v1(protocol), data.read_split(args.data_dir, "test", manifest, report), manifest, protocol)
        result["provenance"] = model["provenance"] | {
            "model_sha256": affine.file_digest(args.model),
            "command": "python -m experiments.particle_graph.materials evaluate --data-dir " + str(args.data_dir)
                       + " --model " + str(args.model) + " --output " + str(args.output)}
        affine.write_json(args.output, result)
        print(canonical(result["macro_mean_scene_rmse"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

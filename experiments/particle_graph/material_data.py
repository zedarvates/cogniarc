"""Paired-material scene generation, numerical checks and verified split reader."""

from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import platform

from .affine import ROOT, file_digest, finite, write_json
from .dense_reference import rk4_step
from .validation import (MANIFEST as V1_MANIFEST, canonical, digest, integrate,
                         make_scene, momentum, rmse, validate_manifest as validate_v1)

MANIFEST = Path(__file__).with_name("material_manifest.json")
PROTOCOL = Path(__file__).with_name("material_protocol.json")
PROTOCOL_REVISION = "0751eb36ca1b01d7a1cc2f1484f49cd563d900e9"
SPLITS = ("train", "validation", "test")
SOURCE_PATHS = ("experiments/particle_graph/material_data.py",
                "experiments/particle_graph/materials.py",
                "experiments/particle_graph/affine.py",
                "experiments/particle_graph/validation.py",
                "experiments/particle_graph/reference.py",
                "experiments/particle_graph/dense_reference.py",
                "tests/test_particle_graph_materials.py")


def validate_manifest(manifest):
    required = {"schema", "units", "dt", "steps", "horizons", "reference_factors", "defaults", "groups", "materials"}
    if set(manifest) != required or manifest["schema"] != "cogniarc.particle-material-scenes.v2":
        raise ValueError("unsupported material manifest")
    # Reuse v1 recipe validation without modifying its frozen implementation.
    base = {k: manifest[k] for k in ("units", "dt", "steps", "horizons", "reference_factors", "defaults")}
    base.update(schema="cogniarc.particle-graph-scenes.v1", refinement_factors=[1, 2, 4],
                scenes=[g | {"config": {}} for g in manifest["groups"]])
    validate_v1(base)
    old_seeds = {s["seed"] for s in json.loads(V1_MANIFEST.read_text())["scenes"]}
    if any(g["seed"] in old_seeds for g in manifest["groups"]):
        raise ValueError("v2 seeds must be absent from v1")
    if set(manifest["materials"]) != set(SPLITS):
        raise ValueError("missing material partition")
    configurations = {}
    for split in SPLITS:
        materials = manifest["materials"][split]
        keys = {"id", "stiffness", "viscosity"} | ({"condition"} if split == "test" else set())
        if len(materials) != 4 or len({m["id"] for m in materials}) != 4:
            raise ValueError("four distinct materials required per split")
        for material in materials:
            if set(material) != keys or not isinstance(material["id"], str) or not material["id"]:
                raise ValueError("invalid material fields")
            values = [material[k] for k in ("stiffness", "viscosity")]
            if not finite(values) or min(values) <= 0:
                raise ValueError("material parameters must be finite and positive")
        configurations[split] = {(m["stiffness"], m["viscosity"]) for m in materials}
        if len(configurations[split]) != 4:
            raise ValueError("duplicate material configuration")
    if any(configurations[a] & configurations[b] for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("material configurations overlap across splits")
    train = configurations["train"]
    stiffness, viscosity = {m[0] for m in train}, {m[1] for m in train}
    if len(stiffness) != 2 or len(viscosity) != 2:
        raise ValueError("training requires a two-by-two material grid")
    def inside(k, mu):
        return min(stiffness) < k < max(stiffness) and min(viscosity) < mu < max(viscosity)
    if not all(inside(*m) for m in configurations["validation"]):
        raise ValueError("validation materials must interpolate training ranges")
    tests = {m["id"]: m for m in manifest["materials"]["test"]}
    conditions = {"base-material": "interpolation_base", "interpolated-material": "interpolation_joint",
                  "high-stiffness": "stiffness_extrapolation", "high-viscosity": "viscosity_extrapolation"}
    if set(tests) != set(conditions) or any(tests[k]["condition"] != v for k, v in conditions.items()):
        raise ValueError("unsupported paired test materials")
    baseline = tests["base-material"]
    if not all(inside(tests[k]["stiffness"], tests[k]["viscosity"])
               for k in ("base-material", "interpolated-material")):
        raise ValueError("interpolation test material lies outside training range")
    if (tests["high-stiffness"]["stiffness"] <= max(stiffness)
            or tests["high-stiffness"]["viscosity"] != baseline["viscosity"]
            or tests["high-viscosity"]["viscosity"] <= max(viscosity)
            or tests["high-viscosity"]["stiffness"] != baseline["stiffness"]):
        raise ValueError("extrapolation pairs must change only the named material parameter")


def read_config():
    manifest, protocol = json.loads(MANIFEST.read_text()), json.loads(PROTOCOL.read_text())
    validate_manifest(manifest)
    return manifest, protocol


def provenance():
    return {"date": "2026-09-13", "protocol_revision": PROTOCOL_REVISION,
            "manifest_sha256": file_digest(MANIFEST), "protocol_sha256": file_digest(PROTOCOL),
            "source_sha256": {p: file_digest(ROOT / p) for p in SOURCE_PATHS},
            "environment": {"python": platform.python_version(), "system": platform.system(),
                            "machine": platform.machine(), "dependencies": "Python standard library only"},
            "license": "MIT (repository license)"}


def verify_provenance(saved):
    current = provenance()
    for key in ("protocol_revision", "manifest_sha256", "protocol_sha256", "source_sha256"):
        if saved[key] != current[key]:
            raise ValueError(f"material evidence {key} mismatch")


def scene_recipes(manifest, split):
    return [(group, material) for group in manifest["groups"] if group["split"] == split
            for material in manifest["materials"][split]]


def initial_scene(manifest, group, material):
    scene = group | {"config": {k: material[k] for k in ("stiffness", "viscosity")}}
    return make_scene(manifest, scene)


def numerical_gates(row, protocol):
    bounds = protocol["numerical_checks"]
    return {key: finite([row[key]]) and 0 <= row[key] <= limit
            for key, limit in (("reference_position_gap", bounds["reference_position_gap_max"]),
                               ("reference_velocity_gap", bounds["reference_velocity_gap_max"]),
                               ("mass_error", bounds["mass_error_max"]),
                               ("momentum_with_gravity_error", bounds["momentum_with_gravity_error_max"]))}


def generate_scene(manifest, protocol, group, material):
    initial, config, gravity = initial_scene(manifest, group, material)
    references = [integrate(initial, config, gravity, manifest["dt"], manifest["steps"], factor,
                            rk4_step, manifest["horizons"]) for factor in manifest["reference_factors"]]
    scene_id = group["id"] + "/" + material["id"]
    rows = []
    mass = math.fsum(p.mass for p in initial)
    for h in manifest["horizons"]:
        states = [reference[h] for reference in references]
        expected = tuple(p + mass * g * h * manifest["dt"] for p, g in zip(momentum(initial), gravity))
        row = {"horizon": h, "reference_position_gap": rmse(*states),
               "reference_velocity_gap": rmse(*states, "velocity"),
               "mass_error": max(abs(math.fsum(p.mass for p in state) - mass) for state in states),
               "momentum_with_gravity_error": max(math.dist(momentum(state), expected) for state in states)}
        row["gates"] = numerical_gates(row, protocol)
        rows.append(row)
    record = {"schema": "cogniarc.particle-material-snapshots.v2", "scene_id": scene_id,
              "group_id": group["id"], "material_id": material["id"], "split": group["split"],
              "geometry_condition": group["condition"], "material_condition": material.get("condition", "development"),
              "seed": group["seed"], "config": asdict(config), "gravity": list(gravity),
              "units": manifest["units"], "particle_ids": list(range(len(initial))),
              "masses": [p.mass for p in initial], "initial_state_sha256": digest([asdict(p) for p in initial]),
              "integrator": "independent_dense_rk4", "internal_dt": manifest["dt"] / 8,
              "snapshots": [{"step": h, "time": h * manifest["dt"],
                             "positions": [list(p.position) for p in state],
                             "velocities": [list(p.velocity) for p in state]}
                            for h, state in sorted(references[-1].items())]}
    if any(not finite(v) for snapshot in record["snapshots"]
           for key in ("positions", "velocities") for v in snapshot[key]):
        raise ValueError("non-finite numerical state")
    return record, {"scene_id": scene_id, "split": group["split"], "group_id": group["id"],
                    "initial_state_sha256": record["initial_state_sha256"], "record_sha256": digest(record),
                    "horizons": rows, "passed": all(all(r["gates"].values()) for r in rows)}


def generate(manifest, protocol, output_dir):
    validate_manifest(manifest)
    report = {"schema": "cogniarc.particle-material-numerics.v2", "scenes": [], "artifacts": {},
              "provenance": provenance()}
    group_by_initial, initial_by_group = {}, {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        records = []
        for group, material in scene_recipes(manifest, split):
            record, measurement = generate_scene(manifest, protocol, group, material)
            initial_hash, group_id = record["initial_state_sha256"], record["group_id"]
            if group_by_initial.setdefault(initial_hash, group_id) != group_id:
                raise ValueError("initial state crosses group boundaries")
            if initial_by_group.setdefault(group_id, initial_hash) != initial_hash:
                raise ValueError("material variants do not share the initial state")
            records.append(record)
            report["scenes"].append(measurement)
        path = output_dir / (split + ".jsonl")
        payload = "".join(canonical(r) + "\n" for r in records).encode()
        path.write_bytes(payload)
        report["artifacts"][split] = {"path": path.name, "bytes": len(payload),
                                       "sha256": sha256(payload).hexdigest(), "scenes": len(records),
                                       "groups": len({r["group_id"] for r in records})}
    report["passed"] = all(row["passed"] for row in report["scenes"])
    return report


def read_report(directory, manifest, protocol):
    report = json.loads((directory / "numerics.json").read_text())
    verify_provenance(report["provenance"])
    if report["schema"] != "cogniarc.particle-material-numerics.v2":
        raise ValueError("unsupported numerical report")
    expected = {g["id"] + "/" + m["id"] for split in SPLITS for g, m in scene_recipes(manifest, split)}
    if {r["scene_id"] for r in report["scenes"]} != expected or len(report["scenes"]) != len(expected):
        raise ValueError("numerical report scene mismatch")
    for row in report["scenes"]:
        if ([r["horizon"] for r in row["horizons"]] != manifest["horizons"] or not row["passed"]
                or any(not all(numerical_gates(r, protocol).values()) for r in row["horizons"])):
            raise ValueError("failed numerical checks; fitting/evaluation refused")
    if not report["passed"]:
        raise ValueError("failed numerical checks; fitting/evaluation refused")
    return report


def read_split(directory, split, manifest, report):
    if split not in SPLITS:
        raise ValueError("unknown split")
    info = report["artifacts"][split]
    if info["path"] != split + ".jsonl":
        raise ValueError("unexpected corpus path")
    raw = (directory / info["path"]).read_bytes()
    if len(raw) != info["bytes"] or sha256(raw).hexdigest() != info["sha256"]:
        raise ValueError("material corpus checksum mismatch")
    records = [json.loads(line) for line in raw.decode().splitlines()]
    recipes = scene_recipes(manifest, split)
    if len(records) != len(recipes) or len(records) != info["scenes"]:
        raise ValueError("material corpus size mismatch")
    measurements = {r["scene_id"]: r for r in report["scenes"]}
    for record, (group, material) in zip(records, recipes):
        initial, config, gravity = initial_scene(manifest, group, material)
        expected = {"schema": "cogniarc.particle-material-snapshots.v2", "scene_id": group["id"] + "/" + material["id"],
                    "group_id": group["id"], "material_id": material["id"], "split": split,
                    "geometry_condition": group["condition"], "material_condition": material.get("condition", "development"),
                    "seed": group["seed"], "config": asdict(config), "gravity": list(gravity), "units": "dimensionless",
                    "particle_ids": list(range(len(initial))), "masses": [p.mass for p in initial],
                    "initial_state_sha256": digest([asdict(p) for p in initial]),
                    "integrator": "independent_dense_rk4", "internal_dt": manifest["dt"] / 8}
        if set(record) != set(expected) | {"snapshots"} or any(record[k] != v for k, v in expected.items()):
            raise ValueError("material snapshot metadata mismatch")
        if digest(record) != measurements[record["scene_id"]]["record_sha256"]:
            raise ValueError("material snapshot differs from numerical report")
        if [s["step"] for s in record["snapshots"]] != [0] + manifest["horizons"]:
            raise ValueError("material snapshot horizons mismatch")
        for snapshot in record["snapshots"]:
            if snapshot["time"] != snapshot["step"] * manifest["dt"]:
                raise ValueError("material snapshot time mismatch")
            for key in ("positions", "velocities"):
                if len(snapshot[key]) != len(initial) or any(len(v) != 3 or not finite(v) for v in snapshot[key]):
                    raise ValueError("invalid material particle vectors")
        if (record["snapshots"][0]["positions"] != [list(p.position) for p in initial]
                or record["snapshots"][0]["velocities"] != [list(p.velocity) for p in initial]):
            raise ValueError("initial state differs from generator")
    return records

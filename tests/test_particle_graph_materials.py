"""Material response, grouped splits, data integrity and frozen-baseline controls."""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.particle_graph import affine, material_data as data, materials
from experiments.particle_graph.validation import canonical, digest


def synthetic_record(manifest, group, material):
    """Analytic acceleration k*centred_position - mu*centred_velocity + gravity."""
    initial, config, gravity = data.initial_scene(manifest, group, material)
    positions, velocities = [list(p.position) for p in initial], [list(p.velocity) for p in initial]
    n = len(initial)
    centre = [sum(p[k] for p in positions) / n for k in range(3)]
    mean_v = [sum(v[k] for v in velocities) / n for k in range(3)]
    accelerations = [[material["stiffness"] * (p[k] - centre[k])
                      - material["viscosity"] * (v[k] - mean_v[k]) + gravity[k] for k in range(3)]
                    for p, v in zip(positions, velocities)]
    snapshots = []
    for h in [0] + manifest["horizons"]:
        t = h * manifest["dt"]
        snapshots.append({"step": h, "time": t,
                          "positions": [[p[k] + t * v[k] + 0.5 * t * t * a[k] for k in range(3)]
                                        for p, v, a in zip(positions, velocities, accelerations)],
                          "velocities": [[v[k] + t * a[k] for k in range(3)] for v, a in zip(velocities, accelerations)]})
    return {"schema": "cogniarc.particle-material-snapshots.v2", "scene_id": group["id"] + "/" + material["id"],
            "group_id": group["id"], "material_id": material["id"], "split": group["split"],
            "geometry_condition": group["condition"], "material_condition": material.get("condition", "development"),
            "seed": group["seed"], "config": asdict(config), "gravity": list(gravity), "units": "dimensionless",
            "particle_ids": list(range(n)), "masses": [p.mass for p in initial],
            "initial_state_sha256": digest([asdict(p) for p in initial]), "integrator": "independent_dense_rk4",
            "internal_dt": manifest["dt"] / 8, "snapshots": snapshots}


class MaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.protocol = data.read_config()
        cls.train = [synthetic_record(cls.manifest, g, m) for g, m in data.scene_recipes(cls.manifest, "train")]
        cls.validation = [synthetic_record(cls.manifest, g, m) for g, m in data.scene_recipes(cls.manifest, "validation")]
        cls.model = materials.select_models(cls.train, cls.validation, cls.manifest, cls.protocol)
        # These targets follow the analytic test law, not the reserved SPH corpus.
        fake_group = {"id": "analytic-test-control", "split": "test", "seed": 9999,
                      "shape": [2, 2, 2], "gravity": [0, -0.3, 0], "condition": "unseen_seed"}
        cls.fake_test = [synthetic_record(cls.manifest, fake_group, m) for m in cls.manifest["materials"]["test"]]

    def test_manifest_preserves_groups_and_uses_new_seeds(self):
        self.assertEqual([len(data.scene_recipes(self.manifest, split)) for split in data.SPLITS], [16, 8, 12])
        for split, expected in zip(data.SPLITS, (4, 2, 3)):
            self.assertEqual(len({g["id"] for g, _ in data.scene_recipes(self.manifest, split)}), expected)
        wrong = deepcopy(self.manifest)
        wrong["groups"][-1]["seed"] = wrong["groups"][0]["seed"]
        with self.assertRaisesRegex(ValueError, "seeds"):
            data.validate_manifest(wrong)
        wrong["groups"][-1]["seed"] = 101
        with self.assertRaisesRegex(ValueError, "absent from v1"):
            data.validate_manifest(wrong)

    def test_material_variants_share_only_their_own_initial_state(self):
        hashes = set()
        for group in self.manifest["groups"]:
            states = [data.initial_scene(self.manifest, group, m)[0] for m in self.manifest["materials"][group["split"]]]
            self.assertTrue(all(s == states[0] for s in states))
            hashes.add(digest([asdict(p) for p in states[0]]))
        self.assertEqual(len(hashes), 9)

    def test_wrong_material_holdout_labels_and_parameters_are_rejected(self):
        cases = []
        wrong = deepcopy(self.manifest); wrong["materials"]["test"][2]["viscosity"] = 0.07; cases.append(wrong)
        wrong = deepcopy(self.manifest); wrong["materials"]["validation"][0]["stiffness"] = 0.6; cases.append(wrong)
        wrong = deepcopy(self.manifest); wrong["materials"]["train"][0]["viscosity"] = math.nan; cases.append(wrong)
        wrong = deepcopy(self.manifest); wrong["materials"]["test"][2]["condition"] = "interpolation_joint"; cases.append(wrong)
        for manifest in cases:
            with self.assertRaises(ValueError):
                data.validate_manifest(manifest)

    def test_numerical_export_matches_single_particle_free_flight(self):
        manifest = deepcopy(self.manifest)
        manifest.update(steps=2, horizons=[1, 2])
        group = manifest["groups"][1] | {"shape": [1, 1, 1]}
        material = manifest["materials"]["train"][0]
        record, report = data.generate_scene(manifest, self.protocol, group, material)
        self.assertTrue(report["passed"])
        initial = record["snapshots"][0]
        for target in record["snapshots"][1:]:
            expected = affine.predict(initial, record["gravity"], target["time"], "known_gravity_ballistic")
            self.assertLess(max(affine.errors(expected, target).values()), 1e-14)
        self.assertEqual(record["group_id"], group["id"])
        self.assertEqual(record["config"]["stiffness"], material["stiffness"])

    def test_numerical_gates_reject_bad_reference_even_with_pass_flag(self):
        row = {"reference_position_gap": 0, "reference_velocity_gap": 0, "mass_error": 0, "momentum_with_gravity_error": 0}
        self.assertTrue(all(data.numerical_gates(row, self.protocol).values()))
        for value in (1e-3, math.nan, -1):
            wrong = row | {"reference_position_gap": value, "passed": True}
            self.assertFalse(all(data.numerical_gates(wrong, self.protocol).values()))

    def test_blind_refit_is_identical_to_the_unchanged_v1_fit_algorithm(self):
        actual = materials.fit_candidate(self.train, [1, 10, 50], 0.01, "blind_v2")
        expected = affine.fit_candidate(self.train, [1, 10, 50], 0.01)
        self.assertEqual(actual, expected)

    def test_conditioned_model_recovers_an_independent_material_acceleration_law(self):
        models = materials.fit_candidate(self.train, [50], 1e-6, "material_v2")
        for scene in self.fake_test:
            predicted = materials.predict(scene["snapshots"][0], scene["gravity"], scene["config"],
                                          0.1, "material_v2", models["50"])
            error = affine.errors(predicted, scene["snapshots"][-1])
            self.assertLess(error["position_rmse"], 1e-8)
            self.assertLess(error["velocity_rmse"], 2e-7)

    def test_validation_targets_cannot_modify_any_candidate_coefficients(self):
        changed = deepcopy(self.validation)
        for scene in changed:
            for target in scene["snapshots"][1:]:
                target["velocities"][0][1] += 10
        second = materials.select_models(self.train, changed, self.manifest, self.protocol)
        for key in materials.VARIANTS:
            self.assertEqual([c["fit_sha256"] for c in self.model["variants"][key]["candidates"]],
                             [c["fit_sha256"] for c in second["variants"][key]["candidates"]])
            self.assertNotEqual([c["validation_score"] for c in self.model["variants"][key]["candidates"]],
                                [c["validation_score"] for c in second["variants"][key]["candidates"]])

    def test_group_and_initial_state_overlap_are_rejected_during_selection(self):
        for key in ("group_id", "seed", "initial_state_sha256"):
            changed = deepcopy(self.validation)
            changed[0][key] = self.train[0][key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "overlap"):
                materials.select_models(self.train, changed, self.manifest, self.protocol)
        with self.assertRaisesRegex(ValueError, "train scenes only"):
            materials.fit_candidate(self.fake_test, [50], 0.01, "material_v2")

    def test_scoring_cannot_refit_and_blind_models_have_zero_material_response(self):
        frozen = canonical(self.model)
        with patch.object(materials, "fit_candidate", side_effect=AssertionError("refit")), \
                patch.object(materials, "select_models", side_effect=AssertionError("retune")):
            report = materials.evaluate_test(self.model, materials.load_frozen_v1(self.protocol),
                                             self.fake_test, self.manifest, self.protocol)
        self.assertEqual(canonical(self.model), frozen)
        self.assertEqual(len(report["paired_material_changes"]), 9)
        for pair in report["paired_material_changes"]:
            for method in (*materials.BASELINES, "frozen_v1", "blind_v2"):
                self.assertEqual(pair["metrics"][method], pair["target_change_rms"])
            self.assertLess(pair["metrics"]["material_v2"]["position_rmse"], pair["target_change_rms"]["position_rmse"])

    def test_reader_verifies_pair_metadata_and_corruption(self):
        manifest = deepcopy(self.manifest)
        manifest["groups"] = manifest["groups"][:1]
        records = self.train[:4]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            def write(records):
                raw = "".join(canonical(r) + "\n" for r in records).encode()
                (directory / "train.jsonl").write_bytes(raw)
                return {"artifacts": {"train": {"path": "train.jsonl", "bytes": len(raw), "sha256": sha256(raw).hexdigest(), "scenes": 4}},
                        "scenes": [{"scene_id": r["scene_id"], "record_sha256": digest(r)} for r in records]}
            report = write(records)
            self.assertEqual(data.read_split(directory, "train", manifest, report), records)
            (directory / "train.jsonl").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                data.read_split(directory, "train", manifest, report)
            changed = deepcopy(records)
            changed[0]["group_id"] = "other-group"
            report = write(changed)
            with self.assertRaisesRegex(ValueError, "metadata"):
                data.read_split(directory, "train", manifest, report)

    def test_macro_error_uses_equal_scene_weights(self):
        rows = [{"particles": n, "metrics": {"blind_v2": {"position_rmse": p, "velocity_rmse": v}}}
                for n, p, v in ((8, 1.0, 2.0), (27, 9.0, 4.0))]
        self.assertEqual(materials.macro_errors(rows, ["blind_v2"]),
                         {"blind_v2": {"position_rmse": 5.0, "velocity_rmse": 3.0}})


if __name__ == "__main__":
    unittest.main()

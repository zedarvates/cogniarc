"""Numerical recovery, observable invariants and split isolation for ridge fits."""

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.particle_graph import affine


class AffineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = affine.read_protocol()
        development = affine.read_corpus(cls.protocol, {"train", "validation"})
        cls.train = [r for r in development if r["split"] == "train"]
        cls.validation = [r for r in development if r["split"] == "validation"]
        cls.model = affine.select_model(cls.train, cls.validation, cls.protocol)

    def test_affine_recovery_and_analytic_ridge_shrinkage(self):
        rows = [[-1.0], [1.0]]
        targets = [[3.0 - 2.0, -7.0 + 4.0], [3.0 + 2.0, -7.0 - 4.0]]
        for alpha in (1e-6, 1.0, 100.0):
            model = affine.ridge_fit(rows, targets, [0.5, 0.5], alpha)
            self.assertEqual(model["bias"], [3.0, -7.0])
            self.assertAlmostEqual(model["coefficients"][0][0], 2 / (1 + alpha), places=12)
            self.assertAlmostEqual(model["coefficients"][1][0], -4 / (1 + alpha), places=12)
        recovered = affine.affine_values(affine.ridge_fit(rows, targets, [1, 1], 1e-12), [[0.3]])[0]
        self.assertAlmostEqual(recovered[0], 3.6, places=10)
        self.assertAlmostEqual(recovered[1], -8.2, places=10)

    def test_collinear_and_constant_features_remain_finite(self):
        model = affine.ridge_fit([[-1, -2, 5], [1, 2, 5]], [[-3], [3]], [1, 1], 1e-6)
        self.assertEqual(model["feature_scale"][2], 1.0)
        self.assertEqual(model["coefficients"][0][2], 0.0)
        predicted = affine.affine_values(model, [[0, 0, 5], [0.5, 1, 5]])
        self.assertAlmostEqual(predicted[0][0], 0.0, places=12)
        self.assertAlmostEqual(predicted[1][0], 1.5, places=5)

    def test_invalid_regression_inputs_are_rejected(self):
        for alpha in (0, -1, math.nan, math.inf, True):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                affine.ridge_fit([[0], [1]], [[1], [2]], [1, 1], alpha)
        for rows, targets, weights in (([[0], [math.nan]], [[1], [2]], [1, 1]),
                                        ([[0], [1]], [[1]], [1, 1]),
                                        ([[0], [1, 2]], [[1], [2]], [1, 1]),
                                        ([[0], [1]], [[1], [2]], [1, -1])):
            with self.assertRaises(ValueError):
                affine.ridge_fit(rows, targets, weights, 0.01)

    def test_baselines_match_analytic_free_flight(self):
        initial = {"positions": [[1.0, -2.0, 3.0]], "velocities": [[0.4, -0.2, 0.1]]}
        gravity, t = [0.0, -0.3, 0.2], 0.7
        target = {"positions": [[1.28, -2.2135, 3.119]], "velocities": [[0.4, -0.41, 0.24]]}
        actual = affine.predict(initial, gravity, t, "known_gravity_ballistic")
        self.assertLess(max(affine.errors(actual, target).values()), 1e-15)
        self.assertEqual(affine.predict(initial, gravity, t, "persistence"), initial)
        cv = affine.predict(initial, gravity, t, "constant_velocity")
        self.assertAlmostEqual(cv["positions"][0][1], -2.14)
        self.assertEqual(cv["velocities"], initial["velocities"])

    def test_fit_rejects_validation_and_test_records(self):
        for split in ("validation", "test"):
            wrong = deepcopy(self.train)
            wrong[0]["split"] = split
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "train scenes only"):
                affine.fit_candidate(wrong, self.protocol["horizons"], 0.01)
        overlap = deepcopy(self.validation)
        overlap[0]["scene_id"] = self.train[0]["scene_id"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            affine.select_model(self.train, overlap, self.protocol)

    def test_validation_targets_cannot_change_candidate_fits_or_normalization(self):
        changed = deepcopy(self.validation)
        for record in changed:
            for snapshot in record["snapshots"][1:]:
                for position in snapshot["positions"]:
                    position[0] += 100.0
        second = affine.select_model(self.train, changed, self.protocol)
        self.assertEqual([c["fit_sha256"] for c in self.model["candidates"]],
                         [c["fit_sha256"] for c in second["candidates"]])
        self.assertNotEqual([c["validation_score"] for c in self.model["candidates"]],
                            [c["validation_score"] for c in second["candidates"]])
        # Expected gravity statistics from three equal-weight training scenes.
        for model in self.model["horizon_models"].values():
            self.assertAlmostEqual(model["feature_mean"][7], -0.1, places=14)
            self.assertAlmostEqual(model["feature_scale"][7], math.sqrt(0.02), places=14)

    def test_duplicating_particles_within_one_scene_does_not_reweight_it(self):
        duplicated = deepcopy(self.train)
        for snapshot in duplicated[0]["snapshots"]:
            for key in ("positions", "velocities"):
                snapshot[key] *= 2
        before = affine.fit_candidate(self.train, [10], 0.01)["10"]
        after = affine.fit_candidate(duplicated, [10], 0.01)["10"]
        for key in ("feature_mean", "feature_scale", "bias"):
            for a, b in zip(before[key], after[key]):
                self.assertAlmostEqual(a, b, places=12)
        for a, b in zip(before["coefficients"], after["coefficients"]):
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y, places=12)

    def test_prediction_respects_translation_boost_and_particle_permutation(self):
        initial = deepcopy(self.validation[0]["snapshots"][0])
        gravity, t = self.validation[0]["gravity"], 0.1
        model = self.model["horizon_models"]["50"]
        base = affine.predict(initial, gravity, t, "affine", model)
        shift, boost = [3.0, -2.0, 1.0], [-0.3, 0.2, 0.4]
        moved = {"positions": [[p[k] + shift[k] for k in range(3)] for p in reversed(initial["positions"])],
                 "velocities": [[v[k] + boost[k] for k in range(3)] for v in reversed(initial["velocities"])]}
        actual = affine.predict(moved, gravity, t, "affine", model)
        for q, u, p, v in zip(actual["positions"], actual["velocities"],
                              reversed(base["positions"]), reversed(base["velocities"])):
            for k in range(3):
                self.assertAlmostEqual(q[k], p[k] + shift[k] + t * boost[k], places=12)
                self.assertAlmostEqual(u[k], v[k] + boost[k], places=12)

    def test_evaluator_cannot_refit_and_target_changes_leave_model_unchanged(self):
        # A synthetic test-shaped copy of validation exercises scoring before
        # the real reserved test targets are evaluated by the experiment CLI.
        fake_test = deepcopy(self.validation[:1])
        fake_test[0].update(split="test", scene_id="synthetic-scoring-control")
        frozen = affine.canonical(self.model)
        with patch.object(affine, "fit_candidate", side_effect=AssertionError("refit")), \
                patch.object(affine, "select_model", side_effect=AssertionError("retune")):
            first = affine.evaluate_test(self.model, fake_test, self.protocol)
            fake_test[0]["snapshots"][-1]["positions"][0][0] += 10.0
            second = affine.evaluate_test(self.model, fake_test, self.protocol)
        self.assertNotEqual(first["scenes"], second["scenes"])
        self.assertEqual(affine.canonical(self.model), frozen)
        self.assertFalse(second["selection_changed_during_evaluation"])

    def test_best_validation_score_selects_alpha_with_declared_tie_break(self):
        expected = min(self.model["candidates"], key=lambda c: (c["validation_score"], -c["alpha"]))
        self.assertEqual(self.model["selected_alpha"], expected["alpha"])
        self.assertEqual(affine.digest(self.model["horizon_models"]), expected["fit_sha256"])
        with patch.object(affine, "errors", return_value={"position_rmse": 1.0, "velocity_rmse": 1.0}):
            tied = affine.select_model(self.train, self.validation, self.protocol)
        self.assertEqual(tied["selected_alpha"], max(self.protocol["ridge_alphas"]))

    def test_corpus_checksum_rejects_modified_input(self):
        source = affine.ROOT / self.protocol["corpus"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshots.jsonl"
            path.write_bytes(source.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                affine.read_corpus(self.protocol, {"train"}, path)

    def test_model_selection_and_provenance_tampering_are_rejected(self):
        valid = deepcopy(self.model)
        valid["provenance"] = affine.provenance(self.protocol)
        affine.verify_model(valid, self.protocol)
        for field in ("protocol_sha256", "corpus_sha256", "source_sha256"):
            broken = deepcopy(valid)
            broken["provenance"][field] = "wrong"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "mismatch"):
                affine.verify_model(broken, self.protocol)
        broken = deepcopy(valid)
        broken["horizon_models"]["1"]["bias"][0] += 1.0
        with self.assertRaisesRegex(ValueError, "selection"):
            affine.verify_model(broken, self.protocol)


if __name__ == "__main__":
    unittest.main()

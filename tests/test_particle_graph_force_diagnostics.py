"""Numerical and evidence-consumption checks on disposable diagnostic fixtures."""

from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import tempfile
import unittest

from experiments.particle_graph import force_diagnostics as diagnostic
from experiments.particle_graph.reference import Config


class ForceDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = diagnostic.read_protocol()
        cls.coefficients = [1.0, 2.0, 3.0, 4.0]

    def observed(self, positions=None, velocities=None, masses=None, config=None):
        positions = positions if positions is not None else [[-0.13, 0.02, 0.0], [0.19, -0.04, 0.01]]
        masses = masses if masses is not None else [0.17, 0.29]
        return {"scene_id": "disposable/unit", "group_id": "disposable", "role": "unit", "path": "unit",
                "step": 0, "time": 0.0, "units": "dimensionless", "config": asdict(config or Config(radius=0.73)),
                "masses": masses, "particle_ids": list(range(len(masses))),
                "material_condition": "unit", "mass_condition": "unit",
                "state": {"positions": positions, "velocities": velocities if velocities is not None
                          else [[-0.07, 0.03, 0.02], [0.11, -0.06, 0.01]]}}

    def measured(self, observed=None):
        return diagnostic.diagnose(observed or self.observed(), self.coefficients, self.protocol)

    def assert_all_checks(self, row):
        self.assertTrue(all(row["checks"].values()), [k for k, v in row["checks"].items() if not v])

    def test_pair_reconstruction_and_components_on_unequal_cloud(self):
        rng = random.Random(733131)
        observed = self.observed([[rng.uniform(-0.31, 0.31) for _ in range(3)] for _ in range(6)],
                                 [[rng.uniform(-0.14, 0.14) for _ in range(3)] for _ in range(6)],
                                 [0.07 + 0.04 * i for i in range(6)])
        self.assert_all_checks(self.measured(observed))

    def test_low_density_pressure_error_is_retained_not_gate_failure(self):
        row = self.measured(self.observed(masses=[0.021, 0.043]))
        self.assertLess(max(row["densities"]), 1)
        self.assertEqual(row["accelerations"]["dense"]["pressure"], [[0.0] * 3] * 2)
        self.assertGreater(diagnostic.particle_stats(row)["components"]["pressure"]["local_ms"], 0)
        self.assert_all_checks(row)

    def test_isolated_particle_has_self_density_and_zero_force(self):
        row = self.measured(self.observed([[0.03, -0.04, 0.01]], [[0.02, 0.03, -0.01]], [0.17]))
        self.assertGreater(row["densities"][0], 0)
        self.assertEqual(row["neighbour_densities"], [0.0])
        self.assertEqual(row["neighbour_counts"], [0])
        self.assertEqual(row["pair_statistics"]["all"], {})
        self.assert_all_checks(row)

    def test_exact_and_outside_support_have_no_pairs(self):
        for separation in (0.73, 0.7300000001):
            with self.subTest(separation=separation):
                row = self.measured(self.observed([[0, 0, 0], [separation, 0, 0]]))
                self.assertEqual(row["neighbour_counts"], [0, 0])
                self.assert_all_checks(row)

    def test_coincident_pressure_zero_but_viscosity_can_act(self):
        row = self.measured(self.observed([[0.01, 0, 0], [0.01, 0, 0]]))
        for name in diagnostic.IMPLEMENTATIONS:
            self.assertEqual(row["accelerations"][name]["pressure"], [[0.0] * 3] * 2)
            self.assertTrue(any(x != 0 for a in row["accelerations"][name]["viscosity"] for x in a))
        self.assert_all_checks(row)

    def test_unequal_mass_accelerations_balance_force_not_acceleration(self):
        row = self.measured()
        masses = row["observation"]["masses"]
        for name in diagnostic.IMPLEMENTATIONS:
            a, b = row["accelerations"][name]["total"]
            for x, y in zip(a, b):
                self.assertAlmostEqual(masses[0] * x, -masses[1] * y, places=13)
            self.assertTrue(any(abs(x + y) > 1e-8 for x, y in zip(a, b)))

    def test_permutation_preserves_descriptors_and_particle_forces(self):
        before = self.observed()
        after = deepcopy(before)
        after["masses"].reverse()
        for field in ("positions", "velocities"):
            after["state"][field].reverse()
        a, b = self.measured(before), self.measured(after)
        self.assertEqual(a["densities"], list(reversed(b["densities"])))
        for name in diagnostic.IMPLEMENTATIONS:
            for mode in diagnostic.MODES:
                self.assertTrue(diagnostic.close_vectors(a["accelerations"][name][mode],
                                                         list(reversed(b["accelerations"][name][mode])), self.protocol))
        self.assert_all_checks(b)

    def test_context_affects_physical_pair_but_not_density_free_pair(self):
        original = self.observed()
        original["path"] = "context"
        extended = deepcopy(original)
        extended["state"]["positions"].append([0.04, 0.21, 0.0])
        extended["state"]["velocities"].append([0.0, 0.0, 0.0])
        extended["masses"].append(0.23)
        extended["particle_ids"].append(2)
        a, b = self.measured(original), self.measured(extended)
        self.assertEqual(a["target_pair"]["forces"]["local"], b["target_pair"]["forces"]["local"])
        self.assertNotEqual(a["target_pair"]["forces"]["dense"]["viscosity"], b["target_pair"]["forces"]["dense"]["viscosity"])
        self.assert_all_checks(b)

    def test_zero_material_component_controls(self):
        for config, mode in ((Config(radius=0.73, stiffness=0), "pressure"),
                              (Config(radius=0.73, viscosity=0), "viscosity")):
            row = self.measured(self.observed(config=config))
            for name in diagnostic.IMPLEMENTATIONS:
                self.assertEqual(row["accelerations"][name][mode], [[0.0] * 3] * 2)
            self.assert_all_checks(row)

    def test_equal_state_weight_not_pooled_particle_weight(self):
        row = self.measured()
        stats = []
        for count, error in ((1, 1.0), (7, 3.0)):
            component = diagnostic.vector_stats([[0.0] * 3] * count, [[error] * 3] * count)
            stats.append((row, {"items": count, "components": {mode: component for mode in diagnostic.MODES}}))
        combined = diagnostic.combine(stats)
        self.assertEqual(combined["items"], 8)
        for mode in diagnostic.MODES:
            self.assertAlmostEqual(combined["components"][mode]["rmse"], math.sqrt(5))
            self.assertIsNone(combined["components"][mode]["relative_rmse"])

    def test_empty_bin_is_null_not_zero_error(self):
        result = diagnostic.combine([])
        self.assertEqual(result["states"], 0)
        self.assertTrue(all(c["rmse"] is None for c in result["components"].values()))

    def test_bin_boundaries_and_mass_roundoff_margin(self):
        self.assertEqual(diagnostic.bucket(1, [1, 2, 4]), "[1,2)")
        edges = self.protocol["bins"]["mass_ratio"]["edges"]
        self.assertEqual(diagnostic.bucket(3, edges), diagnostic.bucket(3.0000000000000004, edges))
        self.assertNotEqual(diagnostic.bucket(3, edges), diagnostic.bucket(3.001, edges))
        self.assertEqual(len(set(diagnostic.labels(edges))), 4)

    def test_reader_consumes_saved_record_and_rejects_tampering(self):
        original, row = self.observed(), self.measured()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(diagnostic.read_records(path, [original], self.coefficients, self.protocol), [row])
            for field in ("state", "density", "acceleration", "pair_statistic"):
                changed = deepcopy(row)
                if field == "state":
                    changed["observation"]["state"]["positions"][0][0] += 0.01
                elif field == "density":
                    changed["densities"][0] += 0.01
                elif field == "acceleration":
                    changed["accelerations"]["dense"]["total"][0][0] += 0.01
                else:
                    changed["pair_statistics"]["all"]["all"]["components"]["pressure"]["mse"] += 0.01
                path.write_text(json.dumps(changed) + "\n")
                with self.subTest(field=field), self.assertRaises(ValueError):
                    diagnostic.read_records(path, [original], self.coefficients, self.protocol)

    def test_reader_rejects_missing_duplicate_and_unexpected_records(self):
        original, row = self.observed(), self.measured()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            for content in ("", (json.dumps(row) + "\n") * 2):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    diagnostic.read_records(path, [original], self.coefficients, self.protocol)
            changed = deepcopy(row)
            changed["observation"]["scene_id"] = "unexpected"
            path.write_text(json.dumps(changed) + "\n")
            with self.assertRaises(ValueError):
                diagnostic.read_records(path, [original], self.coefficients, self.protocol)

    def test_failed_numerics_suppress_all_aggregate_scores(self):
        row = self.measured()
        row["checks"]["dense_pressure_reconstruction"] = False
        result = diagnostic.summarize([row], self.protocol)
        self.assertFalse(result["all_checks_pass"])
        self.assertIsNone(result["main"])
        self.assertIsNone(result["static_controls"])
        self.assertIn("dense_pressure_reconstruction", result["failures"][0]["checks"])


if __name__ == "__main__":
    unittest.main()

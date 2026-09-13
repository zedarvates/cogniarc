"""Numerical-reference controls and negative controls for scene split leakage."""

from copy import deepcopy
import json
import math
import random
import unittest

from experiments.particle_graph.dense_reference import accelerations, rk4_step
from experiments.particle_graph.reference import Config, Particle, evaluate, step
from experiments.particle_graph.validation import (
    MANIFEST, convergence_gate, integrate, make_scene, run, validate_manifest,
)


class NumericalValidationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(MANIFEST.read_text())

    def test_dense_acceleration_matches_graph_with_unequal_masses(self):
        rng = random.Random(91)
        state = tuple(Particle(tuple(rng.uniform(-0.5, 0.5) for _ in range(3)),
                               tuple(rng.uniform(-0.1, 0.1) for _ in range(3)),
                               rng.uniform(0.1, 0.5)) for _ in range(13))
        config, gravity = Config(), (0.1, -0.2, 0.3)
        expected = evaluate(state, config).accelerations
        actual = accelerations(state, config, gravity)
        for a, b in zip(expected, actual):
            for x, y, g in zip(a, b, gravity):
                self.assertAlmostEqual(x+g, y, places=11)

    def test_rk4_matches_analytic_ballistic_motion(self):
        initial = (Particle((0, 0, 0), (0.2, 0.1, -0.3), 0.25),
                   Particle((10, 0, 0), (-0.1, 0.2, 0.3), 0.5))
        gravity, dt, steps = (0.1, -0.3, 0.2), 0.02, 15
        state = initial
        for _ in range(steps):
            state = rk4_step(state, Config(), dt, gravity)
        t = dt*steps
        for p, q in zip(initial, state):
            self.assertEqual(p.mass, q.mass)
            for k in range(3):
                self.assertAlmostEqual(q.position[k], p.position[k]+t*p.velocity[k]+0.5*t*t*gravity[k], places=12)
                self.assertAlmostEqual(q.velocity[k], p.velocity[k]+t*gravity[k], places=12)

    def test_refinements_compare_same_physical_times(self):
        initial = (Particle((0, 0, 0), (0.2, -0.3, 0.4)),)
        for factor in (1, 2, 4, 8):
            with self.subTest(factor=factor):
                snapshots = integrate(initial, Config(), (0, 0, 0), 0.01, 10,
                                      factor, step, (1, 5, 10))
                self.assertEqual(list(snapshots), [0, 1, 5, 10])
                for tick in (1, 5, 10):
                    for v, x in zip(initial[0].velocity, snapshots[tick][0].position):
                        self.assertAlmostEqual(x, tick*0.01*v, places=14)

    def test_manifest_has_distinct_scene_splits_and_replays(self):
        validate_manifest(self.manifest)
        self.assertEqual({s["split"] for s in self.manifest["scenes"]}, {"train", "validation", "test"})
        first = [make_scene(self.manifest, s) for s in self.manifest["scenes"]]
        second = [make_scene(self.manifest, s) for s in self.manifest["scenes"]]
        self.assertEqual(first, second)
        self.assertEqual(len({scene[0] for scene in first}), len(first))

    def test_seed_leakage_and_duplicate_ids_are_rejected(self):
        for field in ("seed", "id"):
            manifest = deepcopy(self.manifest)
            manifest["scenes"][-1][field] = manifest["scenes"][0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_manifest(manifest)

    def test_holdout_label_cannot_hide_development_particle_counts(self):
        manifest = deepcopy(self.manifest)
        scene = next(s for s in manifest["scenes"] if s["condition"] == "held_out_particle_count")
        scene["shape"] = [2, 2, 2]
        with self.assertRaisesRegex(ValueError, "particle count"):
            validate_manifest(manifest)

    def test_holdout_label_cannot_hide_development_parameters(self):
        manifest = deepcopy(self.manifest)
        scene = next(s for s in manifest["scenes"] if s["condition"] == "held_out_parameters")
        scene["config"] = {}
        with self.assertRaisesRegex(ValueError, "parameters"):
            validate_manifest(manifest)

    def test_invalid_protocols_are_rejected(self):
        cases = []
        for key, value in (("dt", math.nan), ("steps", True), ("steps", 101),
                           ("horizons", [10, 1, 50]), ("refinement_factors", [1, 2, 8]),
                           ("reference_factors", [4, 4])):
            manifest = deepcopy(self.manifest); manifest[key] = value; cases.append(manifest)
        for gravity in ((0, math.inf, 0), (0, 0), (True, 0, 0)):
            manifest = deepcopy(self.manifest); manifest["scenes"][0]["gravity"] = gravity; cases.append(manifest)
        for shape in ([2, True, 2], [5, 5, 5], [0, 2, 2]):
            manifest = deepcopy(self.manifest); manifest["scenes"][0]["shape"] = shape; cases.append(manifest)
        for manifest in cases:
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                validate_manifest(manifest)

    def test_gate_rejects_divergence_and_inaccurate_reference(self):
        self.assertTrue(convergence_gate([4e-5, 2e-5, 1e-5], 1e-8))
        self.assertFalse(convergence_gate([1e-5, 2e-5, 4e-5], 1e-8))
        self.assertFalse(convergence_gate([4e-5, 2e-5, 1e-5], 1e-5))
        self.assertFalse(convergence_gate([4e-5, math.nan, 1e-5], 0))
        self.assertFalse(convergence_gate([4e-5, 4e-5, 4e-5], 0))
        self.assertTrue(convergence_gate([0, 0, 0], 0))

    def test_small_end_to_end_retains_ids_splits_times_and_gravity_balance(self):
        manifest = deepcopy(self.manifest)
        manifest["steps"] = 2; manifest["horizons"] = [1, 2]
        manifest["scenes"] = [manifest["scenes"][i] for i in (1, 3, 5)]
        for scene in manifest["scenes"]:
            scene["shape"] = [2, 1, 1]
        report, corpus = run(manifest)
        self.assertTrue(report["passed"])
        self.assertEqual([r["scene_id"] for r in report["scenes"]], [r["scene_id"] for r in corpus])
        self.assertEqual([r["split"] for r in corpus], ["train", "validation", "test"])
        for record in corpus:
            self.assertEqual(record["particle_ids"], [0, 1])
            self.assertEqual([s["time"] for s in record["snapshots"]], [0, 0.002, 0.004])
            for snapshot in record["snapshots"]:
                self.assertEqual(len(snapshot["positions"]), len(record["masses"]))
        self.assertTrue(all(r["momentum_error"] < 1e-10 for r in report["scenes"]))


if __name__ == "__main__":
    unittest.main()

"""Analytic conservation, reserved-data isolation and projection failure controls."""

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.particle_graph import affine, conservation as cons, materials, rollouts
from experiments.particle_graph.validation import digest


class ConservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = cons.read_protocol()
        cls.old_scenes, cls.models, cls.frozen = rollouts.load_inputs(rollouts.read_protocol())
        # Tests use separate disposable recipes, never the reserved benchmark seeds.
        cls.manifest = json.loads(cons.MANIFEST.read_text())
        cls.manifest["groups"] = [{"id": "unit-control", "seed": 9001, "shape": [2, 1, 1]}]

    def test_unequal_mass_projection_has_exact_ledgers_and_preserves_relative_state(self):
        initial = {"positions": [[1.0, 2.0, -1.0], [-2.0, 0.0, 3.0]],
                   "velocities": [[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]]}
        proposed = {"positions": [[2.0, 5.0, -1.5], [-1.2, 1.0, 4.0]],
                    "velocities": [[4.0, -3.0, 2.0], [0.1, 2.0, -1.0]]}
        masses, gravity, dt = [2.0, 0.5], [0.12, -0.24, 0.09], 0.02
        before = deepcopy([initial, proposed, masses])
        corrected = cons.project_proposal(initial, proposed, masses, gravity, dt)
        ledgers = rollouts.diagnostics(corrected, initial, masses, gravity, dt, corrected, self.protocol)
        self.assertLess(ledgers["momentum_error"], 2e-15)
        self.assertLess(ledgers["centre_of_mass_error"], 2e-15)
        for key in ("positions", "velocities"):
            for k in range(3):
                self.assertAlmostEqual(corrected[key][1][k] - corrected[key][0][k],
                                       proposed[key][1][k] - proposed[key][0][k], places=14)
        self.assertEqual([initial, proposed, masses], before)

    def test_500_steps_remove_arbitrary_uniform_model_bias(self):
        initial = {"positions": [[1.0, -2.0, 0.5]], "velocities": [[0.2, 0.3, -0.1]]}
        masses, gravity, dt = [2.5], [0.12, -0.24, 0.09], 0.002
        def advance(state, time):
            raw = {"positions": [[x + 0.7 for x in state["positions"][0]]],
                   "velocities": [[v - 0.9 for v in state["velocities"][0]]]}
            return cons.project_proposal(state, raw, masses, gravity, time)
        result = rollouts.trace(initial, masses, dt, 500, self.protocol["checkpoints"], advance, 1e6)
        self.assertTrue(result["completed"])
        for tick, state in result["snapshots"].items():
            if tick == 0:
                continue
            target = affine.predict(initial, gravity, tick * dt, "known_gravity_ballistic")
            self.assertLess(max(affine.errors(state, target).values()), 5e-13)

    def test_projection_is_idempotent_for_a_ballistic_proposal(self):
        initial = {"positions": [[-1.0, 0, 0], [2.0, 0, 0]], "velocities": [[0.2, 0, 0], [-0.1, 0, 0]]}
        masses, gravity, dt = [0.5, 2.0], [0.12, -0.24, 0.09], 0.1
        target = affine.predict(initial, gravity, dt, "known_gravity_ballistic")
        projected = cons.project_proposal(initial, target, masses, gravity, dt)
        again = cons.project_proposal(initial, projected, masses, gravity, dt)
        self.assertLess(max(affine.errors(projected, target).values()), 1e-15)
        self.assertLess(max(affine.errors(projected, again).values()), 1e-15)

    def test_invalid_raw_proposals_cannot_be_hidden_by_projection(self):
        initial = {"positions": [[0.0, 0.0, 0.0]], "velocities": [[0.0, 0.0, 0.0]]}
        for invalid in (math.nan, math.inf, 1e7):
            with self.subTest(invalid=invalid):
                def advance(state, dt):
                    proposal = {"positions": [[invalid, 0, 0]], "velocities": [[invalid, 0, 0]]}
                    return cons.project_proposal(state, proposal, [1.0], [0, 0, 0], dt)
                result = rollouts.trace(initial, [1.0], 0.002, 2, [1, 2], advance, 1e6)
                self.assertEqual(result["failure"]["step"], 1)
                self.assertEqual(set(result["snapshots"]), {0})
        with self.assertRaises(ValueError):
            cons.project_proposal(initial, {"positions": [], "velocities": []}, [1], [0, 0, 0], 0.1)
        for masses, gravity, dt in (([0], [0, 0, 0], 0.1), ([1], [0, 0], 0.1), ([1], [0, 0, 0], 0)):
            with self.assertRaises(ValueError):
                cons.project_proposal(initial, initial, masses, gravity, dt)

    def test_stepper_ignores_future_targets_and_unused_horizon_models(self):
        scene = deepcopy(self.old_scenes[0])
        altered, models = deepcopy(scene), deepcopy(self.models)
        altered["snapshots"] = altered["snapshots"][:1] + [{"step": 10, "positions": None, "velocities": None}]
        for variant in models["variants"].values():
            variant["horizon_models"]["10"] = None
            variant["horizon_models"]["50"] = None
        for method in ("projected_blind_v2", "projected_material_v2"):
            a = cons.method_stepper(scene, method, self.models, self.frozen, self.protocol)
            b = cons.method_stepper(altered, method, models, self.frozen, self.protocol)
            self.assertEqual(a(rollouts.copy_state(scene["snapshots"][0]), 0.002),
                             b(rollouts.copy_state(scene["snapshots"][0]), 0.002))

    def test_mass_centred_errors_separate_uniform_translation_and_deformation(self):
        initial = {"positions": [[0, 0, 0], [2, 0, 0]], "velocities": [[0, 1, 0], [0, -1, 0]]}
        shifted = {k: [[x + 5 for x in vector] for vector in vectors] for k, vectors in initial.items()}
        masses = [1.0, 2.0]
        self.assertGreater(affine.errors(initial, shifted)["position_rmse"], 4)
        self.assertLess(max(affine.errors(cons.centred(initial, masses), cons.centred(shifted, masses)).values()), 1e-15)
        shifted["positions"][0][0] += 1
        self.assertGreater(affine.errors(cons.centred(initial, masses), cons.centred(shifted, masses))["position_rmse"], 0.1)

    def test_support_counts_use_strict_radius_and_exclude_self(self):
        state = {"positions": [[0, 0, 0], [0.5, 0, 0], [1.5, 0, 0]], "velocities": [[0, 0, 0]] * 3}
        self.assertEqual(cons.support_counts(state, 1), {"neighbour_pairs": 1, "isolated_particles": 1})

    def test_reserved_records_are_paired_and_reject_seed_or_state_reuse(self):
        previous = cons.prior_records()
        records = cons.make_scenes(self.manifest, previous)
        self.assertEqual(len(records), 4)
        self.assertEqual(len({r["initial_state_sha256"] for r in records}), 1)
        with self.assertRaisesRegex(ValueError, "seed overlaps"):
            cons.make_scenes(self.manifest, records)
        renamed = deepcopy(records)
        for record in renamed:
            record["seed"] = 9999
        with self.assertRaisesRegex(ValueError, "initial state overlaps"):
            cons.make_scenes(self.manifest, renamed)

    def test_actual_reader_rejects_tampering_duplicates_and_missing_records(self):
        records = cons.make_scenes(self.manifest, [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "initial.jsonl"
            cons.write_jsonl(path, records)
            self.assertEqual(cons.read_scenes(path, self.manifest, []), records)
            changed = deepcopy(records)
            changed[0]["snapshots"][0]["positions"][0][0] += 0.001
            for invalid in (changed, records[:-1], records + records[:1]):
                cons.write_jsonl(path, invalid)
                with self.assertRaises(ValueError):
                    cons.read_scenes(path, self.manifest, [])

    def test_modified_protocol_is_rejected(self):
        with patch.object(affine, "file_digest", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "protocol checksum"):
                cons.read_protocol()

    def test_complete_small_run_has_no_fitting_and_keeps_relative_trajectories(self):
        protocol = deepcopy(self.protocol)
        protocol.update(steps=3, checkpoints=[1, 3])
        scenes = cons.make_scenes(self.manifest, [])[:1]
        before = digest([self.models, self.frozen, scenes])
        with patch.object(materials, "fit_candidate", side_effect=AssertionError("refit")), \
                patch.object(materials, "select_models", side_effect=AssertionError("retune")), \
                patch.object(affine, "fit_candidate", side_effect=AssertionError("refit")):
            report, references = cons.run(scenes, self.models, self.frozen, protocol)
        self.assertEqual(digest([self.models, self.frozen, scenes]), before)
        for key in ("all_references_qualified", "all_trajectories_completed", "all_projected_ledgers_pass", "all_paired_centred_states_match"):
            self.assertTrue(report[key], key)
        self.assertEqual([r["step"] for r in references[0]["snapshots"]], [0, 1, 3])
        self.assertFalse(report["scenes"][0]["reference_checks"]["1"]["prefix_checked"])
        rows = report["scenes"][0]["methods"]["projected_material_v2"]["checkpoints"]
        failed = deepcopy(rows)
        failed[1].update(available=False, errors=None, centred_errors=None, diagnostics=None)
        self.assertIsNone(cons.aggregate(failed)["mean_centred_position_rmse"])
        unqualified = deepcopy(rows)
        unqualified[1]["reference_qualified"] = False
        self.assertIsNone(cons.aggregate(unqualified)["mean_centred_velocity_rmse"])


if __name__ == "__main__":
    unittest.main()

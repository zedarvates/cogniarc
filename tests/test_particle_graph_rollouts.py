"""Autoregressive feedback, analytic ledgers and failure-aware scoring controls."""

from copy import deepcopy
import math
import unittest
from unittest.mock import patch

from experiments.particle_graph import affine, materials, rollouts
from experiments.particle_graph.validation import digest


class RolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = rollouts.read_protocol()
        cls.scenes, cls.models, cls.frozen = rollouts.load_inputs(cls.protocol)

    def test_feedback_matches_constant_acceleration_and_keeps_snapshots(self):
        initial = {"positions": [[1.0, 2.0, 3.0]], "velocities": [[0.2, -0.1, 0.3]]}
        saved = deepcopy(initial)
        acceleration, received = [0.4, -0.2, 0.1], []
        def advance(state, dt):
            received.append(deepcopy(state))
            # Deliberate mutation exercises snapshot and initial-state isolation.
            for k in range(3):
                state["positions"][0][k] += dt * state["velocities"][0][k] + 0.5 * dt ** 2 * acceleration[k]
                state["velocities"][0][k] += dt * acceleration[k]
            return state
        result = rollouts.trace(initial, [2.0], 0.01, 500, [1, 10, 50, 100, 250, 500], advance, 1e6)
        self.assertTrue(result["completed"])
        self.assertEqual(len(received), 500)
        self.assertEqual(received[1], result["snapshots"][1])
        self.assertEqual(initial, saved)
        self.assertEqual(result["snapshots"][0], initial)
        for h, state in result["snapshots"].items():
            t = h * 0.01
            for k in range(3):
                self.assertAlmostEqual(state["positions"][0][k], initial["positions"][0][k]
                                       + t * initial["velocities"][0][k] + 0.5 * t * t * acceleration[k], places=10)
                self.assertAlmostEqual(state["velocities"][0][k], initial["velocities"][0][k] + t * acceleration[k], places=11)

    def test_stepper_uses_only_step_one_model_and_no_future_state(self):
        scene, altered = deepcopy(self.scenes[0]), deepcopy(self.scenes[0])
        models = deepcopy(self.models)
        for variant in models["variants"].values():
            variant["horizon_models"]["10"] = None
            variant["horizon_models"]["50"] = None
        for snapshot in altered["snapshots"][1:]:
            snapshot["positions"] = [[1e90, 1e90, 1e90]]
            snapshot["velocities"] = [[-1e90, -1e90, -1e90]]
        first = rollouts.trace(scene["snapshots"][0], scene["masses"], 0.002, 5, [1, 5],
                               rollouts.method_stepper(scene, "material_v2", models, self.frozen), 1e6)
        second = rollouts.trace(altered["snapshots"][0], altered["masses"], 0.002, 5, [1, 5],
                                rollouts.method_stepper(altered, "material_v2", models, self.frozen), 1e6)
        self.assertEqual(first, second)
        self.assertTrue(first["completed"])

    def test_unequal_mass_ledgers_match_external_gravity_and_detect_impulse(self):
        initial = {"positions": [[1.0, 0, -1.0], [-2.0, 1.0, 0]], "velocities": [[0.1, 0.2, 0.3], [-0.3, 0.1, -0.2]]}
        masses, gravity, t = [2.0, 0.5], [0.1, -0.3, 0.2], 0.6
        state = {"positions": [[p[k] + t * v[k] + 0.5 * t * t * gravity[k] for k in range(3)]
                               for p, v in zip(initial["positions"], initial["velocities"])],
                 "velocities": [[v[k] + t * gravity[k] for k in range(3)] for v in initial["velocities"]]}
        ledger = rollouts.diagnostics(state, initial, masses, gravity, t, state, self.protocol)
        self.assertLess(ledger["momentum_error"], 1e-14)
        self.assertLess(ledger["centre_of_mass_error"], 1e-14)
        self.assertTrue(ledger["momentum_pass"])
        self.assertEqual(ledger["kinetic_energy_difference"], 0)
        changed = deepcopy(state)
        changed["velocities"][0][2] += 0.2
        ledger = rollouts.diagnostics(changed, initial, masses, gravity, t, state, self.protocol)
        self.assertAlmostEqual(ledger["momentum_error"], 0.4, places=13)
        self.assertFalse(ledger["momentum_pass"])

    def test_first_invalid_step_stops_without_repair_or_later_checkpoints(self):
        initial = {"positions": [[1.0, 0, 0]], "velocities": [[0, 0, 0]]}
        for invalid in (4.0, math.nan, math.inf):
            calls = []
            def advance(state, dt):
                calls.append(1)
                return {"positions": [[2.0 if len(calls) == 1 else invalid, 0, 0]], "velocities": [[0, 0, 0]]}
            result = rollouts.trace(initial, [1], 0.1, 5, [1, 2, 5], advance, 3.0)
            self.assertFalse(result["completed"])
            self.assertEqual(result["last_step"], 1)
            self.assertEqual(result["failure"]["step"], 2)
            self.assertEqual(set(result["snapshots"]), {0, 1})
            self.assertEqual(len(calls), 2)

    def test_particle_loss_is_recorded_as_failure(self):
        initial = {"positions": [[0, 0, 0]], "velocities": [[0, 0, 0]]}
        result = rollouts.trace(initial, [1], 0.1, 2, [1, 2],
                                lambda _state, _dt: {"positions": [], "velocities": []}, 1e6)
        self.assertEqual(result["failure"]["step"], 1)
        self.assertIn("identity/count", result["failure"]["reason"])
        self.assertEqual(result["last_step"], 0)

    def test_reference_requires_resolution_and_exact_short_prefix(self):
        state = {"positions": [[0, 0, 0]], "velocities": [[0, 0, 0]]}
        scene = {"masses": [1], "gravity": [0, 0, 0]}
        self.assertTrue(rollouts.reference_check(state, state, state, scene, 0.1, state, self.protocol)["qualified"])
        coarse = deepcopy(state); coarse["positions"][0][0] = 1e-9
        # The ballistic-error-relative gate is stricter here than the absolute cap.
        check = rollouts.reference_check(coarse, state, state, scene, 0.1, None, self.protocol)
        self.assertFalse(check["qualified"])
        self.assertFalse(check["gates"]["position_resolution"])
        check = rollouts.reference_check(state, state, state, scene, 0.1, coarse, self.protocol)
        self.assertFalse(check["gates"]["short_prefix_replay"])
        self.assertFalse(rollouts.reference_check(None, state, state, scene, 0.1, None, self.protocol)["qualified"])

    def test_aggregates_do_not_hide_failed_or_unqualified_members(self):
        diagnostic = {"mass_pass": True, "momentum_pass": False, "momentum_error": 0.2, "centre_of_mass_error": 0.1}
        rows = [{"available": True, "reference_qualified": True,
                 "errors": {"position_rmse": p, "velocity_rmse": 2 * p}, "diagnostics": diagnostic} for p in (1.0, 3.0)]
        complete = rollouts.aggregate(rows)
        self.assertEqual(complete["mean_position_rmse"], 2.0)
        self.assertEqual(complete["momentum_pass"], 0)
        failed = deepcopy(rows)
        failed[1].update(available=False, errors=None, diagnostics=None)
        result = rollouts.aggregate(failed)
        self.assertIsNone(result["mean_position_rmse"])
        self.assertEqual(result["predictions_available"], 1)
        unqualified = deepcopy(rows)
        unqualified[1]["reference_qualified"] = False
        self.assertIsNone(rollouts.aggregate(unqualified)["mean_velocity_rmse"])
        self.assertEqual(rollouts.aggregate(unqualified)["predictions_available"], 2)

    def test_small_complete_run_preserves_prefix_and_never_fits(self):
        protocol = deepcopy(self.protocol)
        protocol.update(steps=2, checkpoints=[1, 2])
        before = digest([self.models, self.frozen])
        with patch.object(materials, "fit_candidate", side_effect=AssertionError("refit")), \
                patch.object(materials, "select_models", side_effect=AssertionError("retune")), \
                patch.object(affine, "fit_candidate", side_effect=AssertionError("refit v1")):
            report, references = rollouts.run(self.scenes[:1], self.models, self.frozen, protocol)
        self.assertEqual(digest([self.models, self.frozen]), before)
        self.assertTrue(report["all_references_qualified"])
        self.assertTrue(report["all_trajectories_completed"])
        self.assertTrue(report["scenes"][0]["reference_checks"]["1"]["gates"]["short_prefix_replay"])
        self.assertEqual([s["step"] for s in references[0]["snapshots"]], [0, 1, 2])
        self.assertEqual(set(report["scenes"][0]["methods"]), set(protocol["methods"]))

    def test_frozen_input_mismatch_is_rejected_before_evaluation(self):
        protocol = deepcopy(self.protocol)
        first = next(iter(protocol["frozen_inputs_sha256"]))
        protocol["frozen_inputs_sha256"][first] = "0" * 64
        with self.assertRaisesRegex(ValueError, "frozen rollout input checksum"):
            rollouts.load_inputs(protocol)


if __name__ == "__main__":
    unittest.main()

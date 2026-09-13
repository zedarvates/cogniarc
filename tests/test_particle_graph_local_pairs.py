"""Pair conservation, compact support, constrained fitting and reserved controls."""

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.particle_graph import affine, conservation, local_pairs as local, local_pair_evaluation as experiment
from experiments.particle_graph import material_data as data, rollouts
from experiments.particle_graph.reference import Config, Particle
from experiments.particle_graph.validation import digest


class LocalPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = local.read_protocol()
        _, cls.models, cls.frozen = rollouts.load_inputs(rollouts.read_protocol())
        manifest, p = data.read_config()
        report = data.read_report(rollouts.DATA_DIR, manifest, p)
        cls.train = data.read_split(rollouts.DATA_DIR, "train", manifest, report)[:1]
        cls.validation = data.read_split(rollouts.DATA_DIR, "validation", manifest, report)[:1]
        cls.manifest = json.loads(local.MANIFEST.read_text())
        cls.manifest["groups"] = [{"id": "unit-local", "seed": 9101, "shape": [2, 2, 1]}]
        for i, control in enumerate(cls.manifest["controls"]):
            control["seed"] = 9201 + i

    def test_opposite_forces_conserve_momentum_with_unequal_masses(self):
        state = (Particle((0, 0, 0), (0.1, 0.2, -0.1), 0.125),
                 Particle((0.3, 0.2, -0.1), (-0.2, 0.1, 0.3), 0.375),
                 Particle((-0.1, 0.3, 0.2), (0.4, -0.1, -0.2), 0.2))
        acceleration = local.accelerations(state, Config(), [3, 2, 5, 1])
        residual = [math.fsum(p.mass * a[k] for p, a in zip(state, acceleration)) for k in range(3)]
        self.assertLess(math.sqrt(math.fsum(x*x for x in residual)), 1e-15)
        self.assertGreater(max(abs(x) for a in acceleration for x in a), 0)

    def test_compact_support_and_graph_rebuild_after_separation(self):
        config = Config(radius=1)
        inside = (Particle((0, 0, 0)), Particle((0.9, 0, 0)))
        self.assertGreater(abs(local.accelerations(inside, config, [1, 1, 1, 1])[0][0]), 0)
        for distance in (1.0, 1.2):
            outside = (Particle((0, 0, 0), (1, 0, 0)), Particle((distance, 0, 0), (-1, 0, 0)))
            self.assertEqual(local.accelerations(outside, config, [1, 1, 1, 1]), ((0.0,)*3, (0.0,)*3))
        self.assertEqual(local.accelerations(inside[:1], config, [1, 1, 1, 1]), ((0.0,)*3,))

    def test_coincident_pressure_is_zero_and_viscosity_remains_finite(self):
        state = (Particle((0, 0, 0), (1, 0, 0), 0.5), Particle((0, 0, 0), (-1, 0, 0), 1.5))
        self.assertEqual(local.accelerations(state, Config(), [1, 1, 0, 0]), ((0.0,)*3, (0.0,)*3))
        force = local.accelerations(state, Config(), [0, 0, 1, 1])
        self.assertLess(force[0][0], 0)
        self.assertAlmostEqual(0.5*force[0][0]+1.5*force[1][0], 0, places=14)

    def test_viscosity_has_nonpositive_instantaneous_kinetic_power(self):
        state = (Particle((0, 0, 0), (1, 2, -1), 0.5), Particle((0.3, 0.1, 0.2), (-2, 0, 1), 1.5))
        force = local.accelerations(state, Config(), [0, 0, 2, 3])
        power = math.fsum(p.mass * math.fsum(v*a for v,a in zip(p.velocity, acceleration)) for p, acceleration in zip(state, force))
        self.assertLess(power, 0)

    def test_rotation_translation_boost_and_permutation_equivariance(self):
        state = (Particle((0.1, 0.2, 0.3), (0.3, -0.2, 0.1), 0.5), Particle((0.4, -0.1, 0.2), (-0.2, 0.1, 0.4), 1.5))
        coefficients = [1, 3, 2, 4]
        rotation = lambda v: (-v[1], v[0], v[2])
        changed = tuple(Particle(tuple(x+t for x,t in zip(rotation(p.position), (2,-3,1))),
                                 tuple(x+t for x,t in zip(rotation(p.velocity), (0.2,0.3,-0.1))), p.mass) for p in reversed(state))
        original = local.accelerations(state, Config(), coefficients)
        actual = local.accelerations(changed, Config(), coefficients)
        for a,b in zip(actual, reversed(original)):
            self.assertLess(math.dist(a, rotation(b)), 1e-14)

    def test_shared_midpoint_matches_500_step_free_flight(self):
        initial = (Particle((1,-2,0.5), (0.2,0.3,-0.1), 0.7),)
        state, gravity, dt = initial, (0.12,-0.24,0.09), 0.002
        for _ in range(500):
            state = local.midpoint(state, Config(), dt, gravity, lambda p,c: local.accelerations(p,c,[1,2,3,4]))
        expected = affine.predict(local.state_of(initial), gravity, 1.0, "known_gravity_ballistic")
        self.assertLess(max(affine.errors(local.state_of(state), expected).values()), 1e-12)
        self.assertEqual(state[0].mass, initial[0].mass)

    def test_invalid_coefficients_and_midpoint_acceleration_are_rejected(self):
        state = (Particle((0,0,0)),)
        for coefficients in ([1,2,3], [1,2,3,-1], [1,2,3,math.nan]):
            with self.assertRaises(ValueError):
                local.accelerations(state, Config(), coefficients)
        for bad in ((), ((math.inf,0,0),), ((0,0),)):
            with self.assertRaises(ValueError):
                local.midpoint(state, Config(), 0.002, (0,0,0), lambda p,c:bad)

    def test_nonnegative_ridge_matches_analytic_diagonal_solution(self):
        rows = [[float(i==j) for j in range(4)] for i in range(4)]
        fitted = local.nonnegative_ridge(rows, [2.0,-3.0,1.0,0.0], [0.25]*4, 0.5)
        for a,b in zip(fitted["coefficients"], [2/1.5,0,1/1.5,0]):
            self.assertAlmostEqual(a,b,places=13)
        self.assertLess(fitted["kkt_violation"], 1e-14)
        self.assertEqual(fitted["active"], [0,2])

    def test_force_labels_roundtrip_and_reject_tampering(self):
        scenes = self.train+self.validation
        rows = local.label_records(scenes, self.protocol)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"targets.jsonl"
            conservation.write_jsonl(path,rows)
            self.assertEqual(local.read_labels(path,scenes,self.protocol),rows)
            rows[0]["internal_accelerations"][0][0]+=0.01
            conservation.write_jsonl(path,rows)
            with self.assertRaisesRegex(ValueError,"force target"):
                local.read_labels(path,scenes,self.protocol)
        test=deepcopy(self.train);test[0]["split"]="test"
        with self.assertRaisesRegex(ValueError,"development"):
            local.label_records(test,self.protocol)

    def test_validation_cannot_change_candidate_coefficients_or_scaling(self):
        labels=local.label_records(self.train+self.validation,self.protocol)
        first=local.fit_model(self.train,self.validation,labels,self.protocol)
        changed=deepcopy(labels)
        for row in changed:
            if row["split"]=="validation":
                row["internal_accelerations"]=[[3*x for x in vector] for vector in row["internal_accelerations"]]
        second=local.fit_model(self.train,self.validation,changed,self.protocol)
        for a,b in zip(first["candidates"],second["candidates"]):
            for key in ("coefficients","feature_rms","train_acceleration_rmse"):
                self.assertEqual(a[key],b[key])
        self.assertFalse(first["test_used_for_selection"])
        self.assertFalse(first["refit_on_validation"])
        with self.assertRaises(ValueError):
            local.fit_model(self.validation,self.validation,labels,self.protocol)

    def test_model_reader_enforces_saved_coefficients_source_and_label_hashes(self):
        labels=local.label_records(self.train+self.validation,self.protocol)
        model=local.fit_model(self.train,self.validation,labels,self.protocol)
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory)
            artifact=conservation.write_jsonl(d/"force_targets.jsonl",labels)
            model["provenance"]={"source_sha256":local.source_hashes(),"protocol_sha256":local.PROTOCOL_SHA256,"force_targets":artifact}
            affine.write_json(d/"model.json",model)
            affine.write_json(d/"model_lock.json",{"model_sha256":affine.file_digest(d/"model.json")})
            self.assertEqual(local.read_model(d,self.protocol),model)
            modified=deepcopy(model);modified["coefficients"][0]+=1
            affine.write_json(d/"model.json",modified)
            with self.assertRaisesRegex(ValueError,"checksum"):
                local.read_model(d,self.protocol)
            affine.write_json(d/"model_lock.json",{"model_sha256":affine.file_digest(d/"model.json")})
            with self.assertRaisesRegex(ValueError,"validation selection"):
                local.read_model(d,self.protocol)
            affine.write_json(d/"model.json",model)
            affine.write_json(d/"model_lock.json",{"model_sha256":affine.file_digest(d/"model.json")})
            (d/"force_targets.jsonl").write_text('[]\n')
            with self.assertRaisesRegex(ValueError,"labels changed"):
                local.read_model(d,self.protocol)

    def test_new_mass_variants_share_geometry_and_total_mass(self):
        rows=experiment.make_scenes(self.manifest,experiment.previous_records())
        main=[r for r in rows if r["role"]=="main"]
        self.assertEqual(len(main),4)
        self.assertEqual(len({digest(r["snapshots"]) for r in main}),1)
        for row in main:
            self.assertAlmostEqual(math.fsum(row["masses"]),len(row["masses"])*self.manifest["defaults"]["mass"],places=14)
        with self.assertRaisesRegex(ValueError,"overlaps"):
            experiment.make_scenes(self.manifest,rows)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"initial.jsonl"
            conservation.write_jsonl(path,rows)
            self.assertEqual(experiment.read_scenes(path,self.manifest,[]),rows)
            bad=deepcopy(rows);bad[0]["masses"][0]*=2
            conservation.write_jsonl(path,bad)
            with self.assertRaisesRegex(ValueError,"recipe"):
                experiment.read_scenes(path,self.manifest,[])

    def test_frozen_input_hash_is_checked(self):
        with patch.object(affine,"file_digest",return_value="0"*64):
            with self.assertRaisesRegex(ValueError,"protocol checksum"):
                local.read_protocol()

    def test_small_end_to_end_keeps_controls_separate_and_never_refits(self):
        protocol=deepcopy(self.protocol);protocol.update(steps=3,checkpoints=[1,3])
        scenes=experiment.make_scenes(self.manifest,[])
        scenes=[next(s for s in scenes if s["role"]=="main")]+[s for s in scenes if s["role"]=="control"]
        pair_model={"coefficients":[10,0.1,20,0.1],"development_group_ids":[]}
        before=digest([scenes,self.models,self.frozen,pair_model])
        with patch.object(local,"fit_model",side_effect=AssertionError("refit")),patch.object(local,"nonnegative_ridge",side_effect=AssertionError("retune")):
            report,reference=experiment.run(scenes,self.models,self.frozen,pair_model,protocol)
        self.assertEqual(digest([scenes,self.models,self.frozen,pair_model]),before)
        self.assertTrue(report["all_trajectories_completed"])
        self.assertTrue(report["all_references_qualified"])
        self.assertTrue(report["all_local_ledgers_pass"])
        self.assertTrue(report["all_local_analytic_controls_pass"])
        self.assertEqual(report["main_mean_scene_errors"]["3"]["pair_local"]["scenes"],1)
        self.assertEqual(len(report["controls"]),3)
        for key in ("control/isolated","control/separated"):
            for row in report["controls"][key]["pair_local"]["checkpoints"]:
                self.assertEqual(row["force_diagnostics"]["internal_acceleration_rms"],0)
        self.assertEqual([r["step"] for r in reference[0]["snapshots"]],[0,1,3])


if __name__=="__main__":
    unittest.main()

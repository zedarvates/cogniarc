"""Analytic flight, collision-budget and conservation controls for box contacts."""

from dataclasses import replace
import math
import random
import unittest

from experiments.particle_graph.boundaries import Box, drift_in_box, step_in_box, validate_box
from experiments.particle_graph.reference import Config, Particle, step


def kinetic(state):
    return sum(0.5*p.mass*sum(v*v for v in p.velocity) for p in state)


def elastic_fold(position, velocity, dt, lower, upper):
    """Closed-form mirror extension, independent of hit-time event processing."""
    width = upper-lower
    phase = (position-lower+velocity*dt) % (2*width)
    result = lower+phase if phase <= width else lower+2*width-phase
    if phase == 0:
        speed = abs(velocity)
    elif phase == width:
        speed = -abs(velocity)
    else:
        speed = velocity if phase < width else -velocity
    return result, speed


class BoundaryTests(unittest.TestCase):
    def assertVectorClose(self, actual, expected, places=11):
        for x, y in zip(actual, expected):
            self.assertAlmostEqual(x, y, places=places)

    def test_no_contact_matches_original_unbounded_step(self):
        state = (Particle((0, 0, 0), (0.1, 0.2, 0.3), 0.25),
                 Particle((0.3, 0.2, 0.1), (-0.2, 0.1, 0), 0.5))
        config, gravity = Config(), (0, -0.3, 0)
        result = step_in_box(state, config, Box((-5, -5, -5), (5, 5, 5)), 0.01, gravity)
        self.assertEqual(result.particles, step(state, config, 0.01, gravity))
        self.assertEqual(result.impacts, 0)
        self.assertEqual(result.wall_impulses, ((0.0, 0.0, 0.0),)*2)
        self.assertEqual(result.dissipated_energy, 0)

    def test_six_faces_match_single_inelastic_impact(self):
        box = Box((-1, -1, -1), (1, 1, 1), restitution=0.5)
        for axis in range(3):
            for sign in (-1, 1):
                with self.subTest(axis=axis, sign=sign):
                    velocity = [0, 0, 0]; velocity[axis] = sign*3
                    state = (Particle((0, 0, 0), tuple(velocity), 2),)
                    result = drift_in_box(state, box, 0.5)
                    self.assertAlmostEqual(result.particles[0].position[axis], sign*0.75)
                    self.assertAlmostEqual(result.particles[0].velocity[axis], -sign*1.5)
                    self.assertAlmostEqual(result.wall_impulses[0][axis], -sign*9)
                    self.assertAlmostEqual(result.dissipated_energy, 6.75)
                    self.assertEqual(result.impacts, 1)

    def test_tangential_motion_is_unchanged_by_frictionless_contact(self):
        state = (Particle((0, 0, 0), (3, 0.2, -0.1)),)
        result = drift_in_box(state, Box((-1, -1, -1), (1, 1, 1), restitution=0.5), 0.5)
        self.assertVectorClose(result.particles[0].position, (0.75, 0.1, -0.05))
        self.assertVectorClose(result.particles[0].velocity, (-1.5, 0.2, -0.1))

    def test_exact_endpoint_is_post_impact(self):
        for sign in (-1, 1):
            state = (Particle((0, 0, 0), (sign, 0, 0)),)
            result = drift_in_box(state, Box((-1, -1, -1), (1, 1, 1)), 1)
            self.assertEqual(result.impacts, 1)
            self.assertEqual(result.particles[0].position[0], sign)
            self.assertEqual(result.particles[0].velocity[0], -sign)

    def test_initial_wall_outward_inward_and_zero_restitution(self):
        box = Box((0, 0, 0), (1, 1, 1))
        outward = (Particle((0, 0.5, 0.5), (-2, 0, 0)),)
        result = drift_in_box(outward, box, 0.1)
        self.assertAlmostEqual(result.particles[0].position[0], 0.2)
        self.assertEqual(result.particles[0].velocity[0], 2)
        inward = (replace(outward[0], velocity=(2, 0, 0)),)
        self.assertEqual(drift_in_box(inward, box, 0.1).impacts, 0)
        stuck = drift_in_box(outward, replace(box, restitution=0), 0.1)
        self.assertEqual(stuck.particles[0].position[0], 0)
        self.assertEqual(stuck.particles[0].velocity[0], 0)
        self.assertEqual(stuck.impacts, 1)
        self.assertEqual(stuck.dissipated_energy, 2)

    def test_simultaneous_edge_and_corner_contacts(self):
        for axes in (2, 3):
            velocity = tuple(2 if k < axes else 0 for k in range(3))
            result = drift_in_box((Particle((0, 0, 0), velocity),),
                                  Box((-1, -1, -1), (1, 1, 1), restitution=0.5), 0.75)
            self.assertEqual(result.impacts, axes)
            self.assertVectorClose(result.particles[0].position,
                                   tuple(0.75 if k < axes else 0 for k in range(3)))
            self.assertVectorClose(result.particles[0].velocity,
                                   tuple(-1 if k < axes else 0 for k in range(3)))
            self.assertAlmostEqual(result.dissipated_energy, axes*1.5)

    def test_many_elastic_hits_match_independent_mirror_solution(self):
        rng = random.Random(7003)
        box = Box((-1, -2, -3), (1, 2, 3), radius=0.125)
        for _ in range(40):
            position = tuple(rng.uniform(lo+box.radius, hi-box.radius) for lo, hi in zip(box.lower, box.upper))
            velocity = tuple(rng.uniform(-50, 50) for _ in range(3))
            result = drift_in_box((Particle(position, velocity),), box, 1.25)
            for axis in range(3):
                expected_x, expected_v = elastic_fold(position[axis], velocity[axis], 1.25,
                                                       box.lower[axis]+box.radius, box.upper[axis]-box.radius)
                self.assertAlmostEqual(result.particles[0].position[axis], expected_x, places=10)
                self.assertEqual(result.particles[0].velocity[axis], expected_v)
            self.assertEqual(result.dissipated_energy, 0)

    def test_two_inelastic_hits_match_analytic_flight_times(self):
        result = drift_in_box((Particle((0.25, 0.5, 0.5), (3, 0, 0)),),
                              Box((0, 0, 0), (1, 1, 1), restitution=0.5), 2)
        # First contact at 1/4; second at 1/4 + 2/3; remaining drift at +3/4.
        self.assertEqual(result.impacts, 2)
        self.assertAlmostEqual(result.particles[0].position[0], 0.8125)
        self.assertAlmostEqual(result.particles[0].velocity[0], 0.75)

    def test_radius_restricts_centres_and_input_is_immutable(self):
        state = (Particle((0, 0, 0), (2, 0, 0)),)
        before = state
        result = drift_in_box(state, Box((-1, -1, -1), (1, 1, 1), radius=0.125), 0.5)
        self.assertEqual(state, before)
        self.assertEqual(result.particles[0].position[0], 0.75)
        self.assertEqual(result.particles[0].velocity[0], -2)

    def test_impulse_and_energy_ledgers_for_unequal_masses(self):
        state = (Particle((0, 0, 0), (7, -3, 1), 0.25),
                 Particle((0.2, 0.3, -0.1), (-4, 2, 8), 1.75))
        result = drift_in_box(state, Box((-1, -1, -1), (1, 1, 1), restitution=0.7), 2)
        self.assertAlmostEqual(kinetic(state)-kinetic(result.particles), result.dissipated_energy, places=11)
        self.assertGreater(result.dissipated_energy, 0)
        for axis in range(3):
            change = sum(q.mass*q.velocity[axis]-p.mass*p.velocity[axis] for p, q in zip(state, result.particles))
            self.assertAlmostEqual(change, sum(j[axis] for j in result.wall_impulses), places=11)

    def test_invalid_geometry_state_and_impact_budget_fail_without_output(self):
        invalid_boxes = (Box(radius=-1), Box(radius=0.5), Box(restitution=1.1),
                         Box(restitution=math.nan), Box(lower=(0, 0)), Box(upper=(math.inf, 1, 1)),
                         Box(max_impacts_per_axis=True), Box(max_impacts_per_axis=0))
        for box in invalid_boxes:
            with self.subTest(box=box), self.assertRaises(ValueError):
                validate_box(box)
        for p in (Particle((1, 0, 0)), Particle((0, 0, 0), (math.nan, 0, 0)), Particle((0, 0, 0), mass=0)):
            with self.subTest(p=p), self.assertRaises(ValueError):
                drift_in_box((p,), Box(), 1)
        state = (Particle((0, 0, 0), (100, 0, 0)),)
        with self.assertRaisesRegex(ValueError, "impact budget"):
            drift_in_box(state, Box(max_impacts_per_axis=1), 1)
        self.assertEqual(state[0].position, (0, 0, 0))
        for dt in (0, -1, math.inf, math.nan, True):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                drift_in_box((), Box(), dt)

    def test_empty_box_step_is_well_defined(self):
        result = step_in_box((), Config(), Box(), 0.1)
        self.assertEqual(result.particles, ())
        self.assertEqual(result.wall_impulses, ())
        self.assertEqual(result.impacts, 0)


if __name__ == "__main__":
    unittest.main()

"""Independent dense oracle and structural controls for the offline experiment."""

from dataclasses import replace
import math
import random
import unittest

from experiments.particle_graph.reference import Config, Particle, evaluate, neighbor_pairs, step


def dense_oracle(particles, config):
    """O(N^2), directed sums, without the spatial hash or pair-force helper."""
    h = config.radius
    n = len(particles)
    radii = [[math.dist(a.position, b.position) for b in particles] for a in particles]
    densities = [sum(b.mass * 315 / (64 * math.pi * h**9)
                     * max(0.0, h*h - radii[i][j]**2)**3
                     for j, b in enumerate(particles)) for i in range(n)]
    pressures = [max(0.0, config.stiffness * (rho - config.rest_density)) for rho in densities]
    acceleration = []
    for i, a in enumerate(particles):
        row = [0.0, 0.0, 0.0]
        for j, b in enumerate(particles):
            r = radii[i][j]
            if i == j or r >= h:
                continue
            for axis in range(3):
                gradient = (-45 / (math.pi * h**6) * (h-r)**2
                            * (a.position[axis] - b.position[axis]) / r) if r else 0.0
                row[axis] += -b.mass * (pressures[i] + pressures[j]) / (
                    2 * densities[i] * densities[j]) * gradient
                row[axis] += config.viscosity * b.mass / (densities[i] * densities[j]) * (
                    b.velocity[axis] - a.velocity[axis]) * 45 / (math.pi * h**6) * (h-r)
        acceleration.append(tuple(row))
    return densities, acceleration


class ParticleGraphTests(unittest.TestCase):
    def setUp(self):
        rng = random.Random(723)
        self.particles = tuple(Particle(
            tuple(rng.uniform(-1, 1) for _ in range(3)),
            tuple(rng.uniform(-0.2, 0.2) for _ in range(3)),
            rng.uniform(0.5, 1.5)) for _ in range(24))
        self.config = Config(radius=0.9, rest_density=1.0, stiffness=0.3, viscosity=0.1)

    def assertVectorClose(self, a, b, places=10):
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=places)

    def test_graph_matches_dense_search_with_negative_cells(self):
        expected = tuple((i, j) for i in range(len(self.particles))
                         for j in range(i+1, len(self.particles)) if math.dist(
                             self.particles[i].position, self.particles[j].position) < self.config.radius)
        self.assertEqual(neighbor_pairs(self.particles, self.config.radius), expected)

    def test_cutoff_self_density_and_coincident_particles(self):
        p = tuple(Particle((x, 0.0, 0.0)) for x in (0.0, 0.0, 1.0, -1.0))
        self.assertEqual(neighbor_pairs(p, 1.0), ((0, 1),))
        result = evaluate(p)
        self.assertAlmostEqual(result.densities[0], 2 * 315 / (64 * math.pi))
        self.assertEqual(result.accelerations, ((0.0, 0.0, 0.0),) * 4)
        inside = (Particle((0, 0, 0)), Particle((1-1e-9, 0, 0)))
        self.assertEqual(neighbor_pairs(inside, 1.0), ((0, 1),))

    def test_dense_operator_parity_with_unequal_masses(self):
        result = evaluate(self.particles, self.config)
        density, acceleration = dense_oracle(self.particles, self.config)
        self.assertVectorClose(result.densities, density)
        for a, b in zip(result.accelerations, acceleration):
            self.assertVectorClose(a, b)

    def test_total_internal_force_is_zero(self):
        result = evaluate(self.particles, self.config)
        total = tuple(sum(p.mass * a[k] for p, a in zip(self.particles, result.accelerations))
                      for k in range(3))
        self.assertVectorClose(total, (0, 0, 0))

    def test_viscosity_dissipates_instantaneous_kinetic_energy(self):
        result = evaluate(self.particles, replace(self.config, stiffness=0))
        power = sum(p.mass * sum(v*a for v, a in zip(p.velocity, acceleration))
                    for p, acceleration in zip(self.particles, result.accelerations))
        self.assertLess(power, 0)

    def test_translation_and_rotation_equivariance(self):
        def rotate(v):
            return (-v[1], v[0], v[2])
        transformed = tuple(Particle(tuple(x+t for x, t in zip(rotate(p.position), (2.0, -3.0, 0.5))),
                                     rotate(p.velocity), p.mass) for p in self.particles)
        original = evaluate(self.particles, self.config)
        result = evaluate(transformed, self.config)
        self.assertEqual(original.pairs, result.pairs)
        self.assertVectorClose(original.densities, result.densities)
        for a, b in zip(original.accelerations, result.accelerations):
            self.assertVectorClose(rotate(a), b)

    def test_permutation_equivariance(self):
        order = list(reversed(range(len(self.particles))))
        original = evaluate(self.particles, self.config)
        result = evaluate(tuple(self.particles[i] for i in order), self.config)
        for new, old in enumerate(order):
            self.assertAlmostEqual(result.densities[new], original.densities[old])
            self.assertVectorClose(result.accelerations[new], original.accelerations[old])

    def test_replay_and_closed_rollout_momentum(self):
        def rollout():
            state = self.particles
            for _ in range(20):
                state = step(state, self.config, 0.001)
            return state
        first = rollout()
        self.assertEqual(first, rollout())
        self.assertEqual(tuple(p.mass for p in first), tuple(p.mass for p in self.particles))
        for axis in range(3):
            self.assertAlmostEqual(sum(p.mass*p.velocity[axis] for p in first),
                                   sum(p.mass*p.velocity[axis] for p in self.particles), places=10)

    def test_graph_rebuilt_after_motion(self):
        state = (Particle((0, 0, 0), (1, 0, 0)), Particle((1.1, 0, 0), (-1, 0, 0)))
        config = Config(stiffness=0, viscosity=0)
        self.assertEqual(evaluate(state, config).pairs, ())
        self.assertEqual(evaluate(step(state, config, 0.1), config).pairs, ((0, 1),))

    def test_external_gravity_and_empty_state(self):
        result = step((Particle((0, 0, 0), mass=2),), Config(), 0.1, (0, -10, 0))
        self.assertVectorClose(result[0].velocity, (0, -1, 0))
        self.assertVectorClose(result[0].position, (0, -0.1, 0))
        self.assertEqual(evaluate(()).densities, ())
        self.assertEqual(step((), Config(), 0.1), ())

    def test_invalid_inputs_fail(self):
        for config in (Config(radius=0), Config(radius=-1), Config(rest_density=0),
                       Config(stiffness=-1), Config(viscosity=math.nan)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                evaluate(self.particles, config)
        for particle in (Particle((math.inf, 0, 0)), Particle((0, 0)),
                         Particle((0, 0, 0), (0, math.nan, 0)), Particle((0, 0, 0), mass=0)):
            with self.subTest(particle=particle), self.assertRaises(ValueError):
                evaluate((particle,))
        for dt in (0, -1, math.inf, math.nan):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                step(self.particles, self.config, dt)


if __name__ == "__main__":
    unittest.main()

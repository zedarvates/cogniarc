"""Independent directed SPH sums and RK4 for small numerical controls only.

This shares data types with the candidate, not its neighbour graph, kernels,
force accumulation or integrator. It solves the same simplified equations;
agreement is numerical evidence, not independent validation of real water.
"""

import math

from .reference import Config, Particle, Vec3


def accelerations(state: tuple[Particle, ...], config: Config,
                  gravity: Vec3 = (0.0, 0.0, 0.0)) -> tuple[Vec3, ...]:
    """Dense directed sums, including self density and zero-floor pressure."""
    h = config.radius
    radii = [[math.dist(a.position, b.position) for b in state] for a in state]
    density = [sum(b.mass * 315 / (64 * math.pi * h**9)
                   * max(0.0, h*h - radii[i][j]**2)**3
                   for j, b in enumerate(state)) for i in range(len(state))]
    pressure = [max(0.0, config.stiffness * (rho - config.rest_density)) for rho in density]
    result = []
    for i, a in enumerate(state):
        value = list(gravity)
        for j, b in enumerate(state):
            r = radii[i][j]
            if i == j or r >= h:
                continue
            pressure_scale = (b.mass * (pressure[i] + pressure[j])
                              / (2 * density[i] * density[j])
                              * 45 / (math.pi * h**6) * (h-r)**2 / r) if r else 0.0
            viscosity_scale = (config.viscosity * b.mass / (density[i] * density[j])
                               * 45 / (math.pi * h**6) * (h-r))
            for axis in range(3):
                value[axis] += pressure_scale * (a.position[axis] - b.position[axis])
                value[axis] += viscosity_scale * (b.velocity[axis] - a.velocity[axis])
        result.append(tuple(value))
    return tuple(result)


def rk4_step(state: tuple[Particle, ...], config: Config, dt: float,
             gravity: Vec3 = (0.0, 0.0, 0.0)) -> tuple[Particle, ...]:
    """Classical RK4; the caller supplies validated, bounded scene parameters."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")

    def derivative(current):
        return tuple(zip((p.velocity for p in current), accelerations(current, config, gravity)))

    def advance(origin, changes, factor):
        return tuple(Particle(
            tuple(x + factor*dx for x, dx in zip(p.position, position_rate)),
            tuple(v + factor*dv for v, dv in zip(p.velocity, velocity_rate)), p.mass)
            for p, (position_rate, velocity_rate) in zip(origin, changes))

    k1 = derivative(state)
    k2 = derivative(advance(state, k1, dt/2))
    k3 = derivative(advance(state, k2, dt/2))
    k4 = derivative(advance(state, k3, dt))
    weighted = tuple((
        tuple((a + 2*b + 2*c + d)/6 for a, b, c, d in zip(q1[0], q2[0], q3[0], q4[0])),
        tuple((a + 2*b + 2*c + d)/6 for a, b, c, d in zip(q1[1], q2[1], q3[1], q4[1])))
        for q1, q2, q3, q4 in zip(k1, k2, k3, k4))
    result = advance(state, weighted, dt)
    if not all(math.isfinite(x) for p in result for x in p.position + p.velocity):
        raise ValueError("non-finite dense reference state")
    return result

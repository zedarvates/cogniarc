"""Small 3-D SPH-inspired reference operators, using only the standard library.

Kernel equations: Mueller, Charypar & Gross, SCA 2003, sections 3.1-3.5.
This experiment omits boundaries, surface tension and incompressibility. It is
not a calibrated water simulator, a learned graph model, or a runtime adapter.
"""

from dataclasses import dataclass
from itertools import product
from math import floor, isfinite, pi, sqrt

Vec3 = tuple[float, float, float]
Pair = tuple[int, int]


@dataclass(frozen=True)
class Particle:
    position: Vec3
    velocity: Vec3 = (0.0, 0.0, 0.0)
    mass: float = 1.0


@dataclass(frozen=True)
class Config:
    radius: float = 1.0
    rest_density: float = 1.0
    stiffness: float = 0.1
    viscosity: float = 0.05


@dataclass(frozen=True)
class Operators:
    pairs: tuple[Pair, ...]
    densities: tuple[float, ...]
    pressures: tuple[float, ...]
    accelerations: tuple[Vec3, ...]


def _vector(value: Vec3) -> bool:
    return len(value) == 3 and all(isfinite(x) for x in value)


def _validate(particles: tuple[Particle, ...], config: Config) -> None:
    for name in ("radius", "rest_density", "stiffness", "viscosity"):
        value = getattr(config, name)
        if not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if config.radius == 0 or config.rest_density == 0:
        raise ValueError("radius and rest_density must be positive")
    for particle in particles:
        if not _vector(particle.position) or not _vector(particle.velocity):
            raise ValueError("positions and velocities must be finite 3-D vectors")
        if not isfinite(particle.mass) or particle.mass <= 0:
            raise ValueError("mass must be finite and positive")


def _difference(a: Vec3, b: Vec3) -> Vec3:
    return tuple(x - y for x, y in zip(a, b))


def _distance_squared(a: Vec3, b: Vec3) -> float:
    return sum(x * x for x in _difference(a, b))


def neighbor_pairs(particles: tuple[Particle, ...], radius: float) -> tuple[Pair, ...]:
    """Return each pair once, in index order, with distance strictly below radius.

    A cell has width radius; the 27 surrounding cells contain all candidates.
    Self density is handled separately. Coincident distinct particles are pairs.
    """
    _validate(particles, Config(radius=radius))
    cells: dict[tuple[int, int, int], list[int]] = {}
    for i, particle in enumerate(particles):
        cell = tuple(floor(x / radius) for x in particle.position)
        cells.setdefault(cell, []).append(i)
    pairs = []
    for cell, indices in cells.items():
        for offset in product((-1, 0, 1), repeat=3):
            adjacent = tuple(x + dx for x, dx in zip(cell, offset))
            for i in indices:
                for j in cells.get(adjacent, ()):
                    if i < j and _distance_squared(
                        particles[i].position, particles[j].position
                    ) < radius * radius:
                        pairs.append((i, j))
    return tuple(sorted(pairs))


def evaluate(particles: tuple[Particle, ...], config: Config = Config()) -> Operators:
    """Compute density and symmetric pressure/viscosity forces on a fresh graph.

    Pressure uses max(0, k * (rho - rho0)): the zero floor is an explicit
    non-tensile simplification. Forces are accumulated once per unordered pair;
    accelerations divide those forces by each particle's own mass.
    """
    _validate(particles, config)
    h = config.radius
    pairs = neighbor_pairs(particles, h)
    poly6 = 315.0 / (64.0 * pi * h**9)
    spiky = 45.0 / (pi * h**6)
    densities = [particle.mass * poly6 * h**6 for particle in particles]
    for i, j in pairs:
        distance2 = _distance_squared(particles[i].position, particles[j].position)
        weight = poly6 * (h * h - distance2)**3
        densities[i] += particles[j].mass * weight
        densities[j] += particles[i].mass * weight
    pressures = [max(0.0, config.stiffness * (rho - config.rest_density))
                 for rho in densities]
    forces = [[0.0, 0.0, 0.0] for _ in particles]
    for i, j in pairs:
        a, b = particles[i], particles[j]
        delta = _difference(a.position, b.position)
        distance = sqrt(sum(x * x for x in delta))
        # The radial direction is undefined at zero separation: use zero there.
        gradient = (-spiky * (h - distance)**2 / distance) if distance else 0.0
        common = a.mass * b.mass / (densities[i] * densities[j])
        pressure_factor = -common * (pressures[i] + pressures[j]) * 0.5 * gradient
        viscosity_factor = config.viscosity * common * spiky * (h - distance)
        for axis in range(3):
            force = (pressure_factor * delta[axis]
                     + viscosity_factor * (b.velocity[axis] - a.velocity[axis]))
            forces[i][axis] += force
            forces[j][axis] -= force
    accelerations = tuple(tuple(f / p.mass for f in force)
                          for p, force in zip(particles, forces))
    if not all(isfinite(x) for x in densities + pressures):
        raise ValueError("non-finite density or pressure: rescale the fixture")
    if not all(_vector(a) for a in accelerations):
        raise ValueError("non-finite acceleration: rescale the fixture")
    return Operators(pairs, tuple(densities), tuple(pressures), accelerations)


def step(
    particles: tuple[Particle, ...],
    config: Config,
    dt: float,
    gravity: Vec3 = (0.0, 0.0, 0.0),
) -> tuple[Particle, ...]:
    """One semi-implicit Euler step; no automatic stable-time-step guarantee."""
    if not isfinite(dt) or dt <= 0 or not _vector(gravity):
        raise ValueError("dt must be finite and positive; gravity must be finite")
    operators = evaluate(particles, config)
    result = []
    for particle, acceleration in zip(particles, operators.accelerations):
        velocity = tuple(v + dt * (a + g) for v, a, g in
                         zip(particle.velocity, acceleration, gravity))
        position = tuple(x + dt * v for x, v in zip(particle.position, velocity))
        result.append(Particle(position, velocity, particle.mass))
    result = tuple(result)
    _validate(result, config)
    return result

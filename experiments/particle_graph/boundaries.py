"""Frictionless box contacts for the offline particle-graph experiment.

Forces kick velocities once, then straight-line drift resolves every crossed
wall. This is exact for free flight between fixed planar walls, not for curved
trajectories under continuous acceleration. No SPH wall-density correction,
no-slip condition, solid-body coupling or calibrated fluid boundary is supplied.
"""

from dataclasses import dataclass
import math

from .reference import Config, Particle, Vec3, evaluate


@dataclass(frozen=True)
class Box:
    lower: Vec3 = (-0.5, -0.5, -0.5)
    upper: Vec3 = (0.5, 0.5, 0.5)
    radius: float = 0.0
    restitution: float = 1.0
    max_impacts_per_axis: int = 10000


@dataclass(frozen=True)
class BoundaryResult:
    particles: tuple[Particle, ...]
    wall_impulses: tuple[Vec3, ...]
    dissipated_energy: float
    impacts: int


def _finite(value) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _vector(value) -> bool:
    return isinstance(value, (tuple, list)) and len(value) == 3 and all(_finite(x) for x in value)


def validate_box(box: Box) -> None:
    if not _vector(box.lower) or not _vector(box.upper):
        raise ValueError("box limits must be finite 3-D vectors")
    if not _finite(box.radius) or box.radius < 0:
        raise ValueError("contact radius must be finite and nonnegative")
    if not _finite(box.restitution) or not 0 <= box.restitution <= 1:
        raise ValueError("restitution must be in [0, 1]")
    if type(box.max_impacts_per_axis) is not int or not 1 <= box.max_impacts_per_axis <= 10000:
        raise ValueError("impact limit must be an integer in [1, 10000]")
    for lo, hi in zip(box.lower, box.upper):
        width = hi - lo - 2*box.radius
        if not math.isfinite(width) or width <= 0:
            raise ValueError("each box axis must have positive clearance after contact radius")


def _validate(state: tuple[Particle, ...], box: Box, dt: float) -> None:
    validate_box(box)
    if not _finite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    for p in state:
        if not _vector(p.position) or not _vector(p.velocity) or not _finite(p.mass) or p.mass <= 0:
            raise ValueError("particles require finite 3-D state and positive mass")
        if any(not lo+box.radius <= x <= hi-box.radius for x, lo, hi in
               zip(p.position, box.lower, box.upper)):
            raise ValueError("initial particle centre is outside the accessible box")


def _drift_axis(x, velocity, lower, upper, remaining, restitution, limit):
    impacts, speed_squared_loss = 0, 0.0
    while remaining > 0 and velocity != 0:
        wall = upper if velocity > 0 else lower
        time_to_wall = (wall-x)/velocity
        if time_to_wall > remaining:
            # The hit-time comparison proves this segment stays inside. Clamp
            # only its floating-point endpoint to the already bounded interval.
            x = min(upper, max(lower, x+velocity*remaining))
            break
        if impacts == limit:
            raise ValueError("impact budget exceeded; reduce dt or velocity")
        x = wall
        remaining = max(0.0, remaining-time_to_wall)
        outgoing = -restitution*velocity
        speed_squared_loss += velocity*velocity-outgoing*outgoing
        velocity = outgoing
        impacts += 1
    return x, velocity, speed_squared_loss, impacts


def drift_in_box(state: tuple[Particle, ...], box: Box, dt: float) -> BoundaryResult:
    """Resolve straight flights, including repeated and simultaneous axis hits.

    Impulses are those applied by walls to particles, in particle order. Loss
    covers collision dissipation only. Invalid input or exhausted impact budget
    raises before returning a result; the immutable input is never modified.
    """
    _validate(state, box, dt)
    result, impulses = [], []
    loss, hits = 0.0, 0
    for particle in state:
        positions, velocities = [], []
        for axis in range(3):
            position, velocity, speed_loss, count = _drift_axis(
                particle.position[axis], particle.velocity[axis],
                box.lower[axis]+box.radius, box.upper[axis]-box.radius,
                dt, box.restitution, box.max_impacts_per_axis)
            positions.append(position); velocities.append(velocity)
            loss += 0.5*particle.mass*speed_loss
            hits += count
        result.append(Particle(tuple(positions), tuple(velocities), particle.mass))
        impulses.append(tuple(particle.mass*(v-old) for v, old in zip(velocities, particle.velocity)))
    if not math.isfinite(loss) or not all(_vector(j) for j in impulses):
        raise ValueError("non-finite collision ledger; rescale the fixture")
    output = tuple(result)
    _validate(output, box, dt)
    return BoundaryResult(output, tuple(impulses), loss, hits)


def step_in_box(state: tuple[Particle, ...], config: Config, box: Box, dt: float,
                gravity: Vec3 = (0.0, 0.0, 0.0)) -> BoundaryResult:
    """One SPH force kick followed by bounded drift; first-order splitting."""
    _validate(state, box, dt)
    if not _vector(gravity):
        raise ValueError("gravity must be a finite 3-D vector")
    acceleration = evaluate(state, config).accelerations
    kicked = tuple(Particle(p.position, tuple(v+dt*(a+g) for v, a, g in
                                             zip(p.velocity, force, gravity)), p.mass)
                   for p, force in zip(state, acceleration))
    return drift_in_box(kicked, box, dt)

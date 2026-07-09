"""
Organic jitter — correlated noise for handwriting.

Key idea: a hand tremor is smooth, not white noise.
offset(t) = sum of 2-3 sine harmonics with random phase.
This creates fluid, natural-looking variation.

Also: per-point sigma, slant variance, baseline variance,
motor age (global sigma from 0.15 child → 0.01 adult).
"""

import math
import random
import numpy as np
from typing import List, Tuple, Optional

# ─── Motor age sigma mapping ─────────────────────────────────────

# Child: shaky, high variance. Adult: steady, low variance.
# σ_min ≈ 0.008 — never perfectly geometric.
MOTOR_AGES = {
    "toddler": 0.15,      # Gribouillage
    "child": 0.08,        # Lettres tremblées
    "teen": 0.04,         # Écriture qui se forme
    "adult": 0.015,       # Écriture assurée
    "master": 0.008,      # σ_min — jamais parfait
}

SIGMA_MIN = 0.008
SIGMA_MAX = 0.20


def motor_age_to_sigma(age: str) -> float:
    """Map a named motor age to global sigma."""
    return MOTOR_AGES.get(age, 0.04)


def sigma_to_motor_age(sigma: float) -> str:
    """Return the closest named motor age for a sigma value."""
    closest = min(MOTOR_AGES, key=lambda k: abs(MOTOR_AGES[k] - sigma))
    return closest


# ─── Correlated noise ────────────────────────────────────────────

class OrganicJitter:
    """Correlated noise generator for organic handwriting.

    Produces smooth offsets using harmonic synthesis instead of
    independent per-point noise. Each call to jitter_stroke() returns
    a stroke with the same global parameters but different seed-phase.

    Parameters
    ----------
    sigma_global : float
        Base amplitude. 0.15 = toddler, 0.008 = master.
    n_harmonics : int
        Number of sine harmonics (2-3 recommended).
    slant_sigma : float
        Variance of slant angle in radians (default 0.02 ≈ 1°).
    scale_sigma : float
        Variance of overall scale (default 0.01 ≈ 1%).
    baseline_sigma : float
        Vertical drift variance (default 0.005).
    rng : Optional[random.Random]
        Deterministic RNG for reproducibility.
    """

    def __init__(
        self,
        sigma_global: float = 0.04,
        n_harmonics: int = 3,
        slant_sigma: float = 0.02,
        scale_sigma: float = 0.01,
        baseline_sigma: float = 0.005,
        rng: Optional[random.Random] = None,
    ):
        self.sigma_global = max(SIGMA_MIN, min(SIGMA_MAX, sigma_global))
        self.n_harmonics = max(1, min(5, n_harmonics))
        self.slant_sigma = slant_sigma
        self.scale_sigma = scale_sigma
        self.baseline_sigma = baseline_sigma
        self.rng = rng or random.Random()

    def _per_point_noise(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate smooth correlated noise for x and y offsets.

        Uses harmonic synthesis: offset(t) = sum_{k=1}^{n} A_k * sin(freq_k * t + phase_k)
        where t is the point index normalized to [0, 2π].
        """
        t = np.linspace(0, 2 * math.pi, n_points)
        dx = np.zeros(n_points)
        dy = np.zeros(n_points)

        for k in range(1, self.n_harmonics + 1):
            freq = k * 1.5  # Each harmonic slightly higher frequency
            phase_x = self.rng.uniform(0, 2 * math.pi)
            phase_y = self.rng.uniform(0, 2 * math.pi)
            amp = self.sigma_global * (1.0 / k)  # Higher harmonics = lower amplitude
            dx += amp * self.rng.uniform(0.7, 1.3) * np.sin(freq * t + phase_x)
            dy += amp * self.rng.uniform(0.7, 1.3) * np.sin(freq * t + phase_y)

        return dx, dy

    def _slant_offset(self, n_points: int) -> np.ndarray:
        """Apply slant variation: a small consistent angle shift across all points."""
        slant_rad = self.rng.gauss(0, self.slant_sigma)
        # Slant shifts x proportionally to y
        return np.full(n_points, math.tan(slant_rad) * 0.1) if abs(slant_rad) > 0.001 else np.zeros(n_points)

    def _scale_offset(self) -> Tuple[float, float]:
        """Return (scale_x, scale_y) multiplicative factors."""
        sx = 1.0 + self.rng.gauss(0, self.scale_sigma)
        sy = 1.0 + self.rng.gauss(0, self.scale_sigma)
        return sx, sy

    def _baseline_drift(self, n_points: int) -> np.ndarray:
        """Slow vertical drift across the stroke (like hand lowering while writing)."""
        t = np.linspace(0, 1, n_points)
        return self.baseline_sigma * np.sin(t * math.pi) * self.rng.uniform(-1, 1)

    def jitter_stroke(self, stroke: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Apply full organic jitter to one stroke.

        Returns a new stroke with smooth, correlated offsets.
        """
        n = len(stroke)
        if n < 2:
            return stroke

        pts = np.array(stroke, dtype=float)

        # 1. Per-point correlated noise
        dx, dy = self._per_point_noise(n)

        # 2. Slant
        dx += self._slant_offset(n)

        # 3. Baseline drift
        dy += self._baseline_drift(n)

        # 4. Scale variation (one per stroke)
        sx, sy = self._scale_offset()
        center = pts.mean(axis=0)
        pts = (pts - center) * np.array([sx, sy]) + center

        # Apply offsets
        pts[:, 0] += dx
        pts[:, 1] += dy

        # Clip to reasonable bounds (allow slight overflow for organic feel)
        return [(float(x), float(y)) for x, y in pts]


def jitter_strokes(
    strokes: List[List[Tuple[float, float]]],
    sigma_global: float = 0.04,
    n_harmonics: int = 3,
    seed: Optional[int] = None,
) -> List[List[Tuple[float, float]]]:
    """Convenience function: create a one-shot OrganicJitter and jitter all strokes.

    Each stroke gets the same global parameters but independent noise (different phase).
    """
    rng = random.Random(seed)
    jitter = OrganicJitter(sigma_global=sigma_global, n_harmonics=n_harmonics, rng=rng)
    return [jitter.jitter_stroke(s) for s in strokes]

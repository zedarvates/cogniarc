"""
Geometric evaluation — score handwriting strokes against ideal skeletons.

Pure geometric scoring (no vision model needed):
  - Discrete Fréchet distance per stroke
  - Loop closure score
  - Proportion deviation
  - Stroke order correctness
  - Composite score 0-100 per glyph

All pure numpy.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional

# ─── Discrete Fréchet distance ───────────────────────────────────

_CLOSURE_THRESHOLD = 0.1  # strokes whose ideal start-end is closer than this are "closed"


def _is_closed_stroke(ideal_stroke: List[Tuple[float, float]]) -> bool:
    """Check if a stroke is intended to be closed (start and end close in ideal)."""
    if len(ideal_stroke) < 3:
        return False
    start = np.array(ideal_stroke[0])
    end = np.array(ideal_stroke[-1])
    return float(np.linalg.norm(start - end)) < _CLOSURE_THRESHOLD

def _frechet_distance(
    P: np.ndarray,
    Q: np.ndarray,
) -> float:
    """Discrete Fréchet distance between two polygonal curves.

    Measures the similarity of two curves independent of parametrization.
    Lower = more similar.
    """
    n, m = len(P), len(Q)
    ca = np.full((n, m), -1.0)

    def _c(i, j):
        if ca[i, j] > -0.5:
            return ca[i, j]
        d = float(np.linalg.norm(P[i] - Q[j]))
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i == 0:
            ca[i, j] = max(_c(0, j - 1), d)
        elif j == 0:
            ca[i, j] = max(_c(i - 1, 0), d)
        else:
            ca[i, j] = max(min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)), d)
        return ca[i, j]

    return float(_c(n - 1, m - 1))


def stroke_frechet(
    drawn: List[Tuple[float, float]],
    ideal: List[Tuple[float, float]],
) -> float:
    """Fréchet distance between drawn stroke and ideal, normalized to [0, 1].

    Returns 0 = perfect match, 1 = maximally different.
    """
    if not drawn or not ideal:
        return 1.0

    P = np.array(drawn, dtype=float)
    Q = np.array(ideal, dtype=float)

    # Normalize both to unit bounding box
    def _normalize(arr):
        min_vals = arr.min(axis=0)
        max_vals = arr.max(axis=0)
        span = max(max_vals - min_vals)
        if span < 1e-8:
            span = 1.0
        return (arr - min_vals) / span

    Pn = _normalize(P)
    Qn = _normalize(Q)

    # Resample to equal length for Fréchet
    def _resample(arr, n_points=20):
        current_n = len(arr)
        if current_n <= 2:
            return arr
        t_orig = np.linspace(0, 1, current_n)
        t_new = np.linspace(0, 1, n_points)
        x = np.interp(t_new, t_orig, arr[:, 0])
        y = np.interp(t_new, t_orig, arr[:, 1])
        return np.column_stack([x, y])

    Pr = _resample(Pn)
    Qr = _resample(Qn)

    return min(1.0, _frechet_distance(Pr, Qr))


# ─── Loop closure ────────────────────────────────────────────────

def closure_score(
    drawn: List[Tuple[float, float]],
    max_dist: float = 0.08,
) -> float:
    """Score how well a loop closes (0 = open, 1 = perfectly closed).

    Only matters for strokes where start and end should meet (determined
    by the ideal stroke's start-end distance).
    """
    if len(drawn) < 3:
        return 1.0

    start = np.array(drawn[0])
    end = np.array(drawn[-1])
    dist = float(np.linalg.norm(start - end))

    if dist >= max_dist:
        return 0.0
    return 1.0 - (dist / max_dist)


# ─── Proportion score ────────────────────────────────────────────

def proportion_score(
    drawn: List[Tuple[float, float]],
    ideal: List[Tuple[float, float]],
) -> float:
    """Score how well aspect ratio of drawn stroke matches ideal (0-1).

    1 = perfect aspect ratio match, 0 = completely wrong proportions.
    """
    if not drawn or not ideal:
        return 0.5

    d_arr = np.array(drawn, dtype=float)
    i_arr = np.array(ideal, dtype=float)

    def _aspect_ratio(arr):
        span = arr.max(axis=0) - arr.min(axis=0)
        span = np.maximum(span, 1e-8)
        return span[0] / span[1]

    ar_d = _aspect_ratio(d_arr)
    ar_i = _aspect_ratio(i_arr)

    ratio = min(ar_d, ar_i) / max(ar_d, ar_i) if max(ar_d, ar_i) > 0 else 1.0
    return ratio


# ─── Per-point error (for targeted practice) ─────────────────────

def per_point_error(
    drawn: List[Tuple[float, float]],
    ideal: List[Tuple[float, float]],
) -> List[float]:
    """Return error magnitude per point for targeted practice.

    Uses dynamic time warping alignment to match drawn-to-ideal points,
    then returns Euclidean distance for each aligned pair.

    Returns list of errors per drawn point.
    """
    if not drawn or not ideal:
        return [1.0] * len(drawn) if drawn else []

    P = np.array(drawn, dtype=float)
    Q = np.array(ideal, dtype=float)

    # Simple DTW alignment
    n, m = len(P), len(Q)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(P[i - 1] - Q[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i - 1, j - 1], dtw[i, j - 1])

    # Backtrace to find alignment
    i, j = n, m
    alignment = []
    while i > 0 and j > 0:
        alignment.append((i - 1, j - 1))
        min_prev = min(
            (dtw[i - 1, j], (i - 1, j)),
            (dtw[i - 1, j - 1], (i - 1, j - 1)),
            (dtw[i, j - 1], (i, j - 1)),
        )
        _, (i, j) = min_prev

    alignment.reverse()

    # Aggregate errors per drawn point
    drawn_errors = [0.0] * n
    drawn_counts = [0] * n
    for pi, qi in alignment:
        err = float(np.linalg.norm(P[pi] - Q[qi]))
        drawn_errors[pi] += err
        drawn_counts[pi] += 1

    for i in range(n):
        if drawn_counts[i] > 0:
            drawn_errors[i] /= drawn_counts[i]

    return drawn_errors


# ─── Composite score ─────────────────────────────────────────────

def evaluate_strokes(
    drawn_strokes: List[List[Tuple[float, float]]],
    ideal_strokes: List[List[Tuple[float, float]]],
) -> Dict:
    """Full geometric evaluation of drawn vs ideal strokes.

    Returns
    -------
    Dict with:
      - score: composite 0-100 (100 = perfect)
      - frechet_scores: per-stroke Fréchet distance (0-1 lower=better)
      - closure_scores: per-stroke closure (0-1)
      - proportion_scores: per-stroke proportion match (0-1)
      - per_point_errors: list of per-stroke per-point errors
      - details: breakdown of scoring components
    """
    n_strokes = min(len(drawn_strokes), len(ideal_strokes))

    frechet_scores = []
    closure_scores = []
    proportion_scores = []
    all_pp_errors = []

    for i in range(n_strokes):
        drawn = drawn_strokes[i]
        ideal = ideal_strokes[i]

        fd = stroke_frechet(drawn, ideal)
        frechet_scores.append(fd)

        cl = closure_score(drawn) if _is_closed_stroke(ideal) else 1.0
        closure_scores.append(cl)

        prop = proportion_score(drawn, ideal)
        proportion_scores.append(prop)

        pp = per_point_error(drawn, ideal)
        all_pp_errors.append(pp)

    # Composite: weighted average
    # Fréchet: 50% (most important — shape similarity)
    # Closure: 20% (only matters for closed strokes)
    # Proportions: 30% (aspect ratio consistency)
    if frechet_scores:
        # Convert Fréchet (0-1, lower=better) to score (0-1, higher=better)
        frechet_ok = [1.0 - f for f in frechet_scores]
        frechet_avg = float(np.mean(frechet_ok)) if frechet_ok else 0.5
    else:
        frechet_avg = 0.5

    closure_avg = float(np.mean(closure_scores)) if closure_scores else 1.0
    prop_avg = float(np.mean(proportion_scores)) if proportion_scores else 0.5

    composite = 0.50 * frechet_avg + 0.20 * closure_avg + 0.30 * prop_avg
    score = min(100, max(0, round(composite * 100, 1)))

    return {
        "score": score,
        "frechet_avg": round(frechet_avg, 4),
        "closure_avg": round(closure_avg, 4),
        "proportion_avg": round(prop_avg, 4),
        "frechet_scores": [round(f, 4) for f in frechet_scores],
        "closure_scores": [round(c, 4) for c in closure_scores],
        "proportion_scores": [round(p, 4) for p in proportion_scores],
        "per_point_errors": all_pp_errors,
        "details": {
            "frechet_weight": 0.5,
            "closure_weight": 0.2,
            "proportion_weight": 0.3,
        },
    }

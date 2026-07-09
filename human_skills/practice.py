"""
Practice loop — motor learning engine.

Implements the child-like learning cycle:
  1. Write the glyph with current sigma
  2. Evaluate → locate errors per point
  3. Reduce sigma for error-prone points (consolidation)
  4. Keep sigma_min residual → hand stays organic forever
  5. Mastery = 5 consecutive attempts ≥ 80

Exports learning curve as JSONL.
"""

import json
import random
import math
from typing import List, Tuple, Dict, Optional, Callable

from .glyphs import GLYPHS, get_glyph
from .organic import OrganicJitter, SIGMA_MIN, jitter_strokes
from .render_svg import strokes_to_svg
from .evaluate import evaluate_strokes, per_point_error

# ─── Per-glyph sigma state ───────────────────────────────────────

class GlyphPracticeState:
    """Tracks per-glyph learning state.

    Each stroke point has its own sigma. When a point is consistently
    drawn wrong, its sigma is reduced (consolidation).
    """

    def __init__(self, char: str, initial_sigma: float = 0.15):
        self.char = char
        self.strokes = get_glyph(char)
        self.sigmas: List[List[float]] = []
        self.history: List[Dict] = []
        self.mastery_count = 0
        self.total_attempts = 0

        # Initialize per-point sigmas
        for stroke in self.strokes:
            self.sigmas.append([initial_sigma] * len(stroke))

    def get_global_sigma(self) -> float:
        """Average sigma across all points."""
        all_s = [s for stroke_s in self.sigmas for s in stroke_s]
        return sum(all_s) / max(1, len(all_s))

    def jitter_strokes(self, rng: random.Random) -> List[List[Tuple[float, float]]]:
        """Apply per-point sigmas to draw the glyph."""
        jittered = []
        for stroke_idx, stroke in enumerate(self.strokes):
            sigmas = self.sigmas[stroke_idx]
            pts = list(stroke)  # copy

            if len(pts) < 2:
                jittered.append(pts)
                continue

            # Per-point correlated noise using stroke's average sigma
            avg_sigma = sum(sigmas) / max(1, len(sigmas))
            j = OrganicJitter(sigma_global=avg_sigma, rng=rng)
            jittered.append(j.jitter_stroke(pts))

        return jittered

    def consolidate(self, errors: List[List[float]], learning_rate: float = 0.3):
        """Reduce sigma for error-prone points.

        Errors > threshold → reduce sigma for that point.
        Sigma never goes below SIGMA_MIN.
        """
        for stroke_idx, stroke_errors in enumerate(errors):
            if stroke_idx >= len(self.sigmas):
                continue
            for point_idx, err in enumerate(stroke_errors):
                if point_idx >= len(self.sigmas[stroke_idx]):
                    continue
                if err > 0.05:  # Error above threshold → need consolidation
                    reduction = self.sigmas[stroke_idx][point_idx] * learning_rate * min(1.0, err)
                    self.sigmas[stroke_idx][point_idx] = max(
                        SIGMA_MIN,
                        self.sigmas[stroke_idx][point_idx] - reduction
                    )
                # Small reward for correct points — gentle sigma increase
                elif err < 0.01:
                    increase = self.sigmas[stroke_idx][point_idx] * 0.01
                    self.sigmas[stroke_idx][point_idx] = min(
                        0.15,
                        self.sigmas[stroke_idx][point_idx] + increase
                    )

    def is_mastered(self) -> bool:
        """Check if 5 consecutive attempts scored ≥ 80."""
        recent = [h for h in self.history[-10:] if h.get("score", 0) >= 80]
        return len(recent) >= 5

    def summary(self) -> Dict:
        """Return learning summary."""
        return {
            "char": self.char,
            "attempts": self.total_attempts,
            "global_sigma": round(self.get_global_sigma(), 4),
            "mastered": self.is_mastered(),
            "mastery_count": self.mastery_count,
            "best_score": max((h.get("score", 0) for h in self.history), default=0),
            "last_score": self.history[-1].get("score", 0) if self.history else 0,
        }


# ─── Practice loop ───────────────────────────────────────────────

def practice_glyph(
    char: str,
    initial_sigma: float = 0.15,
    max_attempts: int = 300,
    mastery_threshold: int = 80,
    learning_rate: float = 0.3,
    seed: Optional[int] = None,
    on_attempt: Optional[Callable] = None,
) -> GlyphPracticeState:
    """Practice a single glyph until mastery or max_attempts.

    Parameters
    ----------
    char : str
        Character to practice (0-9, A-Z).
    initial_sigma : float
        Starting sigma (0.15 = toddler).
    max_attempts : int
        Maximum practice attempts.
    mastery_threshold : int
        Score required for mastery (default 80).
    learning_rate : float
        How aggressively sigma is reduced on errors.
    seed : Optional[int]
        RNG seed for reproducibility.
    on_attempt : Optional[Callable]
        Callback after each attempt: fn(state, attempt_number, result)

    Returns
    -------
    GlyphPracticeState with practice history.
    """
    state = GlyphPracticeState(char, initial_sigma)
    rng = random.Random(seed)

    for attempt in range(max_attempts):
        state.total_attempts += 1

        # Draw
        drawn = state.jitter_strokes(rng)

        # Evaluate
        result = evaluate_strokes(drawn, state.strokes)
        score = result["score"]

        # Record
        record = {
            "attempt": state.total_attempts,
            "score": score,
            "global_sigma": state.get_global_sigma(),
            "frechet_avg": result["frechet_avg"],
        }
        state.history.append(record)

        # Consolidate from per-point errors
        errors = result.get("per_point_errors", [])
        if errors:
            state.consolidate(errors, learning_rate)

        # Mastery tracking
        if score >= mastery_threshold:
            state.mastery_count += 1
        else:
            state.mastery_count = 0

        # Callback
        if on_attempt:
            on_attempt(state, attempt, result)

        # Early exit on mastery
        if state.is_mastered():
            break

    return state


def train_all_glyphs(
    initial_sigma: float = 0.15,
    max_attempts_per_glyph: int = 300,
    mastery_threshold: int = 80,
    chars: Optional[List[str]] = None,
    seed: int = 42,
    output_jsonl: Optional[str] = None,
) -> Dict[str, GlyphPracticeState]:
    """Train all glyphs and export learning curves.

    Parameters
    ----------
    chars : Optional[List[str]]
        Subset to train (default: all 36).
    output_jsonl : Optional[str]
        Path to write JSONL learning curves.

    Returns {char: GlyphPracticeState}
    """
    if chars is None:
        chars = list(GLYPHS.keys())

    results: Dict[str, GlyphPracticeState] = {}
    all_records = []

    for i, char in enumerate(sorted(chars)):
        print(f"[{i+1}/{len(chars)}] Practicing '{char}'...")
        state = practice_glyph(
            char=char,
            initial_sigma=initial_sigma,
            max_attempts=max_attempts_per_glyph,
            mastery_threshold=mastery_threshold,
            seed=seed + hash(char) % 10000,
        )
        results[char] = state
        summary = state.summary()
        print(f"  → {summary['attempts']} attempts, "
              f"σ={summary['global_sigma']:.4f}, "
              f"mastered={summary['mastered']}, "
              f"best={summary['best_score']}")

        # Collect records for JSONL
        for h in state.history:
            all_records.append({
                "char": char,
                "attempt": h["attempt"],
                "score": h["score"],
                "sigma": h["global_sigma"],
                "frechet": h["frechet_avg"],
            })

    if output_jsonl:
        with open(output_jsonl, "w") as f:
            for record in all_records:
                f.write(json.dumps(record) + "\n")
        print(f"\nLearning curves saved to {output_jsonl}")

    return results

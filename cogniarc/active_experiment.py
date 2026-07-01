"""Active experimentation — choose the action that best disambiguates.

The core of the scientific method the "ScientistAgent" name promises but
doesn't yet deliver: instead of blindly executing the next phase skill, when
competing hypotheses disagree, deliberately pick the action whose outcome
most discriminates between them, take it, observe, and reweight beliefs.

This is classic optimal experiment design: the most informative experiment is
the one whose result you can least predict *because your hypotheses disagree
about it*. An action all hypotheses agree on teaches nothing (zero entropy);
an action that splits them maximizes expected information gain.

Pure and dependency-free (only the stdlib) so the whole loop is unit-testable
without a live arc_agi runtime — see tests/test_active_experiment.py.

Worked example wired into the agent (build_wall_floor_experiment): when static
heuristics flag a colour as an *ambiguous* wall candidate, the competing
hypotheses are literally "colour X is a wall" vs "colour X is floor". The
discriminating action is "move toward a cell of colour X": blocked => wall,
moved => floor. ObjectTracker already records that outcome passively; this
module tells the agent which action resolves the ambiguity fastest instead
of waiting for it to happen by chance.
"""
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class PredictiveHypothesis:
    """A hypothesis that predicts a discrete outcome label for a given action.

    `predict(action) -> outcome_label`. Two hypotheses "disagree" on an action
    when they predict different labels for it. `weight` is a positive
    confidence/prior; it need not be normalized (scoring normalizes).
    """
    name: str
    predict: Callable[[int], str]
    weight: float = 1.0


def outcome_distribution(
    hypotheses: List[PredictiveHypothesis], action: int
) -> Dict[str, float]:
    """Weight each predicted outcome by the total weight of hypotheses
    predicting it, normalized to a probability distribution.

    Returns {} if there are no positively-weighted hypotheses.
    """
    dist: Dict[str, float] = defaultdict(float)
    total = 0.0
    for h in hypotheses:
        if h.weight <= 0:
            continue
        dist[h.predict(action)] += h.weight
        total += h.weight
    if total <= 0:
        return {}
    return {label: w / total for label, w in dist.items()}


def discrimination_score(
    hypotheses: List[PredictiveHypothesis], action: int
) -> float:
    """Shannon entropy (in bits) of the predicted-outcome distribution.

    0.0  => every (weighted) hypothesis predicts the same outcome: the action
            is useless as an experiment, it cannot refute anything.
    high => hypotheses disagree; observing the real outcome refutes whichever
            side mispredicted. Maximized when weight splits evenly across
            distinct outcomes.
    """
    dist = outcome_distribution(hypotheses, action)
    entropy = -sum(p * math.log2(p) for p in dist.values() if p > 0)
    return entropy + 0.0  # normalize -0.0 -> 0.0 for clean reporting


def select_experiment(
    hypotheses: List[PredictiveHypothesis], candidate_actions: List[int]
) -> Optional[Tuple[int, float]]:
    """Pick the action with the highest discrimination score.

    Returns (action, score), or None if there are no candidate actions.
    Ties break toward the lowest action number for deterministic behavior.
    A returned score of 0.0 means no candidate action discriminates at all
    (the caller should fall back to its normal policy rather than "experiment"
    on an action that teaches nothing).
    """
    if not candidate_actions:
        return None
    best_action = None
    best_score = -1.0
    for action in candidate_actions:
        score = discrimination_score(hypotheses, action)
        if score > best_score or (score == best_score and (best_action is None or action < best_action)):
            best_action = action
            best_score = score
    return best_action, best_score


def update_beliefs(
    hypotheses: List[PredictiveHypothesis],
    action: int,
    observed_outcome: str,
    refute: bool = True,
) -> List[PredictiveHypothesis]:
    """Reweight hypotheses after observing an action's real outcome.

    Hypotheses that predicted `observed_outcome` are kept; those that
    mispredicted are refuted (weight -> 0) when `refute=True`, else halved.
    Returns a new list (does not mutate the inputs), preserving order.

    This is the "then update" half of the loop: after the discriminating
    action is taken, whichever side mispredicted is falsified — the essence of
    Popperian testing, not confirmation-seeking.
    """
    updated: List[PredictiveHypothesis] = []
    for h in hypotheses:
        predicted = h.predict(action)
        if predicted == observed_outcome:
            new_weight = h.weight
        else:
            new_weight = 0.0 if refute else h.weight * 0.5
        updated.append(PredictiveHypothesis(name=h.name, predict=h.predict, weight=new_weight))
    return updated


def surviving_hypotheses(hypotheses: List[PredictiveHypothesis]) -> List[PredictiveHypothesis]:
    """Hypotheses still in play (positive weight)."""
    return [h for h in hypotheses if h.weight > 0]


# ─────────────────────────────────────────────────────────────────────────────
# Worked example: disambiguate an uncertain wall colour by choosing the action
# that moves the player toward a cell of that colour.
# ─────────────────────────────────────────────────────────────────────────────
def build_wall_floor_experiment(
    color: int,
    action_directions: Dict[int, Tuple[int, int]],
    player_rc: Tuple[int, int],
    grid,
) -> Tuple[List[PredictiveHypothesis], List[int]]:
    """Build the "is colour X a wall or floor?" experiment.

    Args:
        color: the ambiguous colour to test.
        action_directions: {action -> (d_row, d_col)} unit steps, as *learned*
            by ObjectTracker (not hardcoded — pass its `action_direction`
            results, sign-reduced to unit steps).
        player_rc: current (row, col) of the player.
        grid: 2D array-like of the current frame.

    Returns:
        (hypotheses, candidate_actions) where candidate_actions are exactly the
        actions that step the player onto a cell of `color` (the only actions
        whose outcome differs under the wall vs floor hypotheses). Empty
        candidate list means no adjacent cell of that colour to test right now.

    Hypotheses (outcome label = "blocked" or "moved"):
        - "wall":  moving into colour X => blocked
        - "floor": moving into colour X => moved
    """
    pr, pc = player_rc
    h_rows, w_cols = len(grid), len(grid[0]) if len(grid) else 0

    candidate_actions: List[int] = []
    target_of: Dict[int, bool] = {}  # action -> whether it steps onto `color`
    for action, (dr, dc) in action_directions.items():
        sr = 1 if dr > 0 else (-1 if dr < 0 else 0)
        sc = 1 if dc > 0 else (-1 if dc < 0 else 0)
        if sr == 0 and sc == 0:
            continue
        nr, nc = pr + sr, pc + sc
        steps_onto_color = (
            0 <= nr < h_rows and 0 <= nc < w_cols and int(grid[nr][nc]) == color
        )
        target_of[action] = steps_onto_color
        if steps_onto_color:
            candidate_actions.append(action)

    def wall_predict(action: int) -> str:
        return "blocked" if target_of.get(action, False) else "moved"

    def floor_predict(action: int) -> str:
        return "moved"  # floor never blocks

    hypotheses = [
        PredictiveHypothesis("wall", wall_predict, weight=1.0),
        PredictiveHypothesis("floor", floor_predict, weight=1.0),
    ]
    return hypotheses, sorted(candidate_actions)

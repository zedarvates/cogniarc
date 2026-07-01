"""Program synthesis over a small grid-transformation DSL.

The generalizing alternative to a fixed, game-specific phase machine: given a
few input->output grid examples, search for the *shortest composition of
primitives* that maps every input to its output, then verify that program on a
held-out test input it was never fit against. This is the canonical, strongest
approach on ARC (Icecuber's 2020 Kaggle winner, Hodel's ARC-DSL, and the
program-induction framing in Chollet's ARC paper) — a rule found by search that
survives a held-out example is evidence of generalization, not memorization.

Deliberately pure (numpy only, no arc_agi, no env): the whole search + verify
loop runs on plain grid pairs, so it is fully unit-testable without a live
runtime. See tests/test_program_synthesis.py.

Also includes `next_probe_or_action` / `plan_action_sequence`: the same
"search, don't hardcode" idea applied to a small discrete interactive state
(e.g. a rotation counter) whose transition function isn't known in advance
and can only be learned by taking real actions and observing the result. This
is the pure planner half of a search+execute loop; scientist_agent_skills.py
wires it to real self.step() calls (see _skill_rotate_to_goal), replacing a
previously hardcoded "always press action 4 then 3" cycle with actual search
over whichever actions are available.
"""
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

Grid = np.ndarray


# ─────────────────────────────────────────────────────────────────────────────
# DSL primitives — pure grid -> grid functions.
# The 8 dihedral (D4) symmetries plus a couple of shape-changing ops so that
# some tasks genuinely require multi-step compositions (making the search
# non-trivial rather than a single-op lookup).
# ─────────────────────────────────────────────────────────────────────────────
def _identity(g: Grid) -> Grid:
    return g


def _rot90(g: Grid) -> Grid:
    return np.rot90(g, k=1)


def _rot180(g: Grid) -> Grid:
    return np.rot90(g, k=2)


def _rot270(g: Grid) -> Grid:
    return np.rot90(g, k=3)


def _flip_h(g: Grid) -> Grid:
    return np.fliplr(g)


def _flip_v(g: Grid) -> Grid:
    return np.flipud(g)


def _transpose(g: Grid) -> Grid:
    return g.T


def _anti_transpose(g: Grid) -> Grid:
    return np.fliplr(np.flipud(g.T))


def _tile2x2(g: Grid) -> Grid:
    return np.tile(g, (2, 2))


def _tile_h(g: Grid) -> Grid:
    return np.tile(g, (1, 2))


def _tile_v(g: Grid) -> Grid:
    return np.tile(g, (2, 1))


@dataclass(frozen=True)
class Primitive:
    name: str
    fn: Callable[[Grid], Grid]


DEFAULT_PRIMITIVES: Tuple[Primitive, ...] = (
    Primitive("rot90", _rot90),
    Primitive("rot180", _rot180),
    Primitive("rot270", _rot270),
    Primitive("flip_h", _flip_h),
    Primitive("flip_v", _flip_v),
    Primitive("transpose", _transpose),
    Primitive("anti_transpose", _anti_transpose),
    Primitive("tile2x2", _tile2x2),
    Primitive("tile_h", _tile_h),
    Primitive("tile_v", _tile_v),
)

_BY_NAME: Dict[str, Primitive] = {p.name: p for p in DEFAULT_PRIMITIVES}


Example = Tuple[Grid, Grid]  # (input, output)


def apply_program(grid: Grid, program: Sequence[str],
                  primitives: Sequence[Primitive] = DEFAULT_PRIMITIVES) -> Grid:
    """Apply a program (ordered list of primitive names) to a grid."""
    by_name = {p.name: p for p in primitives}
    out = grid
    for name in program:
        out = by_name[name].fn(out)
    return out


def program_matches(examples: Sequence[Example], program: Sequence[str],
                    primitives: Sequence[Primitive] = DEFAULT_PRIMITIVES) -> bool:
    """True iff `program` maps every example input exactly to its output."""
    for inp, out in examples:
        got = apply_program(inp, program, primitives)
        if got.shape != out.shape or not np.array_equal(got, out):
            return False
    return True


def _signature(grids: Sequence[Grid]) -> Tuple:
    """Hashable signature of the current grid-per-example state, for the visited
    set — dedupes search states that produce identical grids on all inputs."""
    return tuple((g.shape, g.tobytes()) for g in grids)


def synthesize(
    examples: Sequence[Example],
    primitives: Sequence[Primitive] = DEFAULT_PRIMITIVES,
    max_depth: int = 3,
) -> Optional[List[str]]:
    """Breadth-first search for the shortest program mapping ALL example inputs
    to their outputs simultaneously.

    Returns the program (list of primitive names, shortest first found) or None
    if no composition up to `max_depth` fits every example. BFS guarantees the
    returned program is of minimal length; the visited-signature set prunes
    revisited states (e.g. rot90 four times returns to start).

    The empty program (identity) is checked first: if inputs already equal
    outputs, returns [].
    """
    if not examples:
        return None

    inputs = [inp for inp, _ in examples]
    outputs = [out for _, out in examples]

    def is_goal(state: Sequence[Grid]) -> bool:
        return all(
            g.shape == o.shape and np.array_equal(g, o)
            for g, o in zip(state, outputs)
        )

    start = tuple(inputs)
    if is_goal(start):
        return []

    visited = {_signature(start)}
    # queue holds (state_grids, program_so_far)
    queue: deque = deque([(start, [])])

    while queue:
        state, program = queue.popleft()
        if len(program) >= max_depth:
            continue
        for prim in primitives:
            try:
                next_state = tuple(prim.fn(g) for g in state)
            except Exception:
                continue  # a primitive that can't apply to this shape — skip
            sig = _signature(next_state)
            if sig in visited:
                continue
            visited.add(sig)
            next_program = program + [prim.name]
            if is_goal(next_state):
                return next_program
            queue.append((next_state, next_program))

    return None


def verify_on_test(program: Sequence[str], test_input: Grid, test_output: Grid,
                   primitives: Sequence[Primitive] = DEFAULT_PRIMITIVES) -> bool:
    """Check a synthesized program on a held-out (input, output) pair it was
    never fit against. This is the actual generalization test: a program that
    fit the training examples but fails here overfit the training pairs."""
    got = apply_program(test_input, program, primitives)
    return got.shape == test_output.shape and np.array_equal(got, test_output)


# ─────────────────────────────────────────────────────────────────────────────
# Color-mapping inference — a common ARC rule orthogonal to geometry: a single
# consistent recolouring applied to every cell. Kept separate from the search
# so it stays simple and independently testable.
# ─────────────────────────────────────────────────────────────────────────────
def infer_color_map(examples: Sequence[Example]) -> Optional[Dict[int, int]]:
    """If a single consistent per-colour remapping (same shape, cellwise)
    explains every example, return it as {old_color: new_color}; else None.

    Requires input and output shapes to match for every example. Returns None
    if any colour maps inconsistently across cells/examples (i.e. the rule
    isn't a pure recolouring).
    """
    mapping: Dict[int, int] = {}
    for inp, out in examples:
        if inp.shape != out.shape:
            return None
        for iv, ov in zip(inp.flat, out.flat):
            iv, ov = int(iv), int(ov)
            if iv in mapping:
                if mapping[iv] != ov:
                    return None
            else:
                mapping[iv] = ov
    return mapping


def apply_color_map(grid: Grid, mapping: Dict[int, int]) -> Grid:
    """Apply a {old_color: new_color} remapping. Colours absent from the map
    are left unchanged."""
    out = grid.copy()
    for old, new in mapping.items():
        out[grid == old] = new
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Online discrete-state search — for interactive properties (e.g. a rotation
# counter) whose transition function must be LEARNED by taking real actions,
# not assumed. Generalizes the "search for the shortest composition" idea from
# static grids to a live, incrementally-discovered transition graph.
# ─────────────────────────────────────────────────────────────────────────────
State = int  # any hashable discrete state works; typed as int for clarity here
TransitionTable = Dict[Tuple[State, int], State]  # (state, action) -> next_state


def _bfs_known_path(
    table: TransitionTable, start: State, goal: State, actions: Sequence[int],
) -> Optional[List[int]]:
    """Shortest action sequence start -> goal using ONLY edges already present
    in `table` (i.e. transitions already observed by real interaction).
    Returns None if goal is unreachable with current knowledge."""
    if start == goal:
        return []
    visited = {start}
    queue: deque = deque([(start, [])])
    while queue:
        state, path = queue.popleft()
        for action in actions:
            key = (state, action)
            if key not in table:
                continue  # edge not yet observed — can't traverse it blindly
            nxt = table[key]
            if nxt in visited:
                continue
            next_path = path + [action]
            if nxt == goal:
                return next_path
            visited.add(nxt)
            queue.append((nxt, next_path))
    return None


def next_probe_or_action(
    table: TransitionTable, current_state: State, goal_state: State, actions: Sequence[int],
) -> Tuple[str, Optional[int]]:
    """Decide the single next action to take, given everything learned so far.

    Returns (mode, action):
      ("done", None)   — current_state already equals goal_state.
      ("advance", a)   — BFS over already-observed transitions found a path;
                          `a` is its first action. Taking it is planned, not
                          a guess.
      ("probe", a)     — no known path yet; `a` is an untried action from
                          current_state whose effect needs to be observed
                          (real active experimentation: try it, then learn).
      ("stuck", None)  — every action from current_state has already been
                          tried at least once from *some* state and still no
                          path is known; the caller should stop rather than
                          loop forever on a transition graph that provably
                          doesn't connect current_state to goal_state.
    """
    if current_state == goal_state:
        return ("done", None)

    path = _bfs_known_path(table, current_state, goal_state, actions)
    if path:
        return ("advance", path[0])

    untried_here = [a for a in actions if (current_state, a) not in table]
    if untried_here:
        return ("probe", untried_here[0])

    # Every action from this exact state has been tried and none of them
    # (transitively) reaches the goal with current knowledge.
    return ("stuck", None)


def plan_action_sequence(
    transition: Callable[[State, int], State],
    start_state: State,
    goal_state: State,
    actions: Sequence[int],
    max_actions: int = 20,
) -> Tuple[List[int], TransitionTable]:
    """Reference/offline driver for `next_probe_or_action`, useful for tests
    and for any case where `transition` can be called freely (e.g. a
    simulator). Interactive callers (scientist_agent_skills.py) instead call
    next_probe_or_action() themselves once per real self.step(), since a live
    action can't be "tried and rolled back" the way this function assumes.

    Returns (actions_taken, learned_table). Stops early on "done" or "stuck",
    or after max_actions regardless.
    """
    table: TransitionTable = {}
    state = start_state
    taken: List[int] = []

    for _ in range(max_actions):
        mode, action = next_probe_or_action(table, state, goal_state, actions)
        if mode == "done":
            break
        if mode == "stuck":
            break
        state_before = state
        state = transition(state, action)
        table[(state_before, action)] = state
        taken.append(action)

    return taken, table

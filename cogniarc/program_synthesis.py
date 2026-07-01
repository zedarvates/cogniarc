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

Scope note: this is the transform-domain solver (grid in -> grid out). It is a
library primitive, not yet wired into the interactive ScientistAgent loop —
that wiring needs the live game and a holdout game to validate against, per the
observe-before-override discipline in docs/EVALUATION.md.
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

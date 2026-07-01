"""Tests for grid-transformation program synthesis.

All pure/synthetic numpy grid pairs — no live arc_agi runtime. The headline
property is generalization: a program synthesized from training pairs is
verified on a HELD-OUT test pair it was never fit against (ARC's train/test
split), which is what distinguishes a found rule from a memorized mapping.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniarc.program_synthesis import (
    DEFAULT_PRIMITIVES,
    apply_program,
    program_matches,
    synthesize,
    verify_on_test,
    infer_color_map,
    apply_color_map,
)


def _pair(inp, fn):
    inp = np.array(inp)
    return inp, fn(inp)


# ── single-primitive synthesis ────────────────────────────────────────────────
def test_synthesize_identity_when_input_equals_output():
    a = np.array([[1, 2], [3, 4]])
    assert synthesize([(a, a.copy())]) == []


def test_synthesize_rot90():
    examples = [
        _pair([[1, 2], [3, 4]], lambda g: np.rot90(g, 1)),
        _pair([[5, 6], [7, 8]], lambda g: np.rot90(g, 1)),
    ]
    prog = synthesize(examples)
    assert prog == ["rot90"]


def test_synthesize_flip_h():
    # Use a 2-row grid so fliplr is NOT equivalent to rot180 (which would also
    # flip rows). Assert the program reproduces the output rather than a
    # specific name, since several primitives can be equivalent on symmetric
    # inputs and BFS may pick whichever it reaches first.
    examples = [
        _pair([[1, 2, 3], [4, 5, 6]], np.fliplr),
        _pair([[7, 8, 9], [1, 0, 2]], np.fliplr),
    ]
    prog = synthesize(examples)
    assert prog is not None
    assert len(prog) == 1
    assert program_matches(examples, prog)


def test_synthesize_transpose():
    examples = [_pair([[1, 2], [3, 4]], lambda g: g.T)]
    # transpose and anti_transpose etc. could both be single ops; verify the
    # returned program actually reproduces the output rather than asserting name
    prog = synthesize(examples)
    assert prog is not None
    assert program_matches(examples, prog)


# ── multi-step synthesis (BFS finds minimal length) ──────────────────────────
def test_synthesize_two_step_composition():
    def target(g):
        return np.tile(np.fliplr(g), (2, 2))
    examples = [
        _pair([[1, 2], [3, 4]], target),
        _pair([[0, 9], [8, 7]], target),
    ]
    prog = synthesize(examples, max_depth=3)
    assert prog is not None
    assert len(prog) == 2
    assert program_matches(examples, prog)


def test_bfs_returns_minimal_length_program():
    # rot180 is reachable as one primitive AND as flip_h+flip_v (two);
    # BFS must return the length-1 solution.
    examples = [_pair([[1, 2], [3, 4]], lambda g: np.rot90(g, 2))]
    prog = synthesize(examples, max_depth=3)
    assert len(prog) == 1


# ── failure / bounds ─────────────────────────────────────────────────────────
def test_unsatisfiable_returns_none():
    a = np.array([[1, 2], [3, 4]])
    # add 100 to every cell: no geometric primitive can do this
    assert synthesize([(a, a + 100)], max_depth=3) is None


def test_depth_limit_respected():
    def target(g):  # needs 2 steps
        return np.tile(np.fliplr(g), (2, 2))
    examples = [_pair([[1, 2], [3, 4]], target)]
    assert synthesize(examples, max_depth=1) is None
    assert synthesize(examples, max_depth=2) is not None


def test_empty_examples_returns_none():
    assert synthesize([]) is None


def test_inconsistent_examples_have_no_program():
    # example A wants rot90, example B wants flip_v — no single program fits both
    examples = [
        _pair([[1, 2], [3, 4]], lambda g: np.rot90(g, 1)),
        _pair([[5, 6], [7, 8]], np.flipud),
    ]
    assert synthesize(examples, max_depth=2) is None


# ── generalization: held-out verification ────────────────────────────────────
def test_program_generalizes_to_holdout_pair():
    def rule(g):
        return np.tile(np.fliplr(g), (2, 2))
    train = [_pair([[1, 2], [3, 4]], rule), _pair([[0, 9], [8, 7]], rule)]
    prog = synthesize(train, max_depth=3)
    test_in = np.array([[2, 2], [1, 3]])
    assert verify_on_test(prog, test_in, rule(test_in)) is True


def test_holdout_verification_catches_wrong_program():
    a = np.array([[1, 2], [3, 4]])
    # a program that happens to match a symmetric training grid but is the
    # wrong rule must fail on an asymmetric held-out grid
    prog = ["flip_h"]
    test_in = np.array([[1, 2], [3, 4]])
    assert verify_on_test(prog, test_in, np.rot90(test_in, 1)) is False


# ── color-map inference ───────────────────────────────────────────────────────
def test_infer_consistent_color_map():
    inp = np.array([[1, 2], [2, 1]])
    out = np.array([[5, 6], [6, 5]])
    assert infer_color_map([(inp, out)]) == {1: 5, 2: 6}


def test_inconsistent_color_map_returns_none():
    # colour 1 -> 5 in first cell but 1 -> 9 in another: not a pure recolour
    inp = np.array([[1, 1]])
    out = np.array([[5, 9]])
    assert infer_color_map([(inp, out)]) is None


def test_color_map_shape_mismatch_returns_none():
    inp = np.array([[1, 2]])
    out = np.array([[1], [2]])
    assert infer_color_map([(inp, out)]) is None


def test_apply_color_map_leaves_unmapped_colors_untouched():
    grid = np.array([[1, 2, 3]])
    got = apply_color_map(grid, {1: 7})
    assert got.tolist() == [[7, 2, 3]]

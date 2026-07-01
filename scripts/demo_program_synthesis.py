#!/usr/bin/env python3
"""Demo: synthesize a grid transformation from train pairs, verify on holdout.

Shows the generalization discipline in miniature — the rule is found by search
over the DSL using only the training pairs, then checked on a test pair it was
never fit against.

Usage:
    python scripts/demo_program_synthesis.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniarc.program_synthesis import synthesize, verify_on_test, apply_program


def rule(g):
    """The hidden rule the solver must discover: flip horizontally, then tile 2x2."""
    return np.tile(np.fliplr(g), (2, 2))


def main():
    train_inputs = [
        np.array([[1, 2], [3, 4]]),
        np.array([[0, 9], [8, 7]]),
    ]
    train = [(g, rule(g)) for g in train_inputs]

    print("Training pairs (input -> output):")
    for inp, out in train:
        print(f"  {inp.tolist()} -> {out.tolist()}")

    program = synthesize(train, max_depth=3)
    print(f"\nSynthesized program: {program}")

    if program is None:
        print("No program found within depth limit.")
        return

    test_in = np.array([[2, 5], [6, 3]])
    test_out = rule(test_in)
    ok = verify_on_test(program, test_in, test_out)
    print(f"\nHeld-out test input : {test_in.tolist()}")
    print(f"Program output      : {apply_program(test_in, program).tolist()}")
    print(f"Expected output     : {test_out.tolist()}")
    print(f"\nGeneralizes to held-out pair: {ok}")


if __name__ == "__main__":
    main()

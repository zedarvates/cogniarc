"""Tests for the visual abacus reader — cognitive simulation."""

from human_skills.abacus_vision import (
    read_abacus_visually,
    add_visually_on_abacus,
    number_to_beads,
    beads_to_number,
)
from human_skills.abacus import render_abacus_svg


class TestVisualReader:
    def test_read_zero(self):
        svg = render_abacus_svg(0, n_columns=4)
        result = read_abacus_visually(svg, n_columns=4)
        assert result["success"]
        assert result["number"] == 0

    def test_read_small_number(self):
        for num in [1, 5, 7, 9]:
            svg = render_abacus_svg(num, n_columns=4)
            result = read_abacus_visually(svg, n_columns=4)
            assert result["success"], f"Failed on {num}"
            assert result["number"] == num, f"Read {result['number']} != {num}"

    def test_read_two_digits(self):
        for num in [10, 42, 99]:
            svg = render_abacus_svg(num, n_columns=4)
            result = read_abacus_visually(svg, n_columns=4)
            assert result["success"]
            assert result["number"] == num

    def test_read_three_digits(self):
        for num in [100, 123, 999]:
            svg = render_abacus_svg(num, n_columns=4)
            result = read_abacus_visually(svg, n_columns=4)
            assert result["success"]
            assert result["number"] == num

    def test_read_large_number(self):
        svg = render_abacus_svg(123456, n_columns=6)
        result = read_abacus_visually(svg, n_columns=6)
        assert result["success"]
        assert result["number"] == 123456

    def test_read_mystery_number(self):
        import random
        rng = random.Random(42)
        for _ in range(5):
            num = rng.randint(0, 9999)
            svg = render_abacus_svg(num, n_columns=4)
            result = read_abacus_visually(svg, n_columns=4)
            assert result["success"], f"Failed on {num}"
            assert result["number"] == num, f"Read {result['number']} != {num}"

    def test_read_digits_are_parsed(self):
        svg = render_abacus_svg(1825, n_columns=4)
        result = read_abacus_visually(svg, n_columns=4)
        assert result["digits"] == [1, 8, 2, 5]

    def test_reasoning_is_produced(self):
        svg = render_abacus_svg(42, n_columns=4)
        result = read_abacus_visually(svg, n_columns=4)
        assert "raisonnement" in result["reasoning"].lower() or "regarde" in result["reasoning"]
        assert "Colonne" in result["reasoning"]
        assert "bille rouge" in result["reasoning"] or "billes rouges" in result["reasoning"]


class TestVisualAddition:
    def test_add_simple(self):
        res = add_visually_on_abacus(12, 34, n_columns=4)
        assert res["result"] == 46

    def test_add_with_carry(self):
        res = add_visually_on_abacus(47, 35, n_columns=4)
        assert res["result"] == 82

    def test_add_big(self):
        res = add_visually_on_abacus(123, 987, n_columns=4)
        assert res["result"] == 1110

    def test_add_zero(self):
        res = add_visually_on_abacus(42, 0, n_columns=4)
        assert res["result"] == 42

    def test_add_creates_steps(self):
        res = add_visually_on_abacus(5, 5, n_columns=4)
        assert len(res["steps"]) > 1
        assert any("ret" in s["action"].lower() for s in res["steps"]), "No carry step"

    def test_add_reasoning(self):
        res = add_visually_on_abacus(47, 35, n_columns=4)
        assert "47" in res["reasoning"]
        assert "82" in res["reasoning"]
        assert "mouvements" in res["reasoning"]

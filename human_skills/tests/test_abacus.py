"""Tests for the Soroban abacus module."""

from human_skills import abacus


class TestNumberToBeads:
    def test_zero(self):
        beads = abacus.number_to_beads(0, 4)
        assert all(h == 0 and e == 0 for h, e in beads), f"Zero should have no active beads: {beads}"
        assert len(beads) == 4

    def test_single_digit(self):
        beads = abacus.number_to_beads(7, 4)
        # 7 = 1 heaven (5) + 2 earth
        assert beads[-1] == (1, 2), f"7 should be (1,2) in units: {beads}"

    def test_all_columns(self):
        beads = abacus.number_to_beads(123, 6)
        assert beads == [(0, 0), (0, 0), (0, 0), (0, 1), (0, 2), (0, 3)], f"123: {beads}"

    def test_large_number(self):
        beads = abacus.number_to_beads(999999, 6)
        assert all(h == 1 and e == 4 for h, e in beads), f"999999 all (1,4): {beads}"

    def test_nine(self):
        beads = abacus.number_to_beads(9, 1)
        assert beads == [(1, 4)], f"9 should be (1,4): {beads}"

    def test_max_columns(self):
        beads = abacus.number_to_beads(123456, 6)
        assert len(beads) == 6
        assert abacus.beads_to_number(beads) == 123456


class TestBeadsToNumber:
    def test_roundtrip(self):
        for n in [0, 1, 5, 9, 10, 42, 100, 999, 123456]:
            beads = abacus.number_to_beads(n, 6)
            result = abacus.beads_to_number(beads)
            assert result == n, f"Roundtrip failed: {n} -> {beads} -> {result}"

    def test_manual_beads(self):
        assert abacus.beads_to_number([(1, 3)]) == 8  # 5 + 3
        assert abacus.beads_to_number([(0, 4)]) == 4  # 0 + 4
        assert abacus.beads_to_number([(1, 4)]) == 9  # 5 + 4


class TestRenderAbacusSVG:
    def test_svg_valid(self):
        svg = abacus.render_abacus_svg(42, n_columns=4)
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert "viewBox" in svg

    def test_svg_for_zero(self):
        svg = abacus.render_abacus_svg(0, n_columns=3)
        assert "c0392b" not in svg or svg.count("c0392b") < 3
        # Zero has no active beads

    def test_svg_for_nine(self):
        svg = abacus.render_abacus_svg(9, n_columns=3)
        # Nine has active beads — check red appears
        assert "c0392b" in svg

    def test_svg_size_varies(self):
        s1 = abacus.render_abacus_svg(42, n_columns=3)
        s2 = abacus.render_abacus_svg(42, n_columns=6)
        assert len(s2) > len(s1)

    def test_svg_has_frame_and_beam(self):
        svg = abacus.render_abacus_svg(123, n_columns=3)
        assert "stroke" in svg  # frame/beam have strokes
        assert "#5D4037" in svg  # frame color


class TestPractice:
    def test_generate_sequence(self):
        nums = abacus.generate_practice_sequence(n_columns=3, count=10, seed=42)
        assert len(nums) == 10
        assert all(0 <= n < 1000 for n in nums)

    def test_generate_deterministic(self):
        nums1 = abacus.generate_practice_sequence(n_columns=3, count=5, seed=42)
        nums2 = abacus.generate_practice_sequence(n_columns=3, count=5, seed=42)
        assert nums1 == nums2

    def test_practice_session(self):
        session = abacus.AbacusPracticeSession(n_columns=3)
        nums = [0, 1, 5, 9, 42]
        result = session.run_sequence(nums, mode="read")
        assert result["total"] == 5
        assert result["correct"] == 5  # all should be correct (it's a simulation)
        assert result["accuracy"] == 1.0


class TestAddition:
    def test_add_simple_no_carry(self):
        steps = abacus.add_on_abacus(12, 34, n_columns=4)
        last = steps[-1]
        assert last["number"] == 46

    def test_add_with_carry(self):
        steps = abacus.add_on_abacus(47, 35, n_columns=4)
        last = steps[-1]
        assert last["number"] == 82

    def test_add_big_numbers(self):
        steps = abacus.add_on_abacus(123, 987, n_columns=4)
        last = steps[-1]
        assert last["number"] == 1110

    def test_add_zero(self):
        steps = abacus.add_on_abacus(42, 0, n_columns=4)
        last = steps[-1]
        assert last["number"] == 42

    def test_add_steps_have_labels(self):
        steps = abacus.add_on_abacus(47, 35, n_columns=4)
        assert all("label" in s for s in steps)
        assert len(steps) >= 2  # at least initial + some steps

    def test_render_addition_svg(self):
        html = abacus.render_addition_svg(12, 34, n_columns=3)
        assert "12 + 34" in html or "12 + 34" in html
        assert "46" in html

    def test_carry_detected(self):
        steps = abacus.add_on_abacus(5, 5, n_columns=2)
        carries = [s for s in steps if s.get("carry")]
        assert len(carries) >= 1, f"No carry steps detected in {steps}"

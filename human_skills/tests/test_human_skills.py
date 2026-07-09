"""Tests for human_skills package — all pure numpy, no screen needed.
"""

import math
import json
import pytest
import tempfile
import os

from human_skills import (
    GLYPHS, get_glyph,
    OrganicJitter, jitter_strokes, motor_age_to_sigma,
    strokes_to_svg, render_glyph_plate,
    evaluate_strokes,
    practice_glyph, train_all_glyphs,
)
from human_skills.organic import SIGMA_MIN
from human_skills.evaluate import closure_score, proportion_score, per_point_error
from human_skills.practice import GlyphPracticeState
from human_skills import shapes
from human_skills.scenes import build_cluster_house_scene


class TestGlyphs:
    def test_all_36_glyphs_exist(self):
        """36 chars: 10 digits + 26 letters"""
        assert len(GLYPHS) == 36

    def test_digits_0_to_9(self):
        for d in "0123456789":
            assert d in GLYPHS, f"Missing digit {d}"
            assert len(GLYPHS[d]) >= 1, f"Digit {d} has no strokes"

    def test_letters_A_to_Z(self):
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            assert c in GLYPHS, f"Missing letter {c}"
            assert len(GLYPHS[c]) >= 1, f"Letter {c} has no strokes"

    def test_get_glyph_case_insensitive(self):
        assert get_glyph("a") == get_glyph("A")

    def test_get_glyph_invalid_raises(self):
        with pytest.raises(KeyError):
            get_glyph("ñ")

    def test_strokes_have_points(self):
        for char, strokes in GLYPHS.items():
            for i, stroke in enumerate(strokes):
                assert len(stroke) >= 2, f"{char} stroke {i}: <2 points"
                for x, y in stroke:
                    assert 0 <= x <= 1, f"{char} stroke {i}: x={x} out of [0,1]"
                    assert 0 <= y <= 1, f"{char} stroke {i}: y={y} out of [0,1]"


class TestOrganicJitter:
    def test_default_jitter(self):
        stroke = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
        j = OrganicJitter(sigma_global=0.04)
        result = j.jitter_stroke(stroke)
        assert len(result) == 3
        # Should have slight offset
        assert result[0] != (0.0, 0.0)

    def test_deterministic_seed(self):
        stroke = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
        import random
        j1 = OrganicJitter(sigma_global=0.04, rng=random.Random(42))
        j2 = OrganicJitter(sigma_global=0.04, rng=random.Random(42))
        assert j1.jitter_stroke(stroke) == j2.jitter_stroke(stroke)

    def test_motor_age_mapping(self):
        assert motor_age_to_sigma("toddler") == 0.15
        assert motor_age_to_sigma("adult") == 0.015
        assert motor_age_to_sigma("unknown") == 0.04  # default

    def test_jitter_strokes_convenience(self):
        strokes = [[(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)],
                   [(0.2, 0.0), (0.5, 0.5), (0.8, 0.0)]]
        result = jitter_strokes(strokes, sigma_global=0.08, seed=42)
        assert len(result) == 2
        assert len(result[0]) == 3
        assert len(result[1]) == 3

    def test_per_point_noise_harmonics(self):
        """Check that noise is smooth (correlated), not white."""
        import random
        j = OrganicJitter(sigma_global=0.1, n_harmonics=3, rng=random.Random(0))
        stroke = [(i/50, math.sin(i/50 * math.pi)) for i in range(51)]
        result = j.jitter_stroke(stroke)

        # Compute first difference of x offsets — should be smooth
        xs = [p[0] for p in result]
        diffs = [abs(xs[i+1] - xs[i]) for i in range(len(xs)-1)]
        max_diff = max(diffs)
        avg_diff = sum(diffs) / len(diffs)

        # Correlated noise should have moderate max_diff
        assert max_diff < 0.3, f"Jitter too jerky: max_diff={max_diff}"
        assert avg_diff > 0, "No jitter applied"


class TestRenderSVG:
    def test_strokes_to_svg_creates_valid_svg(self):
        strokes = [[(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)]]
        svg = strokes_to_svg(strokes)
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert "path" in svg
        assert "viewBox" in svg

    def test_render_single_point(self):
        strokes = [[(0.5, 0.5)]]
        svg = strokes_to_svg(strokes)
        assert "circle" in svg  # Single point → circle

    def test_empty_strokes(self):
        svg = strokes_to_svg([])
        assert svg.startswith("<svg")

    def test_background_optional(self):
        svg = strokes_to_svg([[(0.0, 0.0), (1.0, 1.0)]], background=None)
        assert "rect" not in svg

    def test_glyph_plate(self):
        subset = {"A": GLYPHS["A"], "B": GLYPHS["B"]}
        svg = render_glyph_plate(subset, chars_per_row=2)
        assert svg.startswith("<svg")
        assert "A" in svg
        assert "B" in svg


class TestEvaluate:
    def test_perfect_match_scores_high(self):
        """A stroke drawn exactly like ideal should score 100."""
        strokes = GLYPHS["A"]
        result = evaluate_strokes(strokes, strokes)
        assert result["score"] >= 95, f"Perfect match scored {result['score']}"

    def test_wildly_wrong_scores_low(self):
        """Completely different strokes should score low."""
        ideal = [[(0.0, 0.0), (1.0, 1.0)]]
        drawn = [[(0.0, 1.0), (1.0, 0.0)]]  # opposite diagonal
        result = evaluate_strokes(drawn, ideal)
        assert result["score"] <= 50, f"Wrong stroke scored {result['score']}"

    def test_empty_strokes_handled(self):
        result = evaluate_strokes([], [])
        assert result["score"] <= 100

    def test_closure_score_open_loop(self):
        """An open stroke should score low on closure."""
        # Start and end far apart
        open_stroke = [(0.0, 0.0), (0.5, 0.5), (1.0, 0.0)]
        score = closure_score(open_stroke, max_dist=0.05)
        assert score == 0.0

    def test_closure_score_closed(self):
        closed_stroke = [(0.0, 0.0), (0.5, 0.5), (1.0, 0.0), (0.5, -0.5), (0.0, 0.0)]
        score = closure_score(closed_stroke, max_dist=0.08)
        assert score > 0.5, f"Closed stroke scored {score}"

    def test_proportion_score(self):
        assert proportion_score([(0.0, 0.0), (1.0, 1.0)],
                                [(0.0, 0.0), (1.0, 1.0)]) > 0.9
        assert proportion_score([(0.0, 0.0), (1.0, 0.0)],
                                [(0.0, 0.0), (0.0, 1.0)]) < 0.01

    def test_per_point_error_length(self):
        drawn = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
        ideal = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
        errors = per_point_error(drawn, ideal)
        assert len(errors) == 3
        assert all(e < 0.01 for e in errors)  # near-perfect match


class TestPractice:
    def test_glyph_practice_state_init(self):
        state = GlyphPracticeState("A", initial_sigma=0.15)
        assert state.char == "A"
        assert not state.is_mastered()
        assert state.total_attempts == 0

    def test_practice_improves_over_time(self):
        """After practicing, sigma should decrease."""
        state = practice_glyph("I", initial_sigma=0.15, max_attempts=20, seed=42)
        assert state.total_attempts > 0
        summary = state.summary()
        assert summary["global_sigma"] <= 0.15, "Sigma should not increase"
        assert summary["best_score"] > 0

    def test_full_training_digits_only(self):
        """Train just digits 0-3 as quick smoke test."""
        results = train_all_glyphs(
            chars=["0", "1", "2", "3"],
            initial_sigma=0.15,
            max_attempts_per_glyph=20,
            seed=42,
        )
        assert len(results) == 4
        for char, state in results.items():
            assert state.total_attempts > 0
            summary = state.summary()
            assert 0 <= summary["best_score"] <= 100

    def test_jsonl_export(self):
        """Check that JSONL export works."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            jsonl_path = f.name

        try:
            results = train_all_glyphs(
                chars=["A", "B"],
                initial_sigma=0.15,
                max_attempts_per_glyph=10,
                seed=42,
                output_jsonl=jsonl_path,
            )
            assert os.path.exists(jsonl_path)
            with open(jsonl_path) as f:
                lines = f.readlines()
            assert len(lines) > 0
            record = json.loads(lines[0])
            assert "char" in record
            assert "score" in record
            assert "sigma" in record
        finally:
            os.unlink(jsonl_path)

    def test_mastery_achievable(self):
        """Very simple glyph like '1' should converge."""
        state = practice_glyph(
            "1", initial_sigma=0.15, max_attempts=30, mastery_threshold=80, seed=42
        )
        summary = state.summary()
        assert summary["attempts"] > 0
        assert summary["global_sigma"] < 0.15  # Should have improved


class TestIntegration:
    def test_end_to_end_glyph_to_svg(self):
        """Render a jittered glyph to SVG."""
        strokes = GLYPHS["A"]
        jittered = jitter_strokes(strokes, sigma_global=0.04, seed=42)
        svg = strokes_to_svg(jittered)
        assert svg.startswith("<svg")
        assert len(svg) > 200

    def test_practice_then_render(self):
        """Practice a glyph, then render the final version."""
        state = practice_glyph("O", initial_sigma=0.15, max_attempts=15, seed=42)
        # Get final version
        import random
        final = state.jitter_strokes(random.Random(0))
        svg = strokes_to_svg(final)
        assert svg.startswith("<svg")
        assert state.summary()["global_sigma"] < 0.15


class TestShapes:
    def test_line_produces_stroke(self):
        s = shapes.line(0, 0, 1, 1)
        assert len(s) == 1
        assert len(s[0]) == 2

    def test_rectangle_four_sides(self):
        s = shapes.rectangle(0.1, 0.1, 0.5, 0.5)
        assert len(s) == 4  # four sides
        for side in s:
            assert len(side) == 2

    def test_rectangle_loop(self):
        s = shapes.rectangle_loop(0.1, 0.1, 0.5, 0.5)
        assert len(s) == 1
        # 5 points: 4 corners + back to start
        assert len(s[0]) == 5

    def test_triangle_three_sides(self):
        s = shapes.triangle(0, 0, 0.5, 1, 1, 0)
        assert len(s) == 3

    def test_triangle_loop(self):
        s = shapes.triangle_loop(0, 0, 0.5, 1, 1, 0)
        assert len(s) == 1
        assert len(s[0]) == 4  # 3 corners + back to start

    def test_circle_points(self):
        s = shapes.circle(0.5, 0.5, 0.2, n_points=12)
        assert len(s) == 1
        assert len(s[0]) == 13  # 12 pts + closure

    def test_arc_not_full_circle(self):
        s = shapes.arc(0.5, 0.5, 0.2, 0.2, 0, 180, n_points=8)
        assert len(s) == 1
        assert len(s[0]) == 9  # 8 segments + 1 = 9

    def test_cross_two_lines(self):
        s = shapes.cross(0, 0, 1, 1)
        assert len(s) == 2

    def test_wavy_line(self):
        s = shapes.wavy_line(0, 1, 0.5, amplitude=0.02, frequency=5, n_points=10)
        assert len(s) == 1
        assert len(s[0]) == 11  # 10 segments + 1

    def test_grass_tufts(self):
        s = shapes.grass_tufts(0, 1, 0.5, count=10)
        assert len(s) == 10

    def test_sun_rays(self):
        s = shapes.sun_rays(0.5, 0.5, 0.1, 0.2, n_rays=8)
        assert len(s) == 8

    def test_ellipse_aspect_ratio(self):
        """Ellipse should produce different x/y spans."""
        s = shapes.ellipse(0.5, 0.5, 0.3, 0.1, n_points=12)
        xs = [p[0] for p in s[0]]
        ys = [p[1] for p in s[0]]
        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        assert x_span > y_span, f"Ellipse not wide: x={x_span:.3f} y={y_span:.3f}"

    def test_plus_sign(self):
        """Plus should produce a horizontal + vertical stroke."""
        s = shapes.plus(0, 0, 1, 1)
        assert len(s) == 2
        # Horizontal: same y
        assert abs(s[0][0][1] - s[0][1][1]) < 0.001
        # Vertical: same x
        assert abs(s[1][0][0] - s[1][1][0]) < 0.001


class TestScenes:
    def test_cluster_house_builds(self):
        scene = build_cluster_house_scene()
        assert len(scene) > 20, f"Too few elements: {len(scene)}"
        # Check all elements have strokes, layer, color
        for strokes, layer, color in scene:
            assert len(strokes) > 0
            assert isinstance(layer, int)
            assert isinstance(color, str)

    def test_cluster_house_has_sun(self):
        scene = build_cluster_house_scene()
        colors = [color for _, _, color in scene]
        assert "#FFD700" in colors, "No gold/sun in scene"

    def test_cluster_house_has_roof(self):
        scene = build_cluster_house_scene()
        colors = [color for _, _, color in scene]
        assert "#CD5C5C" in colors or "#D2691E" in colors, "No roof colors"

    def test_elements_sorted_by_layer(self):
        scene = build_cluster_house_scene()
        # Ground should be before roof
        ground_layers = [
            layer for strokes, layer, color in scene
            if "#4a7c3f" in color or "#5a9c4f" in color
        ]
        roof_layers = [
            layer for strokes, layer, color in scene
            if "#CD5C5C" in color
        ]
        if ground_layers and roof_layers:
            assert max(ground_layers) <= min(roof_layers), "Ground should be behind roof"

    def test_jittered_scene_valid_svg(self):
        """Apply jitter to scene and verify SVG output."""
        scene = build_cluster_house_scene()
        all_paths = []
        for strokes, layer, color in scene:
            j = jitter_strokes(strokes, sigma_global=0.01, seed=42)
            for stroke in j:
                all_paths.append((stroke, color))

        from human_skills.render_svg import _stroke_to_svg_path
        parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1">']
        parts.append('<rect width="100%" height="100%" fill="#f0e6d3"/>')
        for stroke, color in all_paths[:5]:  # just first 5 for speed
            path = _stroke_to_svg_path(stroke, stroke_width=0.004, viewbox_size=1, padding=0)
            if path:
                parts.append(f'<g color="{color}">{path}</g>')
        parts.append("</svg>")
        svg = "\n".join(parts)
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")

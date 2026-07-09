"""human_skills — apprendre à écrire et dessiner comme un enfant.

Package d'apprentissage moteur : squelettes de glyphes, jitter organique,
rendu SVG, évaluation géométrique, boucle de pratique.
"""

from .glyphs import GLYPHS, Stroke, get_glyph
from .organic import OrganicJitter, jitter_strokes, motor_age_to_sigma
from .render_svg import strokes_to_svg, render_glyph_plate
from .evaluate import evaluate_strokes
from .practice import practice_glyph, train_all_glyphs
from . import shapes
from . import scenes

__all__ = [
    "GLYPHS", "Stroke", "get_glyph",
    "OrganicJitter", "jitter_strokes", "motor_age_to_sigma",
    "strokes_to_svg", "render_glyph_plate",
    "evaluate_strokes",
    "practice_glyph", "train_all_glyphs",
    "shapes", "scenes",
]

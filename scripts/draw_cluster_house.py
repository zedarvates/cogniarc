"""
Phase 3 — Dessin vectoriel : la maison cluster ensoleillée 🏠☀️

Dessine la scène avec le jitter organique appris en Phase 2,
signée en lettres réellement pratiquées : « notre maison ».
"""

import os
import sys
import json
import random

# Add parent to path if running as script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from human_skills import (
    GLYPHS, get_glyph,
    jitter_strokes, strokes_to_svg,
    motor_age_to_sigma,
)
from human_skills.organic import OrganicJitter
from human_skills.scenes import build_cluster_house_scene
from human_skills.practice import practice_glyph

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Practice signature letters ──────────────────────────────────

SIGNATURE = "NOTRE MAISON"
SIGNATURE_CHARS = [c for c in SIGNATURE if c != " "]


def practice_signature(sigma_init: float = 0.15, max_attempts: int = 120) -> dict:
    """Practice each letter in the signature and return learned sigmas."""
    print("🎯 Practicing signature letters...")
    sigmas: dict = {}
    for i, char in enumerate(set(SIGNATURE_CHARS)):
        char_upper = char.upper()
        if char_upper not in GLYPHS:
            print(f"  ⚠️  '{char}' not in glyph set, skipping")
            continue
        state = practice_glyph(
            char=char_upper,
            initial_sigma=sigma_init,
            max_attempts=max_attempts,
            mastery_threshold=80,
            seed=42 + ord(char_upper),
        )
        summary = state.summary()
        print(f"  [{i+1}/{len(set(SIGNATURE_CHARS))}] '{char_upper}': "
              f"{summary['attempts']} attempts, "
              f"σ={summary['global_sigma']:.4f}, "
              f"mastered={summary['mastered']}")
        sigmas[char_upper] = {
            "state": state,
            "sigma": summary["global_sigma"],
            "attempts": summary["attempts"],
        }
    return sigmas


def render_signature(sigmas: dict, seed: int = 999) -> list:
    """Render 'notre maison' signature using practiced letter sigmas.

    Returns list of strokes (one per letter) positioned in a row.
    """
    import math

    letters = list(SIGNATURE)  # include spaces, already uppercase
    letter_spacing = 0.055
    start_x = 0.05
    y_center = 0.04  # below the house, in the grass area
    total_width = len(letters) * letter_spacing
    # Center the signature
    offset_x = (1.0 - total_width) / 2

    all_strokes = []
    rng = random.Random(seed)

    for i, char in enumerate(letters):
        x_pos = offset_x + i * letter_spacing
        if char == " ":
            continue

        if char not in sigmas:
            continue

        char_sigma = sigmas[char]["sigma"]
        strokes = get_glyph(char)

        # Scale and position each stroke
        scaled_strokes = []
        for stroke in strokes:
            scaled = [(x * 0.045 + x_pos, y * 0.045 + y_center) for x, y in stroke]
            scaled_strokes.append(scaled)

        # Apply jitter with the practiced sigma
        jittered = jitter_strokes(scaled_strokes, sigma_global=char_sigma, seed=seed + i)
        all_strokes.extend(jittered)

    return all_strokes


# ─── Render the full scene ───────────────────────────────────────

def render_scene_with_jitter(
    elements: list,
    sigma_global: float = 0.015,  # adult-level
    seed: int = 777,
) -> list:
    """Apply organic jitter to every stroke in the scene, sorted by layer."""

    # Sort by layer
    sorted_elements = sorted(elements, key=lambda e: e[1])

    all_paths = []
    rng = random.Random(seed)

    for strokes, layer, color in sorted_elements:
        jitter = OrganicJitter(sigma_global=sigma_global, rng=rng)
        for stroke in strokes:
            jittered = jitter.jitter_stroke(stroke)
            all_paths.append((jittered, color))

    return all_paths


def build_svg_from_paths(paths: list, width_px=800, height_px=600, background="#f0e6d3"):
    """Convert (jittered_strokes, color) pairs to a single SVG."""
    from human_skills.render_svg import _stroke_to_svg_path

    vb = 1.0  # normalized [0,1] coordinate space
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_px}" height="{height_px}" '
        f'viewBox="0 0 {vb} {vb}">\n'
        f'<rect width="100%" height="100%" fill="{background}"/>'
    ]

    for stroke, color in paths:
        # Viewbox=1, padding=0 since coords are already in [0,1]
        path = _stroke_to_svg_path(stroke, stroke_width=0.004, viewbox_size=1, padding=0)
        if path:
            parts.append(f'<g color="{color}">{path}</g>')

    parts.append("</svg>")
    return "\n".join(parts)


# ─── Main ────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("🏠☀️  Maison Cluster Ensoleillée — Phase 3")
    print("=" * 50)

    # 1. Practice signature letters
    sigmas = practice_signature(sigma_init=0.15, max_attempts=120)
    avg_sig = sum(s["sigma"] for s in sigmas.values()) / max(1, len(sigmas))
    mastered = sum(1 for c in SIGNATURE_CHARS
                   if c.upper() in sigmas and c.upper() in GLYPHS)
    print(f"\n📝 Signature sigma avg: {avg_sig:.4f}")

    # 2. Build scene
    print("🏗️  Building scene...")
    scene = build_cluster_house_scene()
    print(f"   {sum(len(s) for s, _, _ in scene)} strokes in scene")

    # 3. Render with adult-level jitter
    print("🎨 Applying organic jitter...")
    paths = render_scene_with_jitter(scene, sigma_global=0.012, seed=777)

    # 4. Add signature
    print("✍️  Adding signature...")
    signature_strokes = render_signature(sigmas, seed=999)
    for stroke in signature_strokes:
        paths.append((stroke, "#2c2c2c"))

    # 5. Build SVG
    print("💾 Building SVG...")
    svg = build_svg_from_paths(paths, width_px=800, height_px=600)

    out_path = os.path.join(OUTPUT_DIR, "maison_cluster_ensoleillee.svg")
    with open(out_path, "w") as f:
        f.write(svg)
    print(f"✅ Saved: {out_path}")

    # Also render toddler version for comparison
    print("👶 Rendering toddler version...")
    paths_toddler = render_scene_with_jitter(scene, sigma_global=0.10, seed=777)
    sig_toddler = render_signature(
        {c: {"sigma": 0.12, "attempts": 0} for c in SIGNATURE_CHARS},
        seed=999,
    )
    for stroke in sig_toddler:
        paths_toddler.append((stroke, "#2c2c2c"))

    svg_toddler = build_svg_from_paths(paths_toddler, width_px=800, height_px=600)
    toddler_path = os.path.join(OUTPUT_DIR, "maison_cluster_toddler.svg")
    with open(toddler_path, "w") as f:
        f.write(svg_toddler)
    print(f"✅ Saved: {toddler_path}")

    # Comparison HTML
    html = f"""<!DOCTYPE html>
<html><head><title>🏠☀️ Maison Cluster Ensoleillée</title>
<style>
body {{ font-family: sans-serif; margin: 30px; background: #1a1a2e; color: #eee; text-align: center; }}
.container {{ display: flex; gap: 30px; justify-content: center; flex-wrap: wrap; }}
.box {{ flex: 1; min-width: 400px; max-width: 600px; }}
.box h2 {{ margin: 10px 0; }}
object {{ width: 100%; border: 2px solid #444; border-radius: 12px; background: #f0e6d3; }}
.footer {{ margin-top: 20px; color: #888; font-size: 0.9em; }}
</style></head><body>
<h1>🏠☀️ Maison Cluster Ensoleillée</h1>
<p>σ_signature = {avg_sig:.4f} · {mastered}/{len(set(SIGNATURE_CHARS))} lettres maîtrisées</p>
<div class="container">
<div class="box"><h2>👶 σ=0.10 (enfant)</h2>
<object data="maison_cluster_toddler.svg" type="image/svg+xml"></object></div>
<div class="box"><h2>🧑 σ=0.012 (appris)</h2>
<object data="maison_cluster_ensoleillee.svg" type="image/svg+xml"></object></div>
</div>
<p class="footer">Signé « NOTRE MAISON » en lettres apprises par la boucle de pratique — pas une police.</p>
</body></html>"""
    html_path = os.path.join(OUTPUT_DIR, "maison_comparaison.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"✅ Comparison: {html_path}")

    print("\n✨ Done! Open outputs/maison_comparaison.html in your browser.")


if __name__ == "__main__":
    main()

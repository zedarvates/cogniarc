"""Demo: train all 36 glyphs and render an alphabet plate + learning curve."""

import json
import os
from human_skills import (
    train_all_glyphs, render_glyph_plate, GLYPHS,
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def demo_practice_curve():
    """Train all 36 glyphs, save learning curve + final plate."""
    jsonl_path = os.path.join(OUTPUT_DIR, "learning_curve.jsonl")
    
    results = train_all_glyphs(
        initial_sigma=0.15,
        max_attempts_per_glyph=120,
        mastery_threshold=80,
        seed=42,
        output_jsonl=jsonl_path,
    )
    
    # Stats
    mastered = sum(1 for s in results.values() if s.is_mastered())
    total = len(results)
    avg_sigma = sum(s.summary()["global_sigma"] for s in results.values()) / total
    total_attempts = sum(s.summary()["attempts"] for s in results.values())
    
    print(f"\n📊 Results: {mastered}/{total} mastered "
          f"(σ_avg={avg_sigma:.4f}, {total_attempts} total attempts)")
    
    # Render final alphabet plate
    final_strokes = {}
    for char in sorted(GLYPHS.keys()):
        # Get the practiced version's final sigmas
        state = results[char]
        import random
        rng = random.Random(42 + hash(char) % 10000)
        final_strokes[char] = state.jitter_strokes(rng)
    
    # Toddler version (high sigma)
    toddler_strokes = {}
    from human_skills.organic import jitter_strokes
    for char in sorted(GLYPHS.keys()):
        toddler_strokes[char] = jitter_strokes(GLYPHS[char], sigma_global=0.15, seed=42)
    
    svg_adult = render_glyph_plate(final_strokes, chars_per_row=8, stroke_width=1.5)
    svg_toddler = render_glyph_plate(toddler_strokes, chars_per_row=8, stroke_width=1.5)
    
    adult_path = os.path.join(OUTPUT_DIR, "alphabet_adult.svg")
    toddler_path = os.path.join(OUTPUT_DIR, "alphabet_toddler.svg")
    comparison_path = os.path.join(OUTPUT_DIR, "alphabet_comparison.html")
    
    with open(adult_path, "w") as f:
        f.write(svg_adult)
    with open(toddler_path, "w") as f:
        f.write(svg_toddler)
    
    # HTML side-by-side
    html = f"""<!DOCTYPE html>
<html><head><title>Écriture Organique — Comparaison</title>
<style>
body {{ font-family: sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
.container {{ display: flex; gap: 20px; }}
.box {{ flex: 1; text-align: center; }}
img {{ width: 100%; max-width: 600px; border: 1px solid #333; border-radius: 8px; }}
</style></head><body>
<h1>🧠 Apprentissage de l'écriture — Phase 2</h1>
<p>{mastered}/{total} glyphes maîtrisés · σ_avg = {avg_sigma:.4f} · {total_attempts} essais</p>
<div class="container">
<div class="box"><h2>👶 Enfant (σ=0.15)</h2>
<object data="alphabet_toddler.svg" type="image/svg+xml" width="500"></object></div>
<div class="box"><h2>🧑 Appris (σ_avg={avg_sigma:.4f})</h2>
<object data="alphabet_adult.svg" type="image/svg+xml" width="500"></object></div>
</div>
</body></html>"""
    with open(comparison_path, "w") as f:
        f.write(html)
    
    print(f"✅ Alphabet plates: {adult_path}")
    print(f"✅ Toddler: {toddler_path}")
    print(f"✅ Comparison: {comparison_path}")
    print(f"✅ Learning curve: {jsonl_path}")
    
    return results


if __name__ == "__main__":
    demo_practice_curve()

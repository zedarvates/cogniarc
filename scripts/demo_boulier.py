"""
Démo : apprendre à compter comme un humain avec un boulier asiatique (soroban) 🧮

Montre :
1. Rendu SVG d'un nombre sur le boulier
2. Séquence de pratique (lecture de nombres)
3. Addition visuelle pas à pas avec mouvement de billes
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from human_skills import abacus

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 50)
print("🧮 Apprendre à compter comme un humain — Soroban")
print("=" * 50)

# ─── 1. Basic number rendering ───
print("\n📐 Exemples de nombres sur le boulier :")
for num in [0, 7, 42, 123, 999, 123456]:
    svg = abacus.render_abacus_svg(num, n_columns=6)
    beads = abacus.number_to_beads(num, 6)
    readback = abacus.beads_to_number(beads)
    print(f"   {num:>6} → billes: {beads} → lu: {readback} {'✅' if readback == num else '❌'}")

# ─── 2. Practice sequence ───
print("\n🎯 Séquence de pratique (lecture) :")
nums = abacus.generate_practice_sequence(n_columns=3, count=5, seed=42)
session = abacus.AbacusPracticeSession(n_columns=3)
result = session.run_sequence(nums, mode="read")
print(f"   {result['correct']}/{result['total']} corrects "
      f"(accuracy: {result['accuracy']:.0%})")

# ─── 3. Visual addition ───
print("\n➕ Addition visuelle pas à pas :")
a, b = 47, 35
steps = abacus.add_on_abacus(a, b, n_columns=4)
print(f"   {a} + {b} = {a+b} en {len(steps)} étapes :")
for step in steps:
    print(f"     Étape {step['step']}: {step['label']}")

# ─── 4. Export SVG files ───
print("\n💾 Export SVG :")

# Number plates
for name, num, cols in [
    ("boulier_zero", 0, 4),
    ("boulier_7", 7, 4),
    ("boulier_42", 42, 4),
    ("boulier_123", 123, 4),
    ("boulier_999", 999, 4),
    ("boulier_123456", 123456, 6),
]:
    svg = abacus.render_abacus_svg(num, n_columns=cols)
    path = os.path.join(OUTPUT_DIR, f"{name}.svg")
    with open(path, "w") as f:
        f.write(svg)
    print(f"   ✅ {name}.svg")

# Addition visualisation HTML
html = abacus.render_addition_svg(47, 35, n_columns=4)
path = os.path.join(OUTPUT_DIR, "addition_47_35.html")
with open(path, "w") as f:
    f.write(html)
print(f"   ✅ addition_47_35.html")

html2 = abacus.render_addition_svg(123, 987, n_columns=4)
path2 = os.path.join(OUTPUT_DIR, "addition_123_987.html")
with open(path2, "w") as f:
    f.write(html2)
print(f"   ✅ addition_123_987.html")

# Anzan practice: plate with multiple numbers
print("\n🧠 Anzan (visualisation mentale) :")
html_parts = ['<!DOCTYPE html><html><head><title>Anzan — nombres au boulier</title>',
              '<style>body{font-family:sans-serif;background:#1a1a2e;color:#eee;margin:20px;text-align:center}',
              '.grid{display:flex;flex-wrap:wrap;gap:10px;justify-content:center}',
              '.card{background:#16213e;border-radius:8px;padding:10px;width:300px}',
              'object{width:100%}</style></head><body>',
              '<h1>🧠 Anzan — Visualisation mentale</h1>',
              '<p>Regarde chaque nombre, puis cache-le. Essaie de le revoir dans ta tête.</p>',
              '<div class="grid">']

for num in [7, 15, 42, 83, 156, 371, 504, 999]:
    svg = abacus.render_abacus_svg(num, n_columns=3)
    svg_path = os.path.join(OUTPUT_DIR, f"anzan_{num}.svg")
    with open(svg_path, "w") as f:
        f.write(svg)
    html_parts.append(
        f'<div class="card"><h2>{num}</h2>'
        f'<object data="anzan_{num}.svg" type="image/svg+xml"></object></div>'
    )

html_parts.append('</div></body></html>')
anzan_path = os.path.join(OUTPUT_DIR, "anzan_practice.html")
with open(anzan_path, "w") as f:
    f.write("\n".join(html_parts))
print(f"   ✅ anzan_practice.html ({8} nombres)")

# ─── Summary ───
print(f"\n{'='*50}")
print("📂 Fichiers générés dans outputs/ :")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f != "alphabet_adult.svg" and f != "alphabet_toddler.svg" and \
       f != "alphabet_comparison.html" and f != "learning_curve.jsonl" and \
       f != "maison_comparaison.html" and f != "maison_cluster_toddler.svg" and \
       f != "maison_cluster_ensoleillee.svg" and f != "perception_gaps.jsonl":
        fpath = os.path.join(OUTPUT_DIR, f)
        print(f"   📄 {f} ({os.path.getsize(fpath)} bytes)")
print(f"\n{'='*50}")
print("🧮 Ouvre outputs/anzan_practice.html ou outputs/addition_47_35.html")
print("dans ton navigateur pour voir le boulier en action !")

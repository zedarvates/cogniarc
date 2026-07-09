"""Démo : lire le boulier visuellement, comme un humain."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from human_skills.abacus_vision import read_abacus_visually, add_visually_on_abacus
from human_skills.abacus import render_abacus_svg

print("=" * 60)
print("👀 Apprendre à lire un boulier — simulation cognitive visuelle")
print("=" * 60)

# ─── 1. Lecture visuelle de nombres ───
print("\n📖 1. Lecture visuelle de nombres SVG :\n")

for num in [7, 42, 123, 999]:
    svg = render_abacus_svg(num, n_columns=4)
    result = read_abacus_visually(svg, n_columns=4)
    status = "✅" if result["number"] == num else "❌"
    print(f"  {status} Nombre attendu: {num} | Lu: {result['number']}")
    # Show first few reasoning lines
    lines = result["reasoning"].split("\n")
    print(f"     {lines[0]}")
    print(f"     {lines[-1]}")
    print()

# ─── 2. Lecture d'un nombre inconnu (déduction) ───
print("\n🔍 2. Deviner un nombre inconnu par inspection visuelle :\n")

import random
rng = random.Random(42)
mystery = rng.randint(1, 9999)
svg = render_abacus_svg(mystery, n_columns=4)
result = read_abacus_visually(svg, n_columns=4)

print(f"  Nombre secret : {mystery}")
print(f"  Lu visuellement : {result['number']}")
print(f"  Résultat : {'✅' if result['number'] == mystery else '❌'}")
print(f"\n  Raisonnement complet :\n{result['reasoning']}")

# ─── 3. Addition visuelle pas à pas ───
print("\n" + "=" * 60)
print("🧮 3. Addition visuelle — simulation cognitive")
print("=" * 60)

a, b = 47, 35
result_add = add_visually_on_abacus(a, b, n_columns=4)
print(f"\n  {a} + {b} = ?")
print(f"  Résultat : {result_add['result']} ({'✅' if result_add['result'] == a+b else '❌'})")
print(f"\n  Étapes ({len(result_add['steps'])} mouvements de billes) :")
for step in result_add['steps']:
    state_str = ' '.join(f"{h*5+e:2d}" for h, e in step['state'])
    print(f"    {step['step']}. {step['action']}")
    if step.get('detail'):
        for line in step['detail'].split('\n'):
            print(f"       {line.strip()}")

# ─── 4. Export HTML de la lecture visuelle ───
print("\n💾 Export...")
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

html = ['<!DOCTYPE html><html><head><title>👀 Lecture visuelle du boulier</title>',
        '<style>',
        'body{font-family:sans-serif;background:#1a1a2e;color:#eee;margin:20px}',
        '.card{background:#16213e;border-radius:8px;padding:15px;margin:10px 0}',
        '.reasoning{background:#0f3460;padding:10px;border-radius:4px;white-space:pre-wrap;font-family:monospace;font-size:0.85em}',
        'object{max-width:500px;display:block;margin:10px 0}',
        '</style></head><body>',
        '<h1>👀 Lecture visuelle du boulier</h1>',
        '<p>L\'IA regarde le SVG, trouve les billes rouges, les compte par colonne, et lit le nombre comme un humain.</p>']

for num in [7, 42, 123, 999, 1234, mystery]:
    svg = render_abacus_svg(num, n_columns=4 if num < 1000 else 6)
    svg_path = os.path.join(output_dir, f"vision_{num}.svg")
    if not os.path.exists(svg_path):
        with open(svg_path, 'w') as f:
            f.write(svg)
    n_cols = 4 if num < 1000 else 6
    result = read_abacus_visually(svg, n_columns=n_cols)
    html.append(f'<div class="card"><h2>Nombre : {num}</h2>'
                f'<object data="vision_{num}.svg" type="image/svg+xml"></object>'
                f'<p>Lu : {result["number"]} '
                f'{"✅" if result["number"] == num else "❌"}</p>'
                f'<div class="reasoning">{result["reasoning"]}</div></div>')

# Addition
html.append('<h2>🧮 Addition visuelle</h2>')
for a, b in [(47, 35), (123, 987)]:
    res = add_visually_on_abacus(a, b, n_columns=4)
    steps_html = '<br>'.join(
        f'<b>{s["step"]}. {s["action"]}</b><br>'
        f'<pre>{s.get("detail", "")}</pre>'
        for s in res['steps']
    )
    html.append(f'<div class="card"><h3>{a} + {b} = {res["result"]}</h3>'
                f'{steps_html}</div>')

html.append('</body></html>')
path = os.path.join(output_dir, "lecture_visuelle_boulier.html")
with open(path, 'w') as f:
    f.write('\n'.join(html))
print(f"   ✅ lecture_visuelle_boulier.html")

print(f"\n✅ Ouvre outputs/lecture_visuelle_boulier.html dans ton navigateur.")

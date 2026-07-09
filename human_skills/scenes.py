"""
Scene composition — arrange shapes into a drawing.

A Scene is a list of (shape_strokes, layer, color) tuples.
Layer = drawing order (0 = background, higher = foreground).
Color = SVG color name or hex.

Provides pre-built scene builders for the cluster house.
"""

from typing import List, Tuple, Optional
from . import shapes as s

# A scene element: (strokes, layer, color)
SceneElement = Tuple[List[List[Tuple[float, float]]], int, str]


# ─── The cluster house scene ─────────────────────────────────────

def build_cluster_house_scene() -> List[SceneElement]:
    """Build the 'maison cluster ensoleillée' scene.

    Layout in [0,1] normalized coordinates:
    - Ground at y≈0.08
    - Main house body: center-left, y=0.08 to 0.50
    - Left module (garage/atelier): attached on the left
    - Right module (extension): attached on the right
    - Roofs above each module
    - Sun top-right
    - Clouds
    - Grass tufts along ground
    """
    elements: List[SceneElement] = []

    # ── Ground line (layer 0) ──
    elements.append((s.wavy_line(0.02, 0.98, 0.08, amplitude=0.01, frequency=5), 0, "#4a7c3f"))

    # ── Grass tufts (layer 0) ──
    elements.append((s.grass_tufts(0.02, 0.98, 0.08, count=20), 0, "#5a9c4f"))

    # ── Main house body (layer 1) ──
    # Center position: x=0.30 to 0.62, y=0.08 to 0.48
    mx, my, mw, mh = 0.30, 0.08, 0.32, 0.40
    elements.append((s.rectangle(mx, my, mw, mh), 1, "#8B4513"))  # brown body

    # ── Left module (layer 1) ──
    # x=0.14 to 0.30, y=0.08 to 0.35
    lx, ly, lw, lh = 0.14, 0.08, 0.16, 0.27
    elements.append((s.rectangle(lx, ly, lw, lh), 1, "#A0522D"))  # sienna

    # ── Right module (layer 1) ──
    # x=0.62 to 0.76, y=0.08 to 0.32
    rx, ry, rw, rh = 0.62, 0.08, 0.14, 0.24
    elements.append((s.rectangle(rx, ry, rw, rh), 1, "#A0522D"))

    # ── Main roof (layer 2) ──
    # Triangle above main body
    roof_peak_x = mx + mw / 2
    roof_peak_y = my + mh + 0.12  # above the body
    elements.append((
        s.triangle(mx, my + mh, roof_peak_x, my + mh + 0.18, mx + mw, my + mh),
        2, "#CD5C5C",  # Indian red
    ))

    # ── Left roof (layer 2) ──
    l_peak_x = lx + lw / 2
    elements.append((
        s.triangle(lx, ly + lh, l_peak_x, ly + lh + 0.12, lx + lw, ly + lh),
        2, "#D2691E",  # Chocolate
    ))

    # ── Right roof (layer 2) ──
    r_peak_x = rx + rw / 2
    elements.append((
        s.triangle(rx, ry + rh, r_peak_x, ry + rh + 0.10, rx + rw, ry + rh),
        2, "#D2691E",
    ))

    # ── Door (layer 3) ──
    # Center door on main body
    door_w, door_h = 0.08, 0.16
    door_x = mx + (mw - door_w) / 2
    elements.append((s.rectangle(door_x, my, door_w, door_h), 3, "#5C4033"))
    # Door knob
    knob_x = door_x + door_w - 0.015
    knob_y = my + door_h * 0.45
    elements.append((s.circle(knob_x, knob_y, 0.008, n_points=8), 3, "#FFD700"))

    # ── Windows (layer 3) ──
    # Main house: left window
    win_w, win_h = 0.06, 0.06
    win1_x = mx + 0.04
    win1_y = my + mh * 0.55
    elements.append((s.rectangle(win1_x, win1_y, win_w, win_h), 3, "#87CEEB"))
    elements.append((s.cross(win1_x, win1_y, win_w, win_h), 3, "#5C4033"))

    # Main house: right window
    win2_x = mx + mw - win_w - 0.04
    win2_y = my + mh * 0.55
    elements.append((s.rectangle(win2_x, win2_y, win_w, win_h), 3, "#87CEEB"))
    elements.append((s.cross(win2_x, win2_y, win_w, win_h), 3, "#5C4033"))

    # Left module: small window
    lw_x = lx + (lw - 0.05) / 2
    lw_y = ly + lh * 0.55
    elements.append((s.rectangle(lw_x, lw_y, 0.05, 0.05), 3, "#87CEEB"))
    elements.append((s.cross(lw_x, lw_y, 0.05, 0.05), 3, "#5C4033"))

    # Right module: small window
    rw_x = rx + (rw - 0.05) / 2
    rw_y = ry + rh * 0.55
    elements.append((s.rectangle(rw_x, rw_y, 0.05, 0.05), 3, "#87CEEB"))
    elements.append((s.cross(rw_x, rw_y, 0.05, 0.05), 3, "#5C4033"))

    # Attic window (main roof)
    attic_x = roof_peak_x - 0.025
    attic_y = my + mh + 0.02
    elements.append((s.circle(attic_x, attic_y, 0.025, n_points=12), 3, "#87CEEB"))
    elements.append((s.cross(attic_x - 0.025, attic_y - 0.025, 0.05, 0.05), 3, "#5C4033"))

    # ── Chimney (layer 2) ──
    chimney_x = mx + mw * 0.75
    chimney_y = my + mh * 0.85
    chimney_w, chimney_h = 0.04, 0.10
    elements.append((s.rectangle(chimney_x, chimney_y, chimney_w, chimney_h), 2, "#8B4513"))
    # Smoke puffs
    import math
    smoke_pts = []
    for i in range(5):
        t = i / 5
        sx = chimney_x + chimney_w / 2 + 0.01 * math.sin(t * math.pi * 3)
        sy = chimney_y + chimney_h + 0.02 + t * 0.06
        smoke_pts.append((sx, sy))
    elements.append(([smoke_pts], 4, "#C0C0C0"))

    # ── Sun (layer 5) ──
    sun_cx, sun_cy = 0.88, 0.82
    elements.append((s.circle(sun_cx, sun_cy, 0.06, n_points=16), 5, "#FFD700"))
    elements.append((s.sun_rays(sun_cx, sun_cy, 0.065, 0.10, n_rays=10), 5, "#FFA500"))

    # ── Light rays to house (layer 4) ──
    elements.append((s.line(sun_cx + 0.03, sun_cy - 0.03, 0.55, 0.50), 4, "#FFD700"))
    elements.append((s.line(sun_cx - 0.02, sun_cy - 0.05, 0.40, 0.45), 4, "#FFD700"))
    elements.append((s.line(sun_cx, sun_cy - 0.06, 0.30, 0.55), 4, "#FFD700"))

    # ── Clouds (layer 5) ──
    # Cloud 1 (left)
    cloud1_cx, cloud1_cy = 0.15, 0.80
    elements.append((s.circle(cloud1_cx, cloud1_cy, 0.04, n_points=10), 5, "#F0F0F0"))
    elements.append((s.circle(cloud1_cx - 0.04, cloud1_cy + 0.01, 0.03, n_points=10), 5, "#F0F0F0"))
    elements.append((s.circle(cloud1_cx + 0.04, cloud1_cy + 0.01, 0.03, n_points=10), 5, "#F0F0F0"))

    # Cloud 2 (right, smaller)
    cloud2_cx, cloud2_cy = 0.72, 0.75
    elements.append((s.circle(cloud2_cx, cloud2_cy, 0.03, n_points=10), 5, "#F0F0F0"))
    elements.append((s.circle(cloud2_cx - 0.03, cloud2_cy + 0.01, 0.025, n_points=10), 5, "#F0F0F0"))
    elements.append((s.circle(cloud2_cx + 0.03, cloud2_cy + 0.01, 0.025, n_points=10), 5, "#F0F0F0"))

    # ── Flower by the house (layer 3) ──
    stem_x = mx + mw + 0.02
    flower_y = 0.08
    elements.append((s.line(stem_x, flower_y, stem_x, flower_y + 0.06), 3, "#228B22"))
    elements.append((s.circle(stem_x, flower_y + 0.06, 0.008, n_points=8), 3, "#FF69B4"))

    return elements

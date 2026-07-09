"""
Primitive shape skeletons for organic drawing.

Each shape is defined as a list of strokes (same as glyphs).
Strokes are lists of (x, y) points normalized to [0, 1].
Can be passed to organic.jitter_strokes() and render_svg.strokes_to_svg().
"""

from typing import List, Tuple

Stroke = List[Tuple[float, float]]
ShapeDef = List[Stroke]


# ─── Line ────────────────────────────────────────────────────────

def line(x1: float, y1: float, x2: float, y2: float) -> ShapeDef:
    """Single straight stroke from (x1,y1) to (x2,y2)."""
    return [[(x1, y1), (x2, y2)]]


# ─── Rectangle ───────────────────────────────────────────────────

def rectangle(x: float, y: float, w: float, h: float) -> ShapeDef:
    """Closed rectangle as four strokes (one per side) for clean jitter.

    Each side is an independent stroke so jitter affects them individually,
    giving a natural hand-drawn look.
    """
    x1, x2 = x, x + w
    y1, y2 = y, y + h
    return [
        [(x1, y1), (x2, y1)],  # top
        [(x2, y1), (x2, y2)],  # right
        [(x2, y2), (x1, y2)],  # bottom
        [(x1, y2), (x1, y1)],  # left
    ]


def rectangle_loop(x: float, y: float, w: float, h: float) -> ShapeDef:
    """Closed rectangle as a single continuous stroke (loop)."""
    x1, x2 = x, x + w
    y1, y2 = y, y + h
    return [[(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]]


# ─── Triangle ────────────────────────────────────────────────────

def triangle(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> ShapeDef:
    """Closed triangle as three independent strokes."""
    return [
        [(x1, y1), (x2, y2)],
        [(x2, y2), (x3, y3)],
        [(x3, y3), (x1, y1)],
    ]


def triangle_loop(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> ShapeDef:
    """Closed triangle as a single continuous stroke (loop)."""
    return [[(x1, y1), (x2, y2), (x3, y3), (x1, y1)]]


# ─── Circle / Ellipse ────────────────────────────────────────────

def ellipse(cx: float, cy: float, rx: float, ry: float, n_points: int = 12) -> ShapeDef:
    """Ellipse/circle as a single closed stroke of n_points around center.

    For a perfect circle: rx == ry.
    Points are evenly distributed around the circumference.
    """
    import math
    pts = []
    for i in range(n_points):
        theta = 2 * math.pi * i / n_points
        px = cx + rx * math.cos(theta)
        py = cy + ry * math.sin(theta)
        pts.append((px, py))
    # Close the loop by repeating start
    pts.append(pts[0])
    return [pts]


def circle(cx: float, cy: float, r: float, n_points: int = 12) -> ShapeDef:
    """Circle shorthand."""
    return ellipse(cx, cy, r, r, n_points)


# ─── Arc ─────────────────────────────────────────────────────────

def arc(cx: float, cy: float, rx: float, ry: float,
        start_deg: float, end_deg: float, n_points: int = 8) -> ShapeDef:
    """Partial ellipse arc.

    Angles in degrees, measured from positive x-axis.
    """
    import math
    start_rad = math.radians(start_deg)
    end_rad = math.radians(end_deg)

    # Ensure we go the short way
    if end_rad < start_rad:
        end_rad += 2 * math.pi

    pts = []
    for i in range(n_points + 1):
        t = start_rad + (end_rad - start_rad) * i / n_points
        px = cx + rx * math.cos(t)
        py = cy + ry * math.sin(t)
        pts.append((px, py))
    return [pts]


# ─── Cross (for window croisillons) ──────────────────────────────

def cross(x: float, y: float, w: float, h: float) -> ShapeDef:
    """Two diagonal lines forming a cross inside a rectangle."""
    return [
        [(x, y), (x + w, y + h)],
        [(x + w, y), (x, y + h)],
    ]


def plus(x: float, y: float, w: float, h: float) -> ShapeDef:
    """Horizontal + vertical lines forming a plus sign."""
    cx, cy = x + w / 2, y + h / 2
    return [
        [(x, cy), (x + w, cy)],
        [(cx, y), (cx, y + h)],
    ]


# ─── Wavy line (for ground) ─────────────────────────────────────

def wavy_line(x1: float, x2: float, y_base: float,
              amplitude: float = 0.02, frequency: float = 6, n_points: int = 20) -> ShapeDef:
    """Sinusoidal wavy line for ground, clouds, etc."""
    import math
    pts = []
    for i in range(n_points + 1):
        t = i / n_points
        x = x1 + (x2 - x1) * t
        y = y_base + amplitude * math.sin(2 * math.pi * frequency * t)
        pts.append((x, y))
    return [pts]


# ─── Grass tufts ─────────────────────────────────────────────────

def grass_tufts(x_start: float, x_end: float, y_base: float,
                count: int = 15, height_range=(0.02, 0.06)) -> ShapeDef:
    """Small vertical marks along the ground line."""
    import random
    strokes = []
    rng = random.Random(42)
    for _ in range(count):
        x = x_start + (x_end - x_start) * rng.random()
        h = height_range[0] + (height_range[1] - height_range[0]) * rng.random()
        strokes.append([(x, y_base), (x, y_base + h)])
    return strokes


# ─── Sun rays ────────────────────────────────────────────────────

def sun_rays(cx: float, cy: float, inner_r: float, outer_r: float,
             n_rays: int = 8) -> ShapeDef:
    """Radiating lines from a sun center."""
    import math
    strokes = []
    for i in range(n_rays):
        theta = 2 * math.pi * i / n_rays
        x1 = cx + inner_r * math.cos(theta)
        y1 = cy + inner_r * math.sin(theta)
        x2 = cx + outer_r * math.cos(theta)
        y2 = cy + outer_r * math.sin(theta)
        strokes.append([(x1, y1), (x2, y2)])
    return strokes

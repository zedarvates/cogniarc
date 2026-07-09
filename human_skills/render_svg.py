"""
SVG rendering — jittered strokes → smooth SVG paths.

Converts point sequences to Catmull-Rom splines (→ cubic Bézier),
with variable stroke width (simulated pen pressure).

No external dependencies — pure string templating.
"""

import math
from typing import List, Tuple, Dict, Optional

# ─── Catmull-Rom → cubic Bézier ──────────────────────────────────

def _catmull_rom_to_bezier(
    p0: Tuple[float, float], p1: Tuple[float, float],
    p2: Tuple[float, float], p3: Tuple[float, float],
    alpha: float = 0.5,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Convert one Catmull-Rom segment to cubic Bézier control points.

    alpha=0.5 = centripetal (default, good for handwriting).
    Returns (p1, cp1, cp2, p2) where p1, p2 are endpoints and cp1, cp2 are control points.
    """
    def _chord_len(p_a, p_b):
        dx = p_b[0] - p_a[0]
        dy = p_b[1] - p_a[1]
        return (dx * dx + dy * dy) ** (alpha / 2.0)

    t0 = 0.0
    t1 = _chord_len(p0, p1) + 1e-10
    t2 = _chord_len(p1, p2) + t1
    t3 = _chord_len(p2, p3) + t2

    def _lerp(a, b, t):
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    # Cubic Bezier control points
    cp1_x = p1[0] + (p2[0] - p0[0]) * (t2 - t1) / (3 * (t2 - t0) + 1e-10) * (t1 - t0) / (t2 - t0 + 1e-10)
    cp1_y = p1[1] + (p2[1] - p0[1]) * (t2 - t1) / (3 * (t2 - t0) + 1e-10) * (t1 - t0) / (t2 - t0 + 1e-10)
    cp1 = (cp1_x, cp1_y)

    cp2_x = p2[0] + (p1[0] - p3[0]) * (t2 - t1) / (3 * (t3 - t1) + 1e-10) * (t2 - t1) / (t3 - t1 + 1e-10)
    cp2_y = p2[1] + (p1[1] - p3[1]) * (t2 - t1) / (3 * (t3 - t1) + 1e-10) * (t2 - t1) / (t3 - t1 + 1e-10)
    cp2 = (cp2_x, cp2_y)

    return p1, cp1, cp2, p2


def _stroke_to_svg_path(
    stroke: List[Tuple[float, float]],
    stroke_width: float = 3.0,
    viewbox_size: float = 400,
    padding: float = 40,
    closed: bool = False,
) -> str:
    """Convert one stroke to an SVG <path> element.

    Uses Catmull-Rom → cubic Bézier interpolation.
    stroke_width = base width (can vary per stroke).
    """
    n = len(stroke)
    if n == 0:
        return ""
    if n == 1:
        x, y = stroke[0]
        sx = x * (viewbox_size - 2 * padding) + padding
        sy = (1 - y) * (viewbox_size - 2 * padding) + padding
        return f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="{stroke_width:.1f}" fill="currentColor"/>'

    # Scale points to viewbox
    scaled = []
    for x, y in stroke:
        sx = x * (viewbox_size - 2 * padding) + padding
        sy = (1 - y) * (viewbox_size - 2 * padding) + padding
        scaled.append((sx, sy))

    # Build cubic Bézier segments
    d_parts: List[str] = []
    for i in range(n - 1):
        p0 = scaled[max(0, i - 1)]
        p1 = scaled[i]
        p2 = scaled[i + 1]
        p3 = scaled[min(n - 1, i + 2)]

        p1_b, cp1, cp2, p2_b = _catmull_rom_to_bezier(p0, p1, p2, p3)

        if i == 0:
            d_parts.append(f"M {p1_b[0]:.2f} {p1_b[1]:.2f}")

        d_parts.append(
            f"C {cp1[0]:.2f} {cp1[1]:.2f} "
            f"{cp2[0]:.2f} {cp2[1]:.2f} "
            f"{p2_b[0]:.2f} {p2_b[1]:.2f}"
        )

    if closed and n > 2:
        # Close path
        d_parts.append("Z")

    return f'<path d="{" ".join(d_parts)}" fill="none" stroke="currentColor" stroke-width="{stroke_width:.2f}" stroke-linecap="round" stroke-linejoin="round"/>'


def strokes_to_svg(
    strokes: List[List[Tuple[float, float]]],
    viewbox_size: float = 400,
    padding: float = 40,
    stroke_width: float = 3.0,
    color: str = "#1a1a2e",
    background: Optional[str] = "#f8f4e8",
    width_px: int = 400,
    height_px: int = 400,
) -> str:
    """Convert multiple strokes to a complete SVG document.

    Each stroke is rendered as a Catmull-Rom smoothed path.
    Variable stroke width is applied per-stroke based on stroke length
    (simulating pen pressure: longer strokes = lighter pressure).
    """
    paths: List[str] = []

    for stroke in strokes:
        if not stroke:
            continue

        # Variable width: longer strokes are slightly thinner (pressure fades)
        total_len = sum(
            math.sqrt((stroke[i][0] - stroke[i-1][0])**2 +
                       (stroke[i][1] - stroke[i-1][1])**2)
            for i in range(1, len(stroke))
        )
        width_factor = max(0.6, min(1.4, 1.0 - total_len * 0.3))
        w = stroke_width * width_factor

        path = _stroke_to_svg_path(stroke, stroke_width=w, viewbox_size=viewbox_size, padding=padding)
        if path:
            paths.append(path)

    bg = f'<rect width="100%" height="100%" fill="{background}"/>' if background else ""

    joined_paths = "\n    ".join(paths)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" height="{height_px}" '
        f'viewBox="0 0 {viewbox_size} {viewbox_size}">\n'
        f'  {bg}\n'
        f'  <g color="{color}">\n'
        f'    {joined_paths}\n'
        f'  </g>\n'
        f'</svg>'
    )

    return svg


def render_glyph_plate(
    glyphs: Dict[str, List[List[Tuple[float, float]]]],
    chars_per_row: int = 8,
    cell_size: float = 80,
    margin: float = 20,
    stroke_width: float = 2.0,
    color: str = "#1a1a2e",
    background: str = "#f8f4e8",
) -> str:
    """Render multiple glyphs onto a single plate SVG (alphabet poster).

    Arranges glyphs in a grid layout, each in its own cell.
    Returns a complete SVG document.
    """
    items = list(glyphs.items())
    n_cols = min(chars_per_row, len(items))
    n_rows = (len(items) + n_cols - 1) // n_cols

    vb = margin * 2 + n_cols * cell_size
    vb_h = margin * 2 + n_rows * cell_size

    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{int(vb)}" height="{int(vb_h)}" '
        f'viewBox="0 0 {vb} {vb_h}">'
    )
    parts.append(f'<rect width="100%" height="100%" fill="{background}"/>')

    for idx, (char, strokes) in enumerate(items):
        col = idx % n_cols
        row = idx // n_cols
        cx = margin + col * cell_size + cell_size / 2 + cell_size * 0.15
        cy = margin + row * cell_size + cell_size / 2 + cell_size * 0.15

        # Label
        label_size = cell_size * 0.12
        parts.append(
            f'<text x="{cx - cell_size * 0.3}" y="{cy - cell_size * 0.3}" '
            f'font-size="{label_size:.1f}" fill="#888" '
            f'font-family="monospace">{char}</text>'
        )

        # Jitter and render within a sub-viewport
        sub_vb = cell_size * 0.6
        offset_x = cx - sub_vb / 2
        offset_y = cy - sub_vb / 2

        for stroke in strokes:
            scaled = []
            for x, y in stroke:
                sx = offset_x + x * sub_vb
                sy = offset_y + (1 - y) * sub_vb
                scaled.append((sx, sy))

            path = _stroke_to_svg_path(scaled, stroke_width=stroke_width * 0.8, viewbox_size=1, padding=0)
            if path:
                parts.append(f'  <g color="{color}">{path}</g>')

    parts.append("</svg>")
    return "\n".join(parts)

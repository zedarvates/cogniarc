"""
Soroban (Japanese abacus) — learn to count visually like a human.

Each column = one decimal digit position:
  - 1 heavenly bead (value 5) — active = pushed toward the beam
  - 4 earthly beads (value 1 each) — active = pushed toward the beam

Number encoding: each digit is a visual configuration of bead positions.
A human learning to count sees the shape of a number, not just the symbol.

Includes: render SVG, number↔beads, practice generator,
and visual addition simulation (bead movements).
"""

import math
import random
from typing import List, Tuple, Dict, Optional


# ─── Bead positions ──────────────────────────────────────────────

def number_to_beads(number: int, n_columns: int = 6) -> List[Tuple[int, int]]:
    """Convert a number to bead positions per column.

    Returns list of (heavenly, earthly) per column.
    heavenly: 0 or 1 (bead active toward beam)
    earthly: 0-4 (beads active toward beam)

    Columns are most-significant first, left-to-right.
    """
    if number < 0:
        number = 0
    if number >= 10 ** n_columns:
        number = 10 ** n_columns - 1

    digits = [int(d) for d in f"{number:0{n_columns}d}"]
    result = []
    for d in digits:
        heaven = 1 if d >= 5 else 0
        earth = d - (5 if heaven else 0)
        result.append((heaven, earth))
    return result


def beads_to_number(beads: List[Tuple[int, int]]) -> int:
    """Convert bead positions back to a number."""
    result = 0
    for heaven, earth in beads:
        result = result * 10 + (heaven * 5 + earth)
    return result


def bead_positions_for_digit(
    heaven: int, earth: int,
    cx: float,  # column center x
    col_w: float,  # column width
    rod_top: float,  # top of column
    rod_bottom: float,  # bottom of column
    beam_y: float,  # the dividing bar
) -> List[Tuple[float, float, float, float]]:
    """Return (x, y, w, h) for each active bead in this column.

    Heavenly bead: above beam, active = moved DOWN to beam
    Earthly beads: below beam, active = moved UP to beam

    Bead dimensions: slightly smaller than column width
    """
    bead_w = col_w * 0.6
    bead_h = (beam_y - rod_top - 0.02) * 0.6  # heavenly bead size
    bead_h_e = (rod_bottom - beam_y - 0.02) * 0.15  # earthly bead size

    positions = []

    # Heavenly bead
    h_y_range = rod_top + 0.01  # inactive (up, away from beam)
    h_y_beam = beam_y - bead_h - 0.01  # active (down, at beam)
    h_y = h_y_beam if heaven == 1 else h_y_range
    positions.append((cx - bead_w / 2, h_y, bead_w, bead_h))

    # Earthly beads (4, bottom to top)
    beam_bottom = beam_y + 0.01
    earth_bottom = rod_bottom - bead_h_e - 0.01

    for i in range(4):
        is_active = i < earth  # bottommost beads are activated first
        if is_active:
            # Active beads are pushed UP toward the beam
            e_y = beam_bottom + (3 - i) * bead_h_e * 1.1
        else:
            # Inactive beads sit at the bottom
            e_y = earth_bottom - (3 - i) * bead_h_e * 1.1
        # We only draw active beads (inactive = out of frame, at bottom)
        positions.append((cx - bead_w / 2, e_y, bead_w, bead_h_e))

    return positions


# ─── SVG rendering ───────────────────────────────────────────────

def render_abacus_svg(
    number: int,
    n_columns: int = 6,
    width_px: int = 600,
    height_px: int = 200,
    bead_color: str = "#2c2c2c",
    active_color: str = "#c0392b",  # red for active (counted) beads
    frame_color: str = "#5D4037",
    beam_color: str = "#8D6E63",
    background: str = "#f5f0e1",
) -> str:
    """Render a number on a soroban as SVG.

    Shows the visual configuration of beads that represents the number.
    """
    beads = number_to_beads(number, n_columns)

    vb_w, vb_h = 1.0, 1.0
    margin = 0.08
    rod_top = margin
    rod_bottom = 1.0 - margin
    beam_y = 0.42  # horizontal bar dividing heaven and earth

    col_w = (1.0 - 2 * margin) / n_columns
    label_size = 0.03

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_px}" height="{height_px}" '
        f'viewBox="0 0 {vb_w} {vb_h}">'
        f'<rect width="100%" height="100%" fill="{background}"/>'
    ]

    # Frame
    frame_w = 1.0 - 2 * margin + 0.02
    frame_h = rod_bottom - rod_top + 0.02
    parts.append(
        f'<rect x="{margin - 0.01}" y="{rod_top - 0.01}" '
        f'width="{frame_w}" height="{frame_h}" '
        f'rx="0.02" fill="none" stroke="{frame_color}" stroke-width="0.008"/>'
    )

    # Beam (division bar)
    beam_w = 1.0 - 2 * margin + 0.01
    parts.append(
        f'<rect x="{margin - 0.005}" y="{beam_y - 0.003}" '
        f'width="{beam_w}" height="0.006" fill="{beam_color}"/>'
    )

    # Rods (vertical lines per column)
    for i in range(n_columns):
        cx = margin + (i + 0.5) * col_w
        parts.append(
            f'<line x1="{cx}" y1="{rod_top}" x2="{cx}" y2="{rod_bottom}" '
            f'stroke="{frame_color}" stroke-width="0.004"/>'
        )

        # Rod label (the decimal position)
        label = f'×{10**(n_columns-1-i)}'
        parts.append(
            f'<text x="{cx}" y="{rod_bottom + 0.025}" '
            f'font-size="{label_size}" fill="#999" '
            f'text-anchor="middle" font-family="monospace">{label}</text>'
        )

    # Beads
    for i, (heaven, earth) in enumerate(beads):
        cx = margin + (i + 0.5) * col_w
        positions = bead_positions_for_digit(
            heaven, earth, cx, col_w,
            rod_top, rod_bottom, beam_y,
        )

        for idx, (bx, by, bw, bh) in enumerate(positions):
            is_active = (idx == 0 and heaven == 1) or (idx > 0 and idx - 1 < earth)
            color = active_color if is_active else "#ddd"

            # Rounded bead
            r = min(bw, bh) * 0.3
            parts.append(
                f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" '
                f'rx="{r}" fill="{color}" '
                f'stroke="none"/>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


# ─── Anzan (mental visualization) practice ───────────────────────

def generate_practice_sequence(
    n_columns: int = 3,
    count: int = 10,
    seed: int = 42,
) -> List[int]:
    """Generate a sequence of random numbers for practice.

    Numbers are within the range of n_columns digits.
    """
    rng = random.Random(seed)
    max_val = 10 ** n_columns
    return [rng.randint(0, max_val - 1) for _ in range(count)]


class AbacusPracticeSession:
    """Practice reading numbers on the abacus.

    Mode 'read': show abacus, student says the number
    Mode 'place': show number, student places beads
    Mode 'anzan': show number briefly, hide it, student recalls
    """

    def __init__(self, n_columns: int = 3):
        self.n_columns = n_columns
        self.results: List[Dict] = []

    def run_sequence(
        self,
        numbers: List[int],
        mode: str = "read",
        show_time_sec: float = 3.0,
    ) -> Dict:
        """Run a practice sequence.

        In real usage, an LLM or human would look at the SVG output
        and try to answer. Here we simulate correct/incorrect.
        """
        correct = 0
        for num in numbers:
            beads = number_to_beads(num, self.n_columns)
            read_back = beads_to_number(beads)
            is_correct = read_back == num

            svg = None
            if mode == "read" or mode == "anzan":
                svg = render_abacus_svg(num, self.n_columns)
            elif mode == "place":
                pass  # student would draw beads

            self.results.append({
                "number": num,
                "read_back": read_back,
                "correct": is_correct,
                "svg": svg[:100] + "..." if svg else None,
            })

            if is_correct:
                correct += 1

        return {
            "mode": mode,
            "total": len(numbers),
            "correct": correct,
            "accuracy": round(correct / max(1, len(numbers)), 4),
            "avg_beads": beads_to_number(number_to_beads(
                sum(numbers) // max(1, len(numbers)),
                self.n_columns
            )),
        }


# ─── Visual addition (bead movement simulation) ─────────────────

def add_on_abacus(a: int, b: int, n_columns: int = 6) -> List[Dict]:
    """Simulate adding two numbers on a soroban, step by step.

    Returns a list of step dicts, each with the intermediate number
    and the bead configuration SVG.

    This is how a human would do it: starting from A, then adding
    B digit by digit from right to left, moving beads.
    """
    steps = []

    # Step 0: initial number
    steps.append({
        "step": 0,
        "label": f"Position initiale : {a}",
        "number": a,
        "beads": number_to_beads(a, n_columns),
    })

    b_str = str(b)
    current = a
    position = 1  # units, tens, hundreds...

    for i, digit_char in enumerate(reversed(b_str)):
        digit = int(digit_char)
        place = 10 ** (i)

        if digit == 0:
            continue

        # Sub-steps for adding this digit
        step_num = i + 1

        # Add the digit
        old_digit = (current // place) % 10
        new_digit = old_digit + digit

        if new_digit < 10:
            # Simple: just add, no carry
            current += digit * place
            steps.append({
                "step": step_num,
                "label": f"+{digit} à la position des {'unités' if i==0 else f'10^{i}'} : {current}",
                "number": current,
                "beads": number_to_beads(current, n_columns),
                "carry": False,
            })
        else:
            # Need to carry
            carry = new_digit // 10
            new_digit_val = new_digit % 10
            # Add carry digit to next position
            # First, set current digit to remainder
            current += (new_digit_val - old_digit) * place
            steps.append({
                "step": step_num,
                "label": f"+{digit} → retenue ! {current} (reste {new_digit_val}, retient {carry})",
                "number": current,
                "beads": number_to_beads(current, n_columns),
                "carry": False,
                "carry_pending": True,
            })
            # Then propagate carry
            current += carry * place * 10
            steps.append({
                "step": step_num + 0.5,
                "label": f"retenue propagée → {current}",
                "number": current,
                "beads": number_to_beads(current, n_columns),
                "carry": True,
            })

    return steps


def render_addition_svg(a: int, b: int, n_columns: int = 6) -> str:
    """Render the full addition as an HTML visualisation with step-by-step SVGs."""
    steps = add_on_abacus(a, b, n_columns)

    svg_parts = [
        '<!DOCTYPE html>',
        '<html><head><title>Addition au boulier</title>',
        '<style>',
        'body { font-family: sans-serif; background: #1a1a2e; color: #eee; margin: 20px; }',
        '.step { background: #16213e; border-radius: 8px; padding: 10px; margin: 10px 0; }',
        '.step h3 { margin: 0 0 5px 0; color: #c0392b; }',
        '.step p { margin: 0 0 5px 0; color: #aaa; font-size: 0.9em; }',
        'object, img { max-width: 500px; display: block; }',
        '.carry { border-left: 3px solid #f39c12; }',
        '</style></head><body>',
        f'<h1>🧮 {a} + {b} = {a+b}</h1>',
    ]

    for step in steps:
        label = step["label"]
        cls = ' carry' if step.get("carry") else ''
        svg = render_abacus_svg(step["number"], n_columns)
        svg_parts.append(
            f'<div class="step{cls}">'
            f'<h3>Étape {step["step"]}</h3>'
            f'<p>{label}</p>'
            f'{svg}'
            f'</div>'
        )

    svg_parts.append(f'<h2>= {a+b}</h2></body></html>')
    return "\n".join(svg_parts)

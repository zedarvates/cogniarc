"""
Vision reader for the soroban — simulates a human looking at an abacus.

Instead of reading bead data directly (cheating), this module:
1. Parses the SVG visually: looks at bead positions and colors
2. Groups beads by column (x position)
3. Separates heavenly vs earthly beads (y relative to beam)
4. Counts active beads per column
5. Outputs the number + step-by-step reasoning

This is the cognitive visual simulation — seeing the abacus like a human does.
"""

import re
import math
from typing import List, Tuple, Dict, Optional


# ─── SVG visual parser ──────────────────────────────────────────

def _parse_svg_rects(svg: str) -> List[Dict]:
    """Extract all rect/rectangle elements from SVG with their attributes.

    Returns list of {x, y, width, height, fill, rx} dicts.
    """
    rects = []

    # Match <rect ... /> or <rect ...>...</rect>
    pattern = r'<rect\s+([^>]*?)/?>'
    for match in re.finditer(pattern, svg):
        attrs_str = match.group(1)
        attrs = {}
        # Extract x, y, width, height, fill, rx
        for key in ['x', 'y', 'width', 'height', 'fill', 'rx', 'stroke']:
            attr_match = re.search(rf'{key}\s*=\s*"([^"]*)"', attrs_str)
            if attr_match:
                val = attr_match.group(1)
                try:
                    attrs[key] = float(val) if key != 'fill' and key != 'stroke' else val
                except ValueError:
                    attrs[key] = val
        if 'x' in attrs and 'y' in attrs:
            rects.append(attrs)

    return rects


def _find_beam_y(svg: str) -> Optional[float]:
    """Find the beam (division bar) y-position in the SVG."""
    rects = _parse_svg_rects(svg)
    # The beam is a thin rectangle spanning the full width
    for r in rects:
        if r.get('height', 0) < 0.01 and r.get('width', 0) > 0.5:
            return r['y']
    # Fallback: look for line that could be the beam
    line_pattern = r'<line\s+([^>]*?)/?>'
    for match in re.finditer(line_pattern, svg):
        attrs_str = match.group(1)
        y1_match = re.search(r'y1\s*=\s*"([^"]*)"', attrs_str)
        y2_match = re.search(r'y2\s*=\s*"([^"]*)"', attrs_str)
        x1_match = re.search(r'x1\s*=\s*"([^"]*)"', attrs_str)
        x2_match = re.search(r'x2\s*=\s*"([^"]*)"', attrs_str)
        if y1_match and y2_match and x1_match and x2_match:
            y1, y2 = float(y1_match.group(1)), float(y2_match.group(1))
            x1, x2 = float(x1_match.group(1)), float(x2_match.group(1))
            if abs(y1 - y2) < 0.01 and (x2 - x1) > 0.5:
                return y1
    return None


# ─── Human-like abacus reading ──────────────────────────────────

def _group_by_column(rects: List[Dict], n_columns: int) -> Dict[int, List[Dict]]:
    """Group bead rects by column based on their x position."""
    if not rects:
        return {}

    xs = [r['x'] for r in rects]
    min_x, max_x = min(xs), max(xs)

    columns: Dict[int, List[Dict]] = {i: [] for i in range(n_columns)}
    col_width = (max_x - min_x) / max(1, n_columns - 1) if n_columns > 1 else max_x - min_x + 1

    for r in rects:
        # Skip non-bead rects (frame, beam — they span full width)
        if r.get('width', 0) > 0.3:
            continue
        # Determine column index
        col = int(round((r['x'] - min_x) / max(col_width, 0.001)))
        col = max(0, min(n_columns - 1, col))
        columns[col].append(r)

    return columns


def _count_beads_visually(
    bead_rects: List[Dict],
    beam_y: float,
    active_color: str = "#c0392b",
) -> Tuple[int, int, str]:
    """Look at beads in one column and figure out the digit.

    Returns (heavenly_count, earthly_count, reasoning_step).

    Heavenly beads: above beam_y, active if color matches active_color
    Earthly beads: below beam_y, active if color matches active_color
    """
    heaven_beads = []
    earth_beads = []

    for r in bead_rects:
        center_y = r['y'] + r.get('height', 0) / 2
        is_active = r.get('fill', '').lower() == active_color

        if center_y < beam_y:
            heaven_beads.append((r, is_active))
        else:
            earth_beads.append((r, is_active))

    # Count active (red) beads
    heaven_active = sum(1 for _, active in heaven_beads if active)
    earth_active = sum(1 for _, active in earth_beads if active)

    # Cap at sensible values (1 heaven max, 4 earth max)
    heaven_active = min(1, heaven_active)
    earth_active = min(4, earth_active)

    digit = heaven_active * 5 + earth_active

    reasoning = (
        f"  • Bandeau ciel : {'1 bille rouge ↓' if heaven_active else 'aucune bille rouge'} → {heaven_active * 5}\n"
        f"  • Bandeau terre : {earth_active} bille{'s' if earth_active > 1 else ''} rouge{'s' if earth_active > 1 else ''} ↑ → {earth_active}\n"
        f"  • → Chiffre = {heaven_active * 5} + {earth_active} = {digit}"
    )

    return heaven_active, earth_active, reasoning


def read_abacus_visually(svg: str, n_columns: int = 6) -> Dict:
    """Read a soroban SVG visually, like a human would.

    This is the cognitive simulation: it looks at the SVG, finds beads,
    counts them by column, and assembles the number.

    Returns step-by-step reasoning and the final number.
    """
    reasoning_lines = [
        "👀 Je regarde le boulier...",
        f"  Je vois {n_columns} colonnes (positions décimales).",
        "",
    ]

    # Step 1: Find the beam
    beam_y = _find_beam_y(svg)
    if beam_y is None:
        reasoning_lines.append("  ⚠️ Je ne trouve pas la barre de séparation.")
        reasoning_lines.append("  Le boulier semble vide ou illisible.")
        return {
            "number": 0,
            "digits": [],
            "reasoning": "\n".join(reasoning_lines),
            "success": False,
        }

    reasoning_lines.append(f"  Barre de séparation trouvée à y={beam_y:.3f}")
    reasoning_lines.append("")

    # Step 2: Parse all rects and group by column
    rects = _parse_svg_rects(svg)
    bead_rects = [r for r in rects if r.get('width', 0) <= 0.3 and r.get('height', 0) < 0.25]
    bead_rects.sort(key=lambda r: r['x'])

    columns = _group_by_column(bead_rects, n_columns)

    # Step 3: Read each column from left (most significant) to right
    digits = []
    column_names = []

    # Determine column place values
    for i in range(n_columns):
        power = n_columns - 1 - i
        if power == 0:
            name = "unités"
        elif power == 1:
            name = "dizaines"
        elif power == 2:
            name = "centaines"
        elif power == 3:
            name = "milliers"
        elif power == 4:
            name = "dizaines de milliers"
        else:
            name = f"10^{power}"
        column_names.append(name)

    for col_idx in range(n_columns):
        col_beads = columns.get(col_idx, [])
        name = column_names[col_idx]
        power = n_columns - 1 - col_idx

        reasoning_lines.append(f"Colonne {col_idx + 1} ({name}) :")

        if not col_beads:
            reasoning_lines.append("  Aucune bille détectée → 0")
            digit = 0
        else:
            heaven, earth, step_reasoning = _count_beads_visually(col_beads, beam_y)
            digit = heaven * 5 + earth
            reasoning_lines.append(step_reasoning)

        digits.append(digit)
        reasoning_lines.append("")

    # Step 4: Assemble the number
    number = 0
    for d in digits:
        number = number * 10 + d

    reasoning_lines.append(f"📝 Lecture complète : {' '.join(str(d) for d in digits)}")
    reasoning_lines.append(f"→ Nombre : {number}")

    return {
        "number": number,
        "digits": digits,
        "reasoning": "\n".join(reasoning_lines),
        "success": True,
        "n_columns_used": len([d for d in digits if d > 0]),
    }


# ─── Visual addition (human-like cognitive sim) ─────────────────

def add_visually_on_abacus(a: int, b: int, n_columns: int = 4) -> Dict:
    """Simulate a human adding two numbers on an abacus.

    Step by step, with visual reasoning at each step:
    1. Set A on the abacus (place beads)
    2. For each digit of B (right to left):
       a. Add earthly beads first
       b. If needed, exchange 5 (heavenly bead)
       c. If needed, carry to next column
    3. Read the result visually
    """
    steps = []
    abacus_state = list(number_to_beads(a, n_columns))

    # Step 0: Initial state
    steps.append({
        "step": 0,
        "action": f"Poser {a} sur le boulier",
        "state": list(abacus_state),
        "number": a,
    })

    b_str = str(b).zfill(n_columns)
    b_digits = [int(c) for c in b_str]

    for col_idx in range(n_columns - 1, -1, -1):  # right to left
        add_val = b_digits[col_idx]
        if add_val == 0:
            continue

        power = n_columns - 1 - col_idx
        place_name = {0: "unités", 1: "dizaines", 2: "centaines", 3: "milliers"}.get(power, f"10^{power}")

        # Current digit in this column
        cur_heaven, cur_earth = abacus_state[col_idx]
        cur_digit = cur_heaven * 5 + cur_earth
        new_digit = cur_digit + add_val

        if new_digit < 10:
            # Simple addition (no carry)
            new_heaven = 1 if new_digit >= 5 else 0
            new_earth = new_digit - (5 if new_heaven else 0)

            steps.append({
                "step": len(steps),
                "action": f"Additionner {add_val} aux {place_name}",
                "detail": (
                    f"  Avant : {cur_digit} ({'ciel=1' if cur_heaven else 'ciel=0'}, "
                    f"{cur_earth} terre)\n"
                    f"  J'ajoute {add_val} bille{'s' if add_val > 1 else ''} en terre\n"
                    f"  → {new_digit} ({'ciel=1' if new_heaven else 'ciel=0'}, "
                    f"{new_earth} terre)"
                ),
                "state": list(abacus_state),
                "number": beads_to_number(abacus_state),
            })

            abacus_state[col_idx] = (new_heaven, new_earth)

            steps.append({
                "step": len(steps),
                "action": f"Résultat intermédiaire",
                "detail": f"  {beads_to_number(abacus_state)}",
                "state": list(abacus_state),
                "number": beads_to_number(abacus_state),
            })

        else:
            # Need to carry
            remainder = new_digit - 10

            # First, add to current column but handle the 5→10 exchange
            # If we go past 10, we reset this column to remainder and carry 1

            # Subtle: human would do earthly first, then heavenly exchange
            # E.g., 8 + 3: earth 3→4? No, 8 = heaven1+earth3, +3 = 11
            # Earth: 3+3=6 → 5 exchange: 5 beads up = 1 heaven, reset earth
            # Earth now: 6-5=1, heaven: 1+1=2 → but 2 heaven = carry!
            # Final: column=1, carry 1

            # Simplified: just set remainder + carry
            new_heaven = 1 if remainder >= 5 else 0
            new_earth = remainder - (5 if new_heaven else 0)

            steps.append({
                "step": len(steps),
                "action": f"Additionner {add_val} aux {place_name} → RETENUE !",
                "detail": (
                    f"  Avant : {cur_digit}\n"
                    f"  +{add_val} = {new_digit} → dépasse 9\n"
                    f"  Je laisse {remainder} dans cette colonne\n"
                    f"  Je retiens 1 pour la colonne suivante"
                ),
                "state": list(abacus_state),
                "number": beads_to_number(abacus_state),
            })

            abacus_state[col_idx] = (new_heaven, new_earth)

            steps.append({
                "step": len(steps),
                "action": f"Propager la retenue",
                "detail": f"  → {beads_to_number(abacus_state)}",
                "state": list(abacus_state),
                "number": beads_to_number(abacus_state),
            })

            # Propagate carry
            carry_col = col_idx - 1
            while carry_col >= 0:
                ch, ce = abacus_state[carry_col]
                cd = ch * 5 + ce + 1  # add the carry
                if cd < 10:
                    abacus_state[carry_col] = (
                        1 if cd >= 5 else 0,
                        cd - (5 if cd >= 5 else 0)
                    )
                    steps.append({
                        "step": len(steps),
                        "action": f"Retenue ajoutée à la colonne {carry_col + 1}",
                        "detail": f"  → {beads_to_number(abacus_state)}",
                        "state": list(abacus_state),
                        "number": beads_to_number(abacus_state),
                    })
                    break
                else:
                    # Cascade carry
                    abacus_state[carry_col] = (0, 0)  # reset
                    steps.append({
                        "step": len(steps),
                        "action": f"Cascade ! Colonne {carry_col + 1} déborde aussi",
                        "detail": f"  → {beads_to_number(abacus_state)}",
                        "state": list(abacus_state),
                        "number": beads_to_number(abacus_state),
                    })
                    carry_col -= 1

    return {
        "a": a,
        "b": b,
        "result": beads_to_number(abacus_state),
        "steps": steps,
        "reasoning": (
            f"🧮 {a} + {b} = {beads_to_number(abacus_state)}\n"
            f"({len(steps)} mouvements de billes)"
        ),
    }


# ─── Reuse bead conversion helpers ───────────────────────────────

def number_to_beads(number: int, n_columns: int = 6) -> List[Tuple[int, int]]:
    """Convert a number to bead positions per column.

    Returns list of (heavenly, earthly) per column.
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

#!/usr/bin/env python3
"""
Common utilities for CogniARC — shared across modules.

Centralizes common constants, fallback classes, and utilities
to avoid duplication across modules.
"""

from __future__ import annotations

# ─── GameAction Fallback ───
# Used by modules that need arcengine.GameAction but want to work
# even when arcengine is not available (testing, CI, etc.)

try:
    from arcengine import GameAction as _ArcGameAction
except ImportError:
    class _FallbackGameAction:
        ACTION1 = 1; ACTION2 = 2; ACTION3 = 3; ACTION4 = 4
        ACTION5 = 5; ACTION6 = 6; ACTION7 = 7; RESET = 0
    _ArcGameAction = _FallbackGameAction

# Export for other modules
GameAction = _ArcGameAction


# ─── Sternberg Representation Markers ───
# Unified markers for causal and dependency detection
# English primary, French optional via env var

import os

# English (primary)
CAUSAL_MARKERS_EN = [
    'because', 'therefore', 'causes', 'leads to', 'results in',
    'due to', 'hence', 'consequently',
]
DEPENDENCY_MARKERS_EN = [
    'depends on', 'requires', 'needs', 'relies on', 'prerequisite',
    'input', 'output', 'depends_on',
]

# French (optional)
CAUSAL_MARKERS_FR = [
    'parce que', 'donc', 'provoque', 'entraîne', 'résulte en',
]
DEPENDENCY_MARKERS_FR = [
    'nécessite', 'dépend de',
]

# Combined (can be filtered by language preference)
USE_FRENCH_MARKERS = os.environ.get("COGNIARC_FRENCH_MARKERS", "false").lower() == "true"

CAUSAL_MARKERS = CAUSAL_MARKERS_EN + (CAUSAL_MARKERS_FR if USE_FRENCH_MARKERS else [])
DEPENDENCY_MARKERS = DEPENDENCY_MARKERS_EN + (DEPENDENCY_MARKERS_FR if USE_FRENCH_MARKERS else [])


# ─── Grid Hashing ───

def hash_grid(grid) -> str:
    """Hash a numpy grid for state tracking. Returns 16-char hex."""
    import hashlib
    import numpy as np
    if grid is None:
        return hashlib.sha256(b"none").hexdigest()[:16]
    return hashlib.sha256(grid.tobytes()).hexdigest()[:16]


# ─── WorkingMemory Config ───

DEFAULT_WORKING_MEMORY_CAPACITY = 7

def get_working_memory_capacity() -> int:
    """Get WorkingMemory capacity from env var or default. Validates >= 1."""
    raw = os.environ.get("COGNIARC_WM_CAPACITY", str(DEFAULT_WORKING_MEMORY_CAPACITY))
    try:
        cap = int(raw)
    except ValueError:
        raise ValueError(f"COGNIARC_WM_CAPACITY must be integer, got '{raw}'")
    if cap < 1:
        raise ValueError(f"COGNIARC_WM_CAPACITY must be >= 1, got {cap}")
    return cap

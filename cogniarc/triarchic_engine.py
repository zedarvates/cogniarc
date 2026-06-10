#!/usr/bin/env python3
"""
Triarchic Engine — Sternberg's WICS model integrated with CogniARC.

Three intelligences that MUST work together:
  1. ANALYTICAL  — Analyze, evaluate, compare, critique (Doubt + Verify drives)
  2. CREATIVE    — Generate, reframe, question assumptions (Curiosity + Impulse drives)
  3. PRACTICAL   — Translate into real-world action (Simplicity + Caution drives)

The key finding (Sternberg, landmark study): the group that engaged ALL THREE
modes simultaneously outperformed every other group on EVERY measure of actual
achievement — homework, midterm exams, final exams, independent projects.

Implementation: wraps CognitiveDrives and enforces triarchic balance.
"""
from dataclasses import dataclass, field
from typing import Optional
import hashlib, json, numpy as np


@dataclass
class TriarchicState:
    """Current balance of the three Sternberg intelligences."""
    analytical: float = 0.33   # 0..1
    creative: float = 0.33     # 0..1
    practical: float = 0.33    # 0..1

    @property
    def balance_score(self) -> float:
        """How well-balanced are the three modes? 1.0 = perfect balance."""
        vals = [self.analytical, self.creative, self.practical]
        if max(vals) == 0:
            return 0.0
        return min(vals) / max(vals)

    @property
    def dominant_mode(self) -> str:
        """Which mode currently dominates?"""
        modes = {'analytical': self.analytical, 'creative': self.creative, 'practical': self.practical}
        return max(modes, key=lambda k: modes[k])

    @property
    def weakest_mode(self) -> str:
        """Which mode needs activation?"""
        modes = {'analytical': self.analytical, 'creative': self.creative, 'practical': self.practical}
        return min(modes, key=modes.get)

    def recommendation(self) -> str:
        """What should the agent do to rebalance?"""
        if self.balance_score > 0.7:
            return "✅ Triarchic balance OK — all three modes active"

        recs = {
            'analytical': "🔍 Boost ANALYTICAL: pause and critique your own plan. Ask 'What am I missing?'",
            'creative':   "🎨 Boost CREATIVE: reframe the problem entirely. Ask 'Is there a completely different approach?'",
            'practical':  "⚡ Boost PRACTICAL: stop theorizing and act. Ask 'What's the smallest step I can take RIGHT NOW?'"
        }
        return recs[self.weakest_mode]


class TriarchicEngine:
    """
    Enforces Sternberg's triarchic balance on top of CogniARC's CognitiveDrives.

    The genius: cognitive drives already map to WICS, but they can become
    imbalanced (e.g., too much doubt without action = paralysis). This engine
    ensures analytical, creative, and practical modes stay balanced.
    """

    def __init__(self):
        self.state = TriarchicState()
        self.history: list = []
        self.stagnation_counter: int = 0
        self.last_recommendation: str = ""

        # Map CogniARC drives to Sternberg modes
        self.drive_to_mode = {
            'novelty':    'creative',
            'simplicity': 'practical',
            'doubt':      'analytical',
            'pleasure':   'creative',
            'caution':    'analytical',
            'impulse':    'creative',
        }

    def update(self, drive_scores: dict) -> TriarchicState:
        """
        Update triarchic balance from raw cognitive drive scores.

        Args:
            drive_scores: dict like {'novelty': 0.8, 'simplicity': 0.5, ...}
        """
        # Aggregate drives into the three Sternberg modes
        analytical_drives = []
        creative_drives = []
        practical_drives = []

        for drive, score in drive_scores.items():
            mode = self.drive_to_mode.get(drive, 'analytical')
            if mode == 'analytical':
                analytical_drives.append(score)
            elif mode == 'creative':
                creative_drives.append(score)
            else:
                practical_drives.append(score)

        self.state.analytical = float(np.mean(analytical_drives)) if analytical_drives else 0.33
        self.state.creative = float(np.mean(creative_drives)) if creative_drives else 0.33
        self.state.practical = float(np.mean(practical_drives)) if practical_drives else 0.33

        self.last_recommendation = self.state.recommendation()
        self.history.append({
            'analytical': round(self.state.analytical, 3),
            'creative': round(self.state.creative, 3),
            'practical': round(self.state.practical, 3),
            'balance': round(self.state.balance_score, 3),
        })

        # Track stagnation of balance
        if len(self.history) >= 3:
            recent = self.history[-3:]
            if all(abs(h['balance'] - recent[0]['balance']) < 0.05 for h in recent):
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

        return self.state

    def needs_reframe(self) -> bool:
        """Should we question the entire frame? (Douglass pattern)"""
        return (
            self.stagnation_counter > 8 and
            self.state.balance_score < 0.4
        )

    def reframe_prompt(self) -> str:
        """Generate a reframing question."""
        prompts = [
            "What if the problem isn't what we think it is?",
            "What frame am I using that might be wrong?",
            "If I had to solve this with the OPPOSITE approach, what would it be?",
            "What would Faraday/Douglass see that I'm missing?",
        ]
        idx = self.stagnation_counter % len(prompts)
        return prompts[idx]

    def status_report(self) -> str:
        te = self.state
        return (
            f"🧠 Triarchic Balance (Sternberg WICS):\n"
            f"  Analytical: {'█' * int(te.analytical * 20)}{'░' * (20 - int(te.analytical * 20))} {te.analytical:.1%}\n"
            f"  Creative:   {'█' * int(te.creative * 20)}{'░' * (20 - int(te.creative * 20))} {te.creative:.1%}\n"
            f"  Practical:  {'█' * int(te.practical * 20)}{'░' * (20 - int(te.practical * 20))} {te.practical:.1%}\n"
            f"  Balance: {te.balance_score:.1%} | Dominant: {te.dominant_mode} | Need: {te.weakest_mode}\n"
            f"  {self.last_recommendation}"
        )

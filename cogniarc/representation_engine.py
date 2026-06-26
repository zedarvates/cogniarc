#!/usr/bin/env python3
"""
Representation Engine — Sternberg's 3-format exercise (Faraday-inspired).

"Most people use one representation and assume they understand the thing.
But the representation you choose is already deciding what you can and
cannot see." — Sternberg, paraphrasing Faraday's notebooks.

Three formats for any problem:
  1. CAUSAL     — Sequence of causes and effects. What produces what, in what order?
  2. RELATIONAL — Dependency graph. Which elements depend on which?
  3. CORE       — Single sentence capturing the fundamental tension/question.

Usage:
  from representation_engine import RepresentationEngine
  engine = RepresentationEngine()
  result = engine.represent("audit this codebase")
  print(result.to_markdown())
"""
from dataclasses import dataclass
from typing import Optional

from .common import CAUSAL_MARKERS, DEPENDENCY_MARKERS


@dataclass
class Representation:
    """A problem seen through three Sternberg lenses."""
    problem: str
    causal: str        # cause → effect sequence
    relational: str    # dependency graph (text description)
    core_question: str # the fundamental tension

    def to_markdown(self) -> str:
        return (
            f"### 🔗 Causal Chain (cause → effect)\n"
            f"{self.causal}\n\n"
            f"### 🕸️ Relational Map (who depends on what)\n"
            f"{self.relational}\n\n"
            f"### ❓ Core Question (the fundamental tension)\n"
            f"> {self.core_question}\n"
        )

    def to_dict(self) -> dict:
        return {
            'problem': self.problem,
            'causal': self.causal,
            'relational': self.relational,
            'core_question': self.core_question,
        }


class RepresentationEngine:
    """
    Transforms any problem into Sternberg's three complementary representations.

    This is the core exercise from Sternberg's curriculum: Faraday didn't just
    record what happened — he recorded the STRUCTURE of the problem, the
    relationship between variables, the pattern underneath the data. This engine
    forces that same discipline on any problem.

    It uses heuristics + pattern matching, not an LLM. For complex problems,
    each representation can be refined iteratively.
    """

    def represent(self, problem: str) -> Representation:
        """Generate all three representations for a problem."""
        return Representation(
            problem=problem,
            causal=self._infer_causal(problem),
            relational=self._infer_relational(problem),
            core_question=self._infer_core_question(problem),
        )

    def _infer_causal(self, text: str) -> str:
        """Infer a cause-effect chain from the problem description."""
        # Check for explicit causal markers
        for marker in CAUSAL_MARKERS:
            if marker in text.lower():
                parts = text.lower().split(marker, 1)
                return f"1. {parts[0].strip().capitalize()}\n2. → This {marker} {parts[1].strip()}"

        # Generic template for code/technical problems
        if any(kw in text.lower() for kw in ['code', 'bug', 'error', 'fix', 'audit', 'build']):
            return (
                f"1. Current state → identify the gap between expected and actual\n"
                f"2. Root cause → trace the gap to its origin (code, config, dependency)\n"
                f"3. Fix → apply the minimal change that closes the gap\n"
                f"4. Verify → confirm the gap is closed and no new gaps opened"
            )

        # Default: generic inquiry chain
        return (
            f"1. Observation → what do we actually see?\n"
            f"2. Hypothesis → what do we think is happening?\n"
            f"3. Test → what would confirm or disprove the hypothesis?\n"
            f"4. Resolution → act on confirmed understanding"
        )

    def _infer_relational(self, text: str) -> str:
        """Infer a dependency graph from the problem description."""
        deps = []

        for marker in DEPENDENCY_MARKERS:
            if marker in text.lower():
                idx = text.lower().find(marker)
                before = text[:idx].strip().split()[-3:]
                after = text[idx + len(marker):].strip().split()[:3]
                deps.append(f"  {' '.join(after)} ──depends on──→ {' '.join(before)}")

        if deps:
            return "Dependency graph:\n" + "\n".join(deps[:5])

        # Default relational structure
        return (
            "Dependency graph (generic):\n"
            "  Goal ──depends on──→ Available tools\n"
            "  Available tools ──depends on──→ Environment\n"
            "  Environment ──depends on──→ Constraints\n"
            "  Solution ──requires──→ Goal + Tools + Environment"
        )

    def _infer_core_question(self, text: str) -> str:
        """Distill the problem into a single fundamental question."""
        # Look for question marks already
        for sentence in text.replace('?', '?|').split('|'):
            s = sentence.strip()
            if '?' in s:
                return s

        # Pattern-based distillation
        text_lower = text.lower()

        if any(kw in text_lower for kw in ['audit', 'review', 'check']):
            return "What is the smallest change that would have the largest impact on quality?"

        if any(kw in text_lower for kw in ['bug', 'error', 'fix', 'broken']):
            return "What is the root cause, and why didn't we catch it earlier?"

        if any(kw in text_lower for kw in ['build', 'create', 'implement', 'make']):
            return "What is the simplest thing that could possibly work?"

        if any(kw in text_lower for kw in ['learn', 'understand', 'study', 'research']):
            return "What is the fundamental structure I'm missing?"

        # Default: the reframing question
        return f"What question, if answered, would make '{text[:60]}...' irrelevant?"

    def refine(self, representation: Representation, mode: str,
               refinement: str) -> Representation:
        """Refine a specific representation mode with additional insight."""
        if mode == 'causal':
            representation.causal += f"\n+ {refinement}"
        elif mode == 'relational':
            representation.relational += f"\n+ {refinement}"
        elif mode == 'core':
            representation.core_question = refinement
        return representation


# ====== Quick Test ======
if __name__ == "__main__":
    engine = RepresentationEngine()

    problems = [
        "Audit this codebase for security vulnerabilities",
        "The build is broken after the last refactor",
        "I need to implement a new authentication system",
        "Why does my model keep overfitting?",
    ]

    for p in problems:
        rep = engine.represent(p)
        print(f"\n{'='*60}")
        print(rep.to_markdown())

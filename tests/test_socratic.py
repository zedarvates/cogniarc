#!/usr/bin/env python3
"""
Tests for ScientificState and SocraticCritic.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cogniarc import ScientificState, SocraticCritic, SocraticReport, SocraticIssue, SocraticIssueType, SophismType
from cogniarc.scientific_state import Hypothesis, Observation, EvidenceReliability, ActionPlan


# ── ScientificState Tests ──

class TestScientificState:
    def test_create_empty(self):
        s = ScientificState()
        assert s.phase == "discovery"
        assert s.uncertainty == 1.0
        assert s.evidence == []
        assert s.current_hypothesis is None

    def test_create_with_game(self):
        s = ScientificState(game_name="ls20-9607627b", domain_type="movement")
        assert s.game_name == "ls20-9607627b"
        assert s.domain_type == "movement"

    def test_record_observation(self):
        s = ScientificState()
        obs = s.record_observation("Player moves up with action 1", EvidenceReliability.CONFIRMED, "scout")
        assert len(s.evidence) == 1
        assert obs.description == "Player moves up with action 1"
        assert obs.reliability == EvidenceReliability.CONFIRMED

    def test_update_hypothesis(self):
        s = ScientificState()
        h = s.update_hypothesis("Rotate to match goal", confidence=0.8)
        assert s.current_hypothesis is not None
        assert s.current_hypothesis.description == "Rotate to match goal"
        assert s.current_hypothesis.version == 1

        # Second hypothesis archives the first
        h2 = s.update_hypothesis("Navigate to lock", confidence=0.6)
        assert s.current_hypothesis.description == "Navigate to lock"
        assert len(s.refuted_hypotheses) == 1
        assert s.refuted_hypotheses[0].description == "Rotate to match goal"

    def test_refute_hypothesis(self):
        s = ScientificState()
        s.update_hypothesis("Try action 5", confidence=0.5)
        s.refute_current_hypothesis("Action 5 had no effect")
        assert s.current_hypothesis is None
        assert len(s.refuted_hypotheses) == 1
        assert "no effect" in s.refuted_hypotheses[0].refuted_by[0]
        assert s.uncertainty == 1.0  # capped at max

    def test_assumptions(self):
        s = ScientificState()
        s.set_assumption("walls_known", True)
        assert s.get_assumption("walls_known") is True
        assert s.get_assumption("nonexistent") is False
        assert s.get_assumption("nonexistent", True) is True

    def test_plan_lifecycle(self):
        s = ScientificState()
        plan = s.record_plan([1, 2, 3], "Move to target position")
        assert plan.actions == [1, 2, 3]
        assert plan.expected_outcome == "Move to target position"

        s.complete_plan("Reached target", success=True)
        plan = s.current_plan
        assert plan is not None
        assert plan.success is True
        assert plan.actual_outcome == "Reached target"

    def test_report_output(self):
        s = ScientificState(game_name="vc33", domain_type="rotation")
        s.record_observation("Action 6 changes rotation")
        s.update_hypothesis("Cycle action 6 until level completes", confidence=0.6)
        report = s.report()
        assert "ScientificState" in report
        assert "vc33" not in report  # game_name not in report currently
        assert "rotation" in report


# ── SocraticCritic Tests ──

class TestSocraticCritic:
    def test_create(self):
        c = SocraticCritic()
        assert c.interrogation_count == 0

    def test_interrogate_returns_report(self):
        critic = SocraticCritic()
        report = critic.interrogate(
            hypothesis="Try action 6 repeatedly",
            domain_type="rotation",
            available_actions=[6],
            evidence=["Action 6 is available"],
            assumptions={"walls_known": True, "player_found": True, "domain_identified": True, "goal_known": False},
            observations={"stagnation_count": 0, "steps_taken": 5, "last_action_result": ""},
        )
        assert isinstance(report, SocraticReport)
        assert len(report.issues) > 0

    def test_vague_verb_detected(self):
        critic = SocraticCritic()
        report = critic.interrogate(
            hypothesis="Just interact with the target somehow",
            domain_type="unknown",
            available_actions=[],
            evidence=[],
            assumptions={},
            observations={},
        )
        clarifications = [i for i in report.issues if i.type == SocraticIssueType.CLARIFICATION]
        assert len(clarifications) > 0
        assert any("interact" in i.question for i in clarifications)

    def test_stagnation_counterexample(self):
        critic = SocraticCritic()
        report = critic.interrogate(
            hypothesis="Keep trying action 3",
            domain_type="movement",
            available_actions=[1, 2, 3, 4],
            evidence=["Action 1 moves right"],
            assumptions={"walls_known": True, "player_found": True, "domain_identified": True, "goal_known": True},
            observations={"stagnation_count": 10, "steps_taken": 50, "last_action_result": "failed"},
        )
        counterexamples = [i for i in report.issues if i.type == SocraticIssueType.COUNTEREXAMPLE]
        assert len(counterexamples) > 0
        assert any("stagnant" in i.question.lower() for i in counterexamples)

    def test_missing_falsification(self):
        critic = SocraticCritic()
        report = critic.interrogate(
            hypothesis="Navigate to the changer and rotate",
            domain_type="movement",
            available_actions=[1, 2, 3, 4, 6],
            evidence=["Changer exists at (10,20)"],
            assumptions={"walls_known": True, "player_found": True, "domain_identified": True, "goal_known": True},
            observations={},
        )
        falsifications = [i for i in report.issues if i.type == SocraticIssueType.FALSIFICATION]
        assert len(falsifications) > 0

    def test_domain_action_mismatch(self):
        critic = SocraticCritic()
        report = critic.interrogate(
            hypothesis="Walk to the exit",
            domain_type="movement",
            available_actions=[6],  # Only rotation, no movement!
            evidence=[],
            assumptions={},
            observations={},
        )
        constraints = [i for i in report.issues if i.type == SocraticIssueType.PHYSICAL_CONSTRAINT]
        assert len(constraints) > 0
        assert any("movement" in i.question.lower() for i in constraints)

    def test_resolution_tracking(self):
        critic = SocraticCritic()
        report = critic.interrogate(
            hypothesis="Try action 5",
            domain_type="unknown",
            available_actions=[],
            evidence=[],
            assumptions={},
            observations={},
        )
        assert report.resolved_count() == 0
        if report.issues:
            report.issues[0].resolved = True
            report.issues[0].resolution = "Confirmed action 5 is interact"
            assert report.resolved_count() == 1

    def test_quick_check_with_state(self):
        critic = SocraticCritic()
        state = ScientificState(domain_type="movement", available_actions=[1, 2, 3, 4])
        state.set_assumption("walls_known", False)
        state.set_assumption("player_found", True)
        state.set_assumption("domain_identified", True)
        state.set_assumption("goal_known", False)
        state.record_observation("Can move up/down/left/right")

        report = critic.quick_check("Navigate to target and use the switch", state)
        assert len(report.issues) > 0

    def test_blocking_severity(self):
        critic = SocraticCritic()
        report = critic.interrogate(
            hypothesis="Walk north to exit",
            domain_type="movement",
            available_actions=[6],  # CRITICAL: no movement actions
            evidence=[],
            assumptions={},
            observations={},
        )
        assert report.blocking is True  # Should flag movement without movement actions


# ═══ Web Anti-Sophism Tests ═══

class TestWebAntiSophism:

    MARKETING_TEXT = """
    Les experts sont unanimes : ce produit est le numéro 1 mondial.
    Des millions de clients lui font confiance. C'est la seule solution
    pour perdre du poids rapidement. Ce révolutionnaire produit change la vie.
    """

    SCIENTIFIC_TEXT = """
    According to Smith et al. (2024), the reaction rate increases by 23%
    when temperature reaches 45 degrees C. The methodology used a randomized
    controlled trial with 500 participants. Published March 2024 in Nature.
    """

    SPONSORED_TEXT = """
    J'utilise NordVPN depuis 2 ans. Lien d'affiliation en description.
    Meilleur VPN du marché. Top 1 VPN 2026 selon les experts.
    """

    def test_detect_sophisms_marketing(self):
        critic = SocraticCritic()
        sophisms = critic.detect_sophisms(self.MARKETING_TEXT)
        assert len(sophisms) > 0
        types = [s["name"] for s in sophisms]
        assert "BANDWAGON" in types  # "numéro 1 mondial"
        assert "NON_FALSIFIABLE" in types  # "révolutionnaire"

    def test_detect_sophisms_scientific_clean(self):
        critic = SocraticCritic()
        sophisms = critic.detect_sophisms(self.SCIENTIFIC_TEXT)
        assert len(sophisms) == 0  # Clean scientific text

    def test_detect_product_placement_brands(self):
        critic = SocraticCritic()
        placements = critic.detect_product_placement(self.SPONSORED_TEXT)
        types = [p["type"] for p in placements]
        assert "product_placement" in types  # NordVPN, affiliation
        assert "brand_praise" in types       # NordVPN + "meilleur"

    def test_detect_product_placement_affiliate(self):
        critic = SocraticCritic()
        text = "Check this deal: https://amazon.fr/dp/B08XYZ"
        placements = critic.detect_product_placement(text)
        assert len(placements) > 0
        assert any("amazon" in p["match"] for p in placements)

    def test_detect_product_placement_clean(self):
        critic = SocraticCritic()
        placements = critic.detect_product_placement(self.SCIENTIFIC_TEXT)
        assert len(placements) == 0  # No brands

    def test_score_epistemic_marketing(self):
        critic = SocraticCritic()
        sophisms = critic.detect_sophisms(self.MARKETING_TEXT)
        score = critic.score_epistemic_confidence(
            self.MARKETING_TEXT, sophisms=sophisms
        )
        assert score < 0.5  # Low confidence

    def test_score_epistemic_scientific(self):
        critic = SocraticCritic()
        score = critic.score_epistemic_confidence(self.SCIENTIFIC_TEXT)
        assert score >= 0.5  # Neutral to high confidence

    def test_analyze_web_source_marketing(self):
        critic = SocraticCritic()
        report = critic.analyze_web_source(
            self.MARKETING_TEXT, "https://example.com/spam"
        )
        assert len(report.issues) > 0
        sophism_issues = [i for i in report.issues if "SOPHISME" in i.question]
        assert len(sophism_issues) > 0

    def test_analyze_web_source_scientific(self):
        critic = SocraticCritic()
        report = critic.analyze_web_source(self.SCIENTIFIC_TEXT)
        assert len(report.issues) > 0
        confidence_issue = [i for i in report.issues if "Confiance" in i.question]
        assert len(confidence_issue) > 0
        assert "🟢" in confidence_issue[0].question or "🟡" in confidence_issue[0].question

    def test_analyze_web_source_sponsored(self):
        critic = SocraticCritic()
        report = critic.analyze_web_source(self.SPONSORED_TEXT)
        placement_issues = [i for i in report.issues if "PLACEMENT" in i.question]
        assert len(placement_issues) > 0

    def test_search_results_ranking(self):
        critic = SocraticCritic()
        results = [
            {"title": "Best VPN", "url": "https://spam.com",
             "description": "Numero 1 mondial, meilleur VPN, experts recommandent"},
            {"title": "RFC 9341 WireGuard", "url": "https://ietf.org",
             "description": "Methodology in section 4. Published March 2024."},
        ]
        scored = critic.analyze_search_results(results)
        assert scored[0]["epistemic_confidence"] >= scored[1]["epistemic_confidence"]

    def test_format_web_analysis(self):
        critic = SocraticCritic()
        report = critic.analyze_web_source(self.MARKETING_TEXT)
        formatted = critic.format_web_analysis(report, "https://example.com")
        assert "Analyse de source" in formatted
        assert "Confiance épistémique" in formatted
        assert "SOPHISME" in formatted or "sophisme" in formatted.lower()

    def test_web_blocking_low_confidence(self):
        critic = SocraticCritic()
        very_spammy = """
        Révolutionnaire ! Incroyable ! Le meilleur produit game-changer
        que vous ayez jamais vu. Unique en son genre. Exceptionnel.
        Les experts du monde entier le recommandent.
        """
        report = critic.analyze_web_source(very_spammy)
        assert report.blocking is True  # Very low confidence

    def test_detect_sophism_types(self):
        """Verify each sophism type can be instantiated."""
        assert SophismType.APPEAL_TO_AUTHORITY.value > 0
        assert SophismType.BANDWAGON != SophismType.FALSE_DILEMMA
        assert len(SophismType) == 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

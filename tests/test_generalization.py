"""Tests for the dev/holdout generalization report.

Uses synthetic SessionResult/GameResult data (duck-typed, no live arc_agi
runtime needed) to verify the report's math and its honesty guardrails:
it must not silently produce a generalization claim when no holdout games
are configured.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniarc.generalization import compute_generalization_report, load_game_sets


@dataclass
class FakeGameResult:
    game_id: str
    solved: bool


@dataclass
class FakeSessionResult:
    games: List[FakeGameResult] = field(default_factory=list)


def test_default_config_has_dev_game_and_no_holdout():
    sets = load_game_sets()
    assert "ls20-9607627b" in sets["dev_games"]
    assert sets["holdout_games"] == []  # honest starting state


def test_no_holdout_games_warns_explicitly():
    sessions = [FakeSessionResult(games=[FakeGameResult("ls20-9607627b", True)])]
    report = compute_generalization_report(sessions, ["ls20-9607627b"], [])
    assert "No holdout games configured" in report
    assert "generalization" in report.lower()


def test_solve_rates_computed_correctly():
    sessions = [FakeSessionResult(games=[
        FakeGameResult("dev-game", True),
        FakeGameResult("dev-game", True),
        FakeGameResult("dev-game", False),
        FakeGameResult("dev-game", True),
        FakeGameResult("holdout-game", True),
        FakeGameResult("holdout-game", False),
    ])]
    report = compute_generalization_report(sessions, ["dev-game"], ["holdout-game"])
    assert "75.0%" in report   # dev: 3/4
    assert "50.0%" in report   # holdout: 1/2


def test_generalization_gap_reported():
    sessions = [FakeSessionResult(games=[
        FakeGameResult("dev-game", True),
        FakeGameResult("holdout-game", False),
    ])]
    report = compute_generalization_report(sessions, ["dev-game"], ["holdout-game"])
    assert "Generalization gap: +100.0%" in report  # 100% dev - 0% holdout


def test_no_data_for_a_set_reports_no_data():
    sessions = [FakeSessionResult(games=[FakeGameResult("dev-game", True)])]
    report = compute_generalization_report(sessions, ["dev-game"], ["never-attempted-game"])
    assert "no data" in report


def test_unclassified_games_are_flagged():
    sessions = [FakeSessionResult(games=[
        FakeGameResult("dev-game", True),
        FakeGameResult("mystery-game", True),
    ])]
    report = compute_generalization_report(sessions, ["dev-game"], [])
    assert "mystery-game" in report
    assert "unclassified" in report.lower()


def test_empty_sessions_no_crash():
    report = compute_generalization_report([], ["dev-game"], ["holdout-game"])
    assert "no data" in report

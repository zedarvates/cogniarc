"""Tests for Phase 1 T1.1 — Perception gap measurement and ObjectTracker counters."""

import numpy as np
import json
import tempfile
import os

from cogniarc.object_perception import ObjectTracker


class TestPerceptionGapCounters:
    def test_object_tracker_tracks_vanished(self):
        """ObjectTracker should count vanished regions."""
        tracker = ObjectTracker()

        before = np.zeros((5, 5), dtype=np.int32)
        before[1:3, 1:3] = 1  # 2x2 region of color 1

        after = np.zeros((5, 5), dtype=np.int32)  # region vanished

        tracker.observe(before, 0, after)
        stats = tracker.perception_gap_stats()
        assert stats["vanished_count"] > 0, "Should detect vanished region"
        assert stats["total_attempts"] > 0

    def test_vanished_rate_with_matches(self):
        """Mix vanished and moving regions."""
        tracker = ObjectTracker()

        # Step 1: region moves (no vanish)
        before1 = np.zeros((5, 5), dtype=np.int32)
        before1[1:3, 1:3] = 1
        after1 = np.zeros((5, 5), dtype=np.int32)
        after1[2:4, 2:4] = 1  # moved

        # Step 2: region vanishes
        before2 = np.zeros((5, 5), dtype=np.int32)
        before2[2:4, 2:4] = 1
        after2 = np.zeros((5, 5), dtype=np.int32)

        # Step 3: no regions at all (both empty)
        before3 = np.zeros((5, 5), dtype=np.int32)
        after3 = np.zeros((5, 5), dtype=np.int32)

        tracker.observe(before1, 1, after1)
        tracker.observe(before2, 2, after2)
        tracker.observe(before3, 3, after3)

        stats = tracker.perception_gap_stats()
        assert stats["vanished_count"] >= 1
        assert stats["total_attempts"] >= 1  # steps 1+2 have regions, step 3 has none
        assert 0 < stats["vanish_rate"] <= 1.0

    def test_perception_gap_stats_structure(self):
        """perception_gap_stats() returns expected keys."""
        tracker = ObjectTracker()
        before = np.zeros((3, 3), dtype=np.int32)
        after = np.zeros((3, 3), dtype=np.int32)
        tracker.observe(before, 0, after)

        stats = tracker.perception_gap_stats()
        assert "vanished_count" in stats
        assert "total_attempts" in stats
        assert "vanish_rate" in stats
        assert "n_observations" in stats

    def test_no_vanished_on_perfect_match(self):
        """When all regions match, vanish_count should be 0."""
        tracker = ObjectTracker()
        before = np.zeros((5, 5), dtype=np.int32)
        before[0:2, 0:2] = 1
        before[3:5, 3:5] = 2

        after = before.copy()  # identical, all should match

        tracker.observe(before, 0, after)
        stats = tracker.perception_gap_stats()
        assert stats["vanished_count"] == 0, (
            f"Expected 0 vanished, got {stats['vanished_count']}"
        )

    def test_color_change_detected_as_vanished(self):
        """A region that changes color should not find a match and counts as vanished."""
        tracker = ObjectTracker()

        before = np.zeros((5, 5), dtype=np.int32)
        before[1:3, 1:3] = 1  # color 1

        after = np.zeros((5, 5), dtype=np.int32)
        after[1:3, 1:3] = 2  # changed to color 2 — no same-color match

        tracker.observe(before, 0, after)
        stats = tracker.perception_gap_stats()
        assert stats["vanished_count"] >= 1, (
            f"Color change should count as vanished, got {stats['vanished_count']}"
        )

    def test_vanished_log_format(self):
        """Vanished log entries contain expected fields."""
        tracker = ObjectTracker()
        before = np.zeros((5, 5), dtype=np.int32)
        before[1:3, 1:3] = 1
        after = np.zeros((5, 5), dtype=np.int32)

        tracker.observe(before, 0, after)

        assert len(tracker.vanished_log) > 0
        entry = tracker.vanished_log[0]
        assert "color" in entry
        assert "area" in entry
        assert "center" in entry
        assert "step" in entry
        assert entry["color"] == 1
        assert entry["step"] == 0

    def test_multiple_observations_accumulate(self):
        """Vanished count accumulates across multiple observe() calls."""
        tracker = ObjectTracker()

        for _ in range(10):
            before = np.zeros((5, 5), dtype=np.int32)
            before[1:3, 1:3] = 1
            after = np.zeros((5, 5), dtype=np.int32)
            tracker.observe(before, 0, after)

        stats = tracker.perception_gap_stats()
        assert stats["vanished_count"] == 10
        assert stats["n_observations"] == 10
        assert stats["total_attempts"] >= 10
        assert stats["vanish_rate"] == 1.0  # all vanished

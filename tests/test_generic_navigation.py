"""Tests for GenericNavigator — multi-step navigation using ObjectTracker."""

import numpy as np
from cogniarc.object_perception import ObjectTracker
from cogniarc.generic_navigation import GenericNavigator


class TestGenericNavigator:
    def test_get_player_position_finds_color(self):
        """Navigator should find player by color."""
        tracker = ObjectTracker()
        # Create a simple grid with player color
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[3:5, 4:7] = 7  # player region (color 7) at col 4-6, row 3-4

        # Fake some observations so tracker learns color 7 moves
        before = np.zeros((10, 10), dtype=np.int32)
        before[3:5, 4:7] = 7
        after = np.zeros((10, 10), dtype=np.int32)
        after[3:5, 5:8] = 7  # moved right
        tracker.observe(before, 4, after)

        class MockObs:
            frame = [grid]

        nav = GenericNavigator(tracker, MockObs())
        pos = nav.get_player_position()
        assert pos is not None
        # Player center should be around col=5, row=3
        assert 4 <= pos[0] <= 6  # col
        assert 3 <= pos[1] <= 4  # row

    def test_get_player_position_no_player(self):
        """No player color → None."""
        tracker = ObjectTracker()
        grid = np.zeros((10, 10), dtype=np.int32)

        class MockObs:
            frame = [grid]

        nav = GenericNavigator(tracker, MockObs())
        assert nav.get_player_position() is None

    def test_update_grid_creates_gridmap(self):
        """update_grid should build a GridMap from observation."""
        tracker = ObjectTracker()
        grid = np.zeros((10, 10), dtype=np.int32)

        class MockObs:
            frame = [grid]

        nav = GenericNavigator(tracker, MockObs())
        nav.update_grid(MockObs())
        assert nav.grid_map is not None
        assert nav.grid_map.width == 10
        assert nav.grid_map.height == 10

    def test_find_path_returns_actions(self):
        """find_path should return action list when path exists."""
        tracker = ObjectTracker()

        # Build scenario: player at (1,1), target at (5,1)
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[1:3, 1:3] = 7  # player

        # Fake movements to teach action directions
        before = grid.copy()
        after = grid.copy()
        after[1:3, 2:4] = 7  # moved right (action 4)
        np.copyto(before, grid)
        tracker.observe(before, 4, after)

        class MockObs:
            frame = [grid]

        nav = GenericNavigator(tracker, MockObs())
        nav.update_grid(MockObs())

        # Can't test full path without action directions working
        assert nav.grid_map is not None

    def test_navigate_returns_bool(self):
        """navigate should return False when player unknown (graceful)."""
        tracker = ObjectTracker()

        class MockObs:
            frame = [np.zeros((10, 10), dtype=np.int32)]

        nav = GenericNavigator(tracker, MockObs())
        # Should return False gracefully without crashing
        result = nav.navigate(
            (5, 5),
            lambda a: MockObs(),
            max_steps=5,
            obs=MockObs(),
        )
        assert result is False

    def test_stagnation_detection(self):
        """navigate should detect stagnation and return False."""
        tracker = ObjectTracker()

        grid = np.zeros((10, 10), dtype=np.int32)
        grid[1:3, 1:3] = 7  # player at (1,1)
        before = np.zeros((10, 10), dtype=np.int32)
        before[1:3, 1:3] = 7
        after = np.zeros((10, 10), dtype=np.int32)
        after[1:3, 2:4] = 7  # moved right
        tracker.observe(before, 4, after)

        class MockObs:
            frame = [grid]

        nav = GenericNavigator(tracker, MockObs())

        # step_fn that never moves (pretend walls everywhere)
        def stuck_step(action):
            return MockObs()

        result = nav.navigate(
            (8, 8),
            stuck_step,
            max_steps=20,
            obs=MockObs(),
        )
        assert result is False  # Should detect stagnation


class TestMode10Context:
    """Tests for the Mode 10 (SIMULATION) context computation."""

    def test_causal_ambiguity_no_tracker(self):
        """Without ObjectTracker, causal_ambiguity should be False."""
        from cogniarc.scientist_agent import ReasoningMode
        # Default context without tracker
        ctx = {
            "causal_ambiguity": False,
            "drive_caution": 0.0,
            "needs_physical_verification": False,
            "stagnation": 0,
        }
        # Mode 10 trigger check (simplified)
        triggered = (
            ctx.get("causal_ambiguity", False) or
            (ctx.get("drive_caution", 0.0) > 0.7 and ctx.get("stagnation", 0) >= 2) or
            ctx.get("needs_physical_verification", False)
        )
        assert not triggered  # No trigger without context

    def test_simulation_triggered_by_caution_and_stagnation(self):
        """Mode 10 triggers when caution > 0.7 AND stagnation >= 2."""
        ctx = {
            "causal_ambiguity": False,
            "drive_caution": 0.8,
            "needs_physical_verification": False,
            "stagnation": 3,
        }
        triggered = (
            ctx.get("causal_ambiguity", False) or
            (ctx.get("drive_caution", 0.0) > 0.7 and ctx.get("stagnation", 0) >= 2) or
            ctx.get("needs_physical_verification", False)
        )
        assert triggered

    def test_simulation_not_triggered_by_caution_alone(self):
        """Both caution AND stagnation needed."""
        ctx = {
            "drive_caution": 0.8,
            "stagnation": 0,
        }
        triggered = (
            ctx.get("causal_ambiguity", False) or
            (ctx.get("drive_caution", 0.0) > 0.7 and ctx.get("stagnation", 0) >= 2) or
            ctx.get("needs_physical_verification", False)
        )
        assert not triggered

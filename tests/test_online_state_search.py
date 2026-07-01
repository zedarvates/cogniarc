"""Tests for the online discrete-state planner (next_probe_or_action /
plan_action_sequence) — the search-based replacement for a hardcoded
"always press action 4 then 3" rotation cycle.

Pure functions over a caller-supplied transition function, so this is fully
testable without a live arc_agi runtime.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniarc.program_synthesis import next_probe_or_action, plan_action_sequence


def cyclic_transition(state, action):
    """Mirrors the LS20 rotation mechanic as described in the README:
    action 4 advances the cycle, action 3 reverses it, mod 4 states."""
    if action == 4:
        return (state + 1) % 4
    if action == 3:
        return (state - 1) % 4
    return state


# ── next_probe_or_action ──────────────────────────────────────────────────────
def test_done_when_already_at_goal():
    assert next_probe_or_action({}, current_state=2, goal_state=2, actions=[3, 4]) == ("done", None)


def test_probes_untried_action_when_nothing_known():
    mode, action = next_probe_or_action({}, current_state=0, goal_state=2, actions=[3, 4])
    assert mode == "probe"
    assert action in (3, 4)


def test_advances_along_a_known_path():
    table = {(0, 4): 1, (1, 4): 2}
    mode, action = next_probe_or_action(table, current_state=0, goal_state=2, actions=[3, 4])
    assert (mode, action) == ("advance", 4)


def test_probes_remaining_untried_action_before_giving_up():
    # action 4 known from state 0 but leads nowhere useful; action 3 untried.
    table = {(0, 4): 3}
    mode, action = next_probe_or_action(table, current_state=0, goal_state=2, actions=[3, 4])
    assert mode == "probe"
    assert action == 3


def test_stuck_when_all_local_actions_tried_and_no_path_found():
    # Every action from state 0 known, none reaches goal (a self-loop world).
    table = {(0, 3): 0, (0, 4): 0}
    mode, action = next_probe_or_action(table, current_state=0, goal_state=1, actions=[3, 4])
    assert (mode, action) == ("stuck", None)


# ── plan_action_sequence (offline reference driver) ───────────────────────────
def test_plans_shortest_path_for_cyclic_rotation():
    actions, table = plan_action_sequence(
        cyclic_transition, start_state=0, goal_state=2, actions=[3, 4], max_actions=20
    )
    # Distance 0->2 is 2 steps either direction on a 4-cycle; must be minimal.
    assert len(actions) == 2
    # Replaying the actions from start must actually reach the goal.
    state = 0
    for a in actions:
        state = cyclic_transition(state, a)
    assert state == 2


def test_already_at_goal_takes_zero_actions():
    actions, table = plan_action_sequence(
        cyclic_transition, start_state=1, goal_state=1, actions=[3, 4]
    )
    assert actions == []


def test_stops_on_stuck_transition_graph():
    def isolated(state, action):
        return state  # nothing ever changes -> unreachable goal
    actions, table = plan_action_sequence(
        isolated, start_state=0, goal_state=1, actions=[3, 4], max_actions=10
    )
    assert len(actions) < 10  # must give up, not spin for max_actions
    assert (0, 3) in table and (0, 4) in table  # tried both before giving up


def test_respects_max_actions_budget():
    # A transition graph requiring more hops than the budget allows.
    def far(state, action):
        return state + 1 if action == 4 else state
    actions, _ = plan_action_sequence(far, start_state=0, goal_state=100, actions=[3, 4], max_actions=5)
    assert len(actions) <= 5


def test_learned_table_is_reusable_across_calls():
    """The whole point: once an edge is learned, later calls don't re-probe it."""
    table = {}
    mode1, a1 = next_probe_or_action(table, 0, 2, [3, 4])
    assert mode1 == "probe"
    table[(0, a1)] = cyclic_transition(0, a1)
    # From the same state with the same goal, the *same* untried action (not
    # the one just learned) should be probed next, not re-probed.
    mode2, a2 = next_probe_or_action(table, 0, 2, [3, 4])
    assert (0, a1) in table
    if mode2 == "probe":
        assert a2 != a1

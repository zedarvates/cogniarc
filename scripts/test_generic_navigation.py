"""Test generic navigation on real ARC-AGI games."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import arc_agi
from cogniarc.object_perception import ObjectTracker
from cogniarc.generic_navigation import GenericNavigator, navigate_to_target_with_tracker

def test_navigation(game_id: str, target_x: int, target_y: int, max_steps: int = 50):
    """Test navigating toward a target on a real game."""
    print(f"\n{'='*50}")
    print(f"🧭 Test navigation on {game_id} → target ({target_x}, {target_y})")
    print('='*50)
    
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    obs = env.reset()
    
    tracker = ObjectTracker()
    navigator = GenericNavigator(tracker, obs)
    
    # Do random exploration to feed ObjectTracker
    print("  🔍 Exploring to feed ObjectTracker...")
    import random
    rng = random.Random(42)
    
    for i in range(20):
        actions = obs.available_actions
        if not actions:
            break
        grid_before = obs.frame[0].copy()
        action = rng.choice(actions)
        obs = env.step(action)
        grid_after = obs.frame[0].copy()
        tracker.observe(grid_before, action, grid_after)
    
    # Check if player found
    pos = navigator.get_player_position()
    pc = tracker.player_color
    print(f"  Player color: {pc}")
    print(f"  Player position: {pos}")
    print(f"  Wall colors: {tracker.wall_colors}")
    
    if pos is None:
        print("  ❌ Cannot find player position")
        return False
    
    # Try to navigate
    navigator.update_grid(obs)
    path = navigator.find_path((target_x, target_y))
    
    if path is None:
        print(f"  ❌ No path found to ({target_x}, {target_y})")
        # Try greedy
        print(f"  Current pos: {pos}")
        print(f"  Grid map exists: {navigator.grid_map is not None}")
        return False
    
    print(f"  🧭 Path: {len(path)} actions")
    print(f"  Actions: {path}")
    
    # Execute
    print(f"  Executing...")
    success = navigator.navigate(
        (target_x, target_y),
        env.step,
        max_steps=max_steps,
        obs=obs,
    )
    
    final_pos = navigator.get_player_position()
    print(f"  Final position: {final_pos}")
    print(f"  Target: ({target_x}, {target_y})")
    print(f"  {'✅' if success else '❌'} Reached target: {success}")
    
    return success


if __name__ == "__main__":
    # Test on wa30 (holdout that was working)
    # Target: center of the grid (32, 32) as a generic target
    test_navigation("wa30", 32, 32, max_steps=30)
    
    # Test on ls20 (known game)
    test_navigation("ls20", 32, 32, max_steps=30)

#!/usr/bin/env python3
"""Benchmark: 10 LS20 runs with persistent world model (fallback encoder for speed).
Measures cross-session learning: does memory accumulation improve solving?"""
import sys, os, time, json
import numpy as np
from pathlib import Path

# Patch WorldModelTool to use fallback encoder (avoid 6s V-JEPA loading per grid)
from cogniarc.world_model import WorldModelConfig
import cogniarc.world_model as wm_module
_original_init = wm_module.WorldModelTool.__init__

def _fast_init(self, config=None, game_id=None):
    config = config or WorldModelConfig(checkpoint_path='/nonexistent/path.pt')
    _original_init(self, config, game_id)

wm_module.WorldModelTool.__init__ = _fast_init

from cogniarc.scientist_agent import ScientistAgent

GAME = 'ls20-9607627b'
RUNS = 10
CACHE = Path.home() / '.cache' / 'cogniarc' / 'world_model'
NPZ_PATH = CACHE / f'{GAME}.npz'

# Clean start
if NPZ_PATH.exists():
    NPZ_PATH.unlink()
    print(f'🧹 Cleaned previous memory for {GAME}')

print(f'🏁 Benchmark: {RUNS} runs on {GAME} (fallback encoder)')
print(f'{"Run":>4} | {"Solved":>6} | {"Steps":>6} | {"Time":>7} | {"WM Mem":>7}')
print('-' * 55)

results = []
for run in range(1, RUNS + 1):
    t0 = time.time()
    
    try:
        agent = ScientistAgent(GAME, enable_world_model=True, enable_benchmark=False)
        result = agent.solve_level(1)
        elapsed = time.time() - t0
        
        wm_mem = agent.world_model.memory_size() if agent.world_model else 0
        
        row = {
            'run': run, 'solved': result, 'steps': agent.steps,
            'time': round(elapsed, 2), 'wm_memory': wm_mem,
        }
        results.append(row)
        
        print(f'{run:4d} | {str(result):>6} | {agent.steps:6d} | {elapsed:6.1f}s | {wm_mem:7d}')
        
    except Exception as e:
        print(f'{run:4d} | {"ERROR":>6} | {str(e)[:40]}')
        results.append({'run': run, 'solved': False, 'steps': 0, 'time': 0, 'wm_memory': 0, 'error': str(e)[:80]})

# Summary
solved_runs = [r for r in results if r['solved']]
print()
print('--- Summary ---')
print(f'Solved: {len(solved_runs)}/{RUNS} ({len(solved_runs)/RUNS*100:.0f}%)')

if len(solved_runs) >= 2:
    times = [r['time'] for r in solved_runs]
    steps = [r['steps'] for r in solved_runs]
    print(f'Avg time: {np.mean(times):.1f}s (±{np.std(times):.1f})')
    print(f'Avg steps: {np.mean(steps):.0f} (±{np.std(steps):.0f})')

if len(results) >= 2:
    first_half = results[:RUNS//2]
    second_half = results[RUNS//2:]
    
    avg_steps_first = np.mean([r['steps'] for r in first_half])
    avg_steps_second = np.mean([r['steps'] for r in second_half])
    solved_first = sum(1 for r in first_half if r['solved'])
    solved_second = sum(1 for r in second_half if r['solved'])
    
    print(f'\nFirst {RUNS//2} runs: {solved_first}/{RUNS//2} solved, avg {avg_steps_first:.0f} steps')
    print(f'Last {RUNS//2} runs:  {solved_second}/{RUNS//2} solved, avg {avg_steps_second:.0f} steps')
    
    if avg_steps_second < avg_steps_first:
        imp = (1 - avg_steps_second / max(1, avg_steps_first)) * 100
        print(f'📉 Steps improved by {imp:.0f}%')
    if solved_second > solved_first:
        print(f'📈 Solve rate improved!')

# Final WM state
if NPZ_PATH.exists():
    size_kb = NPZ_PATH.stat().st_size / 1024
    data = np.load(NPZ_PATH)
    print(f'\n💾 WM file: {size_kb:.0f} KB, {len(data["actions"])} transitions')
    
    from collections import Counter
    action_counts = Counter(int(a) for a in data['actions'])
    print(f'   Actions: {dict(sorted(action_counts.items()))}')

# Save results
results_path = Path.home() / '.cache' / 'cogniarc' / f'benchmark_wm_{GAME}.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

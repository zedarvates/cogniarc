#!/usr/bin/env python3
"""
Benchmark Tracker — Track ARC-AGI-3 solving results across sessions, LLMs, and agents.

Stores results in JSONL for easy querying and cross-session comparison.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict, field


@dataclass
class GameResult:
    """Result for a single game/level attempt."""
    game_id: str
    level: int
    solved: bool
    steps: int
    time_seconds: float
    tokens_used: int = 0
    strategy: str = ""
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass 
class SessionResult:
    """Aggregate results for a session."""
    session_id: str
    llm_model: str
    agent_version: str
    start_time: str
    end_time: Optional[str] = None
    games: List[GameResult] = field(default_factory=list)
    total_tokens: int = 0
    total_time: float = 0.0
    games_solved: int = 0
    games_attempted: int = 0


class BenchmarkTracker:
    """Track and persist ARC-AGI-3 benchmark results."""
    
    def __init__(self, storage_path: str = "~/.cache/cogniarc/benchmarks.jsonl"):
        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[SessionResult] = None
        
    def start_session(self, llm_model: str, agent_version: str = "cogniarc-v1") -> str:
        """Start a new benchmark session."""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = SessionResult(
            session_id=session_id,
            llm_model=llm_model,
            agent_version=agent_version,
            start_time=datetime.now().isoformat()
        )
        return session_id
    
    def record_game(self, game_id: str, level: int, solved: bool, steps: int, 
                    time_seconds: float, tokens_used: int = 0, strategy: str = "", 
                    error: str = "") -> GameResult:
        """Record a single game attempt."""
        if not self.current_session:
            raise RuntimeError("No active session. Call start_session() first.")
        
        result = GameResult(
            game_id=game_id,
            level=level,
            solved=solved,
            steps=steps,
            time_seconds=time_seconds,
            tokens_used=tokens_used,
            strategy=strategy,
            error=error
        )
        
        self.current_session.games.append(result)
        self.current_session.total_tokens += tokens_used
        self.current_session.total_time += time_seconds
        self.current_session.games_attempted += 1
        if solved:
            self.current_session.games_solved += 1
            
        return result

    def flush(self):
        """Persist current session without ending it."""
        if self.current_session:
            self._persist_session(self.current_session)

    def end_session(self) -> SessionResult:
        """End current session and persist."""
        if not self.current_session:
            raise RuntimeError("No active session.")
        
        self.current_session.end_time = datetime.now().isoformat()
        self._persist_session(self.current_session)
        session = self.current_session
        self.current_session = None
        return session
    
    def _persist_session(self, session: SessionResult):
        """Append session to JSONL file."""
        with open(self.storage_path, 'a') as f:
            f.write(json.dumps(asdict(session)) + '\n')
    
    def load_all_sessions(self) -> List[SessionResult]:
        """Load all sessions from storage."""
        if not self.storage_path.exists():
            return []
        
        sessions = []
        with open(self.storage_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    # Reconstruct dataclasses
                    games = [GameResult(**g) for g in data.get('games', [])]
                    data['games'] = games
                    sessions.append(SessionResult(**data))
        return sessions
    
    def get_summary_table(self) -> str:
        """Generate a markdown summary table of all sessions."""
        sessions = self.load_all_sessions()
        if not sessions:
            return "No benchmark data yet."
        
        lines = [
            "# ARC-AGI-3 Benchmark Results",
            "",
            "| Session | LLM Model | Agent Version | Date | Games Solved/Attempted | Total Time | Total Tokens | Solved Games |",
            "|---------|-----------|---------------|------|------------------------|------------|--------------|--------------|"
        ]
        
        for s in sessions:
            start = s.start_time[:10] if s.start_time else "?"
            solved_names = ", ".join([f"{g.game_id} (L{g.level})" for g in s.games if g.solved])
            lines.append(
                f"| {s.session_id} | {s.llm_model} | {s.agent_version} | {start} | "
                f"{s.games_solved}/{s.games_attempted} | {s.total_time:.1f}s | "
                f"{s.total_tokens:,} | {solved_names or '—'} |"
            )
        
        # Aggregate by LLM
        lines.append("")
        lines.append("## Aggregate by LLM Model")
        lines.append("")
        lines.append("| LLM Model | Sessions | Total Games | Solved | Solve Rate | Avg Time/Game | Avg Tokens/Game |")
        lines.append("|-----------|----------|-------------|--------|------------|---------------|-----------------|")
        
        from collections import defaultdict
        by_llm = defaultdict(list)
        for s in sessions:
            by_llm[s.llm_model].append(s)
        
        for llm, sess_list in by_llm.items():
            total_games = sum(s.games_attempted for s in sess_list)
            total_solved = sum(s.games_solved for s in sess_list)
            total_time = sum(s.total_time for s in sess_list)
            total_tokens = sum(s.total_tokens for s in sess_list)
            solve_rate = (total_solved / total_games * 100) if total_games > 0 else 0
            avg_time = total_time / total_games if total_games > 0 else 0
            avg_tokens = total_tokens / total_games if total_games > 0 else 0
            
            lines.append(
                f"| {llm} | {len(sess_list)} | {total_games} | {total_solved} | "
                f"{solve_rate:.1f}% | {avg_time:.1f}s | {avg_tokens:,.0f} |"
            )
        
        # Per-game stats
        lines.append("")
        lines.append("## Per-Game Statistics")
        lines.append("")
        lines.append("| Game ID | Attempts | Solved | Solve Rate | Avg Steps | Avg Time | Strategies Used |")
        lines.append("|---------|----------|--------|------------|-----------|----------|-----------------|")
        
        by_game = defaultdict(list)
        for s in sessions:
            for g in s.games:
                by_game[g.game_id].append(g)
        
        for game_id, games in sorted(by_game.items()):
            attempts = len(games)
            solved = sum(1 for g in games if g.solved)
            solve_rate = (solved / attempts * 100) if attempts > 0 else 0
            avg_steps = sum(g.steps for g in games) / attempts if attempts > 0 else 0
            avg_time = sum(g.time_seconds for g in games) / attempts if attempts > 0 else 0
            strategies = list(set(g.strategy for g in games if g.strategy))
            
            lines.append(
                f"| {game_id} | {attempts} | {solved} | {solve_rate:.1f}% | "
                f"{avg_steps:.1f} | {avg_time:.1f}s | {', '.join(strategies) or '—'} |"
            )
        
        return "\n".join(lines)
    
    def print_summary(self):
        """Print summary to console."""
        print(self.get_summary_table())


# Convenience function
def create_tracker() -> BenchmarkTracker:
    """Create a benchmark tracker with default storage."""
    return BenchmarkTracker()


if __name__ == "__main__":
    # Demo
    tracker = create_tracker()
    tracker.print_summary()
"""
GoalSanityChecker — Détection de wrong-goal loops.

Inspiré par l'interview Tufa Labs (MLST, Juillet 2026):
"if they try once, make it wrong... it's extremely hard to get them out of the loop.
They're not able to see that there's no way that's the actual goal."

4 checks:
  1. Distance au but — l'action rapproche-t-elle du but ?
  2. Action loop — même action × N sans progression ?
  3. SocraticCritic staleness — mêmes problèmes non résolus depuis N itérations ?
  4. Goal plausibility — le goal a-t-il un sens dans le contexte ?

Si 2+ checks échouent → goal invalidation → reset hypothesis → force exploration.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Set
import math


@dataclass
class SanityVerdict:
    """Résultat du check de sanity."""
    sane: bool = True
    reason: str = ""
    failed_checks: List[str] = field(default_factory=list)
    suggested_action: str = "continue"


class GoalSanityChecker:
    """Détecte les wrong-goal loops et force la ré-exploration."""

    def __init__(self, agent):
        self.agent = agent
        # Historique des actions: [(action_num, (x_before, y_before), (x_after, y_after))]
        self._action_history: List[Tuple[int, Tuple[int, int], Tuple[int, int]]] = []
        # Historique des issues SocraticCritic: [(issue_type, iteration)]
        self._critic_issues: List[Tuple[str, int]] = []
        self._iteration = 0
        self._last_goal_check = 0
        # Seuils
        self.loop_threshold = 5       # Même action × N → loop
        self.distance_window = 5       # Vérifier distance sur N dernières actions
        self.critic_stale_threshold = 3  # Même problème × N itérations
        self.goal_check_interval = 3   # Vérifier le goal tous les N échecs de phase

    def record_action(self, action_num: int, pos_before: Tuple[int, int],
                      pos_after: Tuple[int, int]):
        """Enregistre une action avec positions avant/après."""
        self._action_history.append((action_num, pos_before, pos_after))
        # Garder seulement les 20 dernières
        if len(self._action_history) > 20:
            self._action_history = self._action_history[-20:]

    def record_critic_issues(self, issues: List):
        """Enregistre les issues du SocraticCritic pour cette itération."""
        for issue in issues:
            # Extraire le type depuis l'objet issue ou le dict
            issue_type = getattr(issue, 'type', None) or str(issue)[:30]
            self._critic_issues.append((issue_type, self._iteration))
        # Garder seulement les 30 dernières
        if len(self._critic_issues) > 30:
            self._critic_issues = self._critic_issues[-30:]

    def check(self, phase_failed: bool = False) -> SanityVerdict:
        """Vérifie si le goal courant est sain. Appelé après chaque itération de phase.
        
        phase_failed: si True, le seuil de déclenchement est plus bas (1 check suffit).
                      si False, il faut 2+ checks pour invalider (évite les faux positifs)."""
        self._iteration += 1

        # Always run the checks — don't skip just because phase didn't "fail".
        # A phase can succeed technically (burst 10×↓ executed) but be a wrong goal.
        self._last_goal_check = self._iteration

        failed = []
        reasons = []

        # Check 1: Distance au but
        dist_ok, dist_reason = self._check_distance_to_goal()
        if not dist_ok:
            failed.append("distance_to_goal")
            reasons.append(dist_reason)

        # Check 2: Action loop
        loop_ok, loop_reason = self._check_action_loop()
        if not loop_ok:
            failed.append("action_loop")
            reasons.append(loop_reason)

        # Check 3: SocraticCritic staleness
        critic_ok, critic_reason = self._check_critic_staleness()
        if not critic_ok:
            failed.append("critic_staleness")
            reasons.append(critic_reason)

        # Check 4: Goal plausibility
        goal_ok, goal_reason = self._check_goal_plausibility()
        if not goal_ok:
            failed.append("goal_plausibility")
            reasons.append(goal_reason)

        # Décision: seuil dépend de phase_failed
        # phase_failed=True → 1 check suffit (urgence)
        # phase_failed=False → 2+ checks (évite faux positifs sur succès techniques)
        threshold = 1 if phase_failed else 2
        if len(failed) >= threshold:
            return SanityVerdict(
                sane=False,
                reason=" | ".join(reasons),
                failed_checks=failed,
                suggested_action="force_exploration"
            )
        elif len(failed) == 1:
            return SanityVerdict(
                sane=True,
                reason=reasons[0],
                failed_checks=failed,
                suggested_action="warn"
            )

        return SanityVerdict(sane=True)

    # ── Check 1: Distance au but ──────────────────────────────────────────

    def _check_distance_to_goal(self) -> Tuple[bool, str]:
        """Vérifie si les actions récentes réduisent la distance au but."""
        if len(self._action_history) < self.distance_window:
            return True, ""

        agent = self.agent
        target_pos = self._get_current_target()
        if target_pos is None:
            return True, ""

        # Calculer la tendance de distance sur les N dernières actions
        tx, ty = target_pos
        distances = []
        for _, (bx, by), (ax, ay) in self._action_history[-self.distance_window:]:
            dist_before = math.sqrt((tx - bx) ** 2 + (ty - by) ** 2)
            dist_after = math.sqrt((tx - ax) ** 2 + (ty - ay) ** 2)
            distances.append(dist_after - dist_before)  # négatif = rapprochement

        # Si la distance ne diminue jamais → on ne se rapproche pas
        improvements = sum(1 for d in distances if d < 0)
        if improvements == 0 and len(distances) >= self.distance_window:
            last_x, last_y = self._action_history[-1][2]
            avg_dist = abs(tx - last_x) + abs(ty - last_y)
            return False, (
                f"Aucun rapprochement vers le but ({tx},{ty}) "
                f"en {self.distance_window} actions (dist≈{avg_dist:.0f})"
            )

        return True, ""

    def _get_current_target(self) -> Optional[Tuple[int, int]]:
        """Détermine la position cible actuelle."""
        agent = self.agent

        # Chercher le lock dans l'observation courante
        if hasattr(agent, 'obs') and agent.obs and agent.obs.frame:
            grid = agent.obs.frame[0]
            locks = self._find_sprite_by_tag(agent, 'rjlbuycveu')  # tag lock LS20
            if locks:
                return (locks[0].x, locks[0].y)

        # Fallback: position du player (on navigue toujours quelque part)
        if hasattr(agent, 'player') and agent.player:
            return (agent.player.x, agent.player.y)

        return None

    # ── Check 2: Action loop ──────────────────────────────────────────────

    def _check_action_loop(self) -> Tuple[bool, str]:
        """Détecte la même action répétée sans changement de position."""
        if len(self._action_history) < self.loop_threshold:
            return True, ""

        recent = self._action_history[-self.loop_threshold:]
        actions = [a for a, _, _ in recent]
        positions = [(ax, ay) for _, _, (ax, ay) in recent]

        # Même action partout ?
        if len(set(actions)) == 1:
            action = actions[0]
            # Position change-t-elle ?
            unique_positions = set(positions)
            if len(unique_positions) <= 2:  # Peu ou pas de mouvement
                return False, (
                    f"Action {action} répétée {self.loop_threshold}× "
                    f"sans progression (pos={positions[-1]})"
                )

        return True, ""

    # ── Check 3: SocraticCritic staleness ─────────────────────────────────

    def _check_critic_staleness(self) -> Tuple[bool, str]:
        """Détecte si les mêmes problèmes SocraticCritic persistent."""
        if len(self._critic_issues) < self.critic_stale_threshold:
            return True, ""

        # Grouper par type sur les 10 dernières itérations
        recent_iter = self._iteration - 10
        recent_issues = [(t, i) for t, i in self._critic_issues if i >= recent_iter]

        from collections import Counter
        type_counts = Counter(t for t, _ in recent_issues)

        # Un type apparaît 3+ fois ?
        for issue_type, count in type_counts.most_common(3):
            if count >= self.critic_stale_threshold:
                return False, (
                    f"SocraticCritic soulève '{issue_type}' "
                    f"{count}× sans résolution"
                )

        return True, ""

    # ── Check 4: Goal plausibility ────────────────────────────────────────

    def _check_goal_plausibility(self) -> Tuple[bool, str]:
        """Vérifie que le goal a un sens basique dans le contexte."""
        if len(self._action_history) < 5:
            return True, ""

        agent = self.agent
        target_pos = self._get_current_target()
        if target_pos is None:
            return True, ""

        tx, ty = target_pos
        actions = self._action_history[-5:]

        # Vérifier la direction dominante des actions récentes
        # Action 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT (LS20 mapping)
        direction_dy = {1: -5, 2: 5, 3: 0, 4: 0}
        direction_dx = {1: 0, 2: 0, 3: -5, 4: 5}

        total_dy = 0
        total_dx = 0
        for action, _, _ in actions:
            total_dy += direction_dy.get(action, 0)
            total_dx += direction_dx.get(action, 0)

        # Direction vers le but
        px, py = actions[-1][2]  # position actuelle
        need_dy = ty - py  # positif = vers le bas
        need_dx = tx - px  # positif = vers la droite

        # Si l'agent bouge dans la direction OPPOSÉE au but
        if (need_dy < 0 and total_dy > 0) or (need_dy > 0 and total_dy < 0):
            return False, (
                f"Direction opposée au but: besoin dy={need_dy} "
                f"mais actions dy={total_dy} (lock à ({tx},{ty}), "
                f"joueur à ({px},{py}))"
            )

        if (need_dx < 0 and total_dx > 0) or (need_dx > 0 and total_dx < 0):
            return False, (
                f"Direction opposée au but: besoin dx={need_dx} "
                f"mais actions dx={total_dx}"
            )

        return True, ""

    # ── Helpers ───────────────────────────────────────────────────────────

    def _find_sprite_by_tag(self, agent, tag: str) -> List:
        """Cherche les sprites par tag (comme dans scientist_agent_discovery)."""
        try:
            if hasattr(agent, '_find_tagged_sprites'):
                return agent._find_tagged_sprites(tag)
        except Exception:
            pass

        # Fallback: chercher dans le game
        if hasattr(agent, 'game') and agent.game:
            sprites = []
            for attr_name in dir(agent.game):
                obj = getattr(agent.game, attr_name, None)
                if obj is not None and hasattr(obj, 'tag') and obj.tag == tag:
                    sprites.append(obj)
            if sprites:
                return sprites

        return []

    def reset(self):
        """Réinitialise l'historique (après un changement de niveau)."""
        self._action_history = []
        self._critic_issues = []
        self._iteration = 0
        self._last_goal_check = 0

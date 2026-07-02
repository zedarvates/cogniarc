#!/usr/bin/env python3
"""
SocraticCritic — Socratic midwifery for autonomous ARC-AGI-3 reasoning.

Inspired by AHOIS (arXiv:2606.26722) "Socratic physics critic" (Duzhi agent).
The critic does NOT propose solutions — it only questions, exposing:
  - ambiguous definitions
  - unsupported assumptions
  - missing physical constraints
  - incomplete causal chains
  - counterexamples
  - missing falsification criteria

The ScientistAgent must resolve each issue before acting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum, auto


class SocraticIssueType(Enum):
    """Types of issues a Socratic critic can raise."""
    CLARIFICATION = auto()          # "What do you mean by X?"
    ASSUMPTION = auto()             # "You assumed Y without evidence"
    PHYSICAL_CONSTRAINT = auto()    # "Is Z physically possible here?"
    CAUSAL_GAP = auto()             # "What mechanism links A to B?"
    COUNTEREXAMPLE = auto()         # "How do you explain this anomaly?"
    FALSIFICATION = auto()          # "What would prove your theory wrong?"
    INCOMPLETE = auto()             # "Your hypothesis is missing X"


class SophismType(Enum):
    """12 classical logical fallacies / sophisms for web content analysis."""
    APPEAL_TO_AUTHORITY = auto()
    FALSE_DILEMMA = auto()
    SLIPPERY_SLOPE = auto()
    HASTY_GENERALIZATION = auto()
    AD_HOMINEM = auto()
    BANDWAGON = auto()
    FALSE_CAUSE = auto()
    STRAWMAN = auto()
    CHERRY_PICKING = auto()
    APPEAL_TO_NATURE = auto()
    CIRCULAR_REASONING = auto()
    NON_FALSIFIABLE = auto()


@dataclass
class SocraticIssue:
    """A single issue raised by the Socratic critic."""
    type: SocraticIssueType
    question: str                   # The question posed
    context: str                    # What triggered this question
    severity: float = 0.5           # 0.0 (minor) to 1.0 (blocking)
    resolved: bool = False
    resolution: str = ""            # How the agent resolved it

    def __str__(self) -> str:
        icon = {
            SocraticIssueType.CLARIFICATION: "❓",
            SocraticIssueType.ASSUMPTION: "⚠️",
            SocraticIssueType.PHYSICAL_CONSTRAINT: "🔒",
            SocraticIssueType.CAUSAL_GAP: "🔗",
            SocraticIssueType.COUNTEREXAMPLE: "⚡",
            SocraticIssueType.FALSIFICATION: "🎯",
            SocraticIssueType.INCOMPLETE: "🧩",
        }.get(self.type, "❓")
        status = "✅" if self.resolved else "⏳"
        return f"{status} {icon} [{self.type.name}] {self.question} (sev={self.severity:.2f})"


@dataclass
class SocraticReport:
    """Report from a full Socratic interrogation cycle."""
    issues: List[SocraticIssue] = field(default_factory=list)
    hypothesis: str = ""
    domain: str = ""
    blocking: bool = False          # True if any issue has severity > 0.8

    def add(self, issue: SocraticIssue):
        self.issues.append(issue)
        if issue.severity > 0.8 and not self.blocking:
            self.blocking = True

    def unresolved(self) -> List[SocraticIssue]:
        return [i for i in self.issues if not i.resolved]

    def blocking_issues(self) -> List[SocraticIssue]:
        return [i for i in self.issues if i.severity > 0.8 and not i.resolved]

    def summary(self) -> str:
        total = len(self.issues)
        unresolved = len(self.unresolved())
        blocked = len(self.blocking_issues())
        return (f"SocraticReport: {total} issues, {unresolved} unresolved, "
                f"{blocked} blocking | Blocking={self.blocking}")

    def resolved_count(self) -> int:
        return sum(1 for i in self.issues if i.resolved)

    def format_report(self) -> str:
        lines = [
            f"── SocraticCritic Report ──",
            f"  Hypothesis: '{self.hypothesis[:60]}'",
            f"  Domain: {self.domain}",
            f"  Issues: {len(self.issues)} total, {len(self.unresolved())} unresolved",
            f"  Blocking: {self.blocking}",
        ]
        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)


class SocraticCritic:
    """
    Socratic midwifery — questions without proposing solutions.

    Implements the 6 interrogation types from AHOIS (arXiv:2606.26722):
      1. Clarification — "What do you mean by X?"
      2. Assumption exposure — "You assumed Y without evidence"
      3. Physical constraints — "Is Z possible given available actions?"
      4. Causal probing — "What mechanism links A to B?"
      5. Counterexamples — "How do you explain this anomaly?"
      6. Falsification — "What would prove your theory wrong?"
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.interrogation_count = 0

    def interrogate(self, hypothesis: str, domain_type: str,
                    available_actions: List[int],
                    evidence: List[str],
                    assumptions: Dict[str, bool],
                    observations: Dict[str, Any]) -> SocraticReport:
        """
        Full interrogation cycle — runs all 6 checks on a hypothesis.

        Parameters mirror what the ScientistAgent knows at decision time.
        Returns a SocraticReport with all issues found.
        """
        self.interrogation_count += 1
        report = SocraticReport(
            hypothesis=hypothesis,
            domain=domain_type,
        )

        # Run all 6 checks
        report.issues.extend(self._check_clarity(hypothesis))
        report.issues.extend(self._check_assumptions(hypothesis, assumptions))
        report.issues.extend(self._check_physical_constraints(hypothesis, domain_type, available_actions, assumptions))
        report.issues.extend(self._check_causality(hypothesis, evidence))
        report.issues.extend(self._check_counterexamples(hypothesis, observations))
        report.issues.extend(self._check_falsification(hypothesis, domain_type))

        # Update blocking flag
        for issue in report.issues:
            if issue.severity > 0.8:
                report.blocking = True

        if self.verbose:
            print(report.format_report())

        return report

    # ── 1. CLARIFICATION ──

    def _check_clarity(self, hypothesis: str) -> List[SocraticIssue]:
        """
        Check for ambiguous or undefined terms in the hypothesis.

        Looks for weasel words, undefined references, and vague quantifiers.
        """
        issues = []

        # Detect vague verbs
        vague_verbs = {
            "interact": "What specific action constitutes 'interact'? action(5)? touching?",
            "approach": "Does 'approach' mean move towards or reach exactly?",
            "explore": "What exploration strategy? Random? Systematic? Pattern-based?",
            "try": "'Try' what? What's the alternative if this fails?",
            "use": "'Use' how? As a tool? As a target?",
        }

        hyp_lower = hypothesis.lower()
        for verb, question in vague_verbs.items():
            if verb in hyp_lower:
                issues.append(SocraticIssue(
                    type=SocraticIssueType.CLARIFICATION,
                    question=question,
                    context=f"Vague verb '{verb}' in hypothesis: '{hypothesis[:80]}'",
                    severity=0.4,
                ))

        # Detect undefined references
        undefined_indicators = ["it", "that", "this", "there", "somehow"]
        for word in undefined_indicators:
            if f" {word} " in f" {hyp_lower} ":
                # Only flag if the reference is unclear (heuristic: short hypothesis)
                if len(hypothesis.split()) < 15:
                    issues.append(SocraticIssue(
                        type=SocraticIssueType.CLARIFICATION,
                        question=f"Your hypothesis says '{word}' — what specifically does this refer to?",
                        context=f"Undefined reference in hypothesis",
                        severity=0.3,
                    ))
                    break  # One such warning is enough

        return issues

    # ── 2. ASSUMPTION EXPOSURE ──

    def _check_assumptions(self, hypothesis: str,
                           assumptions: Dict[str, bool]) -> List[SocraticIssue]:
        """
        Check which assumptions are unverified or contradicted by evidence.

        Flags assumptions that are still default (False) when they should be
        confirmed before proceeding.
        """
        issues = []

        # Check critical game-playing assumptions
        critical_assumptions = {
            "walls_known": ("wall colors", "You haven't detected wall colors yet — can you navigate safely?"),
            "actions_scouted": ("action semantics", "You haven't verified what each action does"),
            "player_found": ("player reference", "Do you know where the player object is?"),
            "domain_identified": ("domain type", "You haven't classified the game domain"),
            "goal_known": ("goal condition", "Do you know what 'winning' looks like?"),
        }

        for key, (label, question) in critical_assumptions.items():
            if not assumptions.get(key, False):
                issues.append(SocraticIssue(
                    type=SocraticIssueType.ASSUMPTION,
                    question=question,
                    context=f"Unverified critical assumption: {label}",
                    severity=0.7,  # High — these can cause blind exploration
                ))

        # Check for implicit assumptions in hypothesis wording
        hyp_lower = hypothesis.lower()
        if "always" in hyp_lower or "never" in hyp_lower:
            issues.append(SocraticIssue(
                type=SocraticIssueType.ASSUMPTION,
                question="You used an absolute ('always'/'never'). Is this really true in all states?",
                context=f"Absolute statement in hypothesis: '{hypothesis[:80]}'",
                severity=0.5,
            ))

        if "just" in hyp_lower:
            issues.append(SocraticIssue(
                type=SocraticIssueType.ASSUMPTION,
                question="You said 'just' — are you underestimating the complexity here?",
                context=f"Minimizing language in hypothesis: '{hypothesis[:80]}'",
                severity=0.3,
            ))

        return issues

    # ── 3. PHYSICAL CONSTRAINTS ──

    def _check_physical_constraints(self, hypothesis: str,
                                    domain_type: str,
                                    available_actions: List[int],
                                    assumptions: Optional[Dict[str, bool]] = None) -> List[SocraticIssue]:
        """
        Verify that the hypothesis is consistent with available actions and domain.
        """
        issues = []
        assumptions = assumptions or {}

        # Movement actions exist?
        has_movement = any(a in available_actions for a in [1, 2, 3, 4])
        # Rotation isn't always a dedicated action 6 — some games rotate via a
        # different mechanic entirely (e.g. cycling two other actions at a
        # "changer" object, as LS20 does). Assuming action-6-or-nothing here
        # was itself a hardcoded, game-specific assumption; the caller can set
        # assumptions["has_rotation_mechanism"] from whatever it has actually
        # discovered (a rotation-changer mechanic, a known goal rotation, ...)
        # instead of this check silently blocking every non-action-6 game.
        has_rotation = 6 in available_actions or bool(assumptions.get("has_rotation_mechanism", False))

        hyp_lower = hypothesis.lower()

        # Movement hypothesis but no movement actions
        if ("move" in hyp_lower or "walk" in hyp_lower or "go to" in hyp_lower) and not has_movement:
            issues.append(SocraticIssue(
                type=SocraticIssueType.PHYSICAL_CONSTRAINT,
                question="You're planning movement, but no movement actions (1-4) are available. How will you move?",
                context=f"Movement hypothesis with no movement actions. Available: {available_actions}",
                severity=0.9,
            ))

        # Rotation hypothesis but no rotation action
        if ("rotat" in hyp_lower or "turn" in hyp_lower) and not has_rotation:
            issues.append(SocraticIssue(
                type=SocraticIssueType.PHYSICAL_CONSTRAINT,
                question="You're planning rotation, but action 6 is not available. How will you rotate?",
                context=f"Rotation hypothesis with no rotation action. Available: {available_actions}",
                severity=0.9,
            ))

        # Interact hypothesis but no interact action
        if ("interact" in hyp_lower or "use" in hyp_lower) and 5 not in available_actions:
            issues.append(SocraticIssue(
                type=SocraticIssueType.PHYSICAL_CONSTRAINT,
                question="You want to interact/use something, but action 5 is not in available actions.",
                context=f"Interact hypothesis. Available: {available_actions}",
                severity=0.6,
            ))

        # Domain mismatch
        domain_action_map = {
            "movement": [1, 2, 3, 4],
            "rotation": [6],
            "hybrid": [1, 2, 3, 4, 5, 6, 7],
        }
        if domain_type in domain_action_map:
            expected = domain_action_map[domain_type]
            missing = [a for a in expected if a not in available_actions]
            if missing:
                issues.append(SocraticIssue(
                    type=SocraticIssueType.PHYSICAL_CONSTRAINT,
                    question=f"Domain is '{domain_type}' but expected actions {missing} are unavailable.",
                    context=f"Domain-actions mismatch: domain={domain_type}, available={available_actions}",
                    severity=0.5,
                ))

        return issues

    # ── 4. CAUSAL PROBING ──

    def _check_causality(self, hypothesis: str,
                         evidence: List[str]) -> List[SocraticIssue]:
        """
        Check if the hypothesis has a clear causal chain from action to outcome.
        """
        issues = []

        # Check for missing causal links
        causal_gaps = []

        # "Navigate to X and win" — missing the actual winning mechanism
        if "navigate" in hypothesis.lower() and ("win" in hypothesis.lower() or "complete" in hypothesis.lower()):
            causal_gaps.append(
                "You navigate to a position — but does arriving there actually win the level? "
                "What's the mechanism?"
            )

        # "Rotate and then..." — missing what happens after rotation
        if "rotat" in hypothesis.lower() and not any(w in hypothesis.lower() for w in ["lock", "key", "open", "complete", "match", "goal"]):
            causal_gaps.append(
                "You rotate — but what does rotation achieve in terms of level completion? "
                "What changes in the environment?"
            )

        # "Try action X" — no expected outcome
        if hypothesis.lower().startswith("try"):
            causal_gaps.append(
                "You're trying an action without stating what you expect to happen. "
                "What outcome would confirm your hypothesis?"
            )

        # Hypothesis with no "because" / causal link
        causal_markers = ["because", "so that", "which will", "causing", "in order to", "therefore"]
        if not any(m in hypothesis.lower() for m in causal_markers) and len(hypothesis.split()) > 5:
            # Only flag if it's a substantive hypothesis
            if not hypothesis.lower().startswith("try"):
                causal_gaps.append(
                    "Your hypothesis states WHAT you'll do but not WHY it should work. "
                    "What's the causal mechanism?"
                )

        for q in causal_gaps:
            issues.append(SocraticIssue(
                type=SocraticIssueType.CAUSAL_GAP,
                question=q,
                context=f"Causal gap in: '{hypothesis[:80]}'",
                severity=0.6,
            ))

        return issues

    # ── 5. COUNTEREXAMPLES ──

    def _check_counterexamples(self, hypothesis: str,
                               observations: Dict[str, Any]) -> List[SocraticIssue]:
        """
        Check if recent observations contradict the current hypothesis.
        """
        issues = []

        stagnation = observations.get("stagnation_count", 0)
        steps_taken = observations.get("steps_taken", 0)
        last_action_result = observations.get("last_action_result", "")

        # Stagnation is a strong counterexample to the current theory
        if stagnation > 5:
            issues.append(SocraticIssue(
                type=SocraticIssueType.COUNTEREXAMPLE,
                question=f"You've been stagnant for {stagnation} steps. "
                         "If your hypothesis were correct, you'd expect progress. "
                         "What makes you think this theory is still valid?",
                context=f"Stagnation={stagnation} with current hypothesis",
                severity=0.8,
            ))

        # Repeated same action with no effect
        if not last_action_result or "failed" in last_action_result.lower():
            if steps_taken > 10:
                issues.append(SocraticIssue(
                    type=SocraticIssueType.COUNTEREXAMPLE,
                    question="Your last action produced no result. "
                             "Is this consistent with your hypothesis?",
                    context=f"Failed action at step {steps_taken}",
                    severity=0.4,
                ))

        return issues

    # ── 6. FALSIFICATION ──

    def _check_falsification(self, hypothesis: str,
                             domain_type: str) -> List[SocraticIssue]:
        """
        Check if the hypothesis has clearly stated falsification criteria.
        Without these, the agent can never know it's wrong.
        """
        issues = []

        # Generic warning: no falsification criteria
        # We can't check if criteria exist from just the string, but we can
        # flag that most hypotheses lack them
        if len(hypothesis.split()) > 3:  # Substantive enough to check
            falsification_markers = ["if not", "unless", "otherwise", "if wrong", "falsify"]
            has_criteria = any(m in hypothesis.lower() for m in falsification_markers)

            if not has_criteria:
                issues.append(SocraticIssue(
                    type=SocraticIssueType.FALSIFICATION,
                    question="What specific observation would prove your hypothesis wrong? "
                             "Define a falsification criterion.",
                    context=f"No falsification criteria in hypothesis: '{hypothesis[:80]}'",
                    severity=0.5,
                ))

        return issues

    # ── SIMPLIFIED INTERFACE ──

    def quick_check(self, hypothesis: str, state) -> SocraticReport:
        """
        Convience: interrogate using a ScientificState object directly.

        Extracts relevant fields from the state and runs all 6 checks.
        """
        # Build evidence list from state
        evidence_strs = [obs.description for obs in getattr(state, 'evidence', [])]

        # Build observations dict
        observations = {
            "stagnation_count": getattr(state, 'stagnation_count', 0),
            "steps_taken": getattr(state, 'steps_taken', 0),
            "last_action_result": "",
        }

        return self.interrogate(
            hypothesis=hypothesis,
            domain_type=getattr(state, 'domain_type', ''),
            available_actions=getattr(state, 'available_actions', []),
            evidence=evidence_strs,
            assumptions=getattr(state, 'assumptions', {}),
            observations=observations,
        )

    # ═══════════════════════════════════════════════
    # WEB ANTI-SOPHISM EXTENSION
    # ═══════════════════════════════════════════════
    # Détecte les sophismes, placements de produit, et score la confiance
    # épistémique d'un texte web. Pattern AHOIS appliqué à l'information.
    # ═══════════════════════════════════════════════

    # ── 12 Sophism type recognition patterns ──
    SOPHISM_PATTERNS = [
        (SophismType.APPEAL_TO_AUTHORITY, [
            r'\b(experts?|scientifiques?|docteurs?|professeurs?)\s+(disent|affirment|recommandent|prouvent|confirment)\b',
            r'\b(selon|d\'après)\s+(les?\s+)?(experts?|professionnels?|autorités?)\b',
            r'\b(recherches?\s+(montrent?|prouvent?|indiquent?))\b(?!\s*\[)',
            r'\b(spécialistes?|professionnels?)\s+(unanimement|tous|toutes)\b',
        ]),
        (SophismType.FALSE_DILEMMA, [
            r'\b(choix\s+)?(entre|soit)\s+\w+\s+(ou|ou\s+bien)\s+\w+\s*(,|\s+)(sans|il\s+n\'y\s+a\s+pas\s+d\'alternative)\b',
            r'\b(le\s+)?(seul|unique)\s+(moyen|solution|façon|manière)\s+(est|de)\b',
            r'\b(c\'est\s+)?(ça\s+)?(ou\s+)?(rien|ne\s+rien\s+faire)\b',
            r'either\s+\w+\s+or\s+\w+\s*,?\s*(with\s+no|there\s+is\s+no)\s+(alternative|middle|other\s+option)',
        ]),
        (SophismType.SLIPPERY_SLOPE, [
            r'\b(si\s+on\s+)?(commence|accepte|autorise|permet)\s+\w+.*,\s*(ensuite|après|bientôt|finalement)\s+(ce\s+sera|on\s+arrivera|on\s+finira)\b',
            r'\b(pente\s+glissante|slippery\s+slope|effet\s+domino)\b',
        ]),
        (SophismType.HASTY_GENERALIZATION, [
            r'\b(tout\s+le\s+monde|personne|tou(te)?s)\s+(sav(en)?t|disent?|font|pensent?|utilisen?t)\b',
            r'\b(chaque\s+fois|à\s+chaque\s+)(fois\s+)?que\b.*\b(toujours|jamais)\b',
            r'\b(sur\s+)?la\s+base\s+de\s+(un\s+)?(seul|unique)\s+(exemple|cas|témoignage)\b',
            r'\b(all|everyone|nobody|no\s+one)\s+(knows?|says?|does?|uses?)\b',
        ]),
        (SophismType.AD_HOMINEM, [
            r'\b(connaisse?z\s+rien|incompétent|amateur)\b',
            r'\b(au\s+)?(lieu\s+de\s+)?(répondre|discuter)\s*,?\s*(il|elle)\s+(attaque|critique|insulte)\b',
        ]),
        (SophismType.BANDWAGON, [
            r'\b(le\s+)?(plus\s+)?(populaire|vendu|utilisé|téléchargé|recommandé)\b.*\b(france?|mond(?:e|ial)|tous\s+confondu)s?\b',
            r'\b(des\s+)?milliers?\s+de\s+(personnes?|clients?|utilisateurs?)\s+(nous\s+)?(font\s+)?confiance\b',
            r'\b(le\s+)?(numéro\s+1|leader|champion)\s+(mondial|français|européen)\b',
            r'\b(plus\s+de\s+)?\d+\s*(millions?)\s*(de\s+)?(ventes?|clients?|utilisateurs?)\b',
            r'\b(best.?seller|top\s+\d+|classement\s+)\b',
        ]),
        (SophismType.FALSE_CAUSE, [
            r'\b(après\s+\w+\s*,?\s*donc\s+)\b',
            r'\b(puisque\s+\w+\s*a\s+(marché?|fonctionné)\s*,?\s*(alors|donc)\s+)\b',
        ]),
        (SophismType.STRAWMAN, [
            r'\b(vous\s+)?(prétendez?|dites?|croyez?)\s+(que\s+)?.{10,30}\s*(mais\s+en\s+réalité|or\s+|pourtant)\b',
            r'\b(caricatur(e|er)|déformer\s+(les?\s+)?(propos?|arguments?))\b',
        ]),
        (SophismType.CHERRY_PICKING, [
            r'\b(selon\s+)?(certaines?\s+)?(études?|recherches?)\s+(montrent?|indiquent?)\b(?!.*\b(mais|cependant|toutes?)\b)',
        ]),
        (SophismType.APPEAL_TO_NATURE, [
            r'\b(100\s*%\s*)?(naturel|pur|bio|sans\s+chimique)\b.{0,50}\b(donc|voilà)\b.{0,30}\b(meilleur?|supérieur?)\b',
            r'\b(sans\s+produits\s+chimiques?)\b',
            r'\b(all\s+)?(natural|organic|chemical.?free)\b.{0,30}\b(better|safer|healthier)\b',
        ]),
        (SophismType.CIRCULAR_REASONING, [
            r'\b(c\'est?\s+)?vrai?\s+parce\s+que\s+(c\'est?\s+)?vrai?\b',
            r'\b(we\s+)?know\s+this\s+is\s+true\s+because\s+.{0,50}\b(true|proven)\b',
        ]),
        (SophismType.NON_FALSIFIABLE, [
            r'\b(révolutionnair?|révolutionnaire|innovant|unique\s+en\s+son\s+genre|game.?changer)\b',
            r'\b(results?\s+)?(may|might|could|can)\s+vary\b(?!.*\b(but|however|studies)\b)',
            r'\b(life.?changing|breakthrough|ground.?breaking)\b',
        ]),
    ]

    PRODUCT_PLACEMENT_KEYWORDS = [
        r'\b(NordVPN|ExpressVPN|Surfshark|CyberGhost)\b',
        r'\b(Skillshare|Masterclass|Coursera|Udemy)\b',
        r'\b(Hellofresh|Dollar\s*Shave\s*Club|Blue\s*Apron)\b',
        r'\b(Audible|Amazon\s*Prime)\b',
        r'\b(BetterHelp|Talkspace|Calm|Headspace)\b',
        r'\b(Raycon|Bose|Sony\s*WH|AirPods?)\b',
        r'(amazon\.[a-z]+\/(?:dp|gp\/product)\/[\w]+)',
        r'(ref=.*?(?:tag=|spIA))',
        r'(affiliate|aff=(?:\d|\w))',
        r'(partner\.\w+\.\w+\/)',
        r'\b(sponsorisé|sponsor(?:ed|isé)|paid\s+partnership)\b',
        r'\b(contenu\s+?rédactionnel|publicité|annonce|promotion|offre\s+limitée)\b',
        r'\b(en\s+)?(partenariat|collaboration)\s+(avec|commerciale)\b',
        r'\b(lien\s+d\'affiliation)\b',
        r'\b(top\s+\d+\s+des?\s+meilleurs?)\b',
        r'\b(comparatif|comparaison)\s+\d{4}\b.*\b(test|avis|guide)\b',
        r'\b(j\'utilise|j\'ai\s+testé|je\s+recommande)\s+\w+\s+(depuis|pendant)\s+\d+\s*(ans?|mois?)\b',
    ]

    BRAND_LIST = {
        "vpn": ["nordvpn", "expressvpn", "surfshark", "cyberghost", "protonvpn"],
        "formation": ["skillshare", "masterclass", "coursera", "udemy", "brilliant"],
        "repas": ["hellofresh", "dollar shave club", "blue apron"],
        "audio": ["audible", "amazon prime"],
        "bien-être": ["betterhelp", "talkspace", "calm", "headspace"],
        "tech": ["raycon", "bose", "airpods", "sony wh", "anker"],
        "hebergement": ["hostinger", "bluehost", "siteground", "namecheap"],
        "saas": ["notion", "clickup", "asana", "trello", "slack"],
    }

    def detect_sophisms(self, text: str) -> List[dict]:
        """Detect sophisms in text. Returns list of {type, name, match, context, severity, position}."""
        import re
        findings = []
        text_lower = text.lower()

        for sophism_type, patterns in self.SOPHISM_PATTERNS:
            for pattern in patterns:
                for m in re.finditer(pattern, text_lower, re.IGNORECASE):
                    start = max(0, m.start() - 60)
                    end = min(len(text), m.end() + 60)
                    ctx = text[start:end].replace('\\n', ' ').strip()
                    findings.append({
                        "type": sophism_type,
                        "name": sophism_type.name,
                        "match": m.group()[:80],
                        "context": ctx[:120],
                        "severity": 0.6 if sophism_type in (
                            SophismType.APPEAL_TO_AUTHORITY, SophismType.NON_FALSIFIABLE,
                        ) else 0.5,
                        "position": m.start(),
                    })

        # Dedup nearby (same type within 100 chars)
        deduped = []
        for f in findings:
            dup = any(
                f["type"] == e["type"] and abs(f["position"] - e["position"]) < 100
                for e in deduped
            )
            if not dup:
                deduped.append(f)
        return deduped

    def detect_product_placement(self, text: str) -> List[dict]:
        """Detect product placements, affiliate links, sponsored content."""
        import re
        findings = []
        text_lower = text.lower()

        for pattern in self.PRODUCT_PLACEMENT_KEYWORDS:
            for m in re.finditer(pattern, text_lower, re.IGNORECASE):
                start = max(0, m.start() - 50)
                end = min(len(text), m.end() + 50)
                ctx = text[start:end].replace('\\n', ' ').strip()
                findings.append({
                    "type": "product_placement",
                    "match": m.group()[:60],
                    "context": ctx[:100],
                    "severity": 0.7,
                    "position": m.start(),
                })

        # Brand detection + laudative context
        brands_found = set()
        for _, brand_list in self.BRAND_LIST.items():
            for brand in brand_list:
                if brand in text_lower:
                    brands_found.add(brand)
                    idx = text_lower.index(brand)
                    start = max(0, idx - 80)
                    end = min(len(text), idx + len(brand) + 80)
                    ctx = text_lower[start:end]
                    if any(w in ctx for w in ["meilleur", "incroyable", "génial", "top", "parfait",
                                               "best", "amazing", "awesome", "perfect", "incredible"]):
                        findings.append({
                            "type": "brand_praise",
                            "match": brand,
                            "context": ctx[:100],
                            "severity": 0.55,
                            "position": idx,
                        })

        # Brand density penalty
        word_count = len(text.split())
        if word_count > 0 and len(brands_found) / max(1, word_count / 100) > 2:
            findings.append({
                "type": "high_brand_density",
                "match": f"{len(brands_found)} marques pour ~{word_count} mots",
                "context": f"Densité anormale: {len(brands_found)} marques en ~{word_count} mots",
                "severity": 0.6,
                "position": 0,
            })

        return findings

    def score_epistemic_confidence(self, text: str, url: str = "",
                                   sophisms: Optional[List[dict]] = None,
                                   placements: Optional[List[dict]] = None) -> float:
        """
        Score epistemic confidence of a text source (0.0 = pure marketing, 1.0 = verified).

        Factors weighted: citations, data points, author, date, methodology (+),
        sophisms, placements, emotive language, no sources (-).
        """
        import re
        score = 0.5
        text_lower = text.lower()

        # + Citations [1], (Author, 2024)
        citations = len(re.findall(r'\[\d+\]|\(\w+\s+et\s+al\.?\s*,?\s*\d{4}\)|\(\w+,\s*\d{4}\)', text))
        score += min(0.20, citations * 0.04)

        # + Data (%, numbers with units)
        data_pts = len(re.findall(r'\d+\.?\d*\s*%|\d+\.?\d*\s*(€|$|ans|km|kg|g|ml)', text))
        score += min(0.25, data_pts * 0.03)

        # + Named author
        if re.search(r'(par|by|author|écrit par|publié par)\s+[A-Z][a-z]+', text):
            score += 0.10

        # + Date
        if re.search(r'\d{1,2}\s*(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}', text_lower):
            score += 0.10

        # + Methodology
        if re.search(r'(méthodologie|methodology|protocole|protocol|méthode|method)', text_lower):
            score += 0.15

        # - No sources (long text)
        if citations == 0 and data_pts == 0 and len(text.split()) > 200:
            score -= 0.10

        # - Sophisms
        if sophisms:
            types = set(s["type"] for s in sophisms)
            score -= min(0.40, len(types) * 0.08)

        # - Product placements
        if placements:
            unique = len(set(p["match"] for p in placements))
            score -= min(0.35, unique * 0.07)

        # - Emotive language
        emotive = len(re.findall(
            r'(incroyable|extraordinaire|révolutionnaire|exceptionnel|unique|game.?changer|life.?changing|fantastique|incontournable|incroyablement|extraordinairement)',
            text_lower
        ))
        score -= min(0.20, emotive * 0.03)

        return max(0.0, min(1.0, score))

    def analyze_web_source(self, text: str, url: str = "",
                           verbose: bool = False) -> SocraticReport:
        """
        Full web source analysis: sophisms + product placement + epistemic score.

        Returns a SocraticReport with issues for each finding.
        """
        report = SocraticReport(hypothesis="Web source analysis", domain="web")

        # 1. Sophisms
        sophisms = self.detect_sophisms(text)
        for s in sophisms:
            issue_type = {
                SophismType.APPEAL_TO_AUTHORITY: SocraticIssueType.ASSUMPTION,
                SophismType.FALSE_DILEMMA: SocraticIssueType.PHYSICAL_CONSTRAINT,
                SophismType.SLIPPERY_SLOPE: SocraticIssueType.CAUSAL_GAP,
                SophismType.HASTY_GENERALIZATION: SocraticIssueType.ASSUMPTION,
                SophismType.AD_HOMINEM: SocraticIssueType.CLARIFICATION,
                SophismType.BANDWAGON: SocraticIssueType.ASSUMPTION,
                SophismType.FALSE_CAUSE: SocraticIssueType.CAUSAL_GAP,
                SophismType.STRAWMAN: SocraticIssueType.CLARIFICATION,
                SophismType.CHERRY_PICKING: SocraticIssueType.COUNTEREXAMPLE,
                SophismType.APPEAL_TO_NATURE: SocraticIssueType.ASSUMPTION,
                SophismType.CIRCULAR_REASONING: SocraticIssueType.CAUSAL_GAP,
                SophismType.NON_FALSIFIABLE: SocraticIssueType.FALSIFICATION,
            }.get(s["type"], SocraticIssueType.INCOMPLETE)

            report.add(SocraticIssue(
                type=issue_type,
                question=f"SOPHISME [{s['name']}] : \"{s['match'][:60]}...\"",
                context=s["context"][:100],
                severity=s["severity"],
            ))

        # 2. Product placements
        placements = self.detect_product_placement(text)
        for p in placements:
            report.add(SocraticIssue(
                type=SocraticIssueType.ASSUMPTION,
                question=f"PLACEMENT [{p['type']}] : \"{p['match'][:60]}...\"",
                context=p["context"][:100],
                severity=p["severity"],
            ))

        # 3. Epistemic confidence
        confidence = self.score_epistemic_confidence(text, url, sophisms, placements)
        level = "🟢 Élevée" if confidence >= 0.7 else "🟡 Moyenne" if confidence >= 0.4 else "🔴 Faible"
        report.add(SocraticIssue(
            type=SocraticIssueType.INCOMPLETE,
            question=f"Confiance épistémique : {confidence:.0%} ({level}). "
                     f"{len(sophisms)} sophismes, {len(placements)} placements.",
            context=f"URL: {url[:60] if url else 'N/A'}",
            severity=0.5,
        ))

        if confidence < 0.3:
            report.blocking = True

        if verbose:
            print(report.format_report())
        return report

    def analyze_search_results(self, results: List[dict]) -> List[dict]:
        """Score and rank web search results by epistemic confidence."""
        scored = []
        for r in results:
            text = f"{r.get('title', '')} {r.get('description', '')}"
            sophisms = self.detect_sophisms(text)
            confidence = self.score_epistemic_confidence(text, r.get('url', ''), sophisms)
            scored.append({
                **r,
                "epistemic_confidence": round(confidence, 2),
                "sophisms": [s["name"] for s in sophisms],
                "sophism_count": len(sophisms),
            })
        scored.sort(key=lambda x: x["epistemic_confidence"], reverse=True)
        return scored

    def format_web_analysis(self, report: SocraticReport, url: str = "") -> str:
        """Format web analysis in French for user display."""
        lines = []
        sophisms = [i for i in report.issues if "SOPHISME" in i.question]
        placements = [i for i in report.issues if "PLACEMENT" in i.question]
        confidence = [i for i in report.issues if "Confiance" in i.question]

        if report.blocking:
            lines.append("🔴 **ATTENTION — Source non fiable**")
        else:
            lines.append("🟡 **Analyse de source**")

        lines.append("━" * 60)

        if url:
            lines.append(f"Source: {url}")

        if confidence:
            lines.append(f"\n{confidence[0].question}")

        if sophisms:
            lines.append(f"\n⚠️ **{len(sophisms)} sophisme(s) :**")
            for s in sophisms[:5]:
                q = s.question.replace("SOPHISME ", "  • ")
                lines.append(q)
            if len(sophisms) > 5:
                lines.append(f"  ... et {len(sophisms) - 5} autre(s)")

        if placements:
            lines.append(f"\n💰 **{len(placements)} placement(s) produit :**")
            for p in placements[:4]:
                q = p.question.replace("PLACEMENT ", "  • ")
                lines.append(q)
            if len(placements) > 4:
                lines.append(f"  ... et {len(placements) - 4} autre(s)")

        if report.blocking:
            lines.append("\n🚫 **Recommandation :** Source à vérifier indépendamment ou à éviter.")

        lines.append("━" * 60)
        return "\n".join(lines)

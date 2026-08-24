"""
BehaviouralMemo.py
===================
Behavioral Memory: Pluto's PROCEDURAL memory -- it learns HOW to act, not
what to know.

What it is
----------
Behavioral Memory stores structured `Behavior` records: a situation/context
signature, the action taken in it, the outcome, a confidence score, a reward
value, how often it has fired, timestamps, and a link back to the originating
episode. It is the decision-making layer that sits between knowledge and
action:

    Short-Term Memory -> "What is happening now?"
    Semantic Memory   -> "What do I know about this?"
    Episodic Memory   -> "Have I experienced this before?"       [TODO(connect)]
    Behavioral Memory -> "Based on everything I know and have
                          experienced, what is the best action
                          to take now?"                          <-- this file

What it is NOT
--------------
- It never talks to sensors and never processes raw STM rows or SceneBuilder
  Situations directly. It only ever reasons over the CONTEXT SUMMARY Brain
  builds for it (current STM context + relevant SemanticFacts + relevant
  EpisodicMemory records once that subsystem exists).
- It does not store complete conversations, raw events, or verbatim
  Situations. It stores GENERALIZED strategies extracted across many
  experiences, with references to originating episodes, never duplicated
  event data.
- It performs no perception and no long-term factual bookkeeping -- that is
  STM's and Semantic Memory's job respectively.

Where it sits
-------------
    Brain gathers, per decision cycle:
        current_context   <- ShortTermMemory     (who/where/time/emotion/activity/conversation)
        semantic_facts     <- SemanticMemory.get_facts()   (preferences, routines, relationships)
        similar_episodes    <- EpisodicMemory.find_similar() [TODO(connect), returns [] for now]
    Brain -> BehavioralMemory.select_action(current_context, semantic_facts, similar_episodes)
          -> returns the best-ranked Behavior (or None if nothing qualifies)
    Brain executes the chosen action, then calls
        BehavioralMemory.reinforce(behavior_id, reward)
    afterwards with the observed outcome, which is how existing behaviors
    strengthen into trusted habits, decay, or get archived.

How action selection works
---------------------------
1. `select_action()` builds a CONTEXT SIGNATURE from current_context (coarse,
   structured -- not free text) and searches stored behaviors for ones whose
   own signature is context-similar (see `_context_similarity`).
2. Each match is scored using a weighted combination of:
       confidence, success_rate, recency, frequency, contextual_relevance
   (see `_rank_score`). No single factor dominates; a very confident but
   stale behavior can still lose to a slightly-less-confident, recently
   successful, highly-relevant one.
3. The top-ranked ACTIVE behavior is returned. If nothing clears
   MIN_SELECTION_SCORE, None is returned so Brain knows to fall back to some
   default/explicit policy rather than a low-quality guess.

How learning works
-------------------
- `learn_or_reinforce()` is how a new (situation, action) pair enters memory,
  or an existing matching one gets its stats refreshed, BEFORE the outcome is
  known (frequency +1, last_used refreshed).
- `reinforce()` is called AFTER execution with the real-world outcome/reward.
  It updates confidence, success_count/failure_count, moving-average reward,
  and re-evaluates the behavior's lifecycle stage:
      LEARNING  -> newly seen, not yet trusted
      ACTIVE    -> consistently successful, kept indefinitely
      DECAYING  -> rarely used recently; confidence drifts down over time
      ARCHIVED  -> consistently unsuccessful or fully decayed; excluded from
                   selection but kept on disk for inspection/audit
- `decay_unused()` is a periodic maintenance pass (call it from Brain's main
  loop, e.g. once per cycle) that nudges confidence down for behaviors that
  have not been used recently, and archives ones that decay past a floor.

Persistence
------------
Behaviors are stored as structured JSON on disk (same pattern as
SemanticMemo.py's `_JSONStore`), so this class can be inspected, backed up,
or hand-edited without touching code.
"""

import os
import json
import uuid
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

BEHAVIORS_PATH = os.path.join(os.path.dirname(__file__), "behavioral_memory.json")

# Lifecycle stages
STAGE_LEARNING = "learning"
STAGE_ACTIVE = "active"
STAGE_DECAYING = "decaying"
STAGE_ARCHIVED = "archived"

CONFIDENCE_INITIAL = 0.35
CONFIDENCE_GAIN_ON_SUCCESS = 0.15        # scaled by (1 - confidence), diminishing returns
CONFIDENCE_PENALTY_ON_FAILURE = 0.22
CONFIDENCE_ARCHIVE_BELOW = 0.10          # confidence floor -> archive
ACTIVE_PROMOTION_MIN_USES = 4            # min times used before it can become ACTIVE
ACTIVE_PROMOTION_MIN_SUCCESS_RATE = 0.6

DECAY_PER_IDLE_PASS = 0.03               # confidence lost per decay_unused() call if unused
DECAY_IDLE_SECONDS = 60 * 60 * 24 * 3    # 3 days of no use before decay starts applying

MIN_SELECTION_SCORE = 0.30               # behaviors ranked below this are never selected

# Ranking weights -- must sum to 1.0 (not enforced, but kept true by convention)
WEIGHT_CONFIDENCE = 0.30
WEIGHT_SUCCESS_RATE = 0.25
WEIGHT_RECENCY = 0.15
WEIGHT_FREQUENCY = 0.10
WEIGHT_CONTEXTUAL_RELEVANCE = 0.20

MAX_LINKED_EPISODE_IDS = 30


# --------------------------------------------------------------------------
# BEHAVIOR: the structured, learned unit of "what to do when"
# --------------------------------------------------------------------------

@dataclass
class Behavior:
    """
    One learned strategy: "in this kind of situation, this action tends to
    produce this outcome." Never free text for the core fields -- action and
    situation are structured so ranking/matching can be done programmatically.
    """
    behavior_id: str
    situation_signature: Dict[str, Any]   # coarse structured context, e.g. {"emotion": "stressed", "time_of_day": "night"}
    action: Dict[str, Any]                # structured action descriptor, e.g. {"type": "suggest_rest"}
    outcome_summary: Dict[str, Any]       # last observed structured outcome, e.g. {"accepted": True}
    confidence: float                     # 0.0-1.0, how much this behavior should be trusted
    reward_avg: float                     # running average of numeric reward signal
    success_count: int
    failure_count: int
    frequency: int                        # total number of times this behavior has been selected/used
    created_at: str
    updated_at: str
    last_used_at: str
    stage: str = STAGE_LEARNING
    linked_episode_ids: List[str] = field(default_factory=list)
    active: bool = True

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Behavior":
        return Behavior(**d)


# --------------------------------------------------------------------------
# PERSISTENCE (same minimal thread-safe JSON store pattern as SemanticMemo.py)
# --------------------------------------------------------------------------

class _JSONStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def load(self) -> List[Dict[str, Any]]:
        with self._lock:
            if not os.path.exists(self.path):
                return []
            try:
                with open(self.path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return []

    def save(self, items: List[Dict[str, Any]]):
        with self._lock:
            with open(self.path, "w") as f:
                json.dump(items, f, indent=2)


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _seconds_since(timestamp: str) -> float:
    try:
        then = datetime.fromisoformat(timestamp)
    except (ValueError, TypeError):
        return 0.0
    return max(0.0, (datetime.now() - then).total_seconds())


def _build_situation_signature(current_context: Dict[str, Any],
                                semantic_facts: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    Reduces whatever Brain hands over (current STM-derived context, plus
    optionally a list of relevant SemanticFacts) into one coarse, structured
    signature usable both for storing a new behavior and for matching against
    stored ones. Never free text -- only bucketed/structured fields.

    Expected (but not strictly required) keys on `current_context`:
        person_id, emotion, time_of_day, location_signature, activity
    """
    signature = {
        "person_id": current_context.get("person_id", "unknown"),
        "emotion": current_context.get("emotion", "unknown"),
        "time_of_day": current_context.get("time_of_day", "unknown"),
        "location_signature": current_context.get("location_signature", "unknown"),
        "activity": current_context.get("activity", "unknown"),
    }

    if semantic_facts:
        # Fold in a couple of coarse semantic hints (e.g. a matching routine
        # or preference's category) as extra signature fields, without ever
        # embedding the full fact -- Behavioral Memory references knowledge,
        # it doesn't duplicate it.
        categories_present = sorted({getattr(f, "category", None) for f in semantic_facts if getattr(f, "category", None)})
        if categories_present:
            signature["relevant_semantic_categories"] = categories_present

    return signature


def _context_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """
    Fraction of shared signature keys that match exactly, restricted to keys
    present in both. A simple, explainable similarity metric -- good enough
    for coarse bucketed fields; can be swapped for embeddings later without
    changing any caller.
    """
    keys = set(a.keys()) & set(b.keys())
    if not keys:
        return 0.0
    matches = sum(1 for k in keys if a.get(k) == b.get(k))
    return matches / len(keys)


def _recency_score(last_used_at: str) -> float:
    """1.0 for just-used, decaying toward 0 over ~14 days."""
    idle_seconds = _seconds_since(last_used_at)
    half_life_seconds = 14 * 24 * 60 * 60
    if half_life_seconds <= 0:
        return 0.0
    return max(0.0, 2 ** (-idle_seconds / half_life_seconds))


def _frequency_score(frequency: int, cap: int = 20) -> float:
    """Diminishing-returns frequency score, capped at 1.0."""
    return min(1.0, frequency / cap)


def _rank_score(behavior: Behavior, situation_signature: Dict[str, Any]) -> float:
    relevance = _context_similarity(behavior.situation_signature, situation_signature)
    recency = _recency_score(behavior.last_used_at)
    frequency = _frequency_score(behavior.frequency)

    return (
        WEIGHT_CONFIDENCE * behavior.confidence
        + WEIGHT_SUCCESS_RATE * behavior.success_rate
        + WEIGHT_RECENCY * recency
        + WEIGHT_FREQUENCY * frequency
        + WEIGHT_CONTEXTUAL_RELEVANCE * relevance
    )


def _stage_for(behavior: Behavior) -> str:
    total_uses = behavior.success_count + behavior.failure_count
    if behavior.confidence < CONFIDENCE_ARCHIVE_BELOW:
        return STAGE_ARCHIVED
    if total_uses >= ACTIVE_PROMOTION_MIN_USES and behavior.success_rate >= ACTIVE_PROMOTION_MIN_SUCCESS_RATE:
        return STAGE_ACTIVE
    if _seconds_since(behavior.last_used_at) > DECAY_IDLE_SECONDS:
        return STAGE_DECAYING
    return STAGE_LEARNING


# --------------------------------------------------------------------------
# BEHAVIORAL MEMORY
# --------------------------------------------------------------------------

class BehavioralMemory:
    """
    Pluto's procedural memory. Combines current context (from STM), relevant
    knowledge (from Semantic Memory), and relevant past experience (from
    Episodic Memory, once built) to select the best action, then learns from
    the outcome of whatever action was actually taken.
    """

    def __init__(self, behaviors_path: str = BEHAVIORS_PATH):
        self._store = _JSONStore(behaviors_path)
        self._lock = threading.Lock()

        self.behaviors: Dict[str, Behavior] = {}
        for d in self._store.load():
            behavior = Behavior.from_dict(d)
            self.behaviors[behavior.behavior_id] = behavior

        self.ready = True

    # ---- public API: selection ------------------------------------------

    def select_action(self,
                       current_context: Dict[str, Any],
                       semantic_facts: Optional[List[Any]] = None,
                       similar_episodes: Optional[List[Any]] = None) -> Optional[Behavior]:
        """
        Main entry point for Brain during a decision cycle. Ranks all active,
        non-archived behaviors against the current situation and returns the
        single best match, or None if nothing clears MIN_SELECTION_SCORE
        (Brain should fall back to a default/explicit policy in that case).

        `similar_episodes` is accepted now so this call site never has to
        change once Episodic Memory exists; it is currently unused
        [TODO(connect): once EpisodicMemory.find_similar() is implemented,
        fold matching episodes' outcomes into the ranking the same way
        semantic_facts is folded into the situation signature].
        """
        with self._lock:
            situation_signature = _build_situation_signature(current_context, semantic_facts)

            candidates = [b for b in self.behaviors.values() if b.active and b.stage != STAGE_ARCHIVED]
            if not candidates:
                return None

            scored = [(b, _rank_score(b, situation_signature)) for b in candidates]
            scored.sort(key=lambda pair: pair[1], reverse=True)

            best, best_score = scored[0]
            if best_score < MIN_SELECTION_SCORE:
                return None
            return best

    # ---- public API: learning ---------------------------------------------

    def learn_or_reinforce(self,
                            current_context: Dict[str, Any],
                            action: Dict[str, Any],
                            semantic_facts: Optional[List[Any]] = None,
                            episode_id: Optional[str] = None) -> str:
        """
        Called by Brain right before/at the moment an action is taken --
        either the first time this (situation, action) pair is seen (creates
        a new LEARNING-stage Behavior) or again for an existing matching one
        (bumps frequency/last_used_at only; confidence changes happen in
        `reinforce()` once the outcome is known). Returns the behavior_id so
        Brain can pass it to `reinforce()` afterward.
        """
        with self._lock:
            situation_signature = _build_situation_signature(current_context, semantic_facts)
            existing = self._find_matching_behavior(situation_signature, action)

            if existing is not None:
                existing.frequency += 1
                existing.last_used_at = _now()
                existing.updated_at = _now()
                if episode_id and episode_id not in existing.linked_episode_ids:
                    existing.linked_episode_ids.append(episode_id)
                    existing.linked_episode_ids = existing.linked_episode_ids[-MAX_LINKED_EPISODE_IDS:]
                self._persist()
                return existing.behavior_id

            now = _now()
            behavior = Behavior(
                behavior_id=uuid.uuid4().hex,
                situation_signature=situation_signature,
                action=action,
                outcome_summary={},
                confidence=CONFIDENCE_INITIAL,
                reward_avg=0.0,
                success_count=0,
                failure_count=0,
                frequency=1,
                created_at=now,
                updated_at=now,
                last_used_at=now,
                stage=STAGE_LEARNING,
                linked_episode_ids=[episode_id] if episode_id else [],
                active=True,
            )
            self.behaviors[behavior.behavior_id] = behavior
            self._persist()
            return behavior.behavior_id

    def reinforce(self,
                   behavior_id: str,
                   success: bool,
                   reward: float = 0.0,
                   outcome_summary: Optional[Dict[str, Any]] = None):
        """
        Called by Brain after the selected action has actually been executed
        and its result observed (explicit feedback, or an inferred reward
        signal). Updates confidence, success/failure counters, the running
        reward average, and re-evaluates lifecycle stage.

        Repeated success strengthens the behavior toward ACTIVE; repeated
        failure pushes confidence down and, past CONFIDENCE_ARCHIVE_BELOW,
        the behavior is ARCHIVED (kept on disk, excluded from selection).
        """
        with self._lock:
            behavior = self.behaviors.get(behavior_id)
            if behavior is None:
                return

            if success:
                behavior.success_count += 1
                behavior.confidence = min(0.99, behavior.confidence + CONFIDENCE_GAIN_ON_SUCCESS * (1 - behavior.confidence))
            else:
                behavior.failure_count += 1
                behavior.confidence = max(0.01, behavior.confidence - CONFIDENCE_PENALTY_ON_FAILURE)

            total = behavior.success_count + behavior.failure_count
            behavior.reward_avg = ((behavior.reward_avg * (total - 1)) + reward) / total

            if outcome_summary:
                behavior.outcome_summary = outcome_summary

            behavior.updated_at = _now()
            behavior.last_used_at = _now()
            behavior.stage = _stage_for(behavior)
            behavior.active = behavior.stage != STAGE_ARCHIVED

            self._persist()

    # ---- public API: maintenance ------------------------------------------

    def decay_unused(self):
        """
        Periodic maintenance pass -- call once per Brain cycle (or on a
        slower timer). Behaviors idle longer than DECAY_IDLE_SECONDS lose a
        small amount of confidence each pass; behaviors that decay past
        CONFIDENCE_ARCHIVE_BELOW are archived (kept for audit, excluded from
        future selection).
        """
        with self._lock:
            changed = False
            for behavior in self.behaviors.values():
                if not behavior.active:
                    continue
                if _seconds_since(behavior.last_used_at) <= DECAY_IDLE_SECONDS:
                    continue
                behavior.confidence = max(0.0, behavior.confidence - DECAY_PER_IDLE_PASS)
                behavior.updated_at = _now()
                behavior.stage = _stage_for(behavior)
                behavior.active = behavior.stage != STAGE_ARCHIVED
                changed = True
            if changed:
                self._persist()

    # ---- public API: querying ---------------------------------------------

    def get_behaviors(self, stage: Optional[str] = None, active_only: bool = True) -> List[Behavior]:
        """Inspect learned behaviors -- e.g. for the Memory Core GUI screen."""
        behaviors = list(self.behaviors.values())
        if stage:
            behaviors = [b for b in behaviors if b.stage == stage]
        if active_only:
            behaviors = [b for b in behaviors if b.active]
        return behaviors

    # ---- internals ---------------------------------------------------------

    def _find_matching_behavior(self, situation_signature: Dict[str, Any], action: Dict[str, Any]) -> Optional[Behavior]:
        """
        A stored behavior is "the same" if its situation_signature matches
        exactly and its action matches exactly -- distinct actions for the
        same situation are kept as separate competing behaviors so ranking
        can choose between them.
        """
        for behavior in self.behaviors.values():
            if behavior.situation_signature == situation_signature and behavior.action == action:
                return behavior
        return None

    def _persist(self):
        self._store.save([b.to_dict() for b in self.behaviors.values()])


if __name__ == "__main__":
    # Minimal manual smoke test -- exercises learn -> reinforce -> select
    # without needing Brain, STM, or Semantic Memory wired up.
    bm = BehavioralMemory(behaviors_path="/tmp/_behavioral_memory_test.json")

    context = {"person_id": "owner001", "emotion": "stressed", "time_of_day": "night",
               "location_signature": "t22_l50", "activity": "lying_down"}

    action_a = {"type": "suggest_rest"}
    action_b = {"type": "speak_randomly"}

    # Simulate several cycles where "suggest_rest" tends to succeed and
    # "speak_randomly" tends to fail, so ranking should favor action_a.
    for _ in range(5):
        bid_a = bm.learn_or_reinforce(context, action_a)
        bm.reinforce(bid_a, success=True, reward=1.0, outcome_summary={"accepted": True})

        bid_b = bm.learn_or_reinforce(context, action_b)
        bm.reinforce(bid_b, success=False, reward=-0.5, outcome_summary={"accepted": False})

    chosen = bm.select_action(context)
    print(f"Behaviors learned: {len(bm.behaviors)}")
    for b in bm.get_behaviors():
        print(f"  [{b.stage}] {b.action} conf={b.confidence:.2f} success_rate={b.success_rate:.2f} freq={b.frequency}")
    print(f"Selected action: {chosen.action if chosen else None}")
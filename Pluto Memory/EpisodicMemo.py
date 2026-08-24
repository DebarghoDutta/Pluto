"""
EpisodicMemo.py
===============
Episodic Memory: Pluto's long-term AUTOBIOGRAPHICAL memory.

What it is
----------
Episodic Memory stores COMPLETE, specific experiences -- not generalized
knowledge (that's Semantic Memory's job) and not learned action-strategies
(that's Behavioral Memory's job). It answers "what happened, when, to whom,
and how did it go", one full event at a time:

    Semantic Memory   -> "What do I know in general?"
    Behavioral Memory -> "What should I do?"
    Episodic Memory    -> "Have I lived through something like this before,
                           and how did it turn out?"                <-- this file

Where it sits
-------------
Brain is the ONLY thing that ever talks to Episodic Memory. It never
receives raw STM rows and never talks to SceneBuilder, SemanticMemory, or
BehavioralMemory directly -- Brain hands it exactly what it needs, once per
cycle:

    Brain._perceive()         -> Situation           (via SceneBuilder)
    Brain._gather_knowledge()  -> current_context, semantic_facts
                                  + EpisodicMemory.find_similar(current_context)
    Brain._decide()            -> chosen Behavior (or None)
    Brain._execute()           -> action taken
    Brain._observe_feedback()  -> outcome, success, reward
    Brain._update_memories()   -> EpisodicMemory.log(situation, current_context, action,
                                                      outcome, success, reward, ...)

What gets stored
-----------------
Each `Episode` is one complete event record:
    - the full Situation snapshot (people/objects/pose/speech/environment)
    - the structured current_context Brain derived from it
    - the action that was taken and its outcome
    - the owner's emotional state at the time
    - timestamp, people_involved, location_signature
    - an importance_score (how much this episode matters / should be
      remembered strongly)
    - confidence + success_rate (how reliable/positive this kind of episode
      has proven to be, strengthened across repeats rather than duplicated)
    - links to related episodes, and back-references into Semantic Memory
      facts / Behavioral Memory behaviors that this episode reinforces

How duplication is avoided
----------------------------
A brand-new experience is stored as its own independent Episode. But if an
incoming experience closely RESEMBLES an existing episode (same person,
same coarse location/time-of-day/activity, same action taken), Episodic
Memory does not create a duplicate row -- instead it:
    - increments the existing episode's `repetition_count`
    - updates its `confidence` and `success_rate` from the new outcome
    - appends the new occurrence's timestamp to `occurrences`
    - links in any new `related_semantic_fact_ids` / `related_behavior_ids`
    - recalculates `importance_score`
This is the same "reinforce-in-place vs. promote-new" shape used by
SemanticMemo.py and BehaviouralMemo.py, applied here to whole experiences
instead of extracted facts or action strategies.

Similarity / retrieval
------------------------
`find_similar(current_context)` returns past episodes whose stored context
resembles the context Brain is asking about right now, most-relevant and
most-important first, so Brain can hand them to BehavioralMemory as
evidence ("have we been in a situation like this before, and how did it
go?") alongside SemanticMemory's general facts.

Persistence
------------
Episodes are stored as structured JSON on disk (same pattern as
SemanticMemo.py's / BehaviouralMemo.py's `_JSONStore`), so they can be
inspected, backed up, or hand-edited without touching code.

Usage (from Brain.py):
    from EpisodicMemo import EpisodicMemory

    episodic = EpisodicMemory()
    similar = episodic.find_similar(current_context)
    ...
    episode_id = episodic.log(
        situation=situation,
        current_context=current_context,
        action=action,
        outcome=outcome,
        success=success,
        reward=reward,
        emotional_state=current_context.get("emotion"),
        related_semantic_fact_ids=[f.fact_id for f in semantic_facts],
        related_behavior_id=behavior_id,
    )
"""

import os
import json
import uuid
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

EPISODES_PATH = os.path.join(os.path.dirname(__file__), "episodic_memory.json")

# An incoming experience is treated as a repetition of an existing episode
# (rather than a brand-new one) if all of these coarse signature fields match.
SIGNATURE_FIELDS = ("person_id", "location_signature", "time_of_day", "activity")

CONFIDENCE_INITIAL = 0.35
CONFIDENCE_GAIN_ON_SUCCESS = 0.15         # scaled by (1 - confidence), diminishing returns
CONFIDENCE_PENALTY_ON_FAILURE = 0.20

MAX_OCCURRENCES = 50           # cap how many raw occurrence timestamps a single episode keeps
MAX_LINKED_EPISODE_IDS = 30
MAX_RELATED_FACT_IDS = 30
MAX_RELATED_BEHAVIOR_IDS = 30

IMPORTANCE_REPETITION_THRESHOLDS = (3, 10)   # repetition_count -> low/medium/high contribution
DEFAULT_MAX_SIMILAR_RESULTS = 5


# --------------------------------------------------------------------------
# EPISODE: the structured, permanent unit of "something that happened"
# --------------------------------------------------------------------------

@dataclass
class Episode:
    """
    One complete autobiographical experience. Unlike a SemanticFact (a
    generalized belief) or a Behavior (a learned action-strategy), an
    Episode is a full, specific record of a single lived event -- but
    experiences that closely resemble an existing episode strengthen it
    in place rather than duplicating it (see `repetition_count`,
    `occurrences`).
    """
    episode_id: str
    timestamp: str                              # timestamp of first occurrence
    situation_signature: Dict[str, Any]          # coarse matching signature (see SIGNATURE_FIELDS)
    situation_snapshot: Dict[str, Any]           # full Situation, reduced to a plain dict
    current_context: Dict[str, Any]              # the structured context Brain derived from it
    action: Dict[str, Any]                        # action taken during this episode
    outcome: Dict[str, Any]                       # observed outcome
    emotional_state: str                          # owner's emotion at the time
    people_involved: List[str]                    # person_ids present
    location_signature: str
    importance_score: float                       # 0.0-1.0, how strongly this should be remembered
    confidence: float                             # 0.0-1.0, reliability of this episode's outcome pattern
    success_count: int
    failure_count: int
    repetition_count: int                         # how many times a matching experience has occurred
    occurrences: List[str] = field(default_factory=list)          # timestamps of each occurrence
    related_semantic_fact_ids: List[str] = field(default_factory=list)
    related_behavior_ids: List[str] = field(default_factory=list)
    linked_episode_ids: List[str] = field(default_factory=list)    # other episodes judged related/similar
    created_at: str = ""
    updated_at: str = ""
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
    def from_dict(d: Dict[str, Any]) -> "Episode":
        return Episode(**d)


# --------------------------------------------------------------------------
# PERSISTENCE (same minimal thread-safe JSON store pattern as SemanticMemo.py
# and BehaviouralMemo.py)
# --------------------------------------------------------------------------

class _JSONStore:
    """Minimal thread-safe load/save wrapper for a JSON list-of-dicts file."""

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


def _build_situation_signature(current_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coarse, structured signature used to decide whether a new experience
    resembles an existing Episode. Deliberately the same shape/granularity
    BehavioralMemory uses for its own situation_signature, so episodes and
    behaviors stay comparable.
    """
    return {field_name: current_context.get(field_name, "unknown") for field_name in SIGNATURE_FIELDS}


def _situation_to_dict(situation: Any) -> Dict[str, Any]:
    """
    Reduces a SceneBuilder `Situation` object to a plain, JSON-serializable
    dict snapshot. Works whether `situation` is a real `Situation` dataclass
    instance or an already-plain dict/namespace with the same fields.
    """
    if isinstance(situation, dict):
        return situation
    return {
        "observation_id": getattr(situation, "observation_id", None),
        "timestamp": getattr(situation, "timestamp", None),
        "people": getattr(situation, "people", []),
        "objects": getattr(situation, "objects", []),
        "pose": getattr(situation, "pose", None),
        "speech": getattr(situation, "speech", []),
        "environment": getattr(situation, "environment", {}),
        "status_flags": getattr(situation, "status_flags", []),
    }


def _importance_for(repetition_count: int, confidence: float) -> float:
    """
    Importance blends how often something has recurred with how reliably
    it has gone (confidence) -- a rare-but-strong experience and a
    frequently-repeated one can both end up "important", for different
    reasons.
    """
    low, high = IMPORTANCE_REPETITION_THRESHOLDS
    if repetition_count >= high:
        repetition_component = 1.0
    elif repetition_count >= low:
        repetition_component = 0.6
    else:
        repetition_component = 0.3
    return round(min(1.0, 0.5 * repetition_component + 0.5 * confidence), 3)


def _context_overlap_score(signature_a: Dict[str, Any], signature_b: Dict[str, Any]) -> float:
    """Fraction of signature fields that match -- used to rank similar episodes."""
    if not signature_a or not signature_b:
        return 0.0
    fields = set(signature_a) | set(signature_b)
    if not fields:
        return 0.0
    matches = sum(1 for f in fields if signature_a.get(f) == signature_b.get(f))
    return matches / len(fields)


# --------------------------------------------------------------------------
# EPISODIC MEMORY
# --------------------------------------------------------------------------

class EpisodicMemory:
    """
    Pluto's autobiographical memory. Reachable only through Brain -- never
    called directly by ShortTermMemory, SceneBuilder, SemanticMemory, or
    BehavioralMemory.
    """

    def __init__(self, episodes_path: str = EPISODES_PATH):
        self._store = _JSONStore(episodes_path)
        self._lock = threading.Lock()
        self.episodes: Dict[str, Episode] = {}
        self.ready = False
        self._load()

    # ---- setup ---------------------------------------------------------

    def _load(self):
        raw = self._store.load()
        with self._lock:
            self.episodes = {d["episode_id"]: Episode.from_dict(d) for d in raw}
        self.ready = True

    def _persist(self):
        self._store.save([e.to_dict() for e in self.episodes.values()])

    # ---- public API: retrieval ------------------------------------------

    def find_similar(self, current_context: Dict[str, Any], max_results: int = DEFAULT_MAX_SIMILAR_RESULTS) -> List[Episode]:
        """
        Returns past episodes whose situation_signature resembles
        `current_context`, ranked by a blend of context-overlap and each
        episode's importance_score, most relevant first. Returns [] if
        nothing qualifies -- Brain and BehavioralMemory already treat an
        empty list as "no episodic evidence available yet."
        """
        target_signature = _build_situation_signature(current_context)

        scored = []
        with self._lock:
            for episode in self.episodes.values():
                if not episode.active:
                    continue
                overlap = _context_overlap_score(episode.situation_signature, target_signature)
                if overlap <= 0.0:
                    continue
                rank = 0.6 * overlap + 0.4 * episode.importance_score
                scored.append((episode, rank))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [episode for episode, _ in scored[:max_results]]

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        return self.episodes.get(episode_id)

    def get_episodes(self, person_id: Optional[str] = None, active_only: bool = True) -> List[Episode]:
        """Inspect stored episodes -- e.g. for the Memory Core GUI screen."""
        episodes = list(self.episodes.values())
        if person_id:
            episodes = [e for e in episodes if person_id in e.people_involved]
        if active_only:
            episodes = [e for e in episodes if e.active]
        return episodes

    # ---- public API: logging --------------------------------------------

    def log(self,
            situation: Any,
            current_context: Dict[str, Any],
            action: Dict[str, Any],
            outcome: Dict[str, Any],
            success: bool,
            reward: float = 0.0,
            emotional_state: Optional[str] = None,
            related_semantic_fact_ids: Optional[List[str]] = None,
            related_behavior_id: Optional[str] = None) -> Optional[str]:
        """
        Called by Brain once per cycle, after execution + feedback, with
        everything needed to record the complete event:
            situation                 -> full Situation from SceneBuilder
            current_context           -> Brain's structured context summary
            action                    -> the action that was taken
            outcome                   -> the observed outcome
            success / reward          -> feedback signal for this event
            emotional_state           -> owner's emotion at the time
            related_semantic_fact_ids -> SemanticFact ids Brain looked up this cycle
            related_behavior_id       -> the Behavior this episode resulted from

        If a closely-resembling episode already exists (see
        `_find_matching_episode`), it is reinforced in place instead of
        duplicated. Otherwise a brand-new Episode is created. Returns the
        episode_id either way so Brain/BehavioralMemory can link back to it.
        """
        if current_context is None:
            return None

        with self._lock:
            situation_signature = _build_situation_signature(current_context)
            existing = self._find_matching_episode(situation_signature, action)

            if existing is not None:
                self._reinforce(existing, situation, current_context, outcome, success,
                                 emotional_state, related_semantic_fact_ids, related_behavior_id)
                episode_id = existing.episode_id
            else:
                episode_id = self._create(situation, current_context, action, outcome, success,
                                           emotional_state, related_semantic_fact_ids, related_behavior_id)

            self._link_related_episodes(episode_id, situation_signature)
            self._persist()
            return episode_id

    def process(self, situations: List[Any]):
        """
        Batch-log path, kept for parity with SemanticMemory.process()'s call
        shape. Episodic Memory normally logs one full event per cycle via
        `log()` (since it needs the action/outcome Brain only has at that
        point) -- this is here only for callers that want to backfill
        situations with no action/outcome context attached.
        """
        for situation in situations:
            self.log(
                situation=situation,
                current_context={},
                action={"type": "unknown"},
                outcome={"observed": False},
                success=True,
                reward=0.0,
            )

    # ---- internals: create / reinforce -----------------------------------

    def _find_matching_episode(self, situation_signature: Dict[str, Any], action: Dict[str, Any]) -> Optional[Episode]:
        """
        An incoming experience is a "repetition" of an existing episode if
        its coarse situation_signature matches exactly AND the same action
        was taken -- the same shape of match BehavioralMemory uses for its
        own behaviors, so the two subsystems stay conceptually aligned.
        """
        for episode in self.episodes.values():
            if episode.situation_signature == situation_signature and episode.action == action:
                return episode
        return None

    def _create(self, situation, current_context, action, outcome, success,
                emotional_state, related_semantic_fact_ids, related_behavior_id) -> str:
        now = _now()
        situation_snapshot = _situation_to_dict(situation)
        timestamp = situation_snapshot.get("timestamp") or now

        people_involved = sorted({
            p.get("person_id") for p in situation_snapshot.get("people", [])
            if p.get("person_id")
        })

        success_count = 1 if success else 0
        failure_count = 0 if success else 1
        confidence = min(0.99, CONFIDENCE_INITIAL + CONFIDENCE_GAIN_ON_SUCCESS) if success else max(0.01, CONFIDENCE_INITIAL - CONFIDENCE_PENALTY_ON_FAILURE)

        episode = Episode(
            episode_id=uuid.uuid4().hex,
            timestamp=timestamp,
            situation_signature=_build_situation_signature(current_context),
            situation_snapshot=situation_snapshot,
            current_context=current_context,
            action=action,
            outcome=outcome or {},
            emotional_state=emotional_state or current_context.get("emotion", "unknown"),
            people_involved=people_involved,
            location_signature=current_context.get("location_signature", "unknown"),
            importance_score=_importance_for(1, confidence),
            confidence=confidence,
            success_count=success_count,
            failure_count=failure_count,
            repetition_count=1,
            occurrences=[timestamp],
            related_semantic_fact_ids=(related_semantic_fact_ids or [])[-MAX_RELATED_FACT_IDS:],
            related_behavior_ids=[related_behavior_id] if related_behavior_id else [],
            linked_episode_ids=[],
            created_at=now,
            updated_at=now,
            active=True,
        )
        self.episodes[episode.episode_id] = episode
        return episode.episode_id

    def _reinforce(self, episode: Episode, situation, current_context, outcome, success,
                    emotional_state, related_semantic_fact_ids, related_behavior_id):
        """
        Updates an existing episode in place rather than creating a
        duplicate row for a resembling experience: repetition_count grows,
        confidence/success_rate move with the new outcome, and the new
        occurrence's timestamp/context/snapshot supersede the previous
        ones (episodes track the MOST RECENT full snapshot, while
        `occurrences` keeps the history of when this kind of thing happened).
        """
        situation_snapshot = _situation_to_dict(situation)
        timestamp = situation_snapshot.get("timestamp") or _now()

        if success:
            episode.success_count += 1
            episode.confidence = min(0.99, episode.confidence + CONFIDENCE_GAIN_ON_SUCCESS * (1 - episode.confidence))
        else:
            episode.failure_count += 1
            episode.confidence = max(0.01, episode.confidence - CONFIDENCE_PENALTY_ON_FAILURE)

        episode.repetition_count += 1
        episode.occurrences.append(timestamp)
        episode.occurrences = episode.occurrences[-MAX_OCCURRENCES:]

        episode.situation_snapshot = situation_snapshot
        episode.current_context = current_context
        episode.outcome = outcome or episode.outcome
        episode.emotional_state = emotional_state or current_context.get("emotion", episode.emotional_state)
        episode.location_signature = current_context.get("location_signature", episode.location_signature)

        new_people = {
            p.get("person_id") for p in situation_snapshot.get("people", [])
            if p.get("person_id")
        }
        if new_people:
            episode.people_involved = sorted(set(episode.people_involved) | new_people)

        if related_semantic_fact_ids:
            for fact_id in related_semantic_fact_ids:
                if fact_id not in episode.related_semantic_fact_ids:
                    episode.related_semantic_fact_ids.append(fact_id)
            episode.related_semantic_fact_ids = episode.related_semantic_fact_ids[-MAX_RELATED_FACT_IDS:]

        if related_behavior_id and related_behavior_id not in episode.related_behavior_ids:
            episode.related_behavior_ids.append(related_behavior_id)
            episode.related_behavior_ids = episode.related_behavior_ids[-MAX_RELATED_BEHAVIOR_IDS:]

        episode.importance_score = _importance_for(episode.repetition_count, episode.confidence)
        episode.updated_at = _now()

    def _link_related_episodes(self, episode_id: str, situation_signature: Dict[str, Any]):
        """
        Strengthens associations between episodes that resemble each other
        (partial signature overlap, not necessarily an exact match) so
        related-but-not-identical experiences stay discoverable together,
        without merging them into one record.
        """
        episode = self.episodes.get(episode_id)
        if episode is None:
            return
        for other in self.episodes.values():
            if other.episode_id == episode_id:
                continue
            overlap = _context_overlap_score(other.situation_signature, situation_signature)
            if overlap >= 0.75:
                if other.episode_id not in episode.linked_episode_ids:
                    episode.linked_episode_ids.append(other.episode_id)
                    episode.linked_episode_ids = episode.linked_episode_ids[-MAX_LINKED_EPISODE_IDS:]
                if episode_id not in other.linked_episode_ids:
                    other.linked_episode_ids.append(episode_id)
                    other.linked_episode_ids = other.linked_episode_ids[-MAX_LINKED_EPISODE_IDS:]


if __name__ == "__main__":
    # Minimal manual smoke test -- exercises log() -> reinforce -> find_similar
    # without needing Brain, STM, SceneBuilder, Semantic, or Behavioral wired up.
    from types import SimpleNamespace

    def _make_situation(obs_id, timestamp, person_id="owner001"):
        return SimpleNamespace(
            observation_id=obs_id,
            timestamp=timestamp,
            people=[{"person_id": person_id, "position": [1, 2, 3, 4], "confidence": 0.9,
                     "emotion": "stressed", "emotion_confidence": 0.6}],
            objects=[],
            pose={"posture": "lying_down", "torso_len": 0.2, "leg_len": 0.1},
            speech=[],
            environment={"temperature_c": 22.0, "humidity_pct": 40.0, "light_lux": 50.0, "distance_cm": 100.0},
            status_flags=[],
        )

    em = EpisodicMemory(episodes_path="/tmp/_episodic_memory_test.json")

    context = {"person_id": "owner001", "emotion": "stressed", "time_of_day": "night",
               "location_signature": "t22_l50", "activity": "lying_down"}
    action = {"type": "suggest_rest"}

    for i in range(4):
        situation = _make_situation(f"obs-{i}", f"2026-07-18T22:0{i}:00")
        episode_id = em.log(
            situation=situation,
            current_context=context,
            action=action,
            outcome={"accepted": True},
            success=True,
            reward=1.0,
            related_semantic_fact_ids=["fact-123"],
            related_behavior_id="behavior-abc",
        )

    print(f"Episodes stored: {len(em.episodes)}")
    for e in em.get_episodes():
        print(f"  {e.episode_id} reps={e.repetition_count} conf={e.confidence:.2f} "
              f"success_rate={e.success_rate:.2f} importance={e.importance_score}")

    similar = em.find_similar(context)
    print(f"find_similar() -> {len(similar)} match(es)")
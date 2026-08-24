"""
SemanticMemo.py
===============
Semantic Memory: Pluto's PERMANENT knowledge base.

What it is
----------
Semantic Memory stores generalized, structured KNOWLEDGE -- never raw sensor
values, camera frames, audio samples, or one-time events. It answers "what
is generally true" (owner's routines, preferences, habits, communication
style, commonly used objects, familiar locations, environmental patterns,
emotion patterns, interaction preferences) based on ACCUMULATED experience,
not any single observation.

What it is NOT
--------------
- It never talks to sensors and never processes STM's raw rows directly.
- It never stores an individual `Situation` permanently -- a Situation is
  only ever used as transient EVIDENCE toward a concept.
- It performs no decision-making or behavior generation. It is pure
  long-term understanding; acting on that understanding is someone else's
  job (a future planning/behavior module).

Where it sits
--------------
    STM (raw structured rows, tagged with ObservationID)
        -> SceneBuilder (groups same-cycle rows into one complete Situation)
            -> Brain.route_to_higher_memories()
                -> SemanticMemory.process(situations)   <-- this file

Semantic Memory receives ONLY complete `Situation` objects from SceneBuilder
(see SceneBuilder.py) -- never raw STM rows.

How it learns
-------------
Every incoming Situation is run through a set of pattern EXTRACTORS, each of
which looks for one *kind* of regularity (an object appearing repeatedly, a
person's emotion recurring in a context, a posture/time-of-day routine,
etc). Each extractor proposes candidate concepts as (category, conditions,
meaning, situation_id) tuples -- never as English sentences.

For each candidate:
    - If a matching SemanticFact already exists (same category + same
      conditions), Semantic Memory UPDATES it: confidence moves up when the
      new observation agrees, or down when it contradicts; observation_count
      increments; updated_at refreshes; the Situation's ID is added to
      supporting_situation_ids. No duplicate fact is created.
    - If no matching fact exists yet, the candidate is added to a PENDING
      evidence pool (keyed the same way). Only once a pending candidate has
      accumulated enough repeated evidence (PROMOTION_THRESHOLD observations)
      does it graduate into a permanent, structured SemanticFact.

This is how single Situations -- transient by design -- gradually become
stable, generalized knowledge without ever being stored verbatim themselves.
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

SEMANTIC_FACTS_PATH = os.path.join(os.path.dirname(__file__), "semantic_facts.json")
PENDING_EVIDENCE_PATH = os.path.join(os.path.dirname(__file__), "semantic_pending.json")

PROMOTION_THRESHOLD = 3          # repeated observations of the same candidate before it becomes a permanent fact
MAX_SUPPORTING_IDS = 50          # cap how many situation IDs a single fact keeps a reference to

CONFIDENCE_INITIAL = 0.4         # confidence assigned the moment a fact is first promoted
CONFIDENCE_GAIN = 0.12           # how much confidence rises per confirming observation (diminishing via (1-conf))
CONFIDENCE_PENALTY = 0.20        # how much confidence drops per contradicting observation
CONFIDENCE_DEACTIVATE_BELOW = 0.15  # facts whose confidence falls below this are marked inactive, not deleted

IMPORTANCE_THRESHOLDS = (5, 15)  # observation_count thresholds -> ("low", "medium", "high")

# Category constants -- the fixed vocabulary of concept types Semantic Memory learns.
CATEGORY_ROUTINE = "routine"
CATEGORY_PREFERENCE = "preference"
CATEGORY_HABIT = "habit"
CATEGORY_COMMUNICATION_STYLE = "communication_style"
CATEGORY_OBJECT_FAMILIARITY = "object_familiarity"
CATEGORY_LOCATION_FAMILIARITY = "location_familiarity"
CATEGORY_ENVIRONMENTAL_PATTERN = "environmental_pattern"
CATEGORY_EMOTION_PATTERN = "emotion_pattern"
CATEGORY_INTERACTION_PREFERENCE = "interaction_preference"


# --------------------------------------------------------------------------
# SEMANTIC FACT: the structured, permanent unit of knowledge
# --------------------------------------------------------------------------

@dataclass
class SemanticFact:
    """
    One piece of permanent, generalized knowledge. Never plain text --
    every field is structured so downstream code (behavior/planning, once
    built) can query it programmatically rather than parsing sentences.
    """
    fact_id: str
    category: str                              # one of the CATEGORY_* constants
    title: str                                  # short structured label, e.g. "owner_morning_desk_routine"
    conditions: Dict[str, Any]                  # when this knowledge applies, e.g. {"person_id": "owner001", "time_of_day": "morning"}
    meaning: Dict[str, Any]                     # the learned content itself, e.g. {"posture": "seated", "location_signature": "..."}
    confidence: float                           # 0.0-1.0, rises with agreeing evidence, falls with contradicting evidence
    importance: str                             # "low" | "medium" | "high"
    observation_count: int
    created_at: str
    updated_at: str
    supporting_situation_ids: List[str] = field(default_factory=list)
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SemanticFact":
        return SemanticFact(**d)


# --------------------------------------------------------------------------
# PERSISTENCE: thread-safe JSON-backed stores for facts + pending evidence
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

def _time_of_day_bucket(timestamp: str) -> str:
    """Coarse recurring time-of-day bucket used as a matching `condition`, not a conclusion."""
    try:
        hour = datetime.fromisoformat(timestamp).hour
    except (ValueError, TypeError):
        return "unknown"
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def _location_signature(environment: Dict[str, Any]) -> str:
    """
    Coarse, repeatable signature standing in for "which room/spot this is",
    built from bucketed environment readings.
    TODO(connect): replace with a real room/location ID once Pluto has
    beacons, SLAM, or fixed-position sensors -- this is a structural
    placeholder, not a claim about an actual known location.
    """
    if not environment:
        return "unknown"
    temp = environment.get("temperature_c")
    light = environment.get("light_lux")
    temp_bucket = "unknown" if temp is None else str(int(temp // 2) * 2)
    light_bucket = "unknown" if light is None else str(int(light // 50) * 50)
    return f"t{temp_bucket}_l{light_bucket}"


def _importance_for(observation_count: int) -> str:
    low_max, high_min = IMPORTANCE_THRESHOLDS
    if observation_count < low_max:
        return "low"
    if observation_count < high_min:
        return "medium"
    return "high"


def _conditions_match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Two candidates describe the 'same concept' if their conditions match exactly."""
    return a == b


# --------------------------------------------------------------------------
# PATTERN EXTRACTORS
# --------------------------------------------------------------------------
# Each extractor inspects one complete Situation and yields zero or more
# candidates as (category, conditions, meaning) tuples. Extractors never
# compare across situations themselves -- that matching/promotion logic
# lives in SemanticMemory below. Extractors also never write English
# conclusions into `meaning`; every value is a structured field.

def _extract_object_familiarity(situation) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    candidates = []
    for obj in situation.objects:
        object_class = obj.get("object_class")
        if not object_class:
            continue
        conditions = {"object_class": object_class}
        meaning = {
            "object_class": object_class,
            "typical_location_signature": _location_signature(situation.environment),
            "typical_time_of_day": _time_of_day_bucket(situation.timestamp),
        }
        candidates.append((CATEGORY_OBJECT_FAMILIARITY, conditions, meaning))
    return candidates


def _extract_location_familiarity(situation) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    signature = _location_signature(situation.environment)
    if signature == "unknown":
        return []
    conditions = {"location_signature": signature}
    meaning = {
        "location_signature": signature,
        "typical_time_of_day": _time_of_day_bucket(situation.timestamp),
        "typical_environment": situation.environment,
    }
    return [(CATEGORY_LOCATION_FAMILIARITY, conditions, meaning)]


def _extract_environmental_pattern(situation) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    if not situation.environment:
        return []
    conditions = {"time_of_day": _time_of_day_bucket(situation.timestamp)}
    meaning = {"environment": situation.environment}
    return [(CATEGORY_ENVIRONMENTAL_PATTERN, conditions, meaning)]


def _extract_emotion_pattern(situation) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    candidates = []
    for person in situation.people:
        person_id = person.get("person_id")
        emotion = person.get("emotion")
        if not person_id or person_id == "unknown" or not emotion or emotion == "unknown":
            continue
        conditions = {"person_id": person_id, "time_of_day": _time_of_day_bucket(situation.timestamp)}
        meaning = {"emotion": emotion, "location_signature": _location_signature(situation.environment)}
        candidates.append((CATEGORY_EMOTION_PATTERN, conditions, meaning))
    return candidates


def _extract_routine(situation) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    if not situation.pose or not situation.pose.get("posture"):
        return []
    candidates = []
    time_bucket = _time_of_day_bucket(situation.timestamp)
    location_signature = _location_signature(situation.environment)
    posture = situation.pose["posture"]
    known_people = [p.get("person_id") for p in situation.people if p.get("person_id") not in (None, "unknown")]
    for person_id in known_people or ["unknown"]:
        conditions = {"person_id": person_id, "time_of_day": time_bucket, "location_signature": location_signature}
        meaning = {"posture": posture}
        candidates.append((CATEGORY_ROUTINE, conditions, meaning))
    return candidates


def _extract_communication_style(situation) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    candidates = []
    for entry in situation.speech:
        if entry.get("type") != "speech":
            continue
        speaker_id = entry.get("speaker_id")
        if not speaker_id or speaker_id in ("unidentified", "no_registered_owner_voice"):
            continue
        conditions = {"speaker_id": speaker_id, "language": entry.get("language")}
        meaning = {
            "language": entry.get("language"),
            "typical_sound_level_db": entry.get("sound_level_db"),
            "typical_time_of_day": _time_of_day_bucket(situation.timestamp),
        }
        candidates.append((CATEGORY_COMMUNICATION_STYLE, conditions, meaning))
    return candidates


def _extract_interaction_preference(situation) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """
    Looks for co-occurrence of a known speaker with a known emotion in the
    same scene -- a structural proxy for "how this person tends to interact"
    (e.g. speaks while neutral vs. speaks while another emotion is present).
    TODO(connect): broaden once richer interaction signals (gesture,
    interruption patterns, response latency) are available from STM.
    """
    candidates = []
    speakers = {e.get("speaker_id") for e in situation.speech
                if e.get("type") == "speech" and e.get("speaker_id") not in (None, "unidentified", "no_registered_owner_voice")}
    for person in situation.people:
        person_id = person.get("person_id")
        emotion = person.get("emotion")
        if person_id in speakers and emotion and emotion != "unknown":
            conditions = {"person_id": person_id, "context": "speaking"}
            meaning = {"emotion_while_speaking": emotion}
            candidates.append((CATEGORY_INTERACTION_PREFERENCE, conditions, meaning))
    return candidates


_EXTRACTORS = [
    _extract_object_familiarity,
    _extract_location_familiarity,
    _extract_environmental_pattern,
    _extract_emotion_pattern,
    _extract_routine,
    _extract_communication_style,
    _extract_interaction_preference,
]

# NOTE on `habit`: a habit, as distinct from a routine, is a recurring
# ACTION pattern independent of a fixed time slot (e.g. "owner always picks
# up the mug within a minute of sitting down", rather than "owner sits at
# 9am"). TODO(connect): once STM/SceneBuilder carry sequential/temporal
# ordering between consecutive Situations (not just single-scene snapshots),
# add a `_extract_habit` extractor here that looks for object+posture
# co-occurrence across consecutive Situations rather than within just one.


# --------------------------------------------------------------------------
# SEMANTIC MEMORY
# --------------------------------------------------------------------------

class SemanticMemory:
    """
    Pluto's permanent knowledge base. Receives complete `Situation` objects
    (from SceneBuilder, via Brain) and either reinforces/contradicts existing
    `SemanticFact`s or accumulates pending evidence toward a new one.

    Independent of decision-making: this class only answers "what do we
    generally believe is true" -- it never triggers or suggests any action.
    """

    def __init__(self, facts_path: str = SEMANTIC_FACTS_PATH, pending_path: str = PENDING_EVIDENCE_PATH):
        self._facts_store = _JSONStore(facts_path)
        self._pending_store = _JSONStore(pending_path)
        self._lock = threading.Lock()

        self.facts: Dict[str, SemanticFact] = {}
        for d in self._facts_store.load():
            fact = SemanticFact.from_dict(d)
            self.facts[fact.fact_id] = fact

        # pending[key] = {"category", "conditions", "meanings": [meaning, ...], "situation_ids": [...]}
        # Stored on disk as a list of {"key":..., "record":{...}} pairs for JSON-friendliness.
        loaded_pending = self._pending_store.load()
        self.pending: Dict[str, Dict[str, Any]] = {item["key"]: item["record"] for item in loaded_pending} if loaded_pending else {}

        self.ready = True

    # ---- public API ------------------------------------------------------

    def process(self, situations: List[Any]):
        """
        Main entry point, called by Brain.route_to_higher_memories().
        `situations` is a list of SceneBuilder.Situation objects -- never
        raw STM rows. Each Situation is used only as transient evidence;
        none of it is stored verbatim once processed.
        """
        with self._lock:
            for situation in situations:
                self._process_one(situation)
            self._persist()

    def get_facts(self, category: Optional[str] = None, active_only: bool = True) -> List[SemanticFact]:
        """Query accumulated knowledge -- e.g. for a future behavior/planning module."""
        facts = list(self.facts.values())
        if category:
            facts = [f for f in facts if f.category == category]
        if active_only:
            facts = [f for f in facts if f.active]
        return facts

    # ---- internals ---------------------------------------------------------

    def _process_one(self, situation):
        situation_id = getattr(situation, "observation_id", None)
        if not situation_id:
            return

        candidates: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        for extractor in _EXTRACTORS:
            try:
                candidates.extend(extractor(situation))
            except Exception:
                # A single mis-shaped Situation should never take down learning
                # for every other candidate/extractor.
                continue

        for category, conditions, meaning in candidates:
            self._integrate_candidate(category, conditions, meaning, situation_id)

    def _integrate_candidate(self, category: str, conditions: Dict[str, Any], meaning: Dict[str, Any], situation_id: str):
        existing = self._find_matching_fact(category, conditions)
        if existing is not None:
            self._reinforce_or_contradict(existing, meaning, situation_id)
            return

        key = self._candidate_key(category, conditions)
        record = self.pending.get(key)
        if record is None:
            record = {"category": category, "conditions": conditions, "meanings": [], "situation_ids": []}
            self.pending[key] = record

        if situation_id not in record["situation_ids"]:
            record["meanings"].append(meaning)
            record["situation_ids"].append(situation_id)

        if len(record["situation_ids"]) >= PROMOTION_THRESHOLD:
            self._promote(key, record)

    @staticmethod
    def _candidate_key(category: str, conditions: Dict[str, Any]) -> str:
        return json.dumps({"category": category, "conditions": conditions}, sort_keys=True)

    def _find_matching_fact(self, category: str, conditions: Dict[str, Any]) -> Optional[SemanticFact]:
        for fact in self.facts.values():
            if fact.category == category and _conditions_match(fact.conditions, conditions):
                return fact
        return None

    def _promote(self, key: str, record: Dict[str, Any]):
        """Turns sufficiently-repeated pending evidence into a permanent SemanticFact."""
        now = datetime.now().isoformat(timespec="seconds")
        merged_meaning = self._merge_meanings(record["meanings"])
        fact = SemanticFact(
            fact_id=uuid.uuid4().hex,
            category=record["category"],
            title=self._build_title(record["category"], record["conditions"]),
            conditions=record["conditions"],
            meaning=merged_meaning,
            confidence=CONFIDENCE_INITIAL,
            importance=_importance_for(len(record["situation_ids"])),
            observation_count=len(record["situation_ids"]),
            created_at=now,
            updated_at=now,
            supporting_situation_ids=record["situation_ids"][-MAX_SUPPORTING_IDS:],
            active=True,
        )
        self.facts[fact.fact_id] = fact
        del self.pending[key]

    def _reinforce_or_contradict(self, fact: SemanticFact, meaning: Dict[str, Any], situation_id: str):
        """
        Updates an existing fact in place rather than creating a duplicate.
        Agreement raises confidence; disagreement lowers it. "Agreement" is
        judged per-category on whichever meaning field represents the
        concept's core claim (e.g. `posture` for a routine, `emotion` for an
        emotion pattern) -- everything else in `meaning` is treated as
        supplementary context and simply merged in.
        """
        agrees = self._meaning_agrees(fact.category, fact.meaning, meaning)

        if agrees:
            fact.confidence = min(0.99, fact.confidence + CONFIDENCE_GAIN * (1 - fact.confidence))
        else:
            fact.confidence = max(0.01, fact.confidence - CONFIDENCE_PENALTY)

        fact.observation_count += 1
        fact.importance = _importance_for(fact.observation_count)
        fact.updated_at = datetime.now().isoformat(timespec="seconds")

        if situation_id not in fact.supporting_situation_ids:
            fact.supporting_situation_ids.append(situation_id)
            fact.supporting_situation_ids = fact.supporting_situation_ids[-MAX_SUPPORTING_IDS:]

        fact.meaning = self._merge_meanings([fact.meaning, meaning])

        if fact.confidence < CONFIDENCE_DEACTIVATE_BELOW:
            fact.active = False

    @staticmethod
    def _meaning_agrees(category: str, existing_meaning: Dict[str, Any], new_meaning: Dict[str, Any]) -> bool:
        core_field_by_category = {
            CATEGORY_ROUTINE: "posture",
            CATEGORY_EMOTION_PATTERN: "emotion",
            CATEGORY_COMMUNICATION_STYLE: "language",
            CATEGORY_INTERACTION_PREFERENCE: "emotion_while_speaking",
            CATEGORY_OBJECT_FAMILIARITY: "object_class",
            CATEGORY_LOCATION_FAMILIARITY: "location_signature",
        }
        core_field = core_field_by_category.get(category)
        if core_field is None:
            return True  # no single defining field (e.g. environmental_pattern) -> treat as reinforcing
        if core_field not in existing_meaning or core_field not in new_meaning:
            return True
        return existing_meaning[core_field] == new_meaning[core_field]

    @staticmethod
    def _merge_meanings(meanings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Last-value-wins shallow merge across accumulated meaning dicts -- still fully structured, no text synthesis."""
        merged: Dict[str, Any] = {}
        for m in meanings:
            merged.update(m)
        return merged

    @staticmethod
    def _build_title(category: str, conditions: Dict[str, Any]) -> str:
        """Short structured label -- a slug, not a narrative sentence."""
        parts = [category] + [f"{k}={v}" for k, v in sorted(conditions.items())]
        return "_".join(str(p) for p in parts)

    def _persist(self):
        self._facts_store.save([f.to_dict() for f in self.facts.values()])
        self._pending_store.save([{"key": k, "record": v} for k, v in self.pending.items()])


if __name__ == "__main__":
    # Minimal manual smoke test using a hand-built Situation-like object,
    # so this file can be sanity-checked without wiring the whole Brain.
    from types import SimpleNamespace

    def _make_situation(obs_id, timestamp, person_id="owner001", emotion="neutral", posture="seated"):
        return SimpleNamespace(
            observation_id=obs_id,
            timestamp=timestamp,
            people=[{"person_id": person_id, "position": [1, 2, 3, 4], "confidence": 0.9,
                     "emotion": emotion, "emotion_confidence": 0.6}],
            objects=[{"object_class": "mug", "position": [5, 5, 10, 10], "distance_cm": None, "confidence": 0.8}],
            pose={"posture": posture, "torso_len": 0.2, "leg_len": 0.1},
            speech=[{"type": "speech", "text": "good morning", "language": "en",
                     "speaker_id": person_id, "sound_level_db": 41.0}],
            environment={"temperature_c": 24.0, "humidity_pct": 45.0, "light_lux": 220.0, "distance_cm": 120.0},
            status_flags=[],
        )

    sm = SemanticMemory(facts_path="/tmp/_semantic_facts_test.json", pending_path="/tmp/_semantic_pending_test.json")
    for i in range(5):
        situation = _make_situation(f"obs-{i}", f"2026-07-18T09:0{i}:00")
        sm.process([situation])

    print(f"Facts learned: {len(sm.facts)}")
    for fact in sm.get_facts():
        print(f"  [{fact.category}] {fact.title} conf={fact.confidence:.2f} obs={fact.observation_count}")
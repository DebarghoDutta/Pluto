"""
SceneBuilder.py
===============
Sits BETWEEN Short Term Memory (STM) and Semantic Memory.

STM produces a flat stream of structured, timestamped rows, each tagged with
an `observation_id` marking which acquisition cycle it belongs to (see
ShortTermMemo.py). SceneBuilder's only job is to collapse all rows sharing
an `observation_id` into a single, complete `Situation` object -- one
coherent scene combining everything STM saw/heard/measured at that moment
(people present, objects, pose, speech, environment readings).

SceneBuilder does NOT interpret, generalize, or learn anything. It performs
no pattern discovery and produces no English conclusions -- it only
reshapes STM's flat rows into a structured, complete-context object.
Discovering patterns, learning concepts, and turning repeated Situations
into human-readable knowledge is entirely Semantic Memory's job; Semantic
Memory should only ever receive `Situation` objects from SceneBuilder, never
STM's raw rows directly.

Usage (from Brain.py):
    from SceneBuilder import SceneBuilder

    scene_builder = SceneBuilder()
    rows = short_term.get_recent_data()
    situations = scene_builder.build_situations(rows)
    for situation in situations:
        semantic.process(situation)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Situation:
    """
    One complete, structured scene: every STM row sharing a single
    `observation_id`, reshaped into named context buckets. Still just facts --
    no narrative, no conclusions.
    """
    observation_id: str
    timestamp: str                                   # earliest timestamp among the cycle's rows
    people: List[Dict[str, Any]] = field(default_factory=list)      # face + emotion readings, merged by position
    objects: List[Dict[str, Any]] = field(default_factory=list)     # detected objects with position/distance
    pose: Optional[Dict[str, Any]] = None             # posture + raw joint state, if captured this cycle
    speech: List[Dict[str, Any]] = field(default_factory=list)      # recognized speech / ambient sound events
    environment: Dict[str, Any] = field(default_factory=dict)       # sensor readings (temp, humidity, light, distance)
    status_flags: List[Dict[str, Any]] = field(default_factory=list)  # hardware/status rows (e.g. "no_camera_frame_available")
    scene_text: Optional[str] = None                 # one-line NLP-generated description (filled by SceneNarrator, not SceneBuilder)


class SceneBuilder:
    """
    Groups STM's flat structured rows by `observation_id` and assembles each
    group into a `Situation`. Pure aggregation -- no inference.
    """

    def build_situations(self, rows: List[Dict[str, Any]]) -> List[Situation]:
        """
        `rows` is the list of dicts returned by ShortTermMemory.get_recent_data()
        (each row already has its `attributes` field decoded into a dict).
        Returns one Situation per distinct observation_id, ordered by
        earliest timestamp first.
        """
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            obs_id = row.get("observation_id")
            if not obs_id:
                continue
            grouped.setdefault(obs_id, []).append(row)

        situations = [self._build_one(obs_id, group_rows) for obs_id, group_rows in grouped.items()]
        situations.sort(key=lambda s: s.timestamp)
        return situations

    def _build_one(self, observation_id: str, rows: List[Dict[str, Any]]) -> Situation:
        rows_sorted = sorted(rows, key=lambda r: r.get("timestamp", ""))
        situation = Situation(observation_id=observation_id, timestamp=rows_sorted[0]["timestamp"])

        # Face + emotion rows both key off the same face `position`, so merge
        # them into one per-person entry rather than two disconnected rows.
        faces_by_position: Dict[tuple, Dict[str, Any]] = {}

        for row in rows_sorted:
            source = row.get("source")
            data_type = row.get("data_type")
            attrs = row.get("attributes") or {}
            confidence = row.get("confidence")

            if source == "camera" and data_type == "face":
                key = tuple(attrs.get("position") or [])
                entry = faces_by_position.setdefault(key, {})
                entry["person_id"] = attrs.get("person_id")
                entry["position"] = attrs.get("position")
                entry["confidence"] = confidence

            elif source == "camera" and data_type == "emotion":
                key = tuple(attrs.get("position") or [])
                entry = faces_by_position.setdefault(key, {})
                entry["emotion"] = attrs.get("emotion")
                entry["emotion_confidence"] = confidence

            elif source == "camera" and data_type == "object":
                situation.objects.append({
                    "object_class": attrs.get("object_class"),
                    "position": attrs.get("position"),
                    "distance_cm": attrs.get("distance_cm"),
                    "confidence": confidence,
                })

            elif source == "camera" and data_type == "pose":
                situation.pose = {
                    "posture": attrs.get("posture"),
                    "torso_len": attrs.get("torso_len"),
                    "leg_len": attrs.get("leg_len"),
                }

            elif source == "camera" and data_type == "status":
                situation.status_flags.append({"source": "camera", **attrs})

            elif source == "mic" and data_type == "speech":
                situation.speech.append({
                    "type": "speech",
                    "text": attrs.get("text"),
                    "language": attrs.get("language"),
                    "speaker_id": attrs.get("speaker_id"),
                    "sound_level_db": attrs.get("sound_level_db"),
                })

            elif source == "mic" and data_type == "ambient_sound":
                situation.speech.append({
                    "type": "ambient_sound",
                    "sound_level_db": attrs.get("sound_level_db"),
                })

            elif source == "mic" and data_type == "status":
                situation.status_flags.append({"source": "mic", **attrs})

            elif source == "sensor":
                situation.environment[data_type] = attrs.get("value")

        situation.people = list(faces_by_position.values())
        return situation
"""
SceneNarrator.py
=================
Sits BETWEEN SceneBuilder and everything downstream (Semantic Memory,
scene_observation table, etc.).

SceneBuilder groups STM's flat rows into a `Situation` -- still just
structured facts, no sentence. SceneNarrator's only job is to take ONE
Situation and turn it into ONE plain-English sentence describing the scene,
using a locally-hosted Qwen2.5-3B model, instead of a hand-written /
rule-based description.

This module does not decide what a scene means (that's still Semantic
Memory's job downstream) -- it only narrates what SceneBuilder already
assembled. Camera-only for now: people, objects, pose. Speech/environment
are deliberately left out of the prompt (see _situation_to_prompt_facts())
since this test is scoped to what the camera sees.

Mock-safe by design (same pattern as pg_bridge.py): if the Qwen server
isn't reachable, narrate() returns None and callers must treat that as
"no scene_text this cycle", never as a crash.

Model hosting assumption: Qwen2.5-3B served locally via Ollama
(https://ollama.com), e.g. on the Pi itself:
    ollama pull qwen2.5:3b
    ollama serve            # default: http://localhost:11434

If Ollama runs elsewhere (dev machine, another box on the LAN), set the
OLLAMA_HOST env var, e.g. OLLAMA_HOST=http://192.168.1.50:11434

Usage (from Brain.py):
    from SceneNarrator import SceneNarrator

    narrator = SceneNarrator()
    situation.scene_text = narrator.narrate(situation)
"""

import os
import json
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from SceneBuilder import Situation


_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_OLLAMA_GENERATE_URL = _OLLAMA_HOST.rstrip("/") + "/api/generate"
_MODEL_NAME = os.environ.get("PLUTO_NLP_MODEL", "qwen2.5:3b")
_REQUEST_TIMEOUT_SECONDS = 15

_SYSTEM_PROMPT = (
    "You describe a robot's camera scene in exactly ONE short factual "
    "sentence. Only state what is given below. Do not add people, objects, "
    "emotions, or guesses that are not present in the data. Do not add "
    "commentary, opinions, or extra sentences. Output the sentence only -- "
    "no preamble, no quotes, no labels."
)


class SceneNarrator:
    """
    Converts one Situation into one plain-English sentence via a local
    Qwen2.5-3B server. Pure translation -- no inference about meaning,
    patterns, or importance (that stays Semantic Memory's job).
    """

    def __init__(self, model_name: str = _MODEL_NAME, host: str = _OLLAMA_HOST):
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.generate_url = self.host + "/api/generate"

    def narrate(self, situation: Situation) -> Optional[str]:
        """
        Returns one clean sentence describing `situation`, or None if the
        situation has nothing camera-relevant to describe, or if the model
        server can't be reached. Never raises.
        """
        facts = self._situation_to_prompt_facts(situation)
        if not facts:
            return None

        prompt = self._build_prompt(facts)

        try:
            raw = self._call_ollama(prompt)
        except Exception:
            return None

        return self._clean_sentence(raw)

    # ---- fact extraction (camera-only: people, objects, pose) --------------

    @staticmethod
    def _situation_to_prompt_facts(situation: Situation) -> Optional[Dict[str, Any]]:
        """
        Pulls only the camera-relevant fields out of a Situation, in the
        plain form the model prompt needs. Returns None if there is nothing
        camera-related to narrate this cycle (e.g. a mic/sensor-only tick).
        """
        if not situation.people and not situation.objects and not situation.pose:
            return None

        people = []
        for person in situation.people:
            people.append({
                "who": person.get("person_id") or "an unrecognized person",
                "emotion": person.get("emotion"),
            })

        objects = [
            {
                "what": obj.get("object_class"),
                "distance_cm": obj.get("distance_cm"),
            }
            for obj in situation.objects
            if obj.get("object_class")
        ]

        pose = None
        if situation.pose and situation.pose.get("posture"):
            pose = situation.pose.get("posture")

        return {"people": people, "objects": objects, "pose": pose}

    @staticmethod
    def _build_prompt(facts: Dict[str, Any]) -> str:
        return (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Data:\n{json.dumps(facts, ensure_ascii=False)}\n\n"
            f"One-sentence scene description:"
        )

    # ---- model call ----------------------------------------------------

    def _call_ollama(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }).encode("utf-8")

        req = urllib.request.Request(
            self.generate_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body.get("response", "")

    @staticmethod
    def _clean_sentence(raw: str) -> Optional[str]:
        """Strips whitespace/quotes and collapses accidental multi-line
        output down to the first sentence, so a stray extra line from the
        model never leaks into scene_text."""
        if not raw:
            return None
        text = raw.strip().strip('"').strip()
        text = text.splitlines()[0].strip() if text else text
        return text or None
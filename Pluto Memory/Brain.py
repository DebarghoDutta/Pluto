"""
Brain.py
========
Central orchestrator for Pluto's memory architecture AND the full
perception -> knowledge -> decision -> execution -> feedback -> memory-update
pipeline that runs on top of it.

Subsystems
----------
    ShortTermMemory   -> raw sensory input capture (camera / mic / sensors)      [IMPLEMENTED]
    SceneBuilder      -> groups same-cycle STM rows into complete Situations    [IMPLEMENTED]
    SemanticMemory    -> permanent knowledge learned from repeated Situations   [IMPLEMENTED]
    BehavioralMemory  -> selects/learns the best action for the current context [IMPLEMENTED]
    EpisodicMemory    -> timestamped event/experience log                      [IMPLEMENTED]

This mirrors the Memory Core screen already built in gui.py (four subsystem
cards: Short Term, Semantic, Behavioral, Episodic).

Routing rule (unchanged, now enforced more broadly)
----------------------------------------------------
No memory subsystem ever talks to another memory subsystem directly.
Brain is the ONLY thing that reads from one memory and writes into another.
Concretely:
    - STM never talks to Semantic/Behavioral/Episodic Memory.
    - SceneBuilder only reshapes STM rows into Situations; it interprets
      nothing and calls no other memory.
    - SemanticMemory never sees raw STM rows -- only Situations, via Brain.
    - BehavioralMemory NEVER sees raw STM rows and NEVER sees a raw
      `Situation` object either. It only ever receives the small, structured
      `current_context` dict Brain derives from a Situation, plus whatever
      SemanticFacts / episodes Brain looked up on its behalf. This keeps
      Behavioral Memory from ever duplicating what Semantic Memory already
      knows -- Semantic Memory stays the single source of factual knowledge,
      Behavioral Memory only ever stores/refs structured action strategies.
    - EpisodicMemory only receives Situations + current_context + the
      executed action + its outcome, handed to it by Brain -- never pulled
      directly by Semantic or Behavioral Memory.

Pipeline stages (see `run_cycle()`)
------------------------------------
    1. PERCEPTION      : pull STM rows, group into complete Situations via SceneBuilder.
    2. KNOWLEDGE        : for each Situation, derive a structured current_context,
                          look up relevant SemanticFacts, and (once built) relevant
                          similar episodes from Episodic Memory.
    3. DECISION         : BehavioralMemory.select_action(context, facts, episodes)
                          picks the best learned action, or None if nothing qualifies.
    4. EXECUTION        : Brain executes the chosen action [TODO(connect): wire to
                          Pluto's real actuators/output channels -- speech, motion,
                          display, etc]. A learn_or_reinforce() call registers the
                          (context, action) pair the moment it's chosen.
    5. FEEDBACK         : the observed outcome (explicit or inferred) is turned into
                          a success/failure + numeric reward and fed back via
                          BehavioralMemory.reinforce().
    6. MEMORY UPDATE    : SemanticMemory.process(situations) updates permanent
                          knowledge; EpisodicMemory logs the Situation + action +
                          outcome [TODO(connect)]; BehavioralMemory.decay_unused()
                          runs periodically for maintenance.

Owner data note:
    Face/voice registration (name, dob, images, samples) is submitted by the
    desktop software (gui.py) to server.py (FastAPI), which hands it to
    owner_manager.py for validation, storage, and persistence via database.py.
    Brain owns the single OwnerManager instance and shares it with STM, so
    call `brain.reload_owners()` from server.py right after a registration
    (or update) request finishes -- STM's face/speaker recognition data then
    refreshes immediately with no manual file copying or restart needed.

Usage:
    from Brain import Brain

    brain = Brain()
    brain.start()
    ...
    brain.run_cycle()          # perception -> ... -> memory update, once
    ...
    brain.reload_owners()      # after server.py finishes a registration request
    ...
    brain.stop()
"""

import time
import json
import hashlib
import threading
from collections import deque
from typing import Any, Dict, List, Optional

from ShortTermMemo import ShortTermMemory
from SceneBuilder import SceneBuilder, Situation
from SceneNarrator import SceneNarrator
import pg_bridge
from SemanticMemo import SemanticMemory, _time_of_day_bucket, _location_signature
from BehaviouralMemo import BehavioralMemory, Behavior
from EpisodicMemo import EpisodicMemory
from owner_manager import OwnerManager


# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

# Run BehavioralMemory.decay_unused() every N calls to run_cycle(), not every
# single cycle -- it's maintenance, not a per-decision necessity.
DECAY_EVERY_N_CYCLES = 12

# How often the background pipeline loop calls run_cycle(). Deliberately not
# tighter than STM's own camera cadence (CAMERA_SAMPLE_INTERVAL == 2s in
# ShortTermMemo.py) -- polling faster than new observations can possibly
# arrive just burns CPU/heat on the Pi5 for zero benefit.
CYCLE_INTERVAL_SECONDS = 3.0

# Bounds how many observation_ids we remember as "already processed" so the
# same rows sitting in STM's 5-minute rolling window aren't turned into new
# actions/episodes on every tick. Sized comfortably above the max number of
# observation cycles that can exist in that window (300s / 2s ~= 150).
MAX_PROCESSED_OBSERVATION_IDS = 400


# --------------------------------------------------------------------------
# Brain
# --------------------------------------------------------------------------

class Brain:
    """
    Wires together all of Pluto's memory subsystems and runs the full
    perception -> knowledge -> decision -> execution -> feedback ->
    memory-update pipeline every cycle.
    """

    def __init__(self):
        # Single OwnerManager instance shared across everything that needs
        # owner data (STM face/speaker recognition, future higher memories).
        self.owner_manager = OwnerManager()

        self.short_term = ShortTermMemory(owner_manager=self.owner_manager)
        self.scene_builder = SceneBuilder()
        self.scene_narrator = SceneNarrator()
        self.semantic = SemanticMemory()
        self.behavioral = BehavioralMemory()
        self.episodic = EpisodicMemory()

        self._running = False
        self._cycle_count = 0

        # -- continuous-run bookkeeping ------------------------------------
        # Observation_ids already turned into Situations this process, so
        # re-seeing them in STM's rolling window on a later tick is a no-op
        # instead of a duplicate action/episode.
        self._processed_observation_ids: deque = deque(maxlen=MAX_PROCESSED_OBSERVATION_IDS)
        self._processed_observation_id_set: set = set()
        # Content signature of the most recently *stored* Situation, used to
        # skip writing a duplicate action/episode when back-to-back
        # observations describe the same scene (see _situation_signature()).
        self._last_situation_signature: Optional[str] = None

        # Background thread that keeps run_cycle() ticking on its own once
        # start() is called -- this is what makes the full perception ->
        # memory-update pipeline actually run continuously in production
        # (server.py only calls brain.start()/brain.stop(); it never has to
        # know about the cycle loop).
        self._cycle_loop_running = False
        self._cycle_thread: Optional[threading.Thread] = None

    def start(self):
        """Start all input capture AND the background pipeline loop that
        keeps run_cycle() ticking on its own (see CYCLE_INTERVAL_SECONDS)."""
        pg_bridge.begin_session(session_type="perception")
        self.short_term.start()
        self._running = True
        self._start_cycle_loop()

    def stop(self):
        self._stop_cycle_loop()
        self.short_term.stop()
        pg_bridge.close_session()
        self._running = False

    # ---- background pipeline loop --------------------------------------------

    def _cycle_loop(self):
        while self._cycle_loop_running:
            try:
                self.run_cycle()
            except Exception:
                # A single bad cycle must never kill the whole background
                # loop -- same "never let a hiccup stop the loop" pattern
                # used by the websocket/camera/telemetry broadcast loops.
                pass
            time.sleep(CYCLE_INTERVAL_SECONDS)

    def _start_cycle_loop(self):
        if self._cycle_loop_running:
            return
        self._cycle_loop_running = True
        self._cycle_thread = threading.Thread(target=self._cycle_loop, daemon=True)
        self._cycle_thread.start()

    def _stop_cycle_loop(self):
        self._cycle_loop_running = False

    def get_short_term_snapshot(self, seconds=None):
        """Raw rows currently held in the short-term rolling window."""
        return self.short_term.get_recent_data(seconds)

    def reload_owners(self):
        """
        Call this from server.py right after a new owner registration (or an
        update to an existing owner) has been saved via owner_manager.py.
        Refreshes STM's face + voice recognition data immediately -- no
        manual file copying, no restart required.
        """
        self.short_term.reload_owners()

    # ---- full pipeline -----------------------------------------------------

    def run_cycle(self):
        """
        Runs one full pass of the pipeline:
            PERCEPTION -> KNOWLEDGE -> DECISION -> EXECUTION -> FEEDBACK -> MEMORY UPDATE
        Safe to call repeatedly (e.g. from a timed loop) -- each call is a
        self-contained cycle over whatever Situations exist right now.
        """
        situations = self._perceive()
        if not situations:
            return

        for situation in situations:
            # Mark seen immediately so a slow/failed situation is never
            # rebuilt and retried forever on the next tick.
            self._mark_observation_processed(situation.observation_id)

            signature = self._situation_signature(situation)
            if signature == self._last_situation_signature:
                # Same scene as the last one we actually stored (person,
                # objects, pose, speech, environment all identical) -- skip
                # decision/execution/memory-update entirely so no duplicate
                # action/episode gets written, and so the Pi5 doesn't burn
                # cycles re-deciding something it already decided.
                continue
            self._last_situation_signature = signature

            current_context, semantic_facts, similar_episodes = self._gather_knowledge(situation)

            chosen_behavior = self._decide(current_context, semantic_facts, similar_episodes)

            action, behavior_id = self._execute(current_context, semantic_facts, chosen_behavior)

            outcome, success, reward = self._observe_feedback(situation, action, chosen_behavior)

            self._update_memories(situation, action, behavior_id, outcome, success, reward)

        self._cycle_count += 1
        if self._cycle_count % DECAY_EVERY_N_CYCLES == 0:
            self.behavioral.decay_unused()

    # ---- stage 1: perception ------------------------------------------------

    def _perceive(self) -> List[Situation]:
        """
        Pulls STM's raw structured rows and groups them into complete
        Situations via SceneBuilder. STM itself never interprets data, and
        SceneBuilder never interprets data either -- it only reshapes rows
        into complete scenes.

        get_short_term_snapshot() returns STM's whole rolling window (up to
        STM_RETENTION_SECONDS), not just rows written since the last cycle,
        so without filtering here the same observation cycles would be
        rebuilt into "new" Situations and re-run through decision/execution/
        memory-update on every single tick. Drop any observation_id already
        turned into a Situation in a previous cycle.
        """
        rows = self.get_short_term_snapshot()
        situations = self.scene_builder.build_situations(rows)
        fresh = [
            s for s in situations
            if s.observation_id not in self._processed_observation_id_set
        ]

        # NLP narration (Qwen2.5-3B via SceneNarrator) happens here -- right
        # after SceneBuilder assembles the Situation, before anything else
        # (knowledge lookup, decision, memory update) sees it. Mock-safe:
        # narrate() returns None if the model server isn't reachable, and
        # scene_text just stays None for that cycle.
        for situation in fresh:
            situation.scene_text = self.scene_narrator.narrate(situation)
            if situation.scene_text:
                pg_bridge.log_scene_observation(situation.scene_text)

        return fresh

    def _mark_observation_processed(self, observation_id: str):
        """Records that this observation_id has now been turned into a
        Situation and run through the pipeline (or explicitly skipped as a
        duplicate), so _perceive() never rebuilds it again. Bounded deque
        keeps memory flat regardless of uptime."""
        if observation_id in self._processed_observation_id_set:
            return
        if len(self._processed_observation_ids) == self._processed_observation_ids.maxlen:
            oldest = self._processed_observation_ids.popleft()
            self._processed_observation_id_set.discard(oldest)
        self._processed_observation_ids.append(observation_id)
        self._processed_observation_id_set.add(observation_id)

    @staticmethod
    def _situation_signature(situation: Situation) -> str:
        """
        Content fingerprint of a Situation, deliberately excluding
        `observation_id` and `timestamp` (those are always unique/new) so
        two observation cycles that saw/heard/measured the exact same thing
        hash to the same signature. Used to detect "observation A at t1 ==
        observation B at t2" and skip storing a duplicate action for it.
        """
        payload = {
            "people": situation.people,
            "objects": situation.objects,
            "pose": situation.pose,
            "speech": situation.speech,
            "environment": situation.environment,
            "status_flags": situation.status_flags,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    # ---- stage 2: knowledge -------------------------------------------------

    def _gather_knowledge(self, situation: Situation):
        """
        Derives the small, structured `current_context` dict that
        BehavioralMemory is allowed to see (never the raw Situation, never
        raw STM rows), then looks up whatever relevant knowledge/experience
        exists for it:
            - SemanticMemory.get_facts()  -> relevant permanent knowledge
            - EpisodicMemory.find_similar() -> relevant past experience [TODO(connect)]
        """
        current_context = self._build_current_context(situation)

        # Semantic Memory stays the single source of factual knowledge --
        # Brain reads from it here and hands the result to Behavioral Memory
        # as read-only reference material; Behavioral Memory never queries
        # SemanticMemory itself and never copies facts into its own storage.
        semantic_facts = self._relevant_semantic_facts(current_context)

        similar_episodes = self.episodic.find_similar(current_context)

        return current_context, semantic_facts, similar_episodes

    def _build_current_context(self, situation: Situation) -> Dict[str, Any]:
        """
        Reduces one complete Situation down to the coarse, structured
        context BehavioralMemory operates on. This is the ONLY thing derived
        from a Situation that ever crosses into BehavioralMemory -- the
        Situation object itself is never passed onward.
        """
        primary_person_id = "unknown"
        primary_emotion = "unknown"
        if situation.people:
            # Prefer a recognized owner if more than one person is present.
            recognized = [p for p in situation.people if p.get("person_id") not in (None, "unknown")]
            primary = recognized[0] if recognized else situation.people[0]
            primary_person_id = primary.get("person_id") or "unknown"
            primary_emotion = primary.get("emotion") or "unknown"

        activity = "unknown"
        if situation.pose and situation.pose.get("posture"):
            activity = situation.pose["posture"]

        return {
            "person_id": primary_person_id,
            "emotion": primary_emotion,
            "time_of_day": _time_of_day_bucket(situation.timestamp),
            "location_signature": _location_signature(situation.environment),
            "activity": activity,
        }

    def _relevant_semantic_facts(self, current_context: Dict[str, Any]):
        """
        Pulls SemanticMemory's active facts and narrows them to ones plausibly
        relevant to the current context (matching person and/or time-of-day),
        so Behavioral Memory gets a short, targeted list rather than the
        entire knowledge base every cycle.
        """
        all_facts = self.semantic.get_facts(active_only=True)
        person_id = current_context.get("person_id")
        time_of_day = current_context.get("time_of_day")

        relevant = [
            fact for fact in all_facts
            if fact.conditions.get("person_id") in (None, person_id)
            and fact.conditions.get("time_of_day") in (None, time_of_day)
        ]
        return relevant

    # ---- stage 3: decision ---------------------------------------------------

    def _decide(self, current_context, semantic_facts, similar_episodes) -> Optional[Behavior]:
        """
        Asks BehavioralMemory for the best-ranked action given everything
        gathered in the knowledge stage. Returns None if nothing clears
        BehavioralMemory's minimum selection score, in which case execution
        falls back to a safe no-op.
        """
        return self.behavioral.select_action(current_context, semantic_facts, similar_episodes)

    # ---- stage 4: execution ---------------------------------------------------

    def _execute(self, current_context, semantic_facts, chosen_behavior: Optional[Behavior]):
        """
        Executes the chosen action and registers it with BehavioralMemory via
        learn_or_reinforce() -- this happens for EVERY action taken, whether
        it came from an existing learned Behavior or is being tried for the
        first time (the "default/explicit policy" fallback below), so the
        (context, action) pair always ends up tracked for future reinforcement.
        """
        if chosen_behavior is not None:
            action = chosen_behavior.action
        else:
            # TODO(connect): replace with Pluto's actual default/explicit
            # policy for "no confident learned behavior applies yet" (e.g. a
            # safe default like a neutral acknowledgement). Using a
            # structured placeholder here so the pipeline never breaks on
            # sparse behavioral data.
            action = {"type": "no_confident_action", "fallback": True}

        # TODO(connect): route `action` to Pluto's real output layer here --
        # speech synthesis, motion, display update, etc. Execution result
        # (e.g. "was it accepted", "did it complete") should flow into
        # _observe_feedback() below.
        self._perform_action(action)

        behavior_id = self.behavioral.learn_or_reinforce(current_context, action, semantic_facts)
        return action, behavior_id

    def _perform_action(self, action: Dict[str, Any]):
        """
        TODO(connect): the actual hardware/output call for `action` (speech
        synthesis, motion, display, etc). No-op placeholder for now so the
        pipeline is fully runnable end-to-end before actuators exist.
        """
        pass

    # ---- stage 5: feedback ---------------------------------------------------

    def _observe_feedback(self, situation: Situation, action: Dict[str, Any], chosen_behavior: Optional[Behavior]):
        """
        Turns the observed real-world outcome into a (outcome_summary,
        success, reward) triple for BehavioralMemory.reinforce().

        TODO(connect): replace this placeholder with Pluto's real feedback
        signal once available -- e.g. explicit owner feedback captured by
        STM next cycle, or an inferred signal (owner's emotion improving,
        a verbal acknowledgement, task completion). Until that signal
        exists, unproven actions are treated as neutral (neither reinforced
        nor penalized) rather than guessed at.
        """
        outcome_summary = {"observed": False}
        success = True
        reward = 0.0
        return outcome_summary, success, reward

    # ---- stage 6: memory update -----------------------------------------------

    def _update_memories(self, situation, action, behavior_id, outcome, success, reward):
        """
        Updates all long-term memory subsystems from this cycle's single
        Situation + the action taken + its outcome:
            - SemanticMemory.process()     updates/accumulates permanent knowledge
            - BehavioralMemory.reinforce()  strengthens/weakens the action just taken
            - EpisodicMemory.log()          records the full episode (situation +
              context + action + outcome + emotional state + related facts/behavior),
              and its returned episode_id is linked back into the Behavior via
              learn_or_reinforce()'s `episode_id` parameter.
        """
        self.semantic.process([situation])

        self.behavioral.reinforce(behavior_id, success=success, reward=reward, outcome_summary=outcome)

        current_context = self._build_current_context(situation)
        semantic_facts = self._relevant_semantic_facts(current_context)

        episode_id = self.episodic.log(
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
        if episode_id:
            # Link this episode to the behavior it produced, now that we
            # have an episode_id -- re-registering is a cheap idempotent
            # update (see BehavioralMemory.learn_or_reinforce()).
            self.behavioral.learn_or_reinforce(
                self._build_current_context(situation), action, episode_id=episode_id
            )

    # ---- legacy/manual entry point -------------------------------------------

    def route_to_higher_memories(self):
        """
        Kept for backward compatibility with any external callers still
        using the old name. Delegates to the full pipeline in run_cycle().
        """
        self.run_cycle()


if __name__ == "__main__":
    # brain.start() now starts STM capture AND the background cycle loop
    # (see CYCLE_INTERVAL_SECONDS) that keeps run_cycle() ticking on its
    # own -- no manual run_cycle() calls needed here anymore. This block is
    # just a status printer for local/manual testing.
    brain = Brain()
    brain.start()
    try:
        print(f"Pluto Brain running (pipeline ticking every {CYCLE_INTERVAL_SECONDS}s). "
              "Press Ctrl+C to stop.")
        while True:
            time.sleep(5)
            snapshot = brain.get_short_term_snapshot()
            print(f"STM rows in window: {len(snapshot)} | cycles run: {brain._cycle_count}")
    except KeyboardInterrupt:
        brain.stop()
        print("Stopped.")
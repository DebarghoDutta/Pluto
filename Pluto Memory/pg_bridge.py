"""
pg_bridge.py
============
Bridges Pluto's Pi-side perception pipeline (ShortTermMemo.py's
CameraCapture) to the persistent PostgreSQL database defined in the sibling
Pluto_Database package.

Exactly seven tables are touched here, matching what Debargho asked for --
nothing else in Pluto_Database is read or written from this module:

    CameraCapture._detect_objects()  -> Observation/Visual_obs.py (visual_obs)
    CameraCapture._detect_face()     -> Identity/event_identity.py (event_identity)
    face match -> owner recall/create -> Identity/Owners.py (owners)
    one row opened per Brain run     -> Core/sessions.py (sessions)
    SceneNarrator.narrate() output   -> Observation/Scene_obs.py (scene_observation)
    owner_manager.py register/update -> Identity/Owners.py (owners)
    owner_manager.py register/update -> Identity/face_profile.py (face)
    owner_manager.py register/update -> Identity/voice profile.py (voice)

Folder layout assumed (both live on the Pi, side by side):
    <root>/Pluto/Pluto Memory/pg_bridge.py     <- this file
    <root>/Pluto_Database/Pluto_Database/...   <- table modules + .env

If your layout differs, set the PLUTO_DATABASE_ROOT environment variable to
the absolute path of the Pluto_Database/Pluto_Database folder.

Mock-safe by design (same pattern as the rest of ShortTermMemo.py): if
Postgres isn't reachable -- wrong folder layout, DB not running, developing
off-Pi -- every public function here becomes a silent no-op so detection
keeps working off the CSV short-term log even without Postgres wired up.
"""

import os
import sys
import threading

_DEFAULT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Pluto_Database", "Pluto_Database")
)
_PLUTO_DATABASE_ROOT = os.environ.get("PLUTO_DATABASE_ROOT", _DEFAULT_ROOT)

_READY = False
_INIT_ERROR = None
_session_id = None
_lock = threading.Lock()


def _add_path(p):
    if p and os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


def _init():
    """Wires up imports from Pluto_Database and makes sure the four tables
    this module needs exist. Never raises -- failures just leave _READY False."""
    global _READY, _INIT_ERROR
    global _start_session, _end_session
    global _get_or_create_owner_row, _touch_owner_last_seen, _set_owner_status
    global _insert_event_identity, _insert_visual_obs, _insert_scene_observation
    global _insert_face_profile, _insert_voice_profile
    try:
        _add_path(_PLUTO_DATABASE_ROOT)
        for sub in ("Core", "Identity", "Observation"):
            _add_path(os.path.join(_PLUTO_DATABASE_ROOT, sub))

        from sessions import create_sessions_table, start_session, end_session
        from Owners import create_owners_table, get_or_create_owner_row, touch_owner_last_seen, set_owner_status
        from event_identity import create_event_identity_table, insert_event_identity
        from Visual_obs import create_visual_obs_table, insert_visual_obs
        from Scene_obs import create_scene_observation_table, insert_scene_observation
        from face_profile import create_face_table, insert_face_profile

        # "voice profile.py" has a space in its filename, so it can't be
        # imported with a plain `import` statement -- load it by path instead.
        import importlib.util
        _voice_profile_path = os.path.join(_PLUTO_DATABASE_ROOT, "Identity", "voice profile.py")
        _spec = importlib.util.spec_from_file_location("voice_profile", _voice_profile_path)
        _voice_profile = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_voice_profile)
        create_voice_table = _voice_profile.create_voice_table
        insert_voice_profile = _voice_profile.insert_voice_profile

        create_sessions_table()
        create_owners_table()
        create_event_identity_table()
        create_visual_obs_table()
        create_scene_observation_table()
        create_face_table()
        create_voice_table()

        _start_session = start_session
        _end_session = end_session
        _get_or_create_owner_row = get_or_create_owner_row
        _touch_owner_last_seen = touch_owner_last_seen
        _set_owner_status = set_owner_status
        _insert_event_identity = insert_event_identity
        _insert_visual_obs = insert_visual_obs
        _insert_scene_observation = insert_scene_observation
        _insert_face_profile = insert_face_profile
        _insert_voice_profile = insert_voice_profile

        _READY = True
    except Exception as exc:  # noqa: BLE001 - deliberately broad, mirrors rest of file
        _READY = False
        _INIT_ERROR = exc


_init()


def is_ready() -> bool:
    """True once Postgres is reachable and the four tables are confirmed."""
    return _READY


def init_error():
    """Returns the exception hit during setup, if any (for logging/debugging)."""
    return _INIT_ERROR


# --------------------------------------------------------------------------
# SESSION LIFECYCLE (call from Brain.start() / Brain.stop())
# --------------------------------------------------------------------------

def begin_session(session_type="perception", location=None):
    """Opens one sessions row for this Brain run. Safe no-op if Postgres
    isn't reachable -- returns None in that case."""
    global _session_id
    if not _READY:
        return None
    with _lock:
        try:
            _session_id = _start_session(session_type=session_type, location=location)
        except Exception:
            _session_id = None
    return _session_id


def close_session():
    """Closes the sessions row opened by begin_session(), if any."""
    global _session_id
    if not _READY or _session_id is None:
        return
    with _lock:
        try:
            _end_session(_session_id)
        except Exception:
            pass
        _session_id = None


# --------------------------------------------------------------------------
# LOGGING (call from CameraCapture detection methods)
# --------------------------------------------------------------------------

def log_object_detection(object_class, confidence, position):
    """Mirrors one _detect_objects() CSV row into visual_obs."""
    if not _READY or _session_id is None:
        return
    try:
        _insert_visual_obs(
            session_id=_session_id,
            source="camera",
            entity_id=object_class,
            confidence=confidence,
            location_data=str(position) if position is not None else None,
        )
    except Exception:
        pass


def log_face_event(owner_name, confidence, position):
    """
    Mirrors one _detect_face() CSV row into event_identity.

    `owner_name` should be the recognized owner's name, or None/"" if the
    face didn't match any registered owner. When it's a match, the owner is
    recalled (or created, on first sighting) in the Postgres `owners` table
    and its last_seen is stamped -- this is the "recall from Owners.py" step.
    """
    if not _READY or _session_id is None:
        return
    try:
        if owner_name:
            owner_row = _get_or_create_owner_row(owner_name)
            _touch_owner_last_seen(owner_row["owner_id"])
            identity_owner, result = owner_name, "owner_recognized"
        else:
            identity_owner, result = "unknown", "unrecognized"

        _insert_event_identity(
            session_id=_session_id,
            source="camera",
            identity_type="face",
            identity_owner=identity_owner,
            confidence=confidence if confidence != "" else None,
            result=result,
        )
    except Exception:
        pass


def log_scene_observation(description):
    """Mirrors one SceneNarrator.narrate() sentence into scene_observation.
    Called from Brain.py right after a Situation's scene_text is filled in."""
    if not _READY or _session_id is None or not description:
        return
    try:
        _insert_scene_observation(
            session_id=_session_id,
            description=description,
        )
    except Exception:
        pass


# --------------------------------------------------------------------------
# OWNER REGISTRATION SYNC (call from owner_manager.py)
# --------------------------------------------------------------------------
# database.py (SQLite, pluto.db) stays the single source of truth for owner
# metadata used by the desktop GUI and by STM face/speaker recognition --
# these functions only MIRROR that same registration into Postgres so the
# rest of Pluto_Database (event_identity, face, voice) has a real owner_id
# to reference. No read path anywhere is switched to Postgres; this is
# write-only mirroring, same "never let this break the real flow" spirit as
# the rest of this file.

def sync_owner_registration(name, dob="", face_sample_count=0, voice_sample_count=0):
    """
    Call this right after owner_manager.register_owner() commits to SQLite.
    Recalls/creates the matching Postgres `owners` row and logs one `face`
    and one `voice` profile row against it, so Postgres has the owner from
    the moment they're registered through the Software -- not only once
    the camera happens to recognize their face live.

    Returns the Postgres owner_id, or None if Postgres isn't reachable.
    """
    if not _READY or not name:
        return None
    try:
        owner_row = _get_or_create_owner_row(name)
        owner_id = owner_row["owner_id"]
        _touch_owner_last_seen(owner_id)
        if face_sample_count:
            _insert_face_profile(owner_id=owner_id, profile_name=name, sample_count=face_sample_count)
        if voice_sample_count:
            _insert_voice_profile(owner_id=owner_id, profile_name=name, sample_count=voice_sample_count)
        return owner_id
    except Exception:
        return None


def sync_owner_update(name, added_face_sample_count=0, added_voice_sample_count=0):
    """
    Call this after owner_manager.update_owner() commits to SQLite. Recalls
    the existing Postgres owner row (creating it if this owner was somehow
    never mirrored before) and logs any newly added face/voice samples as
    additional profile rows.
    """
    if not _READY or not name:
        return None
    try:
        owner_row = _get_or_create_owner_row(name)
        owner_id = owner_row["owner_id"]
        _touch_owner_last_seen(owner_id)
        if added_face_sample_count:
            _insert_face_profile(owner_id=owner_id, profile_name=name, sample_count=added_face_sample_count)
        if added_voice_sample_count:
            _insert_voice_profile(owner_id=owner_id, profile_name=name, sample_count=added_voice_sample_count)
        return owner_id
    except Exception:
        return None


def sync_owner_delete(name):
    """
    Call this after owner_manager.delete_owner() commits to SQLite. Marks
    the matching Postgres owner row as 'Deleted' rather than removing it --
    event_identity/face/voice history for that owner_id stays intact.
    """
    if not _READY or not name:
        return
    try:
        owner_row = _get_or_create_owner_row(name)
        _set_owner_status(owner_row["owner_id"], "Deleted")
    except Exception:
        pass
"""
owner_manager.py
================
Business-logic layer for Pluto's registered owners. Sits between:

    server.py (FastAPI)  --calls-->  owner_manager.py  --calls-->  database.py

Responsibilities (per Pluto architecture):
    - Validate incoming registration data (name, dob, face images, voice samples)
    - Check for duplicate owners
    - Create per-owner storage directories if missing
    - Save uploaded face images / voice samples to disk
    - Insert/update owner metadata via database.py
    - Load / list / update / delete owners
    - Provide a clean read interface for STM (face recognition + speaker
      recognition) and the memory system in general

database.py is intentionally kept unaware of any of this -- it only executes
SQL. All decision-making (what's valid, where files go, what "duplicate"
means) lives here.

Directory layout produced by this module:

    owner_data/
        <owner_id>/
            face/
                face_1.jpg
                face_2.jpg
                ...
            voice/
                voice_1.wav
                ...

Usage from server.py:

    from owner_manager import OwnerManager

    manager = OwnerManager()
    result = manager.register_owner(
        name="Debargho",
        dob="2000-01-01",
        face_files=[("face_1.jpg", raw_bytes), ...],
        voice_files=[("voice_1.wav", raw_bytes), ...],
    )
"""

import os
import uuid
import json
import shutil
from datetime import datetime

import database
import pg_bridge


# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

OWNER_DATA_ROOT = os.path.join(os.path.dirname(__file__), "owner_data")

VALID_FACE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VALID_VOICE_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}

MIN_FACE_SAMPLES = 1
MIN_VOICE_SAMPLES = 1


class OwnerValidationError(Exception):
    """Raised when incoming registration/update data fails validation."""
    pass


class OwnerManager:
    """
    Single controller responsible for every operation related to owner data.
    Other components (face recognition, speaker recognition, memory system,
    FastAPI server) should only ever talk to owner data through this class.
    """

    def __init__(self, data_root: str = OWNER_DATA_ROOT):
        self.data_root = data_root
        os.makedirs(self.data_root, exist_ok=True)
        database.init_db()

    # ----------------------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------------------

    def _validate_basic_fields(self, name: str, dob: str):
        if not name or not name.strip():
            raise OwnerValidationError("Owner name is required.")
        if dob:
            try:
                datetime.strptime(dob, "%Y-%m-%d")
            except ValueError:
                raise OwnerValidationError("dob must be in YYYY-MM-DD format.")

    def _validate_files(self, files, valid_extensions, min_count, label):
        if not files or len(files) < min_count:
            raise OwnerValidationError(
                f"At least {min_count} {label} file(s) are required."
            )
        for fname, _content in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in valid_extensions:
                raise OwnerValidationError(
                    f"Unsupported {label} file type: '{fname}'."
                )

    def owner_exists(self, name: str) -> bool:
        return database.get_owner_by_name(name) is not None

    # ----------------------------------------------------------------
    # STORAGE HELPERS
    # ----------------------------------------------------------------

    def _owner_dirs(self, owner_id: str):
        base = os.path.join(self.data_root, owner_id)
        face_dir = os.path.join(base, "face")
        voice_dir = os.path.join(base, "voice")
        os.makedirs(face_dir, exist_ok=True)
        os.makedirs(voice_dir, exist_ok=True)
        return face_dir, voice_dir

    def _save_files(self, target_dir: str, files):
        """files: list of (filename, bytes). Returns list of saved filenames."""
        saved = []
        for fname, content in files:
            safe_name = os.path.basename(fname)
            dest = os.path.join(target_dir, safe_name)
            with open(dest, "wb") as f:
                f.write(content)
            saved.append(safe_name)
        return saved

    # ----------------------------------------------------------------
    # REGISTRATION (create)
    # ----------------------------------------------------------------

    def register_owner(self, name: str, dob: str = "", face_files=None,
                        voice_files=None, settings: dict = None):
        """
        Full registration lifecycle:
            validate -> check duplicate -> create dirs -> save files -> persist metadata

        face_files / voice_files: list of (filename, raw_bytes) tuples, as
        forwarded by server.py from the incoming multipart request.

        Returns the newly created owner record (dict).
        """
        face_files = face_files or []
        voice_files = voice_files or []

        self._validate_basic_fields(name, dob)
        self._validate_files(face_files, VALID_FACE_EXTENSIONS, MIN_FACE_SAMPLES, "face")
        self._validate_files(voice_files, VALID_VOICE_EXTENSIONS, MIN_VOICE_SAMPLES, "voice")

        if self.owner_exists(name):
            raise OwnerValidationError(f"Owner '{name}' is already registered.")

        owner_id = str(uuid.uuid4())
        face_dir, voice_dir = self._owner_dirs(owner_id)

        saved_faces = self._save_files(face_dir, face_files)
        saved_voices = self._save_files(voice_dir, voice_files)

        now = datetime.now().isoformat(timespec="seconds")
        owner_row = {
            "owner_id": owner_id,
            "name": name.strip(),
            "dob": dob or "",
            "face_dir": face_dir,
            "voice_dir": voice_dir,
            "face_files": ",".join(saved_faces),
            "voice_files": ",".join(saved_voices),
            "registered_at": now,
            "updated_at": now,
            "settings_json": json.dumps(settings or {}),
        }
        database.insert_owner(owner_row)

        # Mirror this registration into Postgres (Pluto_Database) so the
        # rest of the schema -- event_identity, face, voice -- has a real
        # owner to reference from the moment of registration, not only
        # once the camera happens to recognize this owner's face live.
        # SQLite (database.py) stays the single source of truth; this is a
        # write-only mirror and never blocks registration if Postgres is
        # unreachable (see pg_bridge.py's mock-safe design).
        pg_bridge.sync_owner_registration(
            name=owner_row["name"],
            dob=owner_row["dob"],
            face_sample_count=len(saved_faces),
            voice_sample_count=len(saved_voices),
        )

        return owner_row

    # ----------------------------------------------------------------
    # READ
    # ----------------------------------------------------------------

    def get_owner(self, owner_id: str):
        return database.get_owner(owner_id)

    def get_owner_by_name(self, name: str):
        return database.get_owner_by_name(name)

    def list_owners(self):
        return database.get_all_owners()

    def get_face_image_paths(self, owner_id: str):
        """Absolute paths to every stored face image for one owner."""
        row = database.get_owner(owner_id)
        if not row or not row.get("face_files"):
            return []
        return [
            os.path.join(row["face_dir"], fname)
            for fname in row["face_files"].split(",") if fname
        ]

    def get_voice_sample_paths(self, owner_id: str):
        """Absolute paths to every stored voice sample for one owner."""
        row = database.get_owner(owner_id)
        if not row or not row.get("voice_files"):
            return []
        return [
            os.path.join(row["voice_dir"], fname)
            for fname in row["voice_files"].split(",") if fname
        ]

    def get_all_face_image_paths(self):
        """
        Returns {owner_id: {"name": str, "paths": [face_image_paths]}} for
        every registered owner. This is the interface STM's face recognition
        loader (ShortTermMemo.py) should use to build its encodings, instead
        of scanning a manually-populated folder.
        """
        result = {}
        for row in database.get_all_owners():
            result[row["owner_id"]] = {
                "name": row["name"],
                "paths": self.get_face_image_paths(row["owner_id"]),
            }
        return result

    def get_all_voice_sample_paths(self):
        """
        Returns {owner_id: {"name": str, "paths": [voice_sample_paths]}} for
        every registered owner. Used by STM's speaker-recognition loader.
        """
        result = {}
        for row in database.get_all_owners():
            result[row["owner_id"]] = {
                "name": row["name"],
                "paths": self.get_voice_sample_paths(row["owner_id"]),
            }
        return result

    # ----------------------------------------------------------------
    # UPDATE
    # ----------------------------------------------------------------

    def update_owner(self, owner_id: str, name: str = None, dob: str = None,
                      add_face_files=None, add_voice_files=None,
                      settings: dict = None):
        """Partial update. Only provided fields are changed."""
        row = database.get_owner(owner_id)
        if not row:
            raise OwnerValidationError(f"No owner found with id '{owner_id}'.")

        fields = {}

        if name is not None:
            if not name.strip():
                raise OwnerValidationError("Owner name cannot be empty.")
            fields["name"] = name.strip()

        if dob is not None:
            self._validate_basic_fields(row["name"], dob)
            fields["dob"] = dob

        if add_face_files:
            self._validate_files(add_face_files, VALID_FACE_EXTENSIONS, 1, "face")
            saved = self._save_files(row["face_dir"], add_face_files)
            existing = row["face_files"].split(",") if row["face_files"] else []
            fields["face_files"] = ",".join([f for f in existing if f] + saved)

        if add_voice_files:
            self._validate_files(add_voice_files, VALID_VOICE_EXTENSIONS, 1, "voice")
            saved = self._save_files(row["voice_dir"], add_voice_files)
            existing = row["voice_files"].split(",") if row["voice_files"] else []
            fields["voice_files"] = ",".join([f for f in existing if f] + saved)

        if settings is not None:
            fields["settings_json"] = json.dumps(settings)

        if fields:
            fields["updated_at"] = datetime.now().isoformat(timespec="seconds")
            database.update_owner(owner_id, fields)

        updated_row = database.get_owner(owner_id)

        # Mirror name/added-sample changes into Postgres, same write-only
        # pattern as register_owner() above.
        pg_bridge.sync_owner_update(
            name=updated_row["name"],
            added_face_sample_count=len(add_face_files) if add_face_files else 0,
            added_voice_sample_count=len(add_voice_files) if add_voice_files else 0,
        )

        return updated_row

    # ----------------------------------------------------------------
    # DELETE
    # ----------------------------------------------------------------

    def delete_owner(self, owner_id: str):
        row = database.get_owner(owner_id)
        if not row:
            raise OwnerValidationError(f"No owner found with id '{owner_id}'.")

        owner_dir = os.path.join(self.data_root, owner_id)
        if os.path.isdir(owner_dir):
            shutil.rmtree(owner_dir, ignore_errors=True)

        database.delete_owner(owner_id)

        # Mirror the deletion into Postgres -- marks the row 'Deleted'
        # rather than removing it, so event_identity/face/voice history
        # for this owner_id stays intact.
        pg_bridge.sync_owner_delete(name=row["name"])

        return True

    # ----------------------------------------------------------------
    # VALIDATION HELPER FOR OTHER SUBSYSTEMS
    # ----------------------------------------------------------------

    def validate_owner_ready_for_recognition(self, owner_id: str) -> bool:
        """
        Sanity check used before handing owner data to face/speaker recognition:
        owner must exist and have at least one usable face + voice file still
        present on disk (not just recorded in the DB).
        """
        row = database.get_owner(owner_id)
        if not row:
            return False
        face_paths = self.get_face_image_paths(owner_id)
        voice_paths = self.get_voice_sample_paths(owner_id)
        return (
            any(os.path.isfile(p) for p in face_paths)
            and any(os.path.isfile(p) for p in voice_paths)
        )
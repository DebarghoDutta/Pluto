"""
server.py
=========
FastAPI route/app definitions for the Raspberry Pi 5 server. This module is
NOT independently runnable anymore -- it has no Brain instance of its own
and no `if __name__ == "__main__":` launcher. It only exposes a factory,
`create_app(brain)`, that Brain.py's own `__main__` block calls to build the
FastAPI app around the ONE Brain instance Brain.py creates and owns.

    Brain.py            -- single point of execution: creates the Brain,
                            starts it, builds the app via create_app(brain),
                            and runs uvicorn. This is the only thing you run.
    server.py (this file) -- pure route/app definitions, imported by Brain.py.
                            Running `python server.py` directly does nothing
                            useful on purpose -- see the guard at the bottom.

Responsibilities:
    - Receive owner registration / update / delete requests (name, dob,
      face images, voice samples) as multipart form-data.
    - Forward everything to owner_manager.py for validation + storage +
      persistence. server.py itself contains NO business logic -- it only
      translates HTTP requests into owner_manager.py calls and HTTP
      responses back.
    - After any registration/update/delete, call brain.reload_owners() so
      Pluto's Short Term Memory (face + speaker recognition) picks up the
      change immediately -- no manual file copying, no restart.
    - Expose read endpoints (list/get owners) for the GUI's owner screen.
    - Mount websocket_server.py's /ws/live route + background broadcast
      loop, giving the desktop software a persistent push feed of STM
      updates (owner recognition events, sensor readings, recent raw rows)
      in addition to the request/response endpoints below.
    - Mount camera_stream.py's /ws/camera route + background loop, giving
      the desktop software a live JPEG video feed from the same camera
      CameraCapture already has open for detection.
    - Mount telemetry.py's /ws/telemetry (+ GET /telemetry) route +
      background loop, giving the desktop dashboard live CPU/RAM/battery/
      temperature stats for its stat-strip MiniStatCards.

Every piece of the project layout diagram's Software/RaspberryPi communication
channels is now wired into this single FastAPI app.

Run on the Pi with (Brain.py is the entry point -- see its __main__ block):
    pip install fastapi "uvicorn[standard]" python-multipart
    python Brain.py
"""

from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from Brain import Brain
from owner_manager import OwnerValidationError
from websocket_server import register_websocket_routes
from camera_stream import register_camera_stream_routes
from telemetry import register_telemetry_routes


# --------------------------------------------------------------------------
# APP FACTORY
# --------------------------------------------------------------------------
# No module-level `app` and no module-level `Brain()` instance anymore --
# both are created by Brain.py and handed in here. This is what makes
# Brain.py the single point of execution: server.py can no longer be started
# on its own (there is nothing at module scope to run), it only assembles
# routes around whatever Brain instance it's given.

def create_app(brain: Brain) -> FastAPI:
    """
    Builds and returns the FastAPI app wired around `brain`. Brain.py calls
    this once, after constructing its single Brain instance, and passes the
    returned app to uvicorn itself -- server.py never starts/stops `brain`
    and never launches uvicorn; that lifecycle lives entirely in Brain.py.
    """
    app = FastAPI(title="Pluto Server", version="0.1.0")

    # TODO(connect): lock this down to the desktop app's actual origin/IP
    # once known, instead of allowing all origins.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Wires up WS /ws/live plus the background broadcast loop that pushes
    # STM updates (owner recognition events, sensor snapshot, recent raw
    # rows) to every connected desktop client on a fixed interval.
    register_websocket_routes(app, brain)

    # Wires up WS /ws/camera plus the background loop that streams live
    # JPEG frames from the same camera CameraCapture already has open for
    # detection.
    register_camera_stream_routes(app, brain)

    # Wires up WS /ws/telemetry (+ GET /telemetry) plus the background loop
    # that pushes CPU/RAM/battery/temp stats for the desktop dashboard's
    # stat strip.
    register_telemetry_routes(app)

    # ---------------------------------------------------------------- #
    # HELPERS
    # ---------------------------------------------------------------- #

    async def _read_upload_files(files: Optional[List[UploadFile]]):
        """Converts FastAPI UploadFile objects into (filename, bytes)
        tuples, the format owner_manager.py expects."""
        result = []
        for f in files or []:
            content = await f.read()
            result.append((f.filename, content))
        return result

    # ---------------------------------------------------------------- #
    # OWNER REGISTRATION / MANAGEMENT ENDPOINTS
    # ---------------------------------------------------------------- #

    @app.post("/owners/register")
    async def register_owner(
        name: str = Form(...),
        dob: str = Form(""),
        face_files: List[UploadFile] = File(...),
        voice_files: List[UploadFile] = File(...),
    ):
        """
        Called by the desktop software's owner-registration screen. Accepts
        the owner's name, dob, one or more face images (gui.py now always
        sends front + left-side + right-side samples), and one or more
        voice samples as multipart form-data.
        """
        faces = await _read_upload_files(face_files)
        voices = await _read_upload_files(voice_files)

        try:
            owner_row = brain.owner_manager.register_owner(
                name=name, dob=dob, face_files=faces, voice_files=voices
            )
        except OwnerValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Refresh STM face/speaker recognition immediately.
        brain.reload_owners()

        return {"status": "ok", "owner": owner_row}

    @app.get("/owners")
    def list_owners():
        """Returns every registered owner (for the GUI's owner-management screen)."""
        return {"owners": brain.owner_manager.list_owners()}

    @app.get("/owners/{owner_id}")
    def get_owner(owner_id: str):
        owner = brain.owner_manager.get_owner(owner_id)
        if not owner:
            raise HTTPException(status_code=404, detail="Owner not found.")
        return owner

    @app.put("/owners/{owner_id}")
    async def update_owner(
        owner_id: str,
        name: Optional[str] = Form(None),
        dob: Optional[str] = Form(None),
        add_face_files: Optional[List[UploadFile]] = File(None),
        add_voice_files: Optional[List[UploadFile]] = File(None),
    ):
        """
        Called when the GUI updates an existing owner (rename, new dob, or
        additional face/voice samples added to improve recognition accuracy).
        """
        faces = await _read_upload_files(add_face_files)
        voices = await _read_upload_files(add_voice_files)

        try:
            updated = brain.owner_manager.update_owner(
                owner_id,
                name=name,
                dob=dob,
                add_face_files=faces or None,
                add_voice_files=voices or None,
            )
        except OwnerValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        brain.reload_owners()
        return {"status": "ok", "owner": updated}

    @app.delete("/owners/{owner_id}")
    def delete_owner(owner_id: str):
        try:
            brain.owner_manager.delete_owner(owner_id)
        except OwnerValidationError as e:
            raise HTTPException(status_code=404, detail=str(e))

        brain.reload_owners()
        return {"status": "ok", "deleted": owner_id}

    # ---------------------------------------------------------------- #
    # BASIC HEALTH / STATUS ENDPOINTS
    # ---------------------------------------------------------------- #

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/stm/snapshot")
    def stm_snapshot(seconds: Optional[int] = None):
        """
        Quick read endpoint for the GUI's Memory Core screen to poll raw STM
        rows directly, ahead of websocket_server.py providing a push-based
        feed.
        """
        return {"rows": brain.get_short_term_snapshot(seconds)}

    # ---------------------------------------------------------------- #
    # MEMORY CORE SUMMARY ENDPOINTS (gui.py's Memory Core screen, via
    # api_client.get_memory_summary())
    # ---------------------------------------------------------------- #

    def _memory_card(title: str, short_desc: str, purpose: str, used_mb: float,
                      total_mb: float, entries: list) -> dict:
        return {
            "title": title,
            "short_desc": short_desc,
            "purpose": purpose,
            "used_mb": round(used_mb, 2),
            "total_mb": total_mb,
            "entries": entries,
        }

    @app.get("/memory/{memory_type}/summary")
    def memory_summary(memory_type: str):
        """
        Returns a summary card for one of the four Memory Core subsystems, in
        the exact shape gui.py's MemoryCoreCard / MemoryDetailPopup expect:
        {title, short_desc, purpose, used_mb, total_mb, entries: [(title, detail, ts), ...]}
        """
        if memory_type == "short_term":
            rows = brain.get_short_term_snapshot()
            entries = [
                (f"{r.get('source')}:{r.get('data_type')}",
                 str(r.get("attributes")), r.get("timestamp", ""))
                for r in rows
            ]
            return _memory_card(
                title="Short Term Memory",
                short_desc=f"{len(rows)} raw sensory rows in the rolling window.",
                purpose="Raw camera/mic/sensor observations captured every cycle.",
                used_mb=len(rows) * 0.002,
                total_mb=50,
                entries=entries[-100:],
            )

        if memory_type == "semantic":
            facts = brain.semantic.get_facts(active_only=False)
            entries = [
                (f.title, f"{f.category}: {f.meaning} (confidence={f.confidence:.2f})", f.updated_at)
                for f in facts
            ]
            return _memory_card(
                title="Semantic Memory",
                short_desc=f"{len(facts)} learned facts.",
                purpose="Permanent, generalized knowledge learned from repeated situations.",
                used_mb=len(facts) * 0.01,
                total_mb=50,
                entries=entries,
            )

        if memory_type == "behavioral":
            behaviors = brain.behavioral.get_behaviors(active_only=False)
            entries = [
                (b.action.get("type", b.behavior_id),
                 f"stage={b.stage}, confidence={b.confidence:.2f}, success_rate={b.success_rate:.2f}",
                 b.updated_at)
                for b in behaviors
            ]
            return _memory_card(
                title="Behavioral Memory",
                short_desc=f"{len(behaviors)} learned behaviors.",
                purpose="Selects/learns the best action for the current context.",
                used_mb=len(behaviors) * 0.01,
                total_mb=50,
                entries=entries,
            )

        if memory_type == "episodic":
            episodes = brain.episodic.get_episodes(active_only=False)
            entries = [
                (e.episode_id, f"emotion={e.emotional_state}, importance={e.importance_score:.2f}", e.timestamp)
                for e in episodes
            ]
            return _memory_card(
                title="Episodic Memory",
                short_desc=f"{len(episodes)} logged episodes.",
                purpose="Timestamped log of experiences (situation + action + outcome).",
                used_mb=len(episodes) * 0.02,
                total_mb=50,
                entries=entries,
            )

        raise HTTPException(status_code=404, detail=f"Unknown memory_type '{memory_type}'.")

    return app


if __name__ == "__main__":
    # server.py is intentionally NOT an independent entry point anymore --
    # Brain.py is the single place that creates Brain(), builds the app via
    # create_app(brain), and runs uvicorn. Running this file directly is a
    # no-op by design so there's only ever one way to start Pluto.
    raise SystemExit(
        "server.py is no longer runnable on its own.\n"
        "Run:  python Brain.py\n"
        "Brain.py owns the single Brain instance, builds this app around it "
        "via create_app(brain), and starts uvicorn itself."
    )

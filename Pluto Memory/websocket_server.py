"""
websocket_server.py
====================
Real-time (push-based) communication channel FROM Pluto TO the desktop
software. Complements server.py (which handles request/response traffic
like owner registration) with a persistent WebSocket feed so the GUI's
dashboard/Memory Core screen doesn't have to keep polling `/stm/snapshot`.

Architecture:

    Brain (STM rolling window, owner recognition, sensors)
            |
            v
    websocket_server.py  --broadcast-->  every connected websocket_client.py
            ^
            |
    mounted into server.py (FastAPI) as a router + startup background task

What gets pushed (per message, JSON):
    {
        "type": "stm_update",
        "timestamp": "...",
        "recent_rows": [...],          # newest raw STM rows since last push
        "owner_events": [...],         # rows where a known owner was recognized
        "sensor_snapshot": {...}       # latest reading per sensor key
    }

Usage from server.py:

    from websocket_server import register_websocket_routes

    register_websocket_routes(app, brain)

This attaches:
    - WS  /ws/live         -> live push feed (what this file is about)
    - background task that broadcasts on a fixed interval, started/stopped
      alongside the FastAPI app's own startup/shutdown events.
"""

import json
import asyncio
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect


BROADCAST_INTERVAL_SECONDS = 2.0   # how often Pluto pushes an update
STM_WINDOW_SECONDS = 5             # only push rows newer than this each tick


class ConnectionManager:
    """Tracks every connected desktop client (websocket_client.py) and
    handles broadcasting + safe disconnect cleanup."""

    def __init__(self):
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast(self, message: dict):
        """Sends `message` (as JSON) to every connected client. Any client
        that has gone away is dropped silently rather than raising."""
        payload = json.dumps(message, default=str)
        dead = []
        async with self._lock:
            targets = list(self._connections)
        for ws in targets:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections.discard(ws)

    @property
    def active_count(self) -> int:
        return len(self._connections)


manager = ConnectionManager()
router = APIRouter()


def _build_sensor_snapshot(rows: list) -> dict:
    """Reduces raw STM sensor rows down to the latest value per sensor key,
    e.g. {"temperature": 24.6, "humidity": 41.2, ...}.

    ShortTermMemo.py writes sensor rows as:
        logger.write(obs_id, "sensor", key, attributes={"value": value})
    i.e. the reading lives at row["attributes"]["value"], not a top-level
    "raw_value" field (there is no such field in the current row schema)."""
    snapshot = {}
    for row in rows:
        if row.get("source") == "sensor":
            attrs = row.get("attributes") or {}
            snapshot[row["data_type"]] = attrs.get("value")
    return snapshot


def _extract_owner_events(rows: list) -> list:
    """Pulls out rows where a registered owner was recognized (face or
    voice), so the GUI can show "owner detected" notifications without
    scanning the full raw row list itself.

    ShortTermMemo.py's current row schema nests recognition results inside
    "attributes":
        - face:  source="camera", data_type="face",
                 attributes={"person_id": <owner_id> | "unknown", "position": [...]}
        - voice: source="mic", data_type="speech",
                 attributes={..., "speaker_id": <owner_id> | "unidentified" | "no_registered_owner_voice"}
    There is no top-level "raw_value" field and no "speaker_id" data_type."""
    events = []
    for row in rows:
        attrs = row.get("attributes") or {}
        source = row.get("source")
        data_type = row.get("data_type")

        if source == "camera" and data_type == "face":
            person_id = attrs.get("person_id")
            if person_id and person_id != "unknown":
                events.append(row)

        elif source == "mic" and data_type == "speech":
            speaker_id = attrs.get("speaker_id")
            if speaker_id and speaker_id not in (
                "unidentified", "no_registered_owner_voice",
            ):
                events.append(row)

    return events


@router.websocket("/ws/live")
async def live_feed(websocket: WebSocket):
    """
    Persistent connection used by websocket_client.py on the desktop side.
    Pluto pushes updates on its own schedule (see broadcast_loop below); this
    endpoint just accepts the connection and keeps it open, reading and
    discarding any client pings/keepalives until disconnect.
    """
    await manager.connect(websocket)
    try:
        while True:
            # We don't require the client to send anything, but keep the
            # receive loop alive so we notice a disconnect promptly.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await manager.disconnect(websocket)


async def broadcast_loop(brain):
    """
    Background task: every BROADCAST_INTERVAL_SECONDS, pull the latest STM
    window from `brain` and push a summarized update to every connected
    client. Runs for the lifetime of the FastAPI app.
    """
    while True:
        try:
            if manager.active_count > 0:
                rows = brain.get_short_term_snapshot(seconds=STM_WINDOW_SECONDS)
                message = {
                    "type": "stm_update",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "recent_rows": rows,
                    "owner_events": _extract_owner_events(rows),
                    "sensor_snapshot": _build_sensor_snapshot(rows),
                }
                await manager.broadcast(message)
        except Exception:
            # Never let a broadcast hiccup kill the loop.
            pass
        await asyncio.sleep(BROADCAST_INTERVAL_SECONDS)


def register_websocket_routes(app, brain):
    """
    Wires this module into an existing FastAPI app (server.py):
        - includes the /ws/live router
        - starts/stops the broadcast_loop background task alongside the
          app's own startup/shutdown events
    """
    app.include_router(router)

    @app.on_event("startup")
    async def _start_broadcast_loop():
        app.state._ws_broadcast_task = asyncio.create_task(broadcast_loop(brain))

    @app.on_event("shutdown")
    async def _stop_broadcast_loop():
        task = getattr(app.state, "_ws_broadcast_task", None)
        if task:
            task.cancel()
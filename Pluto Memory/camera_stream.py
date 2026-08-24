"""
camera_stream.py
=================
Live video channel FROM Pluto TO the desktop software's video_client.py.

Reuses the SAME Picamera2 instance that ShortTermMemo.py's CameraCapture
already owns for face/object/pose detection -- there is only one physical
camera, so this module never opens a second one. It reads whatever frame
CameraCapture most recently cached (`camera.get_latest_frame()`), JPEG-encodes
it, and pushes it out over a WebSocket to every connected desktop client.

Architecture:

    CameraCapture (ShortTermMemo.py)
        captures frame -> runs detection -> caches frame in self.latest_frame
                                                    |
                                                    v
    camera_stream.py: stream_loop() reads camera.get_latest_frame()
                       every STREAM_INTERVAL_SECONDS, JPEG-encodes it,
                       and broadcasts the bytes to every connected client
                                                    |
                                                    v
    mounted into server.py (FastAPI) as a router + startup background task,
    same pattern as websocket_server.py

Wire protocol (binary WebSocket frames):
    Each message sent to a client is just the raw JPEG bytes for one frame.
    video_client.py can decode with, e.g.:
        img = Image.open(io.BytesIO(frame_bytes))

Usage from server.py:

    from camera_stream import register_camera_stream_routes

    register_camera_stream_routes(app, brain)

This attaches:
    - WS  /ws/camera       -> live JPEG frame feed
    - background task that reads the shared camera frame cache and
      broadcasts on a fixed interval, started/stopped alongside the
      FastAPI app's own startup/shutdown events.

Required library (already needed by ultralytics/mediapipe on the Pi):
    pip install opencv-python
"""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect


STREAM_INTERVAL_SECONDS = 0.1   # ~10 fps push rate to connected clients
JPEG_QUALITY = 70               # 0-100, lower = smaller/faster over network


class CameraConnectionManager:
    """Tracks every connected video_client.py and handles broadcasting +
    safe disconnect cleanup. Mirrors websocket_server.py's ConnectionManager,
    kept separate so a slow/dead video client can never block the STM
    text-update feed or vice versa."""

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

    async def broadcast_bytes(self, data: bytes):
        dead = []
        async with self._lock:
            targets = list(self._connections)
        for ws in targets:
            try:
                await ws.send_bytes(data)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections.discard(ws)

    @property
    def active_count(self) -> int:
        return len(self._connections)


manager = CameraConnectionManager()
router = APIRouter()


def _encode_jpeg(frame):
    """
    Encodes a numpy BGR/RGB frame (as produced by Picamera2.capture_array())
    to JPEG bytes. Returns None if encoding fails or cv2 isn't available
    (e.g. running off-Pi without opencv-python installed).
    """
    try:
        import cv2
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return None
        return buf.tobytes()
    except Exception:
        return None


@router.websocket("/ws/camera")
async def camera_feed(websocket: WebSocket):
    """
    Persistent connection used by video_client.py on the desktop side.
    Pluto pushes JPEG frames on its own schedule (see stream_loop below);
    this endpoint just accepts the connection and keeps it open, reading
    and discarding any client keepalives until disconnect.
    """
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await manager.disconnect(websocket)


async def stream_loop(brain):
    """
    Background task: every STREAM_INTERVAL_SECONDS, reads the latest cached
    frame from brain.short_term.camera (the one CameraCapture instance
    already running for detection), JPEG-encodes it, and broadcasts it to
    every connected client. Skips work entirely when no client is connected
    or no frame is available yet (mock mode / camera not wired).
    """
    while True:
        try:
            if manager.active_count > 0:
                frame = brain.short_term.camera.get_latest_frame()
                if frame is not None:
                    jpeg_bytes = _encode_jpeg(frame)
                    if jpeg_bytes is not None:
                        await manager.broadcast_bytes(jpeg_bytes)
        except Exception:
            # Never let an encode/broadcast hiccup kill the loop.
            pass
        await asyncio.sleep(STREAM_INTERVAL_SECONDS)


def register_camera_stream_routes(app, brain):
    """
    Wires this module into an existing FastAPI app (server.py):
        - includes the /ws/camera router
        - starts/stops the stream_loop background task alongside the
          app's own startup/shutdown events
    """
    app.include_router(router)

    @app.on_event("startup")
    async def _start_stream_loop():
        app.state._camera_stream_task = asyncio.create_task(stream_loop(brain))

    @app.on_event("shutdown")
    async def _stop_stream_loop():
        task = getattr(app.state, "_camera_stream_task", None)
        if task:
            task.cancel()
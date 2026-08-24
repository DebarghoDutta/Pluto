"""
video_client.py
----------------
Pluto -> Software communication layer (camera stream).

Pluto (Raspberry Pi) runs camera_stream.py, which pushes JPEG frames over a
dedicated WebSocket endpoint (kept separate from telemetry so a slow/busy
video feed never starves small, latency-sensitive telemetry messages).

This client connects, decodes each incoming binary JPEG frame into a PIL
Image, and hands it to a callback. The GUI is expected to convert that to an
ImageTk.PhotoImage on the MAIN thread (Tkinter/PhotoImage is not thread-safe),
e.g.:

    def on_frame(pil_image):
        root.after(0, lambda: canvas_update(pil_image))

Requires: pip install websocket-client pillow
"""

import io
import json
import logging
import threading
import time
from typing import Callable, Optional

import websocket  # from websocket-client package
from PIL import Image

logger = logging.getLogger("PlutoVideoClient")
logging.basicConfig(level=logging.INFO)


class PlutoVideoClient:
    """
    Persistent WebSocket client for receiving the live camera stream from Pluto.

    Usage:
        def on_frame(image: Image.Image):
            # convert to ImageTk.PhotoImage on the main thread and update canvas
            ...

        def on_stats(fps: float, frame_size_kb: float):
            ...

        video = PlutoVideoClient(
            host="192.168.1.50", port=8000, endpoint="/ws/camera",
            on_frame=on_frame, on_stats=on_stats,
        )
        video.start()
        ...
        video.stop()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        endpoint: str = "/ws/camera",
        on_frame: Optional[Callable[[Image.Image], None]] = None,
        on_stats: Optional[Callable[[float, float], None]] = None,
        on_connect: Optional[Callable[[], None]] = None,
        on_disconnect: Optional[Callable[[], None]] = None,
        reconnect_interval: float = 3.0,
    ):
        self.url = f"ws://{host}:{port}{endpoint}"
        self.on_frame_cb = on_frame
        self.on_stats_cb = on_stats
        self.on_connect_cb = on_connect
        self.on_disconnect_cb = on_disconnect
        self.reconnect_interval = reconnect_interval

        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._should_run = False
        self._connected = False

        # simple rolling FPS counter
        self._frame_count = 0
        self._last_fps_ts = time.time()

    # ------------------------------------------------------------------ #
    # Public control
    # ------------------------------------------------------------------ #
    def start(self):
        if self._thread and self._thread.is_alive():
            logger.warning("PlutoVideoClient already running.")
            return
        self._should_run = True
        self._thread = threading.Thread(target=self._run_forever, daemon=True)
        self._thread.start()
        logger.info(f"Started video client -> {self.url}")

    def stop(self):
        self._should_run = False
        if self._ws:
            self._ws.close()
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Stopped video client.")

    def is_connected(self) -> bool:
        return self._connected

    def request_quality(self, quality: str = "medium"):
        """
        Optional: ask Pluto to change stream quality/resolution to save
        bandwidth, e.g. 'low' | 'medium' | 'high'. Requires camera_stream.py
        on the Pi to honor a control message.
        """
        if self._ws and self._connected:
            try:
                self._ws.send(json.dumps({"type": "set_quality", "quality": quality}))
            except Exception as e:
                logger.error(f"Failed to request quality change: {e}")

    # ------------------------------------------------------------------ #
    # Internal: connection loop with auto-reconnect
    # ------------------------------------------------------------------ #
    def _run_forever(self):
        while self._should_run:
            try:
                self._ws = websocket.WebSocketApp(
                    self.url,
                    on_open=self._handle_open,
                    on_message=self._handle_message,
                    on_error=self._handle_error,
                    on_close=self._handle_close,
                )
                # Binary frames can be large; disable auto-ping timeout issues
                # by keeping pings frequent and lightweight.
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                logger.error(f"Video WebSocket loop error: {e}")

            self._connected = False
            if self.on_disconnect_cb:
                self.on_disconnect_cb()

            if self._should_run:
                logger.info(f"Reconnecting video stream in {self.reconnect_interval}s...")
                time.sleep(self.reconnect_interval)

    def _handle_open(self, ws):
        self._connected = True
        self._frame_count = 0
        self._last_fps_ts = time.time()
        logger.info("Connected to Pluto camera stream.")
        if self.on_connect_cb:
            self.on_connect_cb()

    def _handle_message(self, ws, message):
        # Binary frames come through as bytes; control/metadata as text (JSON).
        if isinstance(message, (bytes, bytearray)):
            self._handle_frame(message)
        else:
            self._handle_text(message)

    def _handle_frame(self, raw_bytes: bytes):
        try:
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to decode frame: {e}")
            return

        self._frame_count += 1
        now = time.time()
        elapsed = now - self._last_fps_ts
        if elapsed >= 1.0:
            fps = self._frame_count / elapsed
            frame_size_kb = len(raw_bytes) / 1024
            if self.on_stats_cb:
                try:
                    self.on_stats_cb(fps, frame_size_kb)
                except Exception as e:
                    logger.error(f"on_stats callback raised: {e}")
            self._frame_count = 0
            self._last_fps_ts = now

        if self.on_frame_cb:
            try:
                self.on_frame_cb(image)
            except Exception as e:
                logger.error(f"on_frame callback raised: {e}")

    def _handle_text(self, message: str):
        try:
            data = json.loads(message)
            logger.debug(f"Video channel control message: {data}")
        except json.JSONDecodeError:
            logger.warning(f"Received non-JSON text on video channel: {message[:100]}")

    def _handle_error(self, ws, error):
        logger.error(f"Video WebSocket error: {error}")

    def _handle_close(self, ws, close_status_code, close_msg):
        logger.info(f"Video WebSocket closed (code={close_status_code}, msg={close_msg})")


if __name__ == "__main__":
    def _on_frame(img):
        print("Frame received:", img.size)

    def _on_stats(fps, size_kb):
        print(f"FPS: {fps:.1f}, size: {size_kb:.1f} KB")

    client = PlutoVideoClient(on_frame=_on_frame, on_stats=_on_stats)
    client.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        client.stop()
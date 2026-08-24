"""
websocket_client.py
--------------------
Pluto -> Software communication layer (live/streaming data).

Pluto (Raspberry Pi) runs a WebSocket server (websocket_server.py) that pushes
continuous updates: telemetry (CPU, battery, RAM, temp), memory-core refreshes,
and general events. This client maintains a persistent connection, auto-
reconnects if Pluto drops offline, and hands each message off to a callback.

Runs in a background thread so it never blocks the Tkinter mainloop. The GUI
should marshal any widget updates back onto the main thread (e.g. via
`root.after(0, update_fn, data)`) since Tkinter is not thread-safe.

Requires: pip install websocket-client
"""

import json
import logging
import threading
import time
from typing import Callable, Optional

import websocket  # from websocket-client package

logger = logging.getLogger("PlutoWebSocketClient")
logging.basicConfig(level=logging.INFO)


class PlutoWebSocketClient:
    """
    Persistent WebSocket client for receiving live telemetry/events from Pluto.

    Usage:
        def on_telemetry(data: dict):
            print(data)  # e.g. {"type": "telemetry", "cpu": 42, "battery": 87, ...}

        client = PlutoWebSocketClient(
            host="192.168.1.50", port=8000, endpoint="/ws/telemetry",
            on_message=on_telemetry,
        )
        client.start()
        ...
        client.stop()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        endpoint: str = "/ws/telemetry",
        on_message: Optional[Callable[[dict], None]] = None,
        on_connect: Optional[Callable[[], None]] = None,
        on_disconnect: Optional[Callable[[], None]] = None,
        reconnect_interval: float = 3.0,
    ):
        self.url = f"ws://{host}:{port}{endpoint}"
        self.on_message_cb = on_message
        self.on_connect_cb = on_connect
        self.on_disconnect_cb = on_disconnect
        self.reconnect_interval = reconnect_interval

        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._should_run = False
        self._connected = False

    # ------------------------------------------------------------------ #
    # Public control
    # ------------------------------------------------------------------ #
    def start(self):
        """Starts the background thread that keeps the WS connection alive."""
        if self._thread and self._thread.is_alive():
            logger.warning("PlutoWebSocketClient already running.")
            return
        self._should_run = True
        self._thread = threading.Thread(target=self._run_forever, daemon=True)
        self._thread.start()
        logger.info(f"Started telemetry client -> {self.url}")

    def stop(self):
        """Stops the client and closes the connection."""
        self._should_run = False
        if self._ws:
            self._ws.close()
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Stopped telemetry client.")

    def is_connected(self) -> bool:
        return self._connected

    def send(self, data: dict):
        """Send a JSON message to Pluto over the same socket (if bidirectional)."""
        if self._ws and self._connected:
            try:
                self._ws.send(json.dumps(data))
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
        else:
            logger.warning("Cannot send: not connected.")

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
                # run_forever blocks until the socket closes/errors
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                logger.error(f"WebSocket loop error: {e}")

            self._connected = False
            if self.on_disconnect_cb:
                self.on_disconnect_cb()

            if self._should_run:
                logger.info(f"Reconnecting in {self.reconnect_interval}s...")
                time.sleep(self.reconnect_interval)

    def _handle_open(self, ws):
        self._connected = True
        logger.info("Connected to Pluto telemetry stream.")
        if self.on_connect_cb:
            self.on_connect_cb()

    def _handle_message(self, ws, message):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(f"Received non-JSON message: {message[:100]}")
            return

        if self.on_message_cb:
            try:
                self.on_message_cb(data)
            except Exception as e:
                logger.error(f"on_message callback raised: {e}")

    def _handle_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def _handle_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket closed (code={close_status_code}, msg={close_msg})")


if __name__ == "__main__":
    def _print_msg(data):
        print("Received:", data)

    client = PlutoWebSocketClient(on_message=_print_msg)
    client.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        client.stop()
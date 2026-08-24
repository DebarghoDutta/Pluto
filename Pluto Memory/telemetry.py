"""
telemetry.py
============
System vitals channel FROM Pluto TO the desktop software's live dashboard
(the top/bottom "stat strip" MiniStatCards in gui.py: CPU, RAM, Battery,
Temp). Same push pattern as websocket_server.py and camera_stream.py --
this file owns its own WebSocket route + background broadcast loop, kept
fully independent so a telemetry hiccup can never affect the STM feed or
the camera feed.

Architecture:

    read_system_stats()  -> {cpu_percent, ram_percent, ram_used_mb,
                              ram_total_mb, temperature_c, battery_percent,
                              battery_charging, disk_percent, uptime_seconds}
                                    |
                                    v
    telemetry.py: broadcast_loop() reads stats every TELEMETRY_INTERVAL_SECONDS
                  and pushes them to every connected client
                                    |
                                    v
    mounted into server.py (FastAPI) as a router + startup background task,
    same pattern as websocket_server.py / camera_stream.py

Wire protocol (JSON over WebSocket):
    {
        "type": "telemetry_update",
        "timestamp": "...",
        "cpu_percent": 23.4,
        "ram_percent": 41.2,
        "ram_used_mb": 812.3,
        "ram_total_mb": 1970.0,
        "temperature_c": 47.8,
        "battery_percent": 76,
        "battery_charging": true,
        "disk_percent": 38.1,
        "uptime_seconds": 4213
    }

Usage from server.py:

    from telemetry import register_telemetry_routes

    register_telemetry_routes(app)

This attaches:
    - WS  /ws/telemetry    -> live system-stats push feed
    - GET /telemetry       -> one-shot read (useful for a simple HTTP poll
                               or a quick health check without a socket)
    - background task that broadcasts on a fixed interval, started/stopped
      alongside the FastAPI app's own startup/shutdown events

Required library:
    pip install psutil
"""

import time
import asyncio
import json
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect


TELEMETRY_INTERVAL_SECONDS = 3.0   # how often Pluto pushes a stats update

_BOOT_TIME = time.time()


# --------------------------------------------------------------------------
# STAT COLLECTION
# --------------------------------------------------------------------------

def _read_cpu_ram_disk():
    """
    Reads CPU/RAM/disk usage via psutil. Falls back to None fields if
    psutil isn't installed (off-Pi dev environment).
    """
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return {
            "cpu_percent": round(cpu_percent, 1),
            "ram_percent": round(mem.percent, 1),
            "ram_used_mb": round(mem.used / (1024 * 1024), 1),
            "ram_total_mb": round(mem.total / (1024 * 1024), 1),
            "disk_percent": round(disk.percent, 1),
        }
    except Exception:
        return {
            "cpu_percent": None,
            "ram_percent": None,
            "ram_used_mb": None,
            "ram_total_mb": None,
            "disk_percent": None,
        }


def _read_cpu_temperature():
    """
    Reads the Raspberry Pi 5's SoC temperature.
    TODO(connect): confirm this thermal zone path on the actual Pi 5 image
    (it has been consistent across Pi 3/4/5 running Raspberry Pi OS/Ubuntu,
    but verify with `cat /sys/class/thermal/thermal_zone0/temp` on-device).
    Falls back to psutil's sensors API, then to None if neither is available.
    """
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            millidegrees = int(f.read().strip())
            return round(millidegrees / 1000.0, 1)
    except Exception:
        pass

    try:
        import psutil
        temps = psutil.sensors_temperatures()
        for entries in temps.values():
            if entries:
                return round(entries[0].current, 1)
    except Exception:
        pass

    return None


def _read_battery():
    """
    TODO(connect): Pluto's companion-robot form factor implies a battery
    pack (rather than the Pi always being wall-powered), likely read via a
    UPS HAT's I2C fuel-gauge chip (e.g. INA219, MAX17048) rather than
    psutil, since standard Raspberry Pi boards have no built-in battery
    sensor. Wire the real fuel-gauge read here once the UPS HAT is chosen.
    Falls back to psutil.sensors_battery() (works if a UPS HAT exposes
    itself as a standard Linux power supply), then to None if unavailable.
    """
    try:
        import psutil
        battery = psutil.sensors_battery()
        if battery is not None:
            return round(battery.percent, 1), bool(battery.power_plugged)
    except Exception:
        pass
    return None, None


def read_system_stats() -> dict:
    """Collects one full telemetry snapshot as a plain dict."""
    stats = _read_cpu_ram_disk()
    stats["temperature_c"] = _read_cpu_temperature()
    battery_percent, battery_charging = _read_battery()
    stats["battery_percent"] = battery_percent
    stats["battery_charging"] = battery_charging
    stats["uptime_seconds"] = int(time.time() - _BOOT_TIME)
    return stats


# --------------------------------------------------------------------------
# CONNECTION MANAGER (mirrors websocket_server.py / camera_stream.py)
# --------------------------------------------------------------------------

class TelemetryConnectionManager:
    """Tracks connected clients and broadcasts telemetry updates. Kept
    separate from websocket_server.py's and camera_stream.py's managers so a
    slow client on one feed never blocks the others."""

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


manager = TelemetryConnectionManager()
router = APIRouter()


# --------------------------------------------------------------------------
# ROUTES
# --------------------------------------------------------------------------

@router.websocket("/ws/telemetry")
async def telemetry_feed(websocket: WebSocket):
    """
    Persistent connection for the desktop dashboard's stat strip. Pluto
    pushes updates on its own schedule (see broadcast_loop below); this
    endpoint just accepts the connection and keeps it open, reading and
    discarding any client keepalives until disconnect.
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


@router.get("/telemetry")
def telemetry_snapshot():
    """One-shot HTTP read of current system stats, for a simple poll or a
    quick health check without needing a websocket connection."""
    stats = read_system_stats()
    stats["type"] = "telemetry_update"
    stats["timestamp"] = datetime.now().isoformat(timespec="seconds")
    return stats


# --------------------------------------------------------------------------
# BACKGROUND BROADCAST LOOP
# --------------------------------------------------------------------------

async def broadcast_loop():
    """
    Background task: every TELEMETRY_INTERVAL_SECONDS, reads system stats
    and pushes them to every connected client. Skips work entirely when no
    client is connected.
    """
    while True:
        try:
            if manager.active_count > 0:
                stats = read_system_stats()
                message = {
                    "type": "telemetry_update",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    **stats,
                }
                await manager.broadcast(message)
        except Exception:
            # Never let a broadcast hiccup kill the loop.
            pass
        await asyncio.sleep(TELEMETRY_INTERVAL_SECONDS)


def register_telemetry_routes(app):
    """
    Wires this module into an existing FastAPI app (server.py):
        - includes the /ws/telemetry and /telemetry routes
        - starts/stops the broadcast_loop background task alongside the
          app's own startup/shutdown events
    """
    app.include_router(router)

    @app.on_event("startup")
    async def _start_telemetry_loop():
        app.state._telemetry_task = asyncio.create_task(broadcast_loop())

    @app.on_event("shutdown")
    async def _stop_telemetry_loop():
        task = getattr(app.state, "_telemetry_task", None)
        if task:
            task.cancel()
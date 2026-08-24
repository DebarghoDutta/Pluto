"""
api_client.py
--------------
Software -> Pluto communication layer.

Pluto (Raspberry Pi) runs a FastAPI server (server.py). This client wraps
all REST calls the Software side needs to make TO Pluto: owner registration,
one-off commands, health checks, etc.

Transport: HTTP (request/response). Use this for anything that is a single
"do this / give me this" action. For continuous/live data (telemetry, camera)
use websocket_client.py / video_client.py instead.
"""

import logging
import requests
from typing import Optional, Dict, Any

logger = logging.getLogger("PlutoAPIClient")
logging.basicConfig(level=logging.INFO)


class PlutoAPIClientError(Exception):
    """Raised when a request to Pluto's FastAPI server fails."""
    pass


class PlutoAPIClient:
    """
    Thin REST client for talking to Pluto's FastAPI server (server.py).

    Usage:
        api = PlutoAPIClient(host="192.168.1.50", port=8000)
        api.health_check()
        api.register_owner("Debargho", "2000-01-01", ["/path/to/face.jpg"], ["/path/to/voice.wav"])
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8000, timeout: float = 5.0):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.session = requests.Session()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _request(
        self,
        method: str,
        endpoint: str,
        json_body: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        try:
            resp = self.session.request(
                method=method,
                url=url,
                json=json_body,
                files=files,
                params=params,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            if resp.content:
                return resp.json()
            return {}
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Cannot reach Pluto at {url}: {e}")
            raise PlutoAPIClientError(f"Cannot reach Pluto server at {self.base_url}") from e
        except requests.exceptions.Timeout as e:
            logger.error(f"Request to {url} timed out")
            raise PlutoAPIClientError(f"Request to {endpoint} timed out") from e
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error from {url}: {e}")
            raise PlutoAPIClientError(f"Pluto returned an error on {endpoint}: {e}") from e

    # ------------------------------------------------------------------ #
    # Health / connectivity
    # ------------------------------------------------------------------ #
    def health_check(self) -> bool:
        """Returns True if Pluto's FastAPI server is reachable and healthy."""
        try:
            result = self._request("GET", "/health")
            return result.get("status") == "ok"
        except PlutoAPIClientError:
            return False

    # ------------------------------------------------------------------ #
    # Owner registration (handled by owner_manager.py on the Pi side)
    # ------------------------------------------------------------------ #
    def register_owner(
        self,
        owner_name: str,
        dob: str,
        face_paths: list,
        voice_paths: list,
    ) -> Dict[str, Any]:
        """
        Registers a new owner with Pluto in a single call, matching
        server.py's POST /owners/register endpoint (owner_manager.py
        requires at least one face file AND one voice file together --
        there is no separate face-only / voice-only registration route).

        Args:
            owner_name: full name of the owner.
            dob: date of birth as 'YYYY-MM-DD' (owner_manager.py validates
                 this exact format).
            face_paths: list of local file paths to face images (>= 1).
            voice_paths: list of local file paths to voice samples (>= 1).
        """
        url = f"{self.base_url}/owners/register"
        opened_files = []
        try:
            files = []
            for path in face_paths:
                fh = open(path, "rb")
                opened_files.append(fh)
                files.append(("face_files", (path.split("/")[-1], fh, "image/jpeg")))
            for path in voice_paths:
                fh = open(path, "rb")
                opened_files.append(fh)
                files.append(("voice_files", (path.split("/")[-1], fh, "audio/wav")))

            data = {"name": owner_name, "dob": dob}
            resp = self.session.post(url, data=data, files=files, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except FileNotFoundError as e:
            raise PlutoAPIClientError(f"File not found: {e}") from e
        except requests.exceptions.RequestException as e:
            raise PlutoAPIClientError(f"Owner registration failed: {e}") from e
        finally:
            for fh in opened_files:
                fh.close()

    def list_owners(self) -> Dict[str, Any]:
        """Fetch the list of currently registered owners from Pluto."""
        return self._request("GET", "/owners")

    def get_owner(self, owner_id: str) -> Dict[str, Any]:
        """Fetch a single owner record by owner_id."""
        return self._request("GET", f"/owners/{owner_id}")

    def delete_owner(self, owner_id: str) -> Dict[str, Any]:
        """Remove an owner's face/voice enrollment from Pluto (by owner_id,
        as returned by register_owner()/list_owners() -- not by name)."""
        return self._request("DELETE", f"/owners/{owner_id}")

    # ------------------------------------------------------------------ #
    # Generic command dispatch (Brain.py orchestrator commands)
    # ------------------------------------------------------------------ #
    def send_command(self, command: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generic passthrough for one-off commands to Brain.py, e.g.
        {"command": "sleep_mode", "payload": {"enabled": true}}
        """
        body = {"command": command, "payload": payload or {}}
        return self._request("POST", "/command", json_body=body)

    # ------------------------------------------------------------------ #
    # Memory subsystem reads (Memory Core GUI screen)
    # ------------------------------------------------------------------ #
    def get_memory_summary(self, memory_type: str) -> Dict[str, Any]:
        """
        memory_type: 'short_term' | 'semantic' | 'behavioral' | 'episodic'
        """
        return self._request("GET", f"/memory/{memory_type}/summary")

    def close(self):
        self.session.close()


if __name__ == "__main__":
    # Quick manual smoke test
    api = PlutoAPIClient(host="127.0.0.1", port=8000)
    print("Health:", api.health_check())
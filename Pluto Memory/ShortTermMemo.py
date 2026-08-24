"""
ShortTermMemo.py
================
Short Term Memory (STM) subsystem for Pluto (companion robot).

Purpose
-------
STM is a pure INPUT layer. It continuously captures raw sensory data from:
    - Camera (Pi Camera 3 via Picamera2)   -> face recognition, object detection,
                                               pose, facial emotion
    - Microphone                           -> speech-to-text, speaker ID,
                                               language, ambient sound level
    - Sensors                              -> environment parameters (temp,
                                               humidity, light, distance...)

STM records TIMESTAMPED FACTS ONLY. It never produces an English conclusion
("owner is working") -- that is the job of Semantic Memory, and Semantic
Memory only ever sees the output of the Scene Builder (see SceneBuilder.py),
never STM's raw rows directly.

Grouping: ObservationID
------------------------
Every sensory reading captured during the same acquisition cycle (one pass
of camera + mic + sensors) is tagged with the SAME `observation_id`. This
lets the Scene Builder later collapse all rows sharing an `observation_id`
into a single `Situation` -- one coherent snapshot of "what STM saw/heard/
measured at this moment" -- instead of a flat, disconnected event log.

The current `observation_id` is produced by a lightweight ticking clock
(`_ObservationClock`) that advances on a fixed interval. Camera, mic, and
sensor capture loops each run on their own cadence (a camera frame every
2s, a mic chunk every ~4s, a sensor read every 5s) but all tag whatever
they capture with whichever `observation_id` is "current" at write time.
This keeps STM's three capture loops independent (matching real hardware
timing) while still letting downstream code group same-cycle rows.

Structured values, not descriptive strings
-------------------------------------------
Every row's payload is a small `attributes` dict of structured fields
(e.g. `person_id="owner001"`, `emotion="neutral"`, `confidence=0.83`,
`position=[x, y, w, h]`, `distance_cm=142.0`, `speaker_id="owner001"`,
`language="en"`, `sound_level_db=41.2`) -- never a pre-composed sentence
like "owner detected" or "owner looks neutral".

Target platform: Raspberry Pi 5, Ubuntu, VS Code, Pi Camera Module 3.

Required libraries (install via pip / apt on the Pi):
    pip install ultralytics opencv-python face_recognition mediapipe \
                SpeechRecognition sounddevice numpy
    sudo apt install python3-picamera2 portaudio19-dev

Architecture note (matches gui.py conventions):
    - Hardware access points are marked with TODO(connect) for real Pi5 wiring.
    - Until hardware is wired, methods fall back to safe mock/no-op behavior so
      the module can be imported and tested off-Pi.

Owner data note:
    Face and voice reference samples come from owner_manager.py (populated via
    gui.py -> server.py). Call `ShortTermMemory.reload_owners()` (or
    `Brain.reload_owners()`) right after a registration completes so STM's
    in-memory encodings refresh with no manual file copying or restart.
"""

import os
import csv
import json
import time
import uuid
import threading
from datetime import datetime

import numpy as np

from owner_manager import OwnerManager
import pg_bridge


# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

STM_CSV_PATH = os.path.join(os.path.dirname(__file__), "stm_data.csv")
STM_RETENTION_SECONDS = 300          # keep only last 5 minutes of raw rows in memory/csv
CAMERA_SAMPLE_INTERVAL = 2.0         # seconds between camera capture cycles
AUDIO_CHUNK_SECONDS = 4.0            # length of each audio capture chunk
SENSOR_SAMPLE_INTERVAL = 5.0         # seconds between sensor reads

OBSERVATION_CYCLE_SECONDS = CAMERA_SAMPLE_INTERVAL   # cadence of the shared ObservationID clock

YOLO_MODEL_PATH = "yolo26n.pt"        # Ultralytics YOLO26 nano -- NMS-free, faster CPU inference than YOLOv8n, good for Pi5

# Structured row schema: `attributes` holds a JSON-encoded dict of the
# type-specific structured fields described in the module docstring.
CSV_FIELDS = ["observation_id", "timestamp", "source", "data_type", "attributes", "confidence"]


# --------------------------------------------------------------------------
# OBSERVATION CLOCK: assigns a shared ObservationID to each acquisition cycle
# --------------------------------------------------------------------------

class _ObservationClock:
    """
    Ticks every `OBSERVATION_CYCLE_SECONDS` and mints a fresh `observation_id`
    (uuid4 hex). Camera/mic/sensor capture loops read `current()` whenever
    they finish a capture, tagging their rows with whichever ObservationID is
    "live" at that moment. This is how independently-timed capture loops end
    up grouped into the same scene without being forced onto one blocking loop.
    """

    def __init__(self, interval=OBSERVATION_CYCLE_SECONDS):
        self.interval = interval
        self._lock = threading.Lock()
        self._id = uuid.uuid4().hex
        self._running = False
        self._thread = None

    def current(self):
        with self._lock:
            return self._id

    def _loop(self):
        while self._running:
            time.sleep(self.interval)
            with self._lock:
                self._id = uuid.uuid4().hex

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False


# --------------------------------------------------------------------------
# CSV WRITER (thread-safe, append-only, structured raw rows)
# --------------------------------------------------------------------------

class _CSVLogger:
    """Thread-safe append-only CSV writer + rolling-window pruner for STM."""

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()

    def write(self, observation_id, source, data_type, attributes=None, confidence=""):
        """
        `attributes` is a plain dict of structured fields, e.g.
            {"person_id": "owner001", "position": [x, y, w, h]}
        Never pass a pre-composed English description here -- STM only
        records facts, never conclusions.
        """
        row = {
            "observation_id": observation_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "source": source,               # "camera" | "mic" | "sensor"
            "data_type": data_type,         # e.g. "face", "object", "pose", "speech", "emotion", "ambient_sound", "temperature"
            "attributes": json.dumps(attributes or {}),
            "confidence": confidence,       # optional float 0-1, blank if not applicable
        }
        with self._lock:
            with open(self.path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writerow(row)

    def prune_older_than(self, seconds):
        """Rolling window: drop rows older than `seconds` to keep STM 'short term'."""
        cutoff = time.time() - seconds
        with self._lock:
            if not os.path.exists(self.path):
                return
            kept_rows = []
            with open(self.path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ts = datetime.fromisoformat(row["timestamp"]).timestamp()
                    except (ValueError, KeyError, TypeError):
                        continue
                    if ts >= cutoff:
                        kept_rows.append(row)
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(kept_rows)

    def read_recent(self, seconds=None):
        """
        Return list of dict rows, optionally filtered to last `seconds`.
        Each row's `attributes` field is decoded back into a dict for callers
        (e.g. the Scene Builder) so they don't have to touch JSON directly.
        """
        if not os.path.exists(self.path):
            return []
        rows = []
        cutoff = time.time() - seconds if seconds else None
        with open(self.path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if cutoff:
                    try:
                        ts = datetime.fromisoformat(row["timestamp"]).timestamp()
                    except (ValueError, KeyError, TypeError):
                        continue
                    if ts < cutoff:
                        continue
                try:
                    row["attributes"] = json.loads(row.get("attributes") or "{}")
                except json.JSONDecodeError:
                    row["attributes"] = {}
                rows.append(row)
        return rows


# --------------------------------------------------------------------------
# CAMERA CAPTURE: face recognition + object detection + pose + emotion
# --------------------------------------------------------------------------

class CameraCapture:
    """
    Handles Pi Camera 3 capture and runs four parallel detections per frame:
        1. Face recognition   -> structured person_id + position + confidence
        2. Object detection   -> structured object_class + position + distance
        3. Pose/activity      -> structured posture + raw joint state
        4. Facial emotion     -> structured emotion label + confidence

    All results are pushed to the CSV logger as structured attribute dicts,
    tagged with the ObservationID that is live at capture time -- never as
    pre-composed sentences.
    """

    def __init__(self, logger: _CSVLogger, clock: _ObservationClock, owner_manager: OwnerManager = None):
        self.logger = logger
        self.clock = clock
        self.owner_manager = owner_manager or OwnerManager()
        self._running = False
        self._thread = None

        self.picam2 = None
        self.yolo_model = None

        # owner_id -> {"name": str, "encodings": [face_encoding, ...]}
        self.owners = {}

        # Shared latest-frame cache so other modules (camera_stream.py) can
        # read the most recent frame WITHOUT opening a second Picamera2
        # instance -- there is only one physical camera. Updated every
        # detection cycle in _loop(). TODO(connect): if camera_stream.py
        # needs a noticeably higher frame rate than CAMERA_SAMPLE_INTERVAL
        # provides for smooth live video, split frame acquisition into its
        # own faster loop and have detection read from this same cache.
        self.latest_frame = None
        self._frame_lock = threading.Lock()

        self._init_camera()
        self._init_yolo()
        self._load_owner_faces()

    # ---- init helpers -----------------------------------------------------

    def _init_camera(self):
        """
        TODO(connect): Wire actual Pi Camera Module 3 via picamera2 on Raspberry Pi 5.
        """
        try:
            from picamera2 import Picamera2  # noqa: F401
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(main={"size": (640, 480)})
            self.picam2.configure(config)
            self.picam2.start()
        except Exception:
            # Not running on Pi hardware / picamera2 unavailable -> mock mode
            self.picam2 = None

    def _init_yolo(self):
        """
        TODO(connect): Confirm YOLO26 weights path is present on the Pi
        (yolo26n.pt), or point to a custom-trained model for Pluto-specific
        objects. Weights auto-download from Ultralytics' asset release on
        first use if not already present locally.
        """
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
        except Exception:
            self.yolo_model = None

    def _load_owner_faces(self):
        """
        Builds face encodings for every owner registered through the GUI ->
        server.py -> owner_manager.py pipeline. Calling `reload()` (or
        `Brain.reload_owners()`) after a new registration picks up their
        stored face images automatically.
        """
        self.owners = {}
        try:
            import face_recognition
        except Exception:
            return  # face_recognition not installed -> stay empty (mock mode)

        for owner_id, info in self.owner_manager.get_all_face_image_paths().items():
            encodings = []
            for fpath in info["paths"]:
                try:
                    img = face_recognition.load_image_file(fpath)
                    encs = face_recognition.face_encodings(img)
                    if encs:
                        encodings.append(encs[0])
                except Exception:
                    continue
            if encodings:
                self.owners[owner_id] = {"name": info["name"], "encodings": encodings}

    def reload(self):
        """Call this right after a new/updated owner registration completes."""
        self._load_owner_faces()

    # ---- frame acquisition --------------------------------------------------

    def _get_frame(self):
        """Returns a numpy BGR/RGB frame, or None if camera unavailable (mock mode)."""
        if self.picam2 is not None:
            try:
                return self.picam2.capture_array()
            except Exception:
                return None
        return None  # TODO(connect): no camera hardware detected

    # ---- detection routines --------------------------------------------------

    @staticmethod
    def _bbox_to_position(top, right, bottom, left):
        """Converts face_recognition's (top, right, bottom, left) into a plain
        [x, y, width, height] position, a structured value rather than a string."""
        x, y = left, top
        w, h = right - left, bottom - top
        return [int(x), int(y), int(w), int(h)]

    def _detect_face(self, frame, obs_id):
        try:
            import face_recognition
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            for location, encoding in zip(face_locations, face_encodings):
                top, right, bottom, left = location
                position = self._bbox_to_position(top, right, bottom, left)

                best_owner_id, best_confidence = None, -1.0
                for owner_id, info in self.owners.items():
                    matches = face_recognition.compare_faces(
                        info["encodings"], encoding, tolerance=0.5
                    )
                    distances = face_recognition.face_distance(info["encodings"], encoding)
                    if True in matches:
                        local_best_idx = int(np.argmin(distances))
                        confidence = round(1 - distances[local_best_idx], 2)
                        if confidence > best_confidence:
                            best_owner_id, best_confidence = owner_id, confidence

                person_id = best_owner_id if best_owner_id is not None else "unknown"
                confidence = best_confidence if best_owner_id is not None else ""
                self.logger.write(
                    obs_id, "camera", "face",
                    attributes={"person_id": person_id, "position": position},
                    confidence=confidence,
                )
                owner_name = self.owners[best_owner_id]["name"] if best_owner_id is not None else None
                pg_bridge.log_face_event(owner_name, confidence, position)
                self._detect_emotion(frame, position, obs_id)
        except Exception:
            pass  # face_recognition not available / no frame

    def _detect_emotion(self, frame, position, obs_id):
        """
        Facial emotion recognition on the detected face region.
        TODO(connect): Wire a real emotion classifier (e.g. FER, DeepFace,
        or a small custom CNN) against the cropped face region defined by
        `position`. Until wired, this is a safe no-op stub -- it does NOT
        invent an emotion; it logs "unknown" with zero confidence so the
        field exists in the schema without fabricating a reading.
        """
        emotion, confidence = "unknown", 0.0
        self.logger.write(
            obs_id, "camera", "emotion",
            attributes={"emotion": emotion, "position": position},
            confidence=confidence,
        )

    def _detect_objects(self, frame, obs_id):
        if self.yolo_model is None:
            return
        try:
            results = self.yolo_model.predict(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = self.yolo_model.names.get(cls_id, str(cls_id))
                    conf = round(float(box.conf[0]), 2)
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = xyxy
                    position = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                    # TODO(connect): derive real distance from a depth sensor /
                    # stereo pair; no depth hardware wired yet.
                    distance_cm = None
                    self.logger.write(
                        obs_id, "camera", "object",
                        attributes={
                            "object_class": label,
                            "position": position,
                            "distance_cm": distance_cm,
                        },
                        confidence=conf,
                    )
                    pg_bridge.log_object_detection(label, conf, position)
        except Exception:
            pass

    def _detect_pose(self, frame, obs_id):
        """
        Raw pose/activity signature capture using mediapipe.
        Logs a structured posture label (e.g. 'seated', 'standing') plus the
        raw joint measurements that produced it -- never a full sentence like
        "owner is working".
        TODO(connect): tune landmark thresholds against real Pi Cam 3 footage.
        """
        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            with mp_pose.Pose(static_image_mode=True) as pose:
                results = pose.process(frame)
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
                    shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                    knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
                    torso_len = abs(hip_y - shoulder_y)
                    leg_len = abs(knee_y - hip_y)
                    posture = "seated" if leg_len < torso_len * 0.6 else "standing"
                    self.logger.write(
                        obs_id, "camera", "pose",
                        attributes={
                            "posture": posture,
                            "torso_len": round(torso_len, 4),
                            "leg_len": round(leg_len, 4),
                        },
                    )
        except Exception:
            pass

    # ---- main loop --------------------------------------------------

    def _loop(self):
        while self._running:
            obs_id = self.clock.current()
            frame = self._get_frame()
            if frame is not None:
                with self._frame_lock:
                    self.latest_frame = frame
                self._detect_face(frame, obs_id)
                self._detect_objects(frame, obs_id)
                self._detect_pose(frame, obs_id)
            else:
                # No camera hardware wired yet -> mock placeholder row
                self.logger.write(obs_id, "camera", "status", attributes={"state": "no_camera_frame_available"})
            time.sleep(CAMERA_SAMPLE_INTERVAL)

    def get_latest_frame(self):
        """
        Thread-safe read of the most recently captured frame (numpy array),
        or None if no frame has been captured yet / camera unavailable.
        Used by camera_stream.py to serve live video without needing its
        own Picamera2 instance.
        """
        with self._frame_lock:
            return self.latest_frame

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass


# --------------------------------------------------------------------------
# MIC CAPTURE: speech-to-text + speaker ID + language + ambient sound level
# --------------------------------------------------------------------------

class MicCapture:
    """
    Captures audio in fixed-length chunks and logs structured results:
        - speech: recognized raw text, plus language + speaker_id
        - ambient_sound: raw sound level (dB) when no clear speech detected
    """

    def __init__(self, logger: _CSVLogger, clock: _ObservationClock, owner_manager: OwnerManager = None):
        self.logger = logger
        self.clock = clock
        self.owner_manager = owner_manager or OwnerManager()
        self._running = False
        self._thread = None
        self._recognizer = None
        self._mic_available = self._init_mic()

        # owner_id -> {"name": str, "voice_paths": [wav_path, ...]}
        self.owner_voice_profiles = {}
        self._load_owner_voices()

    def _load_owner_voices(self):
        """
        Loads registered owners' voice sample paths from owner_manager.py.
        TODO(connect): feed these paths into a real speaker-id model (e.g.
        speechbrain, resemblyzer) to build embeddings for comparison in
        `_recognize_speaker`.
        """
        self.owner_voice_profiles = self.owner_manager.get_all_voice_sample_paths()

    def reload(self):
        """Call this right after a new/updated owner registration completes."""
        self._load_owner_voices()

    def _init_mic(self):
        """
        TODO(connect): Confirm USB/onboard mic device index on Raspberry Pi 5
        (check with `python -m sounddevice`), and set as default input device.
        """
        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
            sr.Microphone()  # raises if no mic present
            return True
        except Exception:
            return False

    def _capture_chunk(self):
        import speech_recognition as sr
        with sr.Microphone() as source:
            audio = self._recognizer.listen(source, timeout=AUDIO_CHUNK_SECONDS, phrase_time_limit=AUDIO_CHUNK_SECONDS)
        return audio

    @staticmethod
    def _estimate_sound_level_db(audio):
        """
        Rough raw sound-level estimate (dBFS-like) from the captured audio's
        sample data. Structured numeric value, not a description like "loud".
        """
        try:
            raw = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
            if raw.size == 0:
                return None
            rms = np.sqrt(np.mean(raw.astype(np.float64) ** 2))
            if rms <= 0:
                return None
            return round(float(20 * np.log10(rms)), 1)
        except Exception:
            return None

    def _recognize_speech(self, audio, obs_id):
        """
        TODO(connect): For fully offline operation on Pi5, swap Google STT for
        an offline engine (e.g. Vosk) to avoid network dependency.
        TODO(connect): wire real language identification (currently assumes
        "en" as a structured placeholder rather than a guess dressed as fact).
        """
        import speech_recognition as sr
        sound_level_db = self._estimate_sound_level_db(audio)
        try:
            text = self._recognizer.recognize_google(audio)
            speaker_id = self._identify_speaker(audio)
            self.logger.write(
                obs_id, "mic", "speech",
                attributes={
                    "text": text,
                    "language": "en",
                    "speaker_id": speaker_id,
                    "sound_level_db": sound_level_db,
                },
            )
        except sr.UnknownValueError:
            self.logger.write(
                obs_id, "mic", "ambient_sound",
                attributes={"sound_level_db": sound_level_db},
            )
        except sr.RequestError:
            self.logger.write(obs_id, "mic", "status", attributes={"state": "stt_service_unavailable"})

    def _identify_speaker(self, audio):
        """
        TODO(connect): Run a real speaker-id model (e.g. resemblyzer,
        speechbrain) comparing captured `audio` against embeddings built from
        each owner's stored voice samples (self.owner_voice_profiles). Until
        wired, returns a structured placeholder rather than a sentence.
        """
        if self.owner_voice_profiles:
            return "unidentified"
        return "no_registered_owner_voice"

    def _loop(self):
        while self._running:
            obs_id = self.clock.current()
            if not self._mic_available:
                self.logger.write(obs_id, "mic", "status", attributes={"state": "no_microphone_available"})
                time.sleep(AUDIO_CHUNK_SECONDS)
                continue
            try:
                audio = self._capture_chunk()
                obs_id = self.clock.current()  # re-check: chunk capture may span a tick
                self._recognize_speech(audio, obs_id)
            except Exception:
                self.logger.write(obs_id, "mic", "status", attributes={"state": "mic_capture_error"})
            time.sleep(0.1)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False


# --------------------------------------------------------------------------
# SENSOR CAPTURE: environment parameters
# --------------------------------------------------------------------------

class SensorCapture:
    """
    Reads ambient environment sensors (temperature, humidity, light, distance, etc).
    All physical sensor wiring is stubbed with TODO(connect) mock values until
    the Pi5 GPIO/I2C sensors are physically attached.
    """

    def __init__(self, logger: _CSVLogger, clock: _ObservationClock):
        self.logger = logger
        self.clock = clock
        self._running = False
        self._thread = None

    def _read_sensors(self):
        """
        TODO(connect): Replace mock values with real sensor reads, e.g.:
            - DHT22 temp/humidity via Adafruit_DHT or adafruit-circuitpython-dht
            - BH1750 ambient light sensor via smbus2 (I2C)
            - HC-SR04 / VL53L0X distance sensor via GPIO/I2C
        """
        return {
            "temperature_c": round(np.random.uniform(20, 30), 1),
            "humidity_pct": round(np.random.uniform(30, 70), 1),
            "light_lux": round(np.random.uniform(50, 500), 1),
            "distance_cm": round(np.random.uniform(30, 300), 1),
        }

    def _loop(self):
        while self._running:
            obs_id = self.clock.current()
            readings = self._read_sensors()
            for key, value in readings.items():
                self.logger.write(obs_id, "sensor", key, attributes={"value": value})
            time.sleep(SENSOR_SAMPLE_INTERVAL)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False


# --------------------------------------------------------------------------
# SHORT TERM MEMORY (facade class used by Brain.py)
# --------------------------------------------------------------------------

class ShortTermMemory:
    """
    Facade over the ObservationClock, CameraCapture, MicCapture, and
    SensorCapture. Provides start/stop control and read access to the raw
    rolling-window CSV.

    This is the class Brain.py should import and instantiate. STM stays a
    pure data acquisition layer: it never groups rows into a Situation
    itself -- that's SceneBuilder's job (see SceneBuilder.py), which Brain.py
    runs between STM and Semantic Memory.
    """

    def __init__(self, csv_path=STM_CSV_PATH, retention_seconds=STM_RETENTION_SECONDS,
                 owner_manager: OwnerManager = None):
        self.logger = _CSVLogger(csv_path)
        self.retention_seconds = retention_seconds
        self.owner_manager = owner_manager or OwnerManager()
        self.clock = _ObservationClock()

        self.camera = CameraCapture(self.logger, self.clock, owner_manager=self.owner_manager)
        self.mic = MicCapture(self.logger, self.clock, owner_manager=self.owner_manager)
        self.sensors = SensorCapture(self.logger, self.clock)

        self._pruner_running = False
        self._pruner_thread = None

    def _pruner_loop(self):
        while self._pruner_running:
            self.logger.prune_older_than(self.retention_seconds)
            time.sleep(10)

    def start(self):
        self.clock.start()
        self.camera.start()
        self.mic.start()
        self.sensors.start()
        self._pruner_running = True
        self._pruner_thread = threading.Thread(target=self._pruner_loop, daemon=True)
        self._pruner_thread.start()

    def stop(self):
        self.camera.stop()
        self.mic.stop()
        self.sensors.stop()
        self.clock.stop()
        self._pruner_running = False

    def get_recent_data(self, seconds=None):
        """Returns raw structured rows (list of dicts) from the current short-term window."""
        return self.logger.read_recent(seconds or self.retention_seconds)

    def reload_owners(self):
        """
        Refreshes face + voice recognition data from OwnerManager. Call this
        right after server.py finishes handling a new owner registration (or
        an update) so newly registered owners are recognized immediately,
        with no manual file copying and no restart required.
        """
        self.camera.reload()
        self.mic.reload()


if __name__ == "__main__":
    # Simple manual test loop (safe to run off-Pi; falls back to mock/no-op mode).
    stm = ShortTermMemory()
    stm.start()
    try:
        print("ShortTermMemory running. Press Ctrl+C to stop.")
        while True:
            time.sleep(5)
            rows = stm.get_recent_data()
            obs_ids = {r["observation_id"] for r in rows}
            print(f"Rows in window: {len(rows)} across {len(obs_ids)} observation cycles")
    except KeyboardInterrupt:
        stm.stop()
        print("Stopped.")
"""
================================================================================
 voice_capture.py
================================================================================
Standalone module responsible for talking to the MICROPHONE and recording a
short voice sample for an owner profile.

Kept in its own file (separate from gui.py) on purpose, so the audio logic
can be developed, tested, and later upgraded (e.g. to a real speaker-
verification / voiceprint-embedding pipeline running on the Raspberry Pi 5)
without needing to touch any GUI code. gui.py only ever calls
`record_voice(...)` from this file.

DEPENDENCY
----------
    pip install pyaudio

On Raspberry Pi / Debian you may first need the PortAudio system library:
    sudo apt-get install portaudio19-dev
    pip install pyaudio

WHERE RECORDINGS ARE SAVED
----------------------------
    <this folder>/captured_data/voices/<name>_<timestamp>.wav

FUTURE UPGRADE PATH (Raspberry Pi 5)
-------------------------------------
Right now this just records N seconds of raw audio to a WAV file. Later
this can be swapped to:
    - Trim leading/trailing silence automatically
    - Run wake-word / speaker-embedding models (e.g. via a lightweight
      on-device voiceprint model) to enroll the owner's voice for real
      speaker recognition, instead of / in addition to storing the raw clip
Only the inside of `record_voice()` needs to change for any of that.
================================================================================
"""

import os
import time
import wave

import pyaudio

VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captured_data", "voices")

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


def _ensure_dir():
    os.makedirs(VOICES_DIR, exist_ok=True)


def _safe_filename(name):
    cleaned = "".join(c for c in name if c.isalnum() or c in ("_", "-")) or "owner"
    return cleaned


def record_voice(owner_name="owner", duration=4, device_index=None, progress_callback=None):
    """
    Records `duration` seconds of audio from the default (or specified)
    microphone and saves it as a 16-bit mono WAV file.

    Args:
        owner_name:        used to build a readable filename.
        duration:           how many seconds to record.
        device_index:       optional specific input device index; None uses
                            the system default microphone.
        progress_callback:  optional function called as
                            progress_callback(elapsed_seconds, total_seconds)
                            roughly once per audio chunk, so a GUI can show
                            live "Recording... Xs left" feedback.

    Returns:
        The absolute path to the saved WAV file, or None on failure (e.g.
        no microphone available).

    NOTE: This function is BLOCKING for the full `duration`. Call it from a
    background thread if using it from a GUI, so the GUI doesn't freeze
    while recording.
    """
    _ensure_dir()

    audio = pyaudio.PyAudio()
    try:
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                             input_device_index=device_index, frames_per_buffer=CHUNK)
    except Exception:
        audio.terminate()
        return None

    frames = []
    total_chunks = max(1, int(RATE / CHUNK * duration))
    start = time.time()

    try:
        for _ in range(total_chunks):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            if progress_callback:
                progress_callback(time.time() - start, duration)
    finally:
        stream.stop_stream()
        stream.close()

    sample_width = audio.get_sample_size(FORMAT)
    audio.terminate()

    filename = f"{_safe_filename(owner_name)}_{int(time.time())}.wav"
    filepath = os.path.join(VOICES_DIR, filename)

    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(sample_width)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    return filepath


if __name__ == "__main__":
    # Quick manual test: run "python3 voice_capture.py" directly.
    def _print_progress(elapsed, total):
        print(f"\rRecording... {elapsed:.1f}/{total}s", end="")

    result = record_voice(owner_name="test_user", duration=4, progress_callback=_print_progress)
    print()
    if result:
        print(f"Saved voice recording to: {result}")
    else:
        print("Voice recording failed - no microphone found?")
"""
================================================================================
 face_capture.py
================================================================================
Standalone module responsible for talking to the CAMERA and capturing a
face image for an owner profile.

Kept in its own file (separate from gui.py) on purpose, so the camera /
face-detection logic can be developed, tested, and later upgraded (e.g. to a
real face-recognition / embedding pipeline running on the Raspberry Pi 5)
without needing to touch any GUI code. gui.py only ever calls
`capture_face(...)` from this file.

DEPENDENCY
----------
    pip install opencv-python

On Raspberry Pi OS you may also need:
    sudo apt-get install libatlas-base-dev

WHERE IMAGES ARE SAVED
-----------------------
    <this folder>/captured_data/faces/<name>_<timestamp>.png

FUTURE UPGRADE PATH (Raspberry Pi 5)
-------------------------------------
Right now this just grabs a single frame. Later this can be swapped to:
    - Use `picamera2` instead of cv2.VideoCapture if using the Pi Camera Module
    - Run face detection (e.g. Haar cascade / MediaPipe) to auto-crop the face
    - Generate and store a face embedding (e.g. via `face_recognition` or a
      lightweight on-device model) instead of / in addition to the raw image
Only the inside of `capture_face()` needs to change for any of that.
================================================================================
"""

import os
import time

import cv2

FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captured_data", "faces")


def _ensure_dir():
    os.makedirs(FACES_DIR, exist_ok=True)


def _safe_filename(name):
    cleaned = "".join(c for c in name if c.isalnum() or c in ("_", "-")) or "owner"
    return cleaned


def capture_face(owner_name="owner", camera_index=0, countdown=3, instruction=None):
    """
    Opens the default camera, shows a live preview window with a short
    countdown, and saves a snapshot as a PNG once the countdown reaches zero.

    Args:
        owner_name:   used to build a readable filename.
        camera_index: which camera to open (0 = default/first camera).
        countdown:    seconds to show a live preview before auto-capturing.
        instruction:  optional short pose instruction shown as an overlay
                      (e.g. "Turn your head to show your LEFT side"), so
                      gui.py can reuse this same function once per required
                      face angle (front / left side / right side) with the
                      right guidance shown each time. Defaults to a generic
                      "Look at the camera" prompt when not given.

    Returns:
        The absolute path to the saved PNG file, or None if no camera was
        found or the user cancelled (pressed ESC / closed the window).

    NOTE: This function is BLOCKING (it runs its own OpenCV preview loop).
    Call it from a background thread if using it from a GUI, so the GUI's
    own event loop doesn't freeze while the camera window is open.
    """
    _ensure_dir()
    instruction = instruction or "Look at the camera"

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        return None

    window_name = "Face Capture - press ESC to cancel"
    start_time = time.time()
    saved_path = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            elapsed = time.time() - start_time
            remaining = max(0, countdown - int(elapsed))
            h, _ = frame.shape[:2]

            display = frame.copy()
            cv2.putText(display, instruction, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(display, f"Capturing in {remaining}...", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 230, 40), 2, cv2.LINE_AA)
            cv2.putText(display, "Press ESC to cancel", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            # if the user closes the window with the mouse
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if elapsed >= countdown:
                filename = f"{_safe_filename(owner_name)}_{int(time.time())}.png"
                filepath = os.path.join(FACES_DIR, filename)
                cv2.imwrite(filepath, frame)
                saved_path = filepath

                confirm = frame.copy()
                cv2.putText(confirm, "Captured!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (60, 220, 60), 2, cv2.LINE_AA)
                cv2.imshow(window_name, confirm)
                cv2.waitKey(500)
                break
    finally:
        cap.release()
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass

    return saved_path


if __name__ == "__main__":
    # Quick manual test: run "python3 face_capture.py" directly.
    result = capture_face(owner_name="test_user")
    if result:
        print(f"Saved face image to: {result}")
    else:
        print("Face capture cancelled or no camera found.")

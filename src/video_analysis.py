"""
video_analysis.py
Analyzes candidate video frames for face presence and engagement using
OpenCV's built-in Haar Cascade face detector.

(Note: an earlier version of this module used MediaPipe Face Mesh, but recent
MediaPipe releases removed the legacy `solutions` API, causing compatibility
issues. OpenCV's Haar Cascade is more stable across versions and needs no
extra model download.)
"""

import cv2
import numpy as np
import os

_CASCADE_PATH = os.path.join(os.path.dirname(__file__), "models", "haarcascade_frontalface_default.xml")
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)

if _face_cascade.empty():
    raise RuntimeError(
        f"Could not load Haar Cascade file at {_CASCADE_PATH}. "
        "Make sure src/models/haarcascade_frontalface_default.xml exists."
    )


def analyze_video(video_path: str, sample_rate: int = 5) -> dict:
    """
    Analyzes a video file for face-presence percentage and positional
    stability/movement (used as an engagement proxy).

    Args:
        video_path: path to the candidate's video file
        sample_rate: analyze every Nth frame (for speed)

    Returns:
        dict with 'face_presence_pct' and 'expression_variety' (0-1 scale)
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    analyzed_frames = 0
    face_detected_frames = 0
    centers = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % sample_rate != 0:
            continue

        analyzed_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) > 0:
            face_detected_frames += 1
            # Use the largest detected face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            cx, cy = x + w / 2, y + h / 2
            centers.append((cx / frame.shape[1], cy / frame.shape[0]))  # normalized

    cap.release()

    face_presence_pct = face_detected_frames / analyzed_frames if analyzed_frames else 0

    # Expression/engagement proxy: how much the face position shifts frame-to-frame
    # (a bit of natural movement is good; wild jumps or a frozen frame score lower)
    if len(centers) > 1:
        centers_arr = np.array(centers)
        movement = np.mean(np.abs(np.diff(centers_arr, axis=0)))
        expression_variety = min(movement * 25, 1.0)
    else:
        expression_variety = 0

    return {
        "face_presence_pct": round(face_presence_pct, 3),
        "expression_variety": round(float(expression_variety), 3),
    }


def video_engagement_score(video_path: str) -> float:
    """Combines face presence and expression variety into one engagement score."""
    result = analyze_video(video_path)
    score = 0.6 * result["face_presence_pct"] + 0.4 * result["expression_variety"]
    return round(score, 3)


if __name__ == "__main__":
    score = video_engagement_score("sample_data/candidate1.mp4")
    print("Video Engagement Score:", score)

from __future__ import annotations

from pathlib import Path


def open_video(path: str | Path):
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("opencv-python is required to read video") from exc
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}. Place demo.mp4 in data/input_videos/ or pass --video.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return cap

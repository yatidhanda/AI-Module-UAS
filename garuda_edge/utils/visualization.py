from __future__ import annotations

from typing import Iterable

from garuda_edge.ingestion.schemas import Track


def draw_tracks(frame, tracks: Iterable[Track]):
    try:
        import cv2
    except Exception:
        return frame
    for track in tracks:
        x1, y1, x2, y2 = [int(v) for v in track.last_bbox_xyxy]
        label = f"{track.track_id} {track.class_name} {track.last_confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 220, 120), 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 220, 120), 2)
    return frame

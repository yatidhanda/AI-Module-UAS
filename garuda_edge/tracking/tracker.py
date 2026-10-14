from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from garuda_edge.ingestion.schemas import Detection, Track
from garuda_edge.tracking.kalman import BoxKalmanFilter

_NO_MATCH_COST = 1e6


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _State:
    track: Track
    kf: BoxKalmanFilter
    missed: int = 0
    hits: int = 1


class ObjectTracker:
    """Stable track IDs from a SORT-style Kalman + Hungarian tracker.

    Each track carries a constant-velocity box filter. Detections are matched to the
    tracks' predicted boxes (same class only) with optimal assignment, so tracks coast
    through short misses instead of being re-created with a new ID.
    """

    def __init__(self, max_missed_frames: int = 30, iou_threshold: float = 0.25, max_coast_frames: int = 5):
        self.max_missed_frames = max_missed_frames
        self.iou_threshold = iou_threshold
        self.max_coast_frames = max_coast_frames
        self._next_id = 1
        self._tracks: dict[str, _State] = {}

    def update(self, detections: list[Detection], frame_id: int, timestamp_s: float) -> list[Track]:
        track_ids = list(self._tracks)
        for track_id in track_ids:
            self._tracks[track_id].kf.predict(timestamp_s)

        matches: list[tuple[str, Detection]] = []
        unmatched_detections = list(detections)
        if track_ids and detections:
            cost = np.full((len(track_ids), len(detections)), _NO_MATCH_COST)
            for i, track_id in enumerate(track_ids):
                state = self._tracks[track_id]
                predicted = state.kf.bbox_xyxy()
                for j, det in enumerate(detections):
                    if det.class_name == state.track.class_name:
                        cost[i, j] = 1.0 - _iou(predicted, det.bbox_xyxy)
            matched_dets: set[int] = set()
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] <= 1.0 - self.iou_threshold:
                    matches.append((track_ids[i], detections[j]))
                    matched_dets.add(j)
            unmatched_detections = [det for j, det in enumerate(detections) if j not in matched_dets]

        matched_track_ids = {track_id for track_id, _ in matches}
        for track_id, det in matches:
            state = self._tracks[track_id]
            state.kf.update(det.bbox_xyxy)
            state.hits += 1
            age = state.track.track_age_frames + 1
            stability = min(1.0, 0.35 + (state.hits / max(10, age)) * 0.65)
            state.track = Track(
                track_id=track_id,
                class_name=det.class_name,
                last_bbox_xyxy=state.kf.bbox_xyxy(),
                last_confidence=det.confidence,
                track_age_frames=age,
                track_stability=stability,
                last_seen_timestamp_s=timestamp_s,
                velocity_px_s=state.kf.velocity_px_s(),
            )
            state.missed = 0
        for track_id in track_ids:
            if track_id not in matched_track_ids:
                self._tracks[track_id].missed += 1

        for det in unmatched_detections:
            track_id = f"T-{self._next_id:04d}"
            self._next_id += 1
            kf = BoxKalmanFilter(det.bbox_xyxy, timestamp_s)
            self._tracks[track_id] = _State(
                track=Track(
                    track_id=track_id,
                    class_name=det.class_name,
                    last_bbox_xyxy=det.bbox_xyxy,
                    last_confidence=det.confidence,
                    track_age_frames=1,
                    track_stability=0.35,
                    last_seen_timestamp_s=timestamp_s,
                    velocity_px_s=kf.velocity_px_s(),
                ),
                kf=kf,
            )

        for track_id, state in list(self._tracks.items()):
            if state.missed > self.max_missed_frames:
                del self._tracks[track_id]

        return [state.track for state in self._tracks.values() if state.missed == 0]

    def predict_tracks(self, timestamp_s: float) -> list[dict[str, Any]]:
        """Predicted box of every recently seen track at timestamp_s; does not change track state."""
        predictions: list[dict[str, Any]] = []
        for track_id, state in self._tracks.items():
            if state.missed > self.max_coast_frames:
                continue
            predictions.append(
                {
                    "track_id": track_id,
                    "class_name": state.track.class_name,
                    "bbox_xyxy": state.kf.peek(timestamp_s),
                    "velocity_px_s": state.kf.velocity_px_s(),
                    "coasting": state.missed > 0,
                }
            )
        return predictions

    def all_tracks(self) -> list[Track]:
        return [state.track for state in self._tracks.values()]

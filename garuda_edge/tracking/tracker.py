from __future__ import annotations

from dataclasses import dataclass

from garuda_edge.ingestion.schemas import Detection, Track


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
    missed: int = 0
    hits: int = 1


class ObjectTracker:
    """Stable track IDs using a robust IoU fallback tracker.

    Supervision ByteTrack can be added without changing callers; this fallback
    is intentionally dependency-light and deterministic for the MVP.
    """

    def __init__(self, max_missed_frames: int = 30, iou_threshold: float = 0.25):
        self.max_missed_frames = max_missed_frames
        self.iou_threshold = iou_threshold
        self._next_id = 1
        self._tracks: dict[str, _State] = {}

    def update(self, detections: list[Detection], frame_id: int, timestamp_s: float) -> list[Track]:
        matched_track_ids: set[str] = set()
        unmatched_detections = list(detections)

        for track_id, state in list(self._tracks.items()):
            best_det = None
            best_iou = 0.0
            for det in unmatched_detections:
                score = _iou(state.track.last_bbox_xyxy, det.bbox_xyxy)
                if score > best_iou and det.class_name == state.track.class_name:
                    best_iou = score
                    best_det = det
            if best_det is not None and best_iou >= self.iou_threshold:
                unmatched_detections.remove(best_det)
                state.hits += 1
                age = state.track.track_age_frames + 1
                stability = min(1.0, 0.35 + (state.hits / max(10, age)) * 0.65)
                state.track = Track(
                    track_id=track_id,
                    class_name=best_det.class_name,
                    last_bbox_xyxy=best_det.bbox_xyxy,
                    last_confidence=best_det.confidence,
                    track_age_frames=age,
                    track_stability=stability,
                    last_seen_timestamp_s=timestamp_s,
                )
                state.missed = 0
                matched_track_ids.add(track_id)
            else:
                state.missed += 1

        for det in unmatched_detections:
            track_id = f"T-{self._next_id:04d}"
            self._next_id += 1
            self._tracks[track_id] = _State(
                track=Track(
                    track_id=track_id,
                    class_name=det.class_name,
                    last_bbox_xyxy=det.bbox_xyxy,
                    last_confidence=det.confidence,
                    track_age_frames=1,
                    track_stability=0.35,
                    last_seen_timestamp_s=timestamp_s,
                )
            )
            matched_track_ids.add(track_id)

        for track_id, state in list(self._tracks.items()):
            if state.missed > self.max_missed_frames:
                del self._tracks[track_id]

        return [state.track for state in self._tracks.values() if state.missed == 0]

    def all_tracks(self) -> list[Track]:
        return [state.track for state in self._tracks.values()]

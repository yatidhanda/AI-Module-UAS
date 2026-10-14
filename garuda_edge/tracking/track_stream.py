from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from garuda_edge.tracking.tracker import ObjectTracker

_EPS = 1e-9


class TrackStateStream:
    """High-rate predicted track states written as JSONL, one line per output tick.

    Ticks lie on a global grid (k / rate_hz) so consecutive frames produce no gaps or
    duplicates. Predictions are causal: each tick extrapolates from the latest frame's
    filter state. This is kinematic data only; it creates no ISR events and does not
    pass through the analyst or policy gate. rate_hz <= 0 disables the stream.
    """

    def __init__(self, path: str | Path, rate_hz: float):
        self.path = Path(path)
        self.rate_hz = float(rate_hz)
        self.ticks_written = 0
        self._fp = None
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = self.path.open("w", encoding="utf-8")

    @property
    def enabled(self) -> bool:
        return self.rate_hz > 0

    def emit_interval(self, tracker: ObjectTracker, t_start: float, t_end: float, frame_id: int | None = None) -> int:
        """Write every tick in [t_start, t_end); returns the number of ticks written."""
        if self._fp is None:
            return 0
        k = math.ceil(t_start * self.rate_hz - _EPS)
        written = 0
        while True:
            t = k / self.rate_hz
            if t >= t_end - _EPS:
                break
            tracks = [
                {
                    "track_id": item["track_id"],
                    "class_name": item["class_name"],
                    "bbox_xyxy": [round(v, 2) for v in item["bbox_xyxy"]],
                    "velocity_px_s": [round(v, 2) for v in item["velocity_px_s"]],
                    "coasting": item["coasting"],
                }
                for item in tracker.predict_tracks(t)
            ]
            self._fp.write(json.dumps({"t": round(t, 6), "frame_id": frame_id, "tracks": tracks}) + "\n")
            written += 1
            k += 1
        self.ticks_written += written
        return written

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def __enter__(self) -> TrackStateStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

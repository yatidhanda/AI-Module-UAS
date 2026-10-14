from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from garuda_edge.dashboard.ownship import extract_latest_ownship_state
from garuda_edge.ingestion.schemas import ISREvent


def _track_row(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "track_id": event["track_id"],
        "object_class": event["object_class"],
        "last_seen_timestamp_s": event["timestamp_s"],
        "track_age_frames": event["track_age_frames"],
        "track_stability": event["track_stability"],
        "confidence_level": event["confidence"]["level"],
        "final_confidence": event["confidence"]["final_confidence"],
        "estimated_lat": event["geo_estimate"]["estimated_lat"],
        "estimated_lon": event["geo_estimate"]["estimated_lon"],
        "threat_assessment": event.get("threat_assessment"),
        "geo_lock_readiness": event.get("geo_lock_readiness"),
    }


def _latest_decision(event: dict[str, Any] | None) -> str | None:
    if not event:
        return None
    return (event.get("policy_result") or {}).get("final_decision", {}).get("decision")


class EventStore:
    """Append-only events.jsonl with derived tracks.json and summary.json.

    The writer keeps per-track state and summary counters in memory and rewrites the
    derived files at most every `flush_interval_s` seconds (and on flush/close), so
    appending stays O(1) instead of re-reading the whole log per event. Readers (the
    API) still derive everything from events.jsonl on disk.
    """

    def __init__(self, output_dir: str | Path = "data/outputs", flush_interval_s: float = 1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"
        self.tracks_path = self.output_dir / "tracks.json"
        self.summary_path = self.output_dir / "summary.json"
        self.flush_interval_s = flush_interval_s
        self._reset_state()

    def append_event(self, event: ISREvent | dict[str, Any]) -> None:
        payload = event.model_dump(mode="json") if hasattr(event, "model_dump") else dict(event)
        if not self._state_loaded:
            self._load_state_from_disk()
        with self.events_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload) + "\n")
        self._observe(payload)
        self._dirty = True
        if time.monotonic() - self._last_flush >= self.flush_interval_s:
            self.flush()

    def flush(self) -> None:
        """Write tracks.json and summary.json from the in-memory state."""
        if not self._dirty:
            return
        self.tracks_path.write_text(json.dumps(list(self._latest_by_track.values()), indent=2), encoding="utf-8")
        self.summary_path.write_text(json.dumps(self._summary_from_state(), indent=2), encoding="utf-8")
        self._dirty = False
        self._last_flush = time.monotonic()

    close = flush

    def read_events(self, limit: int = 100) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        lines = self.events_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines[-limit:] if line.strip()]

    def read_latest_event(self) -> dict[str, Any] | None:
        events = self.read_events(limit=1)
        return events[0] if events else None

    def read_tracks(self) -> list[dict[str, Any]]:
        latest_by_track: dict[str, dict[str, Any]] = {}
        for event in self.read_events(limit=10_000):
            latest_by_track[event["track_id"]] = _track_row(event)
        return list(latest_by_track.values())

    def read_summary(self) -> dict[str, Any]:
        if self.summary_path.exists():
            return json.loads(self.summary_path.read_text(encoding="utf-8"))
        return self._summary_from_events()

    def clear(self) -> None:
        for path in [self.events_path, self.tracks_path, self.summary_path]:
            if path.exists():
                path.unlink()
        self._reset_state()

    def _reset_state(self) -> None:
        self._latest_by_track: dict[str, dict[str, Any]] = {}
        self._event_count = 0
        self._confidence_sum = 0.0
        self._last_event: dict[str, Any] | None = None
        self._ownship_event: dict[str, Any] | None = None
        self._state_loaded = False
        self._dirty = False
        self._last_flush = time.monotonic()

    def _load_state_from_disk(self) -> None:
        self._state_loaded = True
        if not self.events_path.exists():
            return
        with self.events_path.open(encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    self._observe(json.loads(line))

    def _observe(self, event: dict[str, Any]) -> None:
        self._latest_by_track[event["track_id"]] = _track_row(event)
        self._event_count += 1
        self._confidence_sum += event["confidence"]["final_confidence"]
        self._last_event = event
        # Keep the event carrying the latest valid ownship state, as extract_latest_ownship_state would pick.
        state = extract_latest_ownship_state([event])
        if state is not None and (
            self._ownship_event is None or extract_latest_ownship_state([self._ownship_event, event]) == state
        ):
            self._ownship_event = event

    def _summary_from_state(self) -> dict[str, Any]:
        return {
            "total_events": self._event_count,
            "active_tracks": len(self._latest_by_track),
            "average_confidence": round(self._confidence_sum / self._event_count, 3) if self._event_count else 0.0,
            "latest_decision": _latest_decision(self._last_event),
            "latest_platform_state": extract_latest_ownship_state([self._ownship_event]) if self._ownship_event else None,
        }

    def _summary_from_events(self) -> dict[str, Any]:
        events = self.read_events(limit=10_000)
        tracks = self.read_tracks()
        avg_conf = 0.0
        if events:
            avg_conf = sum(event["confidence"]["final_confidence"] for event in events) / len(events)
        return {
            "total_events": len(events),
            "active_tracks": len(tracks),
            "average_confidence": round(avg_conf, 3),
            "latest_decision": _latest_decision(events[-1] if events else None),
            "latest_platform_state": extract_latest_ownship_state(events),
        }

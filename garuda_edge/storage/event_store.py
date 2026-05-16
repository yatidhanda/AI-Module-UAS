from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from garuda_edge.ingestion.schemas import ISREvent


class EventStore:
    def __init__(self, output_dir: str | Path = "data/outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"
        self.tracks_path = self.output_dir / "tracks.json"
        self.summary_path = self.output_dir / "summary.json"

    def append_event(self, event: ISREvent | dict[str, Any]) -> None:
        payload = event.model_dump(mode="json") if hasattr(event, "model_dump") else dict(event)
        with self.events_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload) + "\n")
        self._write_tracks()
        self._write_summary()

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
            latest_by_track[event["track_id"]] = {
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
        return list(latest_by_track.values())

    def read_summary(self) -> dict[str, Any]:
        if self.summary_path.exists():
            return json.loads(self.summary_path.read_text(encoding="utf-8"))
        return self._summary_from_events()

    def clear(self) -> None:
        for path in [self.events_path, self.tracks_path, self.summary_path]:
            if path.exists():
                path.unlink()

    def _write_tracks(self) -> None:
        self.tracks_path.write_text(json.dumps(self.read_tracks(), indent=2), encoding="utf-8")

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
            "latest_decision": (events[-1].get("policy_result") or {}).get("final_decision", {}).get("decision") if events else None,
        }

    def _write_summary(self) -> None:
        self.summary_path.write_text(json.dumps(self._summary_from_events(), indent=2), encoding="utf-8")

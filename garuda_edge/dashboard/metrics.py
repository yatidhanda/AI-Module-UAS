from __future__ import annotations

from collections import Counter
from typing import Any

from garuda_edge.dashboard.data_loader import (
    extract_confidence_level,
    extract_degraded_flags,
    extract_final_decision,
    extract_geo,
    extract_gemma_decision,
    extract_policy_corrections,
    extract_risk_flags,
    extract_threat_assessment,
)
from garuda_edge.dashboard.ownship import extract_platform_state


def compute_dashboard_metrics(
    events: list[dict[str, Any]],
    active_tracks: list[dict[str, Any]],
    operator_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    confidence_counts = Counter(track.get("confidence_level", "unknown") for track in active_tracks)
    policy_corrections = sum(len(extract_policy_corrections(event)) for event in events)
    gemma_events = sum(1 for event in events if extract_gemma_decision(event))
    latest_frame = max([event.get("frame_id", 0) or 0 for event in events], default=0)
    latest_phase = _latest_mission_phase(events)
    threat_counts = Counter(str(track.get("threat_level") or "unknown").lower() for track in active_tracks)
    return {
        "total_events": len(events),
        "active_tracks": len(active_tracks),
        "high_confidence_tracks": confidence_counts.get("high", 0),
        "medium_confidence_tracks": confidence_counts.get("medium", 0),
        "low_confidence_tracks": confidence_counts.get("low", 0),
        "high_threat_tracks": threat_counts.get("high", 0),
        "critical_threat_tracks": threat_counts.get("critical", 0),
        "active_alerts": threat_counts.get("high", 0) + threat_counts.get("critical", 0),
        "detection_track_updates": len(events),
        "gemma_analyzed_events": gemma_events,
        "policy_corrections": policy_corrections,
        "operator_actions": len(operator_actions),
        "latest_frame": latest_frame,
        "latest_mission_phase": latest_phase or "Not specified",
    }


def build_alerts(events: list[dict[str, Any]], active_tracks: list[dict[str, Any]]) -> list[dict[str, str]]:
    low_conf = [track for track in active_tracks if track.get("confidence_level") == "low"]
    high_uncertainty = [
        track for track in active_tracks if (track.get("geo_uncertainty_m") is not None and track["geo_uncertainty_m"] >= 50)
    ]
    degraded_flags = sorted({flag for event in events for flag in extract_degraded_flags(event)})
    corrections = sum(len(extract_policy_corrections(event)) for event in events)
    high_threat = [
        track for track in active_tracks if str(track.get("threat_level") or "").lower() in {"high", "critical"}
    ]
    fallback_or_unavailable = any(
        bool(event.get("analyst_model_fallback_used") or event.get("fallback_used"))
        or "llm_unavailable" in extract_risk_flags(event)
        for event in events
    )

    alerts: list[dict[str, str]] = []
    if low_conf:
        alerts.append({"level": "red", "label": f"Low confidence tracks: {len(low_conf)}"})
    if high_uncertainty:
        alerts.append({"level": "amber", "label": f"High geo uncertainty: {len(high_uncertainty)}"})
    if degraded_flags:
        alerts.append({"level": "amber", "label": f"Degraded flags: {len(degraded_flags)}"})
    if corrections:
        alerts.append({"level": "amber", "label": f"Policy corrections: {corrections}"})
    if high_threat:
        critical = sum(1 for track in high_threat if str(track.get("threat_level")).lower() == "critical")
        level = "red" if critical else "orange"
        alerts.append({"level": level, "label": f"High/Critical mission risk tracks: {len(high_threat)}"})
    if fallback_or_unavailable:
        alerts.append({"level": "amber", "label": "Analyst fallback/unavailable"})
    if not alerts:
        alerts.append({"level": "green", "label": "No active dashboard alerts"})
    return alerts


def object_class_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(event.get("object_class") or "unknown") for event in events)
    return dict(counts.most_common())


def confidence_distribution(events: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(extract_confidence_level(event) for event in events)
    return {level: counts.get(level, 0) for level in ["high", "medium", "low", "unknown"]}


def link_sensor_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {}
    latest = events[-1]
    state = extract_platform_state(latest)
    platform = latest.get("platform") if isinstance(latest.get("platform"), dict) else {}
    link = latest.get("link") if isinstance(latest.get("link"), dict) else {}
    return {
        "sensor_health": state.get("sensor_health") or platform.get("sensor_health", "Not specified"),
        "mission_phase": platform.get("mission_phase", "Not specified"),
        "gps_quality": state.get("gps_quality") or "Not specified",
        "link_quality": state.get("link_quality") or link.get("link_quality", "Not specified"),
        "packet_loss": state.get("packet_loss") if state.get("packet_loss") is not None else link.get("packet_loss", "Not specified"),
        "latency_ms": state.get("latency_ms") if state.get("latency_ms") is not None else link.get("latency_ms", "Not specified"),
        "sync_delta_s": link.get("sync_delta_s", "Not specified"),
    }


def readiness_status(
    health: dict[str, Any],
    events: list[dict[str, Any]],
    operator_actions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {"item": "Events loaded", "status": "ready" if events else "missing"},
        {"item": "Output folder", "status": "ready" if health.get("events_exists") else "missing"},
        {"item": "Annotated video", "status": "ready" if health.get("video_exists") else "missing"},
        {"item": "Telemetry", "status": "simulated"},
        {"item": "Gemma analysis", "status": "present" if health.get("gemma_analysis_exists") else "absent"},
        {"item": "Policy engine", "status": "present"},
        {"item": "Operator log", "status": "present" if operator_actions else "empty"},
        {"item": "Threat library", "status": "present"},
    ]


def track_timeline(events: list[dict[str, Any]], track_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        event_track_id = str(event.get("track_id") or event.get("event_id") or "unknown")
        if event_track_id != track_id:
            continue
        final_decision = extract_final_decision(event)
        geo = extract_geo(event)
        confidence = event.get("confidence") if isinstance(event.get("confidence"), dict) else {}
        rows.append(
            {
                "frame_id": event.get("frame_id"),
                "timestamp_s": event.get("timestamp_s"),
                "confidence": confidence.get("final_confidence") or event.get("final_confidence"),
                "confidence_level": extract_confidence_level(event),
                "track_stability": event.get("track_stability"),
                "geo_uncertainty_m": geo["geo_uncertainty_m"],
                "threat_level": (extract_threat_assessment(event).get("level") or "unknown"),
                "threat_score": extract_threat_assessment(event).get("score"),
                "decision": final_decision.get("decision") or "notify_operator",
            }
        )
    return sorted(rows, key=lambda row: (row.get("frame_id") or -1, row.get("timestamp_s") or -1))


def _latest_mission_phase(events: list[dict[str, Any]]) -> str | None:
    for event in reversed(events):
        platform = event.get("platform") if isinstance(event.get("platform"), dict) else {}
        if platform.get("mission_phase"):
            return str(platform["mission_phase"])
    return None

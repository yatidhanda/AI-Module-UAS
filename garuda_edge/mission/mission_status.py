from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


def evaluate_geo_lock_readiness(event: dict[str, Any], mission_context: dict[str, Any]) -> dict[str, Any]:
    mission = _raw_mission(mission_context)
    objectives = mission.get("geo_lock_objectives") if isinstance(mission.get("geo_lock_objectives"), dict) else {}
    if not objectives.get("enabled", False):
        return {"ready": False, "status": "not_applicable", "reasons": ["geo_lock_objectives_disabled"]}

    object_class = _norm(event.get("object_class") or event.get("class_name"))
    target_classes = {_norm(item) for item in objectives.get("target_classes", [])}
    if object_class not in target_classes:
        return {
            "ready": False,
            "status": "not_applicable",
            "reasons": [f"class_not_in_geo_lock_objectives:{object_class or 'unknown'}"],
        }

    confidence = _extract_confidence(event)
    geo_uncertainty = _extract_geo_uncertainty(event)
    age_frames = _int(event.get("track_age_frames"), default=0)
    min_confidence = _float(objectives.get("min_confidence"), default=0.0)
    max_uncertainty = _float(objectives.get("max_geo_uncertainty_m"), default=float("inf"))
    min_age = _int(objectives.get("min_track_age_frames"), default=0)

    reasons: list[str] = []
    ready = True
    if confidence is None or confidence < min_confidence:
        ready = False
        reasons.append(f"confidence_below_threshold:{_format_value(confidence)}<{min_confidence:g}")
    else:
        reasons.append(f"confidence_ok:{confidence:.2f}>={min_confidence:g}")

    if geo_uncertainty is None or geo_uncertainty > max_uncertainty:
        ready = False
        reasons.append(f"geo_uncertainty_above_threshold:{_format_value(geo_uncertainty)}>{max_uncertainty:g}")
    else:
        reasons.append(f"geo_uncertainty_ok:{geo_uncertainty:.1f}<={max_uncertainty:g}")

    if age_frames < min_age:
        ready = False
        reasons.append(f"track_age_below_threshold:{age_frames}<{min_age}")
    else:
        reasons.append(f"track_age_ok:{age_frames}>={min_age}")

    return {"ready": ready, "status": "ready" if ready else "not_ready", "reasons": reasons}


def compute_mission_status(
    events: list[dict[str, Any]],
    tracks: list[dict[str, Any]] | dict[str, Any] | None,
    summary: dict[str, Any] | None,
    mission_context: dict[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    mission = _raw_mission(mission_context)
    summary = summary or {}
    latest_tracks = _latest_events_by_track(events)
    active_track_events = list(latest_tracks.values())
    confidence_counts = Counter(_extract_confidence_level(event) for event in active_track_events)
    threat_counts = Counter(_extract_threat_level(event) for event in active_track_events)
    degraded_flags = [flag for event in events for flag in _extract_degraded_flags(event)]
    geo_lock_ready = [
        str(event.get("track_id") or event.get("event_id") or "unknown")
        for event in active_track_events
        if evaluate_geo_lock_readiness(event, mission).get("ready")
    ]
    completion = _completion_estimate(events, active_track_events, summary, mission, output_root)
    analyst_present = any(isinstance(event.get("gemma_decision"), dict) for event in events)
    health_status = _mission_health_status(threat_counts, degraded_flags, completion["percent_complete"])

    return {
        "scenario_name": _scenario_name(output_root),
        "total_events": len(events),
        "active_tracks": len(active_track_events),
        "confidence_distribution": {
            "high": confidence_counts.get("high", 0),
            "medium": confidence_counts.get("medium", 0),
            "low": confidence_counts.get("low", 0),
            "unknown": confidence_counts.get("unknown", 0),
        },
        "high_threat_tracks": threat_counts.get("high", 0),
        "critical_threat_tracks": threat_counts.get("critical", 0),
        "degraded_flags_count": len(degraded_flags),
        "geo_lock_ready_tracks": geo_lock_ready,
        "mission_completion_estimate": completion,
        "mission_health": {
            "status": health_status,
            "perception_health": "events_present" if events else "no_events",
            "tracking_health": "tracks_present" if active_track_events else "no_tracks",
            "geo_confidence_health": "geo_lock_ready" if geo_lock_ready else "needs_review",
            "link_health": "degraded" if degraded_flags else "nominal",
            "analyst_health": "present" if analyst_present else "not_run",
            "key_degradations": sorted(set(degraded_flags))[:12],
        },
        "human_review_required": True,
    }


def _completion_estimate(
    events: list[dict[str, Any]],
    active_track_events: list[dict[str, Any]],
    summary: dict[str, Any],
    mission: dict[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    completion_cfg = mission.get("mission_completion") if isinstance(mission.get("mission_completion"), dict) else {}
    rules = completion_cfg.get("completion_rules") if isinstance(completion_cfg.get("completion_rules"), dict) else {}
    min_events = max(1, _int(rules.get("min_events_per_scenario"), default=50))
    min_tracks = max(1, _int(rules.get("min_tracks_per_scenario"), default=3))
    events_count = int(summary.get("events_count") or len(events))
    tracks_count = int(summary.get("tracks_count") or len(active_track_events))
    events_ratio = min(1.0, events_count / min_events)
    tracks_ratio = min(1.0, tracks_count / min_tracks)
    percent = round((events_ratio * 0.55 + tracks_ratio * 0.45) * 100, 1)
    scenario = _scenario_name(output_root)
    required = completion_cfg.get("required_scenarios") if isinstance(completion_cfg.get("required_scenarios"), list) else []
    remaining_gaps: list[str] = []
    if events_count < min_events:
        remaining_gaps.append(f"collect_at_least_{min_events}_events")
    if tracks_count < min_tracks:
        remaining_gaps.append(f"maintain_at_least_{min_tracks}_tracks")
    if required and scenario not in {"default_output", "outputs", "visdrone"} and scenario not in required:
        remaining_gaps.append("selected_scenario_not_listed_in_mission_requirements")
    status = "complete" if percent >= 100 and not remaining_gaps else "in_progress" if percent >= 50 else "insufficient_data"
    return {
        "percent_complete": percent,
        "status": status,
        "rationale": f"Scenario-level estimate from {events_count} loaded events and {tracks_count} active tracks.",
        "remaining_gaps": remaining_gaps,
    }


def _mission_health_status(threat_counts: Counter, degraded_flags: list[str], percent_complete: float) -> str:
    if threat_counts.get("critical", 0) or percent_complete < 35:
        return "needs_review"
    if threat_counts.get("high", 0) or degraded_flags:
        return "degraded"
    return "nominal"


def _latest_events_by_track(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in events:
        track_id = str(event.get("track_id") or event.get("event_id") or "unknown")
        prior = latest.get(track_id)
        if prior is None or _event_sort_key(event) >= _event_sort_key(prior):
            latest[track_id] = event
    return latest


def _extract_confidence(event: dict[str, Any]) -> float | None:
    confidence = event.get("confidence") if isinstance(event.get("confidence"), dict) else {}
    for value in [confidence.get("final_confidence"), event.get("final_confidence"), event.get("detection_confidence")]:
        parsed = _float_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _extract_confidence_level(event: dict[str, Any]) -> str:
    confidence = event.get("confidence") if isinstance(event.get("confidence"), dict) else {}
    return str(confidence.get("level") or event.get("confidence_level") or "unknown").lower()


def _extract_geo_uncertainty(event: dict[str, Any]) -> float | None:
    geo = event.get("geo_estimate") if isinstance(event.get("geo_estimate"), dict) else {}
    return _float_or_none(geo.get("geo_uncertainty_m") if geo else event.get("geo_uncertainty_m"))


def _extract_threat_level(event: dict[str, Any]) -> str:
    threat = event.get("threat_assessment") if isinstance(event.get("threat_assessment"), dict) else {}
    return str(threat.get("level") or "unknown").lower()


def _extract_degraded_flags(event: dict[str, Any]) -> list[str]:
    confidence = event.get("confidence") if isinstance(event.get("confidence"), dict) else {}
    flags = confidence.get("degraded_flags") if isinstance(confidence.get("degraded_flags"), list) else []
    return [str(flag) for flag in flags]


def _event_sort_key(event: dict[str, Any]) -> tuple[float, float]:
    return (_float(event.get("frame_id"), default=-1.0), _float(event.get("timestamp_s"), default=-1.0))


def _scenario_name(output_root: str | Path) -> str:
    path = Path(output_root)
    if path.name == "outputs":
        return "default_output"
    return path.name or "unknown"


def _raw_mission(mission_context: dict[str, Any]) -> dict[str, Any]:
    raw = mission_context.get("raw") if isinstance(mission_context.get("raw"), dict) else None
    return raw if raw is not None else mission_context


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _format_value(value: float | None) -> str:
    return "missing" if value is None else f"{value:.2f}"


def _float(value: Any, default: float) -> float:
    parsed = _float_or_none(value)
    return default if parsed is None else parsed


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default

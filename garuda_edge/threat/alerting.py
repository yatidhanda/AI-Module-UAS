from __future__ import annotations

from typing import Any

from garuda_edge.threat.schemas import ThreatAlert


def build_threat_alerts(events: list[dict[str, Any]], threshold: int = 60, max_alerts: int = 8) -> list[ThreatAlert]:
    latest_by_track: dict[str, dict[str, Any]] = {}
    for event in events:
        track_id = str(event.get("track_id") or event.get("event_id") or "unknown")
        prior = latest_by_track.get(track_id)
        if prior is None or _event_sort_key(event) >= _event_sort_key(prior):
            latest_by_track[track_id] = event

    alerts: list[ThreatAlert] = []
    for track_id, event in latest_by_track.items():
        threat = event.get("threat_assessment") if isinstance(event.get("threat_assessment"), dict) else {}
        score = _float(threat.get("score"), default=0.0)
        if score < threshold:
            continue
        level = str(threat.get("level") or "unknown")
        object_class = str(event.get("object_class") or "unknown")
        alerts.append(
            ThreatAlert(
                alert_id=f"risk-{track_id}-{event.get('frame_id', 'na')}",
                timestamp_s=_float_or_none(event.get("timestamp_s")),
                frame_id=_int_or_none(event.get("frame_id")),
                track_id=track_id,
                object_class=object_class,
                threat_level=level,
                threat_score=round(score, 2),
                summary=_short_summary(threat, object_class, level, score),
                recommended_actions=[str(action) for action in threat.get("recommended_operator_actions", [])],
                human_review_required=True,
            )
        )

    alerts.sort(key=lambda alert: (alert.threat_score, alert.frame_id or -1), reverse=True)
    return alerts[:max_alerts]


def _short_summary(threat: dict[str, Any], object_class: str, level: str, score: float) -> str:
    rationale = str(threat.get("rationale") or "Mission/platform risk requires human review.")
    return f"{object_class} track assessed {level} mission/platform risk ({score:.1f}). {rationale[:180]}"


def _event_sort_key(event: dict[str, Any]) -> tuple[float, float]:
    return (_float(event.get("frame_id"), default=-1.0), _float(event.get("timestamp_s"), default=-1.0))


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


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None

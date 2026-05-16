from __future__ import annotations

from typing import Any

from garuda_edge.threat.schemas import ThreatAssessment
from garuda_edge.threat.threat_library import resolve_object_profile


SAFE_THREAT_ACTIONS = [
    "continue_observation",
    "request_recheck",
    "request_sensor_confirmation",
    "store_evidence",
    "notify_operator",
    "escalate_to_c2",
    "mark_low_confidence",
]


def score_threat_for_track(
    track_event: dict[str, Any],
    mission_context: dict[str, Any],
    threat_library: dict[str, Any],
    operator_watchlist: dict[str, Any] | None = None,
) -> ThreatAssessment:
    mission = _raw_mission(mission_context)
    profile = resolve_object_profile(str(track_event.get("object_class") or track_event.get("class_name") or "unknown"), threat_library)
    library_available = bool(threat_library.get("_library_available", True))
    if not library_available:
        return ThreatAssessment(
            score=0.0,
            level="low",
            label="Low",
            category="unknown",
            rationale="Threat library unavailable. Mission/platform risk was not elevated; human review remains required.",
            contributing_factors=["threat_library_unavailable"],
            recommended_operator_actions=["continue_observation", "store_evidence"],
            human_review_required=True,
        )

    score = _float(profile.get("base_score"), default=10.0)
    factors = [f"base_profile:{profile.get('name', 'unknown')}={score:.1f}"]
    modifiers = threat_library.get("context_modifiers") if isinstance(threat_library.get("context_modifiers"), dict) else {}
    object_class = _norm(track_event.get("object_class") or track_event.get("class_name"))
    category = str(profile.get("category") or "unknown")
    confidence = _extract_confidence(track_event)
    geo_uncertainty = _extract_geo_uncertainty(track_event)
    age_frames = _int(track_event.get("track_age_frames"), default=0)

    priority_classes = {_norm(item) for item in mission.get("priority_classes", []) if item is not None}
    if object_class in priority_classes:
        bonus = _float(modifiers.get("mission_priority_class_bonus"), default=0.0)
        score += bonus
        factors.append(f"mission_priority_class_bonus:+{bonus:.1f}")

    if _watchlist_match(track_event, mission, operator_watchlist):
        bonus = _float(modifiers.get("watchlist_track_bonus"), default=0.0)
        score += bonus
        factors.append(f"watchlist_track_bonus:+{bonus:.1f}")

    geo_lock = mission.get("geo_lock_objectives") if isinstance(mission.get("geo_lock_objectives"), dict) else {}
    geo_lock_classes = {_norm(item) for item in geo_lock.get("target_classes", [])}
    if geo_lock.get("enabled") and object_class in geo_lock_classes:
        bonus = _float(modifiers.get("geo_lock_requested_bonus"), default=0.0)
        score += bonus
        factors.append(f"geo_lock_requested_bonus:+{bonus:.1f}")

    persistent = modifiers.get("persistent_track_bonus") if isinstance(modifiers.get("persistent_track_bonus"), dict) else {}
    if age_frames >= _int(persistent.get("min_age_frames"), default=10_000):
        bonus = _float(persistent.get("bonus"), default=0.0)
        score += bonus
        factors.append(f"persistent_track_bonus:+{bonus:.1f}")

    high_conf = modifiers.get("high_confidence_bonus") if isinstance(modifiers.get("high_confidence_bonus"), dict) else {}
    if confidence is not None and confidence >= _float(high_conf.get("min_confidence"), default=2.0):
        bonus = _float(high_conf.get("bonus"), default=0.0)
        score += bonus
        factors.append(f"high_confidence_bonus:+{bonus:.1f}")

    low_conf = modifiers.get("low_confidence_penalty") if isinstance(modifiers.get("low_confidence_penalty"), dict) else {}
    if confidence is not None and confidence <= _float(low_conf.get("max_confidence"), default=-1.0):
        penalty = _float(low_conf.get("penalty"), default=0.0)
        score -= penalty
        factors.append(f"low_confidence_penalty:-{penalty:.1f}")

    high_uncertainty = modifiers.get("high_geo_uncertainty_penalty")
    if isinstance(high_uncertainty, dict) and geo_uncertainty is not None:
        if geo_uncertainty >= _float(high_uncertainty.get("min_uncertainty_m"), default=10_000.0):
            penalty = _float(high_uncertainty.get("penalty"), default=0.0)
            score -= penalty
            factors.append(f"high_geo_uncertainty_penalty:-{penalty:.1f}")

    degraded_link = modifiers.get("degraded_link_bonus_for_ew_categories")
    if isinstance(degraded_link, dict) and _link_degraded(track_event, degraded_link):
        categories = {_norm(item) for item in degraded_link.get("categories", [])}
        if _norm(category) in categories:
            bonus = _float(degraded_link.get("bonus"), default=0.0)
            score += bonus
            factors.append(f"degraded_link_bonus_for_{category}:+{bonus:.1f}")

    degraded_sensor = modifiers.get("degraded_sensor_penalty") if isinstance(modifiers.get("degraded_sensor_penalty"), dict) else {}
    if _sensor_degraded(track_event):
        penalty = _float(degraded_sensor.get("penalty"), default=0.0)
        score -= penalty
        factors.append(f"degraded_sensor_penalty:-{penalty:.1f}")

    score = _clamp(score, 0.0, 100.0)
    level, label = _level_for_score(score, threat_library)
    rationale = (
        f"{profile.get('rationale', 'Review depends on mission context.')} "
        "This is a mission/platform risk estimate for human review, not hostile intent or engagement authorization."
    )
    return ThreatAssessment(
        score=round(score, 2),
        level=level,
        label=label,
        category=category,
        rationale=rationale,
        contributing_factors=factors,
        recommended_operator_actions=_actions_for_level(level, confidence),
        human_review_required=True,
    )


def _actions_for_level(level: str, confidence: float | None) -> list[str]:
    if level == "critical":
        return ["notify_operator", "escalate_to_c2", "store_evidence", "request_sensor_confirmation"]
    if level == "high":
        return ["notify_operator", "store_evidence", "request_recheck"]
    if confidence is not None and confidence <= 0.45:
        return ["mark_low_confidence", "request_sensor_confirmation", "store_evidence"]
    return ["continue_observation", "store_evidence"]


def _level_for_score(score: float, library: dict[str, Any]) -> tuple[str, str]:
    levels = library.get("levels") if isinstance(library.get("levels"), dict) else {}
    for name, config in levels.items():
        if not isinstance(config, dict):
            continue
        if _float(config.get("min"), default=0) <= score <= _float(config.get("max"), default=100):
            return str(name), str(config.get("label") or name.title())
    return "low", "Low"


def _watchlist_match(event: dict[str, Any], mission: dict[str, Any], watchlist: dict[str, Any] | None) -> bool:
    watchlist = watchlist or {}
    track_id = str(event.get("track_id") or event.get("event_id") or "")
    object_class = _norm(event.get("object_class") or event.get("class_name"))
    track_ids = {str(item) for item in watchlist.get("track_ids", [])}
    classes = {_norm(item) for item in watchlist.get("object_classes", [])}
    geo_classes = {_norm(item) for item in watchlist.get("geo_lock_requested_classes", [])}
    mission_watchlist = mission.get("watchlist") if isinstance(mission.get("watchlist"), dict) else {}
    track_ids.update(str(item) for item in mission_watchlist.get("track_ids", []))
    classes.update(_norm(item) for item in mission_watchlist.get("object_classes", []))
    return bool((track_id and track_id in track_ids) or (object_class and object_class in classes | geo_classes))


def _link_degraded(event: dict[str, Any], config: dict[str, Any]) -> bool:
    link = event.get("link") if isinstance(event.get("link"), dict) else {}
    packet_loss = _float(link.get("packet_loss"), default=0.0)
    latency = _float(link.get("latency_ms"), default=0.0)
    return packet_loss >= _float(config.get("packet_loss_threshold"), default=1.0) or latency >= _float(
        config.get("latency_threshold_ms"), default=10_000.0
    )


def _sensor_degraded(event: dict[str, Any]) -> bool:
    platform = event.get("platform") if isinstance(event.get("platform"), dict) else {}
    sensor_health = str(platform.get("sensor_health") or "").lower()
    confidence = event.get("confidence") if isinstance(event.get("confidence"), dict) else {}
    flags = confidence.get("degraded_flags") if isinstance(confidence.get("degraded_flags"), list) else []
    if flags:
        return True
    return bool(sensor_health and sensor_health not in {"nominal", "good", "healthy", "ok"})


def _extract_confidence(event: dict[str, Any]) -> float | None:
    confidence = event.get("confidence") if isinstance(event.get("confidence"), dict) else {}
    for value in [confidence.get("final_confidence"), event.get("final_confidence"), event.get("detection_confidence")]:
        parsed = _float_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _extract_geo_uncertainty(event: dict[str, Any]) -> float | None:
    geo = event.get("geo_estimate") if isinstance(event.get("geo_estimate"), dict) else {}
    return _float_or_none(geo.get("geo_uncertainty_m") if geo else event.get("geo_uncertainty_m"))


def _raw_mission(mission_context: dict[str, Any]) -> dict[str, Any]:
    raw = mission_context.get("raw") if isinstance(mission_context.get("raw"), dict) else None
    return raw if raw is not None else mission_context


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

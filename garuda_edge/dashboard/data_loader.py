from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from garuda_edge.mission.mission_status import evaluate_geo_lock_readiness
from garuda_edge.threat.threat_library import load_threat_library
from garuda_edge.threat.threat_scoring import score_threat_for_track


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VISDRONE_SCENARIOS = [
    "vehicle_heavy_road",
    "urban_intersection",
    "pedestrian_overhead",
    "high_altitude_tiny_vehicle",
    "moving_drifting_camera",
]

MISSION_KEYS = [
    "mission_id",
    "mission_name",
    "mission_phase",
    "sector_id",
    "area_of_interest",
    "task_type",
    "ruleset",
    "human_review_required",
    "priority_classes",
    "operator_policy",
    "allowed_operator_actions",
    "confidence_behavior",
    "geo_lock_objectives",
    "mission_completion",
    "notes",
    "gemma_context",
]

ACTION_LABELS = {
    "continue_observation": "Continue Observing",
    "request_recheck": "Request Re-check",
    "escalate_to_c2": "Escalate to C2",
    "return_to_base": "Return to Base",
    "hold_position": "Hold Position",
    "mark_false_positive": "Mark False Positive",
    "store_evidence": "Store Evidence",
    "notify_operator": "Notify Operator",
    "request_sensor_confirmation": "Request Sensor Confirmation",
    "mark_low_confidence": "Mark Low Confidence",
    "acknowledge": "Acknowledge",
    "reject": "Reject",
    "end_mission_review": "End Mission Review",
}


def discover_output_roots(base_output: str | Path = "data/outputs") -> list[Path]:
    base = resolve_project_path(base_output)
    roots = [base]
    visdrone_root = base / "visdrone"
    if visdrone_root.exists():
        for scenario_dir in sorted([p for p in visdrone_root.iterdir() if p.is_dir()]):
            roots.append(scenario_dir)
    return roots


def load_events(output_root: str | Path, limit: int = 1000) -> list[dict[str, Any]]:
    path = resolve_project_path(output_root) / "events.jsonl"
    return load_jsonl(path, limit=limit)


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if limit is not None and limit > 0:
        lines = lines[-limit:]
    records: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
        except json.JSONDecodeError:
            continue
    return records


def load_summary(output_root: str | Path) -> dict[str, Any]:
    path = resolve_project_path(output_root) / "summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_tracks(output_root: str | Path) -> list[dict[str, Any]]:
    path = resolve_project_path(output_root) / "tracks.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    except json.JSONDecodeError:
        return []


def load_operator_actions(output_root: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    return load_jsonl(operator_action_log_path(output_root), limit=limit)


def append_operator_action(
    output_root: str | Path,
    track_id: str,
    action: str,
    notes: str = "",
    event_id: str | None = None,
    mission_id: str | None = None,
    scenario_name: str | None = None,
    threat_score: float | None = None,
    threat_level: str | None = None,
    alert_id: str | None = None,
    preview_frame_id: int | None = None,
    preview_image_path: str | None = None,
    selected_object_class: str | None = None,
    selected_confidence: float | None = None,
    selected_threat_level: str | None = None,
    selected_threat_score: float | None = None,
    selected_geo_lock_status: str | None = None,
) -> dict[str, Any]:
    output_root = resolve_project_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = build_operator_action(
        output_root=output_root,
        track_id=track_id,
        action=action,
        notes=notes,
        event_id=event_id,
        mission_id=mission_id,
        scenario_name=scenario_name,
        threat_score=threat_score,
        threat_level=threat_level,
        alert_id=alert_id,
        preview_frame_id=preview_frame_id,
        preview_image_path=preview_image_path,
        selected_object_class=selected_object_class,
        selected_confidence=selected_confidence,
        selected_threat_level=selected_threat_level,
        selected_threat_score=selected_threat_score,
        selected_geo_lock_status=selected_geo_lock_status,
    )
    log_path = operator_action_log_path(output_root)
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload) + "\n")
    payload["operator_log_path"] = str(log_path)
    return payload


def build_operator_action(
    output_root: str | Path,
    track_id: str,
    action: str,
    notes: str = "",
    event_id: str | None = None,
    mission_id: str | None = None,
    scenario_name: str | None = None,
    threat_score: float | None = None,
    threat_level: str | None = None,
    alert_id: str | None = None,
    preview_frame_id: int | None = None,
    preview_image_path: str | None = None,
    selected_object_class: str | None = None,
    selected_confidence: float | None = None,
    selected_threat_level: str | None = None,
    selected_threat_score: float | None = None,
    selected_geo_lock_status: str | None = None,
) -> dict[str, Any]:
    normalized_action = str(action)
    output_root = resolve_project_path(output_root)
    log_path = operator_action_log_path(output_root)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_id": event_id,
        "output_root": str(output_root),
        "operator_log_path": str(log_path),
        "track_id": track_id,
        "action": normalized_action,
        "action_label": ACTION_LABELS.get(normalized_action, normalized_action.replace("_", " ").title()),
        "notes": notes,
        "operator_mode": "human_review",
        "mission_id": mission_id,
        "scenario_name": scenario_name,
        "threat_score": threat_score,
        "threat_level": threat_level,
        "alert_id": alert_id,
        "preview_frame_id": preview_frame_id,
        "preview_image_path": preview_image_path,
        "selected_object_class": selected_object_class,
        "selected_confidence": selected_confidence,
        "selected_threat_level": selected_threat_level,
        "selected_threat_score": selected_threat_score,
        "selected_geo_lock_status": selected_geo_lock_status,
    }


def load_mission_context(path: str | Path = "configs/mission.yaml") -> dict[str, Any]:
    mission_path = resolve_project_path(path)
    result: dict[str, Any] = {
        "path": str(mission_path),
        "loaded": False,
        "error": None,
        "raw": {},
        "display": {key: "Not specified" for key in MISSION_KEYS},
    }
    if not mission_path.exists():
        result["error"] = "Mission context unavailable. Configure configs/mission.yaml."
        return result
    try:
        raw = yaml.safe_load(mission_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raw = {}
        display = {key: raw.get(key, "Not specified") for key in MISSION_KEYS}
        result.update({"loaded": True, "raw": raw, "display": display})
        return result
    except Exception as exc:
        result["error"] = f"Mission context failed to load: {exc}"
        return result


def scenario_name_from_output_root(output_root: str | Path) -> str:
    path = resolve_project_path(output_root)
    if path.name == "outputs":
        return "default_output"
    return path.name or "unknown"


def output_root_health(output_root: str | Path, events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    output_root = resolve_project_path(output_root)
    events = events or []
    gemma_present = any(extract_gemma_decision(event) for event in events)
    fallback_present = any(
        bool(event.get("analyst_model_fallback_used") or event.get("fallback_used")) for event in events
    )
    return {
        "output_root": str(output_root),
        "scenario_name": scenario_name_from_output_root(output_root),
        "events_exists": (output_root / "events.jsonl").exists(),
        "summary_exists": (output_root / "summary.json").exists(),
        "tracks_exists": (output_root / "tracks.json").exists(),
        "video_exists": (output_root / "annotated_demo.mp4").exists(),
        "operator_log_exists": (output_root / "operator_actions.jsonl").exists(),
        "gemma_analysis_exists": gemma_present,
        "gemma_fallback_or_unavailable": fallback_present
        or any("llm_unavailable" in extract_risk_flags(event) for event in events),
    }


def extract_latest_track_state(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_track: dict[str, dict[str, Any]] = {}
    for event in events:
        track_id = str(event.get("track_id") or event.get("event_id") or "unknown")
        prior = latest_by_track.get(track_id)
        if prior is None or _event_sort_key(event) >= _event_sort_key(prior):
            latest_by_track[track_id] = event
    rows = [_track_row(track_id, event) for track_id, event in latest_by_track.items()]
    rows.sort(key=lambda row: (row.get("latest_frame") or -1, row.get("last_seen_timestamp_s") or -1), reverse=True)
    return rows


def extract_confidence_level(event: dict[str, Any]) -> str:
    confidence = event.get("confidence")
    if isinstance(confidence, dict):
        level = confidence.get("level")
        if level:
            return str(level).lower()
    return str(event.get("confidence_level") or event.get("level") or "unknown").lower()


def extract_confidence_value(event: dict[str, Any]) -> float | None:
    confidence = event.get("confidence")
    value = None
    if isinstance(confidence, dict):
        value = confidence.get("final_confidence")
    if value is None:
        value = event.get("final_confidence") or event.get("detection_confidence")
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def enrich_events_with_risk_context(
    events: list[dict[str, Any]],
    mission_context: dict[str, Any],
    threat_library: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not events:
        return []
    mission = mission_context.get("raw") if isinstance(mission_context.get("raw"), dict) else mission_context
    library = threat_library or load_threat_library("configs/threat_library.yaml")
    enriched: list[dict[str, Any]] = []
    for event in events:
        item = dict(event)
        if not isinstance(item.get("threat_assessment"), dict):
            item["threat_assessment"] = score_threat_for_track(item, mission, library).model_dump(mode="json")
        if not isinstance(item.get("geo_lock_readiness"), dict):
            item["geo_lock_readiness"] = evaluate_geo_lock_readiness(item, mission)
        enriched.append(item)
    return enriched


def extract_threat_assessment(event: dict[str, Any]) -> dict[str, Any]:
    threat = event.get("threat_assessment")
    return threat if isinstance(threat, dict) else {}


def extract_geo_lock_readiness(event: dict[str, Any]) -> dict[str, Any]:
    readiness = event.get("geo_lock_readiness")
    return readiness if isinstance(readiness, dict) else {}


def extract_geo(event: dict[str, Any]) -> dict[str, float | None]:
    geo = event.get("geo_estimate") if isinstance(event.get("geo_estimate"), dict) else {}
    lat = geo.get("estimated_lat") if geo else event.get("estimated_lat")
    lon = geo.get("estimated_lon") if geo else event.get("estimated_lon")
    uncertainty = geo.get("geo_uncertainty_m") if geo else event.get("geo_uncertainty_m")
    return {
        "lat": _float_or_none(lat),
        "lon": _float_or_none(lon),
        "geo_uncertainty_m": _float_or_none(uncertainty),
    }


def extract_gemma_decision(event: dict[str, Any]) -> dict[str, Any]:
    decision = event.get("gemma_decision")
    return decision if isinstance(decision, dict) else {}


def extract_policy_result(event: dict[str, Any]) -> dict[str, Any]:
    policy = event.get("policy_result")
    return policy if isinstance(policy, dict) else {}


def extract_final_decision(event: dict[str, Any]) -> dict[str, Any]:
    policy = extract_policy_result(event)
    final_decision = policy.get("final_decision") if isinstance(policy.get("final_decision"), dict) else {}
    if final_decision:
        return final_decision
    gemma = extract_gemma_decision(event)
    return gemma if gemma else {}


def extract_decision_label(event: dict[str, Any]) -> str:
    return str(extract_final_decision(event).get("decision") or "notify_operator")


def extract_policy_corrections(event: dict[str, Any]) -> list[str]:
    policy = extract_policy_result(event)
    corrections = policy.get("corrections") if isinstance(policy, dict) else []
    return [str(item) for item in corrections] if isinstance(corrections, list) else []


def extract_risk_flags(event: dict[str, Any]) -> list[str]:
    final_decision = extract_final_decision(event)
    flags = final_decision.get("risk_flags") or []
    if not flags:
        gemma = extract_gemma_decision(event)
        flags = gemma.get("risk_flags") or []
    return [str(flag) for flag in flags] if isinstance(flags, list) else []


def extract_degraded_flags(event: dict[str, Any]) -> list[str]:
    confidence = event.get("confidence")
    flags = confidence.get("degraded_flags") if isinstance(confidence, dict) else []
    return [str(flag) for flag in flags] if isinstance(flags, list) else []


def filter_events(
    events: list[dict[str, Any]],
    confidence_level: str = "all",
    object_class: str = "all",
    track_id: str = "all",
) -> list[dict[str, Any]]:
    filtered = events
    if confidence_level != "all":
        filtered = [event for event in filtered if extract_confidence_level(event) == confidence_level]
    if object_class != "all":
        filtered = [event for event in filtered if str(event.get("object_class") or "unknown") == object_class]
    if track_id != "all":
        filtered = [event for event in filtered if str(event.get("track_id") or event.get("event_id")) == track_id]
    return filtered


def distinct_object_classes(events: list[dict[str, Any]]) -> list[str]:
    values = sorted({str(event.get("object_class") or "unknown") for event in events})
    return ["all", *values]


def distinct_track_ids(events: list[dict[str, Any]]) -> list[str]:
    values = sorted({str(event.get("track_id") or event.get("event_id") or "unknown") for event in events})
    return ["all", *values]


def latest_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not events:
        return None
    return max(events, key=_event_sort_key)


def find_existing_evidence_frames(output_root: str | Path) -> list[Path]:
    evidence_dir = resolve_project_path(output_root) / "evidence"
    if not evidence_dir.exists():
        return []
    return sorted(
        [path for path in evidence_dir.glob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda path: path.name,
    )


def extract_keyframes_from_video(
    video_path: str | Path,
    output_dir: str | Path,
    fractions: list[float] | None = None,
) -> list[Path]:
    fractions = fractions or [0.25, 0.75]
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if not video_path.exists():
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            cap.release()
            return []
        extracted: list[Path] = []
        for fraction in fractions:
            safe_fraction = min(0.99, max(0.0, float(fraction)))
            frame_index = max(0, min(frame_count - 1, int(frame_count * safe_fraction)))
            output_path = output_dir / f"dashboard_keyframe_{int(safe_fraction * 100):02d}.jpg"
            if output_path.exists():
                extracted.append(output_path)
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if cv2.imwrite(str(output_path), frame):
                extracted.append(output_path)
        cap.release()
        return extracted
    except Exception:
        return []


def select_important_frames(events: list[dict[str, Any]], output_root: str | Path, limit: int = 2) -> list[dict[str, Any]]:
    output_root = resolve_project_path(output_root)
    limit = max(1, int(limit))
    event_candidates = [_frame_card_from_event(event, output_root) for event in events]
    event_candidates = [candidate for candidate in event_candidates if candidate]

    selected: list[dict[str, Any]] = []
    if event_candidates:
        highest = max(event_candidates, key=lambda item: item.get("final_confidence") or -1.0)
        _append_unique_frame(selected, highest)
        degraded = [item for item in event_candidates if item.get("degraded_flags") or item.get("confidence_level") == "low"]
        if degraded:
            _append_unique_frame(selected, min(degraded, key=lambda item: item.get("final_confidence") or 999.0))
        else:
            _append_unique_frame(selected, min(event_candidates, key=lambda item: item.get("final_confidence") or 999.0))
        latest = max(event_candidates, key=lambda item: ((item.get("frame_id") or -1), (item.get("timestamp_s") or -1)))
        _append_unique_frame(selected, latest)
        for candidate in sorted(
            event_candidates,
            key=lambda item: ((item.get("frame_id") or -1), (item.get("final_confidence") or -1.0)),
            reverse=True,
        ):
            _append_unique_frame(selected, candidate)
            if len(selected) >= limit:
                break

    if len(selected) < limit:
        linked_paths = {Path(item["image_path"]).resolve() for item in selected if item.get("image_path")}
        for frame_path in find_existing_evidence_frames(output_root):
            if frame_path.resolve() in linked_paths:
                continue
            _append_unique_frame(selected, _frame_card_from_path(frame_path, source="evidence"))
            if len(selected) >= limit:
                break

    if len(selected) < limit:
        keyframes = extract_keyframes_from_video(
            output_root / "annotated_demo.mp4",
            output_root / "dashboard_frames",
            fractions=[0.25, 0.75],
        )
        for keyframe in keyframes:
            _append_unique_frame(selected, _frame_card_from_path(keyframe, source="video_keyframe"))
            if len(selected) >= limit:
                break

    return selected[:limit]


def _track_row(track_id: str, event: dict[str, Any]) -> dict[str, Any]:
    geo = extract_geo(event)
    final_decision = extract_final_decision(event)
    confidence_level = extract_confidence_level(event)
    confidence_value = extract_confidence_value(event)
    threat = extract_threat_assessment(event)
    geo_lock = extract_geo_lock_readiness(event)
    evidence_path = event.get("evidence_frame_path")
    return {
        "track_id": track_id,
        "object_class": event.get("object_class") or "unknown",
        "latest_frame": event.get("frame_id"),
        "age_frames": event.get("track_age_frames"),
        "latest_confidence": confidence_value,
        "confidence_level": confidence_level,
        "geo_uncertainty_m": geo["geo_uncertainty_m"],
        "latest_lat": geo["lat"],
        "latest_lon": geo["lon"],
        "last_decision": final_decision.get("decision") or "notify_operator",
        "human_review_required": final_decision.get("human_review_required", True),
        "last_seen_timestamp_s": event.get("timestamp_s") or event.get("last_seen_timestamp_s"),
        "evidence_available": bool(evidence_path and Path(str(evidence_path)).exists()),
        "degraded_flags": ", ".join(extract_degraded_flags(event)),
        "threat_score": _float_or_none(threat.get("score")),
        "threat_level": threat.get("level") or "unknown",
        "threat_category": threat.get("category") or "unknown",
        "threat_rationale_short": _short_text(threat.get("rationale"), 110),
        "geo_lock_status": geo_lock.get("status") or "unknown",
    }


def _frame_card_from_event(event: dict[str, Any], output_root: Path) -> dict[str, Any] | None:
    evidence = event.get("evidence_frame_path")
    if not evidence:
        return None
    image_path = _resolve_evidence_path(evidence, output_root)
    if not image_path or not image_path.exists():
        return None
    geo = extract_geo(event)
    confidence = extract_confidence_value(event)
    return {
        "image_path": str(image_path),
        "source": "evidence",
        "event": event,
        "frame_id": event.get("frame_id"),
        "timestamp_s": event.get("timestamp_s"),
        "track_id": event.get("track_id") or event.get("event_id") or "unknown",
        "object_class": event.get("object_class") or "unknown",
        "confidence_level": extract_confidence_level(event),
        "final_confidence": confidence,
        "geo_uncertainty_m": geo["geo_uncertainty_m"],
        "decision": extract_decision_label(event),
        "degraded_flags": extract_degraded_flags(event),
    }


def _frame_card_from_path(image_path: Path, source: str) -> dict[str, Any]:
    return {
        "image_path": str(image_path),
        "source": source,
        "event": None,
        "frame_id": "Not available",
        "track_id": "Not available",
        "object_class": "Not available",
        "confidence_level": "unknown",
        "final_confidence": None,
        "geo_uncertainty_m": None,
        "decision": "Not available",
        "degraded_flags": [],
    }


def _resolve_evidence_path(value: Any, output_root: Path) -> Path | None:
    try:
        path = Path(str(value))
    except TypeError:
        return None
    if path.is_absolute():
        return path
    candidates = [output_root / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _append_unique_frame(selected: list[dict[str, Any]], candidate: dict[str, Any] | None) -> None:
    if not candidate:
        return
    candidate_path = Path(str(candidate.get("image_path", ""))).resolve()
    if any(Path(str(item.get("image_path", ""))).resolve() == candidate_path for item in selected):
        return
    selected.append(candidate)


def _event_sort_key(event: dict[str, Any]) -> tuple[float, float]:
    return (_float_or_none(event.get("frame_id")) or -1.0, _float_or_none(event.get("timestamp_s")) or -1.0)


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _short_text(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def operator_action_log_path(output_root: str | Path) -> Path:
    return resolve_project_path(output_root) / "operator_actions.jsonl"

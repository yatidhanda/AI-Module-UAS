from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from garuda_edge.dashboard.components import (
    render_alert_strip,
    render_badges,
    render_downloads,
    render_header,
    render_human_review_pill,
    render_important_frame_cards,
    render_map,
    render_mission_panel,
    render_ownship_telemetry_panel,
    render_status_panel,
    render_track_table,
    render_threat_alerts,
)
from garuda_edge.dashboard.data_loader import (
    ACTION_LABELS,
    append_operator_action,
    distinct_object_classes,
    distinct_track_ids,
    enrich_events_with_risk_context,
    discover_output_roots,
    extract_confidence_value,
    extract_decision_label,
    extract_degraded_flags,
    extract_final_decision,
    extract_geo_lock_readiness,
    extract_geo,
    extract_gemma_decision,
    extract_latest_track_state,
    extract_policy_corrections,
    extract_policy_result,
    extract_threat_assessment,
    filter_events,
    latest_event,
    load_events,
    load_mission_context,
    load_operator_actions,
    load_summary,
    load_tracks,
    operator_action_log_path,
    output_root_health,
    resolve_project_path,
    scenario_name_from_output_root,
    select_important_frames,
)
from garuda_edge.dashboard.ownship import (
    compute_distance_bearing_between_ownship_and_tracks,
    extract_latest_ownship_state,
    extract_ownship_trail,
)
from garuda_edge.dashboard.metrics import (
    build_alerts,
    compute_dashboard_metrics,
    confidence_distribution,
    link_sensor_summary,
    object_class_counts,
    readiness_status,
    track_timeline,
)
from garuda_edge.dashboard.styles import CONSOLE_CSS
from garuda_edge.dashboard.track_preview import get_selected_track_preview, render_track_highlight_image
from garuda_edge.mission.mission_status import compute_mission_status
from garuda_edge.threat.alerting import build_threat_alerts
from garuda_edge.threat.threat_library import load_threat_library


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("data/outputs")


@st.cache_data(show_spinner=False)
def cached_events(output_root: str, limit: int) -> list[dict[str, Any]]:
    return load_events(output_root, limit=limit)


@st.cache_data(show_spinner=False)
def cached_summary(output_root: str) -> dict[str, Any]:
    return load_summary(output_root)


@st.cache_data(show_spinner=False)
def cached_tracks(output_root: str) -> list[dict[str, Any]]:
    return load_tracks(output_root)


@st.cache_data(show_spinner=False)
def cached_operator_actions(output_root: str) -> list[dict[str, Any]]:
    return load_operator_actions(output_root)


@st.cache_data(show_spinner=False)
def cached_mission_context(path: str) -> dict[str, Any]:
    return load_mission_context(path)


@st.cache_data(show_spinner=False)
def cached_threat_library(path: str = "configs/threat_library.yaml") -> dict[str, Any]:
    return load_threat_library(path)


def main() -> None:
    st.set_page_config(page_title="GarudaEdge ISR Fusion Module", page_icon="🛰️", layout="wide")
    st.markdown(CONSOLE_CSS, unsafe_allow_html=True)

    selected_output, mission_path, controls = render_sidebar()
    if controls["refresh"]:
        st.cache_data.clear()
        st.rerun()

    summary = cached_summary(str(selected_output))
    persisted_tracks = cached_tracks(str(selected_output))
    operator_actions = cached_operator_actions(str(selected_output))
    mission_context = cached_mission_context(mission_path)
    threat_library = cached_threat_library()
    events = enrich_events_with_risk_context(
        cached_events(str(selected_output), controls["max_events"]),
        mission_context,
        threat_library,
    )
    ownship_state = extract_latest_ownship_state(events)
    ownship_trail = extract_ownship_trail(events, max_points=controls["ownship_trail_max_points"])
    health = output_root_health(selected_output, events)
    scenario_name = scenario_name_from_output_root(selected_output)

    filtered_events = filter_events(
        events,
        confidence_level=controls["confidence_filter"],
        object_class=controls["object_class_filter"],
        track_id=controls["track_filter"],
    )
    active_tracks = compute_distance_bearing_between_ownship_and_tracks(
        ownship_state,
        extract_latest_track_state(filtered_events),
    )
    metrics = compute_dashboard_metrics(filtered_events, active_tracks, operator_actions)
    latest = latest_event(filtered_events)
    alert_cfg = threat_library.get("alerting") if isinstance(threat_library.get("alerting"), dict) else {}
    threat_alerts = build_threat_alerts(
        filtered_events,
        threshold=int(alert_cfg.get("high_alert_threshold", 60)),
        max_alerts=int(alert_cfg.get("max_alerts_visible", 8)),
    )
    mission_status = compute_mission_status(filtered_events, persisted_tracks, summary, mission_context, selected_output)

    render_header(selected_output, scenario_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    render_mission_panel(mission_context)
    render_ownship_telemetry_panel(ownship_state)
    render_kpis(metrics)
    render_alert_strip(build_alerts(filtered_events, active_tracks))
    render_threat_alerts(threat_alerts)

    if not health["events_exists"]:
        render_no_events(selected_output, scenario_name)
        return

    st.markdown('<div class="panel-title">Common Operating Picture</div>', unsafe_allow_html=True)
    render_map(
        active_tracks,
        filtered_events,
        mission_context,
        ownship_state=ownship_state,
        ownship_trail=ownship_trail,
        selected_track_id=controls["track_filter"],
        show_all_trails=controls["show_all_trails"],
        show_drone_trail=controls["show_drone_trail"],
        show_selected_track_range_line=controls["show_selected_track_range_line"],
        map_center_mode=controls["map_center_mode"],
    )

    st.markdown('<div class="panel-title">Active Tracks</div>', unsafe_allow_html=True)
    render_track_table(active_tracks)
    render_operator_action_console(
        output_root=selected_output,
        scenario_name=scenario_name,
        events=filtered_events,
        active_tracks=active_tracks,
        latest=latest,
        operator_actions=operator_actions,
        mission_context=mission_context,
        ownship_state=ownship_state,
        key_prefix="top",
    )

    render_tabs(
        output_root=selected_output,
        scenario_name=scenario_name,
        events=filtered_events,
        all_events=events,
        active_tracks=active_tracks,
        persisted_tracks=persisted_tracks,
        summary=summary,
        latest=latest,
        mission_context=mission_context,
        operator_actions=operator_actions,
        threat_alerts=threat_alerts,
        mission_status=mission_status,
        health=health,
        ownship_state=ownship_state,
        show_raw_json=controls["show_raw_json"],
        selected_track=controls["track_filter"],
    )


def render_sidebar() -> tuple[Path, str, dict[str, Any]]:
    st.sidebar.markdown("### Output Root Selector")
    discovered = discover_output_roots(DEFAULT_OUTPUT_ROOT)
    labels = [f"Default output | {DEFAULT_OUTPUT_ROOT}"]
    scenario_roots = [path for path in discovered if path != DEFAULT_OUTPUT_ROOT]
    labels.extend([f"VisDrone scenario | {path.name} | {path}" for path in scenario_roots])
    label_to_path = {labels[0]: DEFAULT_OUTPUT_ROOT}
    for label, path in zip(labels[1:], scenario_roots):
        label_to_path[label] = path
    selected_label = st.sidebar.selectbox("Output root", labels, index=0)
    manual_path = st.sidebar.text_input("Manual output root", value="")
    selected_output = resolve_project_path(Path(manual_path).expanduser() if manual_path.strip() else label_to_path[selected_label])

    st.sidebar.markdown("### Mission Context")
    mission_path = st.sidebar.text_input("Mission YAML path", value="configs/mission.yaml")
    mission_context = cached_mission_context(mission_path)
    st.sidebar.caption("Mission context loaded" if mission_context.get("loaded") else "Mission context unavailable")
    display = mission_context.get("display") or {}
    for key in ["mission_id", "mission_name", "mission_phase", "sector_id", "area_of_interest", "task_type", "ruleset"]:
        st.sidebar.write(f"`{key}`: {display.get(key, 'Not specified')}")
    with st.sidebar.expander("Raw mission YAML"):
        st.json(mission_context.get("raw") or {})

    st.sidebar.markdown("### Scenario Info")
    scenario_name = scenario_name_from_output_root(selected_output)
    preview_events = cached_events(str(selected_output), 200)
    health = output_root_health(selected_output, preview_events)
    st.sidebar.write(f"`selected_output_root`: {selected_output}")
    st.sidebar.write(f"`scenario`: {scenario_name}")
    st.sidebar.write(f"`events.jsonl`: {health['events_exists']}")
    st.sidebar.write(f"`annotated_demo.mp4`: {health['video_exists']}")
    st.sidebar.write(f"`gemma_analysis`: {health['gemma_analysis_exists']}")

    st.sidebar.markdown("### Controls")
    refresh = st.sidebar.button("Refresh Data", use_container_width=True)
    max_events = st.sidebar.slider("Max events to load", min_value=100, max_value=10000, value=1500, step=100)
    control_events = cached_events(str(selected_output), max_events)
    confidence_filter = st.sidebar.selectbox("Confidence filter", ["all", "high", "medium", "low", "unknown"])
    object_class_filter = st.sidebar.selectbox("Object class", distinct_object_classes(control_events))
    track_filter = _searchable_track_selectbox(
        st.sidebar,
        "Track ID",
        distinct_track_ids(control_events),
        key_prefix="sidebar_track",
        placeholder="Type or paste track ID",
    )
    st.sidebar.markdown("### Ownship / Map")
    show_drone_trail = st.sidebar.toggle("Show drone trail", value=True)
    show_selected_track_range_line = st.sidebar.toggle("Show selected track range line", value=True)
    ownship_trail_max_points = st.sidebar.slider(
        "Ownship trail max points",
        min_value=20,
        max_value=1000,
        value=200,
        step=20,
    )
    map_center_mode = st.sidebar.selectbox(
        "Center map on",
        ["Drone / Ownship", "Active Tracks", "Selected Track", "Auto"],
        index=3,
    )
    show_raw_json = st.sidebar.toggle("Show raw JSON", value=False)
    show_all_trails = st.sidebar.toggle("Show object track trail points", value=False)
    auto_refresh = st.sidebar.toggle("Auto-refresh", value=False)
    if auto_refresh:
        st.sidebar.caption("Auto-refresh is enabled for operator review; use Refresh Data to force an immediate reload.")

    return selected_output, mission_path, {
        "refresh": refresh,
        "max_events": max_events,
        "confidence_filter": confidence_filter,
        "object_class_filter": object_class_filter,
        "track_filter": track_filter,
        "show_raw_json": show_raw_json,
        "show_all_trails": show_all_trails,
        "show_drone_trail": show_drone_trail,
        "show_selected_track_range_line": show_selected_track_range_line,
        "ownship_trail_max_points": ownship_trail_max_points,
        "map_center_mode": map_center_mode,
        "auto_refresh": auto_refresh,
    }


def render_kpis(metrics: dict[str, Any]) -> None:
    cols = st.columns(6)
    cols[0].metric("Total Events", metrics["total_events"])
    cols[1].metric("Active Tracks", metrics["active_tracks"])
    cols[2].metric("High Confidence", metrics["high_confidence_tracks"])
    cols[3].metric("Medium Confidence", metrics["medium_confidence_tracks"])
    cols[4].metric("Low Confidence", metrics["low_confidence_tracks"])
    cols[5].metric("Latest Frame", metrics["latest_frame"])

    cols = st.columns(6)
    cols[0].metric("Detection / Track Updates", metrics["detection_track_updates"])
    cols[1].metric("High Threat Tracks", metrics["high_threat_tracks"])
    cols[2].metric("Critical Threat Tracks", metrics["critical_threat_tracks"])
    cols[3].metric("Active Alerts", metrics["active_alerts"])
    cols[4].metric("Gemma Analyzed Events", metrics["gemma_analyzed_events"])
    cols[5].metric("Policy Corrections", metrics["policy_corrections"])

    cols = st.columns(2)
    cols[0].metric("Operator Actions", metrics["operator_actions"])
    cols[1].metric("Latest Mission Phase", metrics["latest_mission_phase"])


def render_no_events(selected_output: Path, scenario_name: str) -> None:
    st.warning("No events found. Run the pipeline for this output root first.")
    scenario_video = Path("data/input_videos/visdrone") / f"{scenario_name}.mp4"
    telemetry = Path("data/telemetry/visdrone") / f"{scenario_name}.csv"
    if scenario_video.exists():
        st.code(
            "venv/bin/python scripts/run_pipeline.py "
            f"--video {scenario_video} --telemetry {telemetry} --output {selected_output} --max-frames 300 --use-gemma false",
            language="bash",
        )
    else:
        st.code(
            "venv/bin/python scripts/run_pipeline.py --video data/input_videos/sampleDronefootage.mov "
            "--telemetry data/telemetry/demo_telemetry.csv --output data/outputs --max-frames 300 --use-gemma true",
            language="bash",
        )


def render_tabs(
    output_root: Path,
    scenario_name: str,
    events: list[dict[str, Any]],
    all_events: list[dict[str, Any]],
    active_tracks: list[dict[str, Any]],
    persisted_tracks: list[dict[str, Any]],
    summary: dict[str, Any],
    latest: dict[str, Any] | None,
    mission_context: dict[str, Any],
    operator_actions: list[dict[str, Any]],
    threat_alerts: list[Any],
    mission_status: dict[str, Any],
    health: dict[str, Any],
    ownship_state: dict[str, Any] | None,
    show_raw_json: bool,
    selected_track: str,
) -> None:
    tabs = st.tabs(["Mission Overview", "Tracks", "Evidence", "Analyst & Policy", "Events / JSON", "Operator Log"])
    with tabs[0]:
        render_mission_overview(
            output_root,
            events,
            summary,
            latest,
            mission_context,
            health,
            operator_actions,
            threat_alerts,
            mission_status,
            ownship_state,
        )
    with tabs[1]:
        render_tracks_tab(events, active_tracks, persisted_tracks, selected_track)
    with tabs[2]:
        render_evidence_tab(output_root, latest)
    with tabs[3]:
        render_analyst_policy_tab(events, latest, mission_context)
    with tabs[4]:
        render_events_json_tab(output_root, events, latest, show_raw_json)
    with tabs[5]:
        render_operator_log_tab(
            output_root,
            scenario_name,
            events,
            active_tracks,
            latest,
            operator_actions,
            mission_context,
            ownship_state,
        )


def render_mission_overview(
    output_root: Path,
    events: list[dict[str, Any]],
    summary: dict[str, Any],
    latest: dict[str, Any] | None,
    mission_context: dict[str, Any],
    health: dict[str, Any],
    operator_actions: list[dict[str, Any]],
    threat_alerts: list[Any],
    mission_status: dict[str, Any],
    ownship_state: dict[str, Any] | None,
) -> None:
    st.markdown('<div class="section-callout">Human review required for all operational decisions.</div>', unsafe_allow_html=True)
    render_mission_health_panels(output_root, mission_status, latest, mission_context, ownship_state)
    render_threat_alerts(threat_alerts)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Scenario Summary")
        st.json(summary or {"status": "summary.json not available"})
        st.subheader("Detection Counts By Object Class")
        st.dataframe(_dict_table(object_class_counts(events), "object_class", "count"), use_container_width=True, hide_index=True)
        st.subheader("Confidence Distribution")
        st.dataframe(_dict_table(confidence_distribution(events), "confidence_level", "count"), use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Readiness / System Status")
        render_status_panel(readiness_status(health, events, operator_actions))
        st.subheader("Link / Sensor Health")
        st.json(link_sensor_summary(events) or {"status": "Not available"})
        st.subheader("Latest Analyst Summary")
        st.write(_latest_summary_text(latest))
        st.subheader("Latest Policy Corrections")
        st.write(extract_policy_corrections(latest or {}) or ["No policy corrections in latest event."])
        st.subheader("Recent Degraded Flags")
        flags = sorted({flag for event in events[-200:] for flag in extract_degraded_flags(event)})
        st.write(flags or ["No degraded flags in recent loaded events."])

    st.subheader("Important Processed Frames")
    important_frames = select_important_frames(events, output_root, limit=2)
    render_important_frame_cards(important_frames)
    with st.expander("Mission Context Raw YAML"):
        st.json(mission_context.get("raw") or {})


def render_mission_health_panels(
    output_root: Path,
    mission_status: dict[str, Any],
    latest: dict[str, Any] | None,
    mission_context: dict[str, Any],
    ownship_state: dict[str, Any] | None,
) -> None:
    completion = mission_status.get("mission_completion_estimate") if isinstance(mission_status, dict) else {}
    health = mission_status.get("mission_health") if isinstance(mission_status, dict) else {}
    raw_mission = mission_context.get("raw") if isinstance(mission_context.get("raw"), dict) else {}
    geo_objectives = raw_mission.get("geo_lock_objectives") if isinstance(raw_mission.get("geo_lock_objectives"), dict) else {}
    final_decision = extract_final_decision(latest or {})
    next_step = final_decision.get("next_step_recommendation")
    if not isinstance(next_step, dict):
        next_step = _deterministic_next_step(mission_status)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.subheader("Mission Health")
        st.metric("Health Status", health.get("status", "unknown"))
        st.metric("Mission Completion", f"{completion.get('percent_complete', 0)}%")
        st.write(
            {
                "scenario": mission_status.get("scenario_name", "unknown"),
                "total_events": mission_status.get("total_events", 0),
                "active_tracks": mission_status.get("active_tracks", 0),
                "high_threat_tracks": mission_status.get("high_threat_tracks", 0),
                "critical_threat_tracks": mission_status.get("critical_threat_tracks", 0),
                "analyst_health": health.get("analyst_health", "unknown"),
            }
        )
    with c2:
        st.subheader("Next Step")
        primary_action = str(next_step.get("primary_action") or "continue_observation")
        st.write(f"Primary: `{primary_action}`")
        st.caption(str(next_step.get("rationale") or "Deterministic recommendation from current mission status."))
        render_badges(next_step.get("supporting_actions", []), level="blue")
        allowed_actions = raw_mission.get("allowed_operator_actions") if isinstance(raw_mission.get("allowed_operator_actions"), list) else []
        if primary_action in allowed_actions:
            if st.button("Log Recommended Action", key="overview_log_recommended_action", use_container_width=True):
                latest_event_payload = latest or {}
                threat = extract_threat_assessment(latest_event_payload)
                payload = append_operator_action(
                    output_root,
                    str(latest_event_payload.get("track_id") or "mission"),
                    primary_action,
                    notes="Logged from Mission Overview next-step panel.",
                    event_id=str(latest_event_payload.get("event_id") or ""),
                    mission_id=raw_mission.get("mission_id"),
                    scenario_name=scenario_name_from_output_root(output_root),
                    threat_score=threat.get("score"),
                    threat_level=threat.get("level"),
                    ownship_lat=(ownship_state or {}).get("lat"),
                    ownship_lon=(ownship_state or {}).get("lon"),
                    ownship_altitude_m=(ownship_state or {}).get("altitude_m"),
                    ownship_heading_deg=(ownship_state or {}).get("heading_deg"),
                )
                st.success(f"Operator action logged to {payload['operator_log_path']}")
                st.cache_data.clear()
                st.rerun()
    with c3:
        st.subheader("Geo-lock Readiness")
        ready_tracks = mission_status.get("geo_lock_ready_tracks") or []
        render_badges(ready_tracks, level="green")
        st.write(
            {
                "target_classes": geo_objectives.get("target_classes", []),
                "min_confidence": geo_objectives.get("min_confidence", "n/a"),
                "max_geo_uncertainty_m": geo_objectives.get("max_geo_uncertainty_m", "n/a"),
                "min_track_age_frames": geo_objectives.get("min_track_age_frames", "n/a"),
            }
        )


def render_tracks_tab(
    events: list[dict[str, Any]],
    active_tracks: list[dict[str, Any]],
    persisted_tracks: list[dict[str, Any]],
    selected_track: str,
) -> None:
    st.subheader("Full Active Track Table")
    render_track_table(active_tracks)
    if persisted_tracks:
        with st.expander("Persisted tracks.json"):
            st.dataframe(pd.DataFrame(persisted_tracks), use_container_width=True, hide_index=True)

    track_id = selected_track if selected_track != "all" else (active_tracks[0]["track_id"] if active_tracks else None)
    if not track_id:
        st.info("Select a track ID to inspect details.")
        return
    track_events = [event for event in events if str(event.get("track_id") or event.get("event_id")) == track_id]
    track_latest = latest_event(track_events)
    st.subheader(f"Selected Track Detail: {track_id}")
    if not track_latest:
        st.info("No event data available for selected track.")
        return

    c1, c2 = st.columns([1, 1])
    with c1:
        confidence = track_latest.get("confidence") if isinstance(track_latest.get("confidence"), dict) else {}
        threat = extract_threat_assessment(track_latest)
        geo_lock = extract_geo_lock_readiness(track_latest)
        st.write(
            {
                "object_class": track_latest.get("object_class"),
                "latest_frame": track_latest.get("frame_id"),
                "track_age_frames": track_latest.get("track_age_frames"),
                "track_stability": track_latest.get("track_stability"),
                "confidence": extract_confidence_value(track_latest),
                "confidence_level": confidence.get("level", "unknown"),
                "geo": extract_geo(track_latest),
            }
        )
        st.write({"confidence_reasons": confidence.get("reasons", []), "degraded_flags": extract_degraded_flags(track_latest)})
        st.subheader("Threat Assessment")
        st.write(
            {
                "score": threat.get("score", "n/a"),
                "level": threat.get("level", "unknown"),
                "category": threat.get("category", "unknown"),
                "rationale": threat.get("rationale", "Threat assessment unavailable."),
            }
        )
        render_badges(threat.get("contributing_factors", []), level="gray")
        st.caption("Mission/platform risk, not hostile intent or engagement authorization.")
        st.subheader("Geo-lock Readiness")
        st.write({"status": geo_lock.get("status", "unknown"), "ready": geo_lock.get("ready", False)})
        render_badges(geo_lock.get("reasons", []), level="gray")
    with c2:
        st.write({"gemma_summary": _latest_summary_text(track_latest)})
        st.write({"policy_corrections": extract_policy_corrections(track_latest)})
        evidence = track_latest.get("evidence_frame_path")
        if evidence and Path(str(evidence)).exists():
            st.image(str(evidence), caption=f"Evidence frame for {track_id}")

    st.subheader("Track Timeline")
    timeline = track_timeline(events, track_id)
    if timeline:
        st.line_chart(pd.DataFrame(timeline).set_index("frame_id")[["confidence"]])
        st.dataframe(pd.DataFrame(timeline), use_container_width=True, hide_index=True)


def render_evidence_tab(output_root: Path, latest: dict[str, Any] | None) -> None:
    video_path = output_root / "annotated_demo.mp4"
    if video_path.exists():
        with st.expander("Open annotated video preview", expanded=False):
            st.video(str(video_path))
    else:
        st.info("No annotated video found in selected output root.")

    evidence_dir = output_root / "evidence"
    frames = sorted([p for p in evidence_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]) if evidence_dir.exists() else []
    st.subheader("Latest Evidence Frame")
    latest_evidence = Path(str(latest.get("evidence_frame_path"))) if latest and latest.get("evidence_frame_path") else None
    if latest_evidence and latest_evidence.exists():
        st.image(str(latest_evidence), caption=latest_evidence.name)
    else:
        st.caption("No evidence frame saved for the latest event.")

    st.subheader("Evidence Gallery")
    if not frames:
        st.info("No evidence frames found in this output root.")
        return
    cols = st.columns(4)
    for idx, frame_path in enumerate(frames[:80]):
        cols[idx % 4].image(str(frame_path), caption=frame_path.name, use_container_width=True)


def render_analyst_policy_tab(
    events: list[dict[str, Any]],
    latest: dict[str, Any] | None,
    mission_context: dict[str, Any],
) -> None:
    latest = latest or {}
    gemma_present = any(extract_gemma_decision(event) for event in events)
    if not gemma_present:
        st.info(
            "This run was likely executed with --use-gemma false. Run a short Gemma pass with --use-gemma true --max-frames 60."
        )

    final_decision = extract_final_decision(latest)
    policy = extract_policy_result(latest)
    threat = extract_threat_assessment(latest)
    geo_lock = extract_geo_lock_readiness(latest)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Latest Analyst Summary")
        st.write(final_decision.get("summary", "No analyst summary available."))
        st.write(
            {
                "analyst_model": latest.get("analyst_model", "Not available"),
                "fallback_used": latest.get("analyst_model_fallback_used", latest.get("fallback_used", "Not available")),
                "analyst_latency_ms": latest.get("analyst_latency_ms", "Not available"),
                "risk_flags": final_decision.get("risk_flags", []),
            }
        )
        st.subheader("Recommended Actions")
        render_badges(final_decision.get("recommended_actions", []), level="blue")
        st.subheader("Next Step Recommendation")
        next_step = final_decision.get("next_step_recommendation")
        if isinstance(next_step, dict):
            st.write(f"Primary: `{next_step.get('primary_action', 'notify_operator')}`")
            st.caption(str(next_step.get("rationale") or "No rationale supplied."))
            render_badges(next_step.get("supporting_actions", []), level="blue")
        else:
            st.caption("No next-step recommendation available in the latest analyst output.")
        st.subheader("Operator Available Actions")
        mission_display = mission_context.get("display") or {}
        allowed_actions = mission_display.get("allowed_operator_actions")
        if isinstance(allowed_actions, list):
            render_badges([ACTION_LABELS.get(action, str(action)) for action in allowed_actions], level="gray")
        else:
            render_badges([], level="gray")
    with c2:
        st.subheader("Policy Gate")
        render_human_review_pill(required=True)
        st.write(
            {
                "decision": final_decision.get("decision", "notify_operator"),
                "human_review_required": final_decision.get("human_review_required", True),
                "policy_corrected": policy.get("corrected", False),
                "allowed": policy.get("allowed", True),
            }
        )
        st.subheader("Policy Corrections")
        corrections_latest = extract_policy_corrections(latest)
        if corrections_latest:
            for correction in corrections_latest:
                st.write(f"- {correction}")
        else:
            st.caption("No policy corrections in latest event.")
        corrections = [correction for event in events for correction in extract_policy_corrections(event)]
        st.write({"all_loaded_policy_corrections": corrections[:200] or []})
        st.subheader("Threat / Risk Passed To Analyst")
        st.write(
            {
                "threat_level": threat.get("level", "unknown"),
                "threat_score": threat.get("score", "n/a"),
                "threat_category": threat.get("category", "unknown"),
                "geo_lock_status": geo_lock.get("status", "unknown"),
            }
        )
        threat_summary = final_decision.get("threat_summary")
        if isinstance(threat_summary, dict):
            st.write(threat_summary)
        mission_health = final_decision.get("mission_health")
        if isinstance(mission_health, dict):
            st.subheader("Gemma Mission Health")
            st.write(mission_health)


def render_events_json_tab(output_root: Path, events: list[dict[str, Any]], latest: dict[str, Any] | None, show_raw_json: bool) -> None:
    st.subheader("Event Table")
    rows = [_event_row(event) for event in events]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No events match current filters.")

    st.subheader("Downloads")
    render_downloads(output_root)

    st.subheader("Latest Event Packet")
    if latest:
        st.json(latest if show_raw_json else _event_row(latest))
    else:
        st.info("No latest event available.")
    if show_raw_json:
        with st.expander("Loaded Events JSON"):
            st.json(events[-50:])


def render_operator_log_tab(
    output_root: Path,
    scenario_name: str,
    events: list[dict[str, Any]],
    active_tracks: list[dict[str, Any]],
    latest: dict[str, Any] | None,
    operator_actions: list[dict[str, Any]],
    mission_context: dict[str, Any],
    ownship_state: dict[str, Any] | None,
) -> None:
    st.subheader("Human Operator Action Panel")
    render_operator_action_console(
        output_root=output_root,
        scenario_name=scenario_name,
        events=events,
        active_tracks=active_tracks,
        latest=latest,
        operator_actions=operator_actions,
        mission_context=mission_context,
        ownship_state=ownship_state,
        key_prefix="tab",
    )

    st.subheader("Operator Action History")
    st.caption(f"Scenario: {scenario_name}")
    if operator_actions:
        st.dataframe(pd.DataFrame(operator_actions), use_container_width=True, hide_index=True)
    else:
        st.info("No operator actions logged for this output root.")


def render_operator_action_console(
    output_root: Path,
    scenario_name: str,
    events: list[dict[str, Any]],
    active_tracks: list[dict[str, Any]],
    latest: dict[str, Any] | None,
    operator_actions: list[dict[str, Any]],
    mission_context: dict[str, Any],
    ownship_state: dict[str, Any] | None,
    key_prefix: str,
) -> None:
    st.markdown('<div class="action-console"><div class="panel-title">Operator Action Console</div>', unsafe_allow_html=True)
    log_path = operator_action_log_path(output_root)
    message_key = f"{key_prefix}_last_operator_action_message"
    if st.session_state.get(message_key):
        st.success(st.session_state.pop(message_key))
    st.caption(f"Log file: `{log_path}`")
    track_options = [track["track_id"] for track in active_tracks] or [str((latest or {}).get("track_id") or "unknown")]
    common_actions = [
        "continue_observation",
        "request_recheck",
        "escalate_to_c2",
        "return_to_base",
        "hold_position",
        "mark_false_positive",
    ]
    label_to_action = {ACTION_LABELS[action]: action for action in common_actions}
    selected_track = _searchable_track_selectbox(
        st,
        "Selected track_id",
        track_options,
        key_prefix=f"{key_prefix}_operator_track",
        placeholder="Type or paste track ID",
    )
    preview = get_selected_track_preview(selected_track, events, output_root)
    preview_event = preview.get("event") if isinstance(preview, dict) and isinstance(preview.get("event"), dict) else {}
    preview_metadata = _preview_metadata(preview_event, preview, selected_track)
    selected_track_row = _active_track_by_id(active_tracks, selected_track)
    left, right = st.columns([1, 1.2])
    with left:
        st.markdown('<div class="track-preview-card"><div class="image-card-title">Selected Track Preview</div>', unsafe_allow_html=True)
        if preview:
            highlighted_path = render_track_highlight_image(
                str(preview["image_path"]),
                preview.get("bbox_xyxy"),
                output_root,
                selected_track,
                preview.get("preview_frame_id"),
                object_class=str(preview_event.get("object_class") or ""),
            )
            preview_metadata["preview_image_path"] = highlighted_path
            st.image(highlighted_path, use_container_width=True)
            if not preview.get("bbox_xyxy"):
                st.warning("Bounding box unavailable for selected track.")
        else:
            st.info("No preview frame available for selected track.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        selected_label = st.radio(
            "Quick action",
            list(label_to_action.keys()),
            horizontal=True,
            key=f"{key_prefix}_action",
        )
        notes = st.text_input("Operator notes", value="", key=f"{key_prefix}_notes")
        st.markdown("Ownship-relative context")
        st.write(
            {
                "range_from_drone_m": _round_or_na(selected_track_row.get("range_from_drone_m"), 1),
                "bearing_from_drone_deg": _round_or_na(selected_track_row.get("bearing_from_drone_deg"), 1),
                "drone_heading_deg": _round_or_na((ownship_state or {}).get("heading_deg"), 1),
                "object_geo_lock_status": selected_track_row.get("geo_lock_status") or preview_metadata.get("geo_lock_status"),
                "threat_level": selected_track_row.get("threat_level") or preview_metadata.get("threat_level"),
            }
        )
    latest_for_track = _latest_action_for_track(operator_actions, selected_track)
    if latest_for_track:
        st.caption(
            f"Latest logged action for {selected_track}: {latest_for_track.get('action_label') or latest_for_track.get('action')} "
            f"at {latest_for_track.get('timestamp_utc', 'unknown time')}"
        )
    else:
        st.caption(f"No logged operator action for {selected_track}.")
    if st.button("Submit / Log Action", use_container_width=True, key=f"{key_prefix}_submit_action"):
        event_id = str(preview_event.get("event_id") or (latest or {}).get("event_id") or "")
        mission_display = mission_context.get("display") or {}
        alert_id = None
        if str(selected_track_row.get("threat_level", "")).lower() in {"high", "critical"}:
            alert_id = f"risk-{selected_track}-{selected_track_row.get('latest_frame', 'na')}"
        payload = append_operator_action(
            output_root,
            selected_track,
            label_to_action[selected_label],
            notes=notes,
            event_id=event_id,
            mission_id=mission_display.get("mission_id"),
            scenario_name=scenario_name,
            threat_score=selected_track_row.get("threat_score"),
            threat_level=selected_track_row.get("threat_level"),
            alert_id=alert_id,
            preview_frame_id=preview_metadata.get("preview_frame_id"),
            preview_image_path=preview_metadata.get("preview_image_path"),
            selected_object_class=preview_metadata.get("object_class"),
            selected_confidence=preview_metadata.get("confidence"),
            selected_threat_level=preview_metadata.get("threat_level"),
            selected_threat_score=preview_metadata.get("threat_score"),
            selected_geo_lock_status=preview_metadata.get("geo_lock_status"),
            ownship_lat=(ownship_state or {}).get("lat"),
            ownship_lon=(ownship_state or {}).get("lon"),
            ownship_altitude_m=(ownship_state or {}).get("altitude_m"),
            ownship_heading_deg=(ownship_state or {}).get("heading_deg"),
            range_from_drone_m=selected_track_row.get("range_from_drone_m"),
            bearing_from_drone_deg=selected_track_row.get("bearing_from_drone_deg"),
        )
        st.session_state[message_key] = f"Operator action logged to {payload['operator_log_path']}"
        st.cache_data.clear()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _latest_action_for_track(operator_actions: list[dict[str, Any]], track_id: str) -> dict[str, Any] | None:
    for action in reversed(operator_actions):
        if str(action.get("track_id")) == str(track_id):
            return action
    return None


def _searchable_track_selectbox(
    container: Any,
    label: str,
    options: list[str],
    key_prefix: str,
    placeholder: str,
) -> str:
    normalized_options = [str(option) for option in options if str(option)]
    if not normalized_options:
        normalized_options = ["unknown"]

    query = container.text_input(
        f"{label} search / paste",
        value="",
        key=f"{key_prefix}_query",
        placeholder=placeholder,
    ).strip()
    filtered_options = _filter_track_options(normalized_options, query)
    if not filtered_options:
        container.caption(f"No track IDs match `{query}`. Clear search to show all tracks.")
        filtered_options = normalized_options

    preferred = _preferred_track_option(normalized_options, filtered_options, query)
    index = filtered_options.index(preferred) if preferred in filtered_options else 0
    return str(container.selectbox(label, filtered_options, index=index, key=f"{key_prefix}_select"))


def _filter_track_options(options: list[str], query: str) -> list[str]:
    if not query:
        return options
    needle = query.strip().lower()
    exact = [option for option in options if option.lower() == needle]
    partial = [option for option in options if needle in option.lower() and option not in exact]
    return [*exact, *partial]


def _preferred_track_option(all_options: list[str], filtered_options: list[str], query: str) -> str:
    if query:
        for option in all_options:
            if option.lower() == query.strip().lower():
                return option
    return filtered_options[0] if filtered_options else all_options[0]


def _preview_metadata(
    event: dict[str, Any],
    preview: dict[str, Any] | None,
    selected_track: str,
) -> dict[str, Any]:
    threat = extract_threat_assessment(event)
    geo_lock = extract_geo_lock_readiness(event)
    return {
        "track_id": selected_track,
        "object_class": event.get("object_class") or "unknown",
        "frame_id": event.get("frame_id"),
        "preview_frame_id": preview.get("preview_frame_id") if isinstance(preview, dict) else None,
        "preview_image_path": preview.get("image_path") if isinstance(preview, dict) else None,
        "confidence": extract_confidence_value(event),
        "threat_level": threat.get("level") or "unknown",
        "threat_score": threat.get("score"),
        "geo_lock_status": geo_lock.get("status") or "unknown",
        "decision": extract_decision_label(event) if event else "notify_operator",
    }


def _active_track_by_id(active_tracks: list[dict[str, Any]], track_id: str) -> dict[str, Any]:
    for track in active_tracks:
        if str(track.get("track_id")) == str(track_id):
            return track
    return {}


def _round_or_na(value: Any, precision: int = 1) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return "N/A"


def _deterministic_next_step(mission_status: dict[str, Any]) -> dict[str, Any]:
    if mission_status.get("critical_threat_tracks", 0):
        return {
            "primary_action": "escalate_to_c2",
            "rationale": "Critical mission/platform risk track is present; keep human review active and notify C2.",
            "supporting_actions": ["notify_operator", "store_evidence", "request_sensor_confirmation"],
        }
    if mission_status.get("high_threat_tracks", 0):
        return {
            "primary_action": "notify_operator",
            "rationale": "High mission/platform risk track is present; preserve evidence and request operator review.",
            "supporting_actions": ["store_evidence", "request_recheck"],
        }
    if mission_status.get("geo_lock_ready_tracks"):
        return {
            "primary_action": "store_evidence",
            "rationale": "One or more tracks meet geo-lock readiness criteria for evidence packaging.",
            "supporting_actions": ["continue_observation"],
        }
    return {
        "primary_action": "continue_observation",
        "rationale": "No high-risk alert is active in the loaded output; continue observation and evidence logging.",
        "supporting_actions": ["store_evidence"],
    }


def _event_row(event: dict[str, Any]) -> dict[str, Any]:
    geo = extract_geo(event)
    final_decision = extract_final_decision(event)
    threat = extract_threat_assessment(event)
    geo_lock = extract_geo_lock_readiness(event)
    return {
        "event_id": event.get("event_id"),
        "track_id": event.get("track_id"),
        "object_class": event.get("object_class"),
        "frame_id": event.get("frame_id"),
        "timestamp_s": event.get("timestamp_s"),
        "confidence": extract_confidence_value(event),
        "confidence_level": (event.get("confidence") or {}).get("level") if isinstance(event.get("confidence"), dict) else None,
        "lat": geo["lat"],
        "lon": geo["lon"],
        "geo_uncertainty_m": geo["geo_uncertainty_m"],
        "threat_score": threat.get("score"),
        "threat_level": threat.get("level"),
        "threat_category": threat.get("category"),
        "geo_lock_status": geo_lock.get("status"),
        "decision": final_decision.get("decision", extract_decision_label(event)),
        "human_review_required": final_decision.get("human_review_required", True),
        "policy_corrections": ", ".join(extract_policy_corrections(event)),
        "degraded_flags": ", ".join(extract_degraded_flags(event)),
        "analyst_model": event.get("analyst_model"),
    }


def _latest_summary_text(event: dict[str, Any] | None) -> str:
    if not event:
        return "No analyst summary available."
    final_decision = extract_final_decision(event)
    return str(final_decision.get("summary") or "No analyst summary available.")


def _dict_table(values: dict[str, Any], key_name: str, value_name: str) -> pd.DataFrame:
    return pd.DataFrame([{key_name: key, value_name: value} for key, value in values.items()])


if __name__ == "__main__":
    main()

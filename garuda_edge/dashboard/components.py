from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from garuda_edge.dashboard.data_loader import extract_decision_label, extract_geo
from garuda_edge.dashboard.styles import alert_badge_class, threat_badge_class


def render_header(selected_output: Path, scenario_name: str, last_updated: str) -> None:
    st.markdown(
        f"""
        <div class="isr-header">
          <div class="isr-title">GarudaEdge ISR Fusion Module</div>
          <div class="isr-subtitle">Human-in-the-loop aerial perception, geo-track confidence and analyst review</div>
          <div class="isr-meta">OUTPUT_ROOT={html.escape(str(selected_output))} &nbsp; SCENARIO={html.escape(scenario_name)} &nbsp; UPDATED={html.escape(last_updated)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mission_panel(mission_context: dict[str, Any]) -> None:
    display = mission_context.get("display") or {}
    if not mission_context.get("loaded"):
        st.warning(mission_context.get("error") or "Mission context unavailable. Configure configs/mission.yaml.")
        return
    notes = display.get("notes")
    note_preview = notes[:2] if isinstance(notes, list) else notes
    fields = [
        ("Mission ID", display.get("mission_id")),
        ("Mission Name", display.get("mission_name")),
        ("Mission Phase", display.get("mission_phase")),
        ("Sector ID", display.get("sector_id")),
        ("Area of Interest", display.get("area_of_interest")),
        ("Task Type", display.get("task_type")),
        ("Ruleset", display.get("ruleset")),
        ("Human Review Required", display.get("human_review_required", True)),
        ("Priority Classes", display.get("priority_classes")),
        ("Notes", note_preview),
    ]
    items = "\n".join(
        f"""
        <div class="mission-item">
          <div class="mission-label">{html.escape(label)}</div>
          <div class="mission-value">{html.escape(_stringify(value))}</div>
        </div>
        """
        for label, value in fields
    )
    st.markdown(
        f"""
        <div class="panel">
          <div class="panel-title">Mission Context</div>
          <div class="mission-grid">{items}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if isinstance(notes, list) and len(notes) > 2:
        with st.expander("All Mission Notes"):
            for note in notes:
                st.write(f"- {note}")


def render_alert_strip(alerts: list[dict[str, str]]) -> None:
    badges = "\n".join(
        f'<span class="badge {alert_badge_class(alert.get("level"))}">{html.escape(alert.get("label", ""))}</span>'
        for alert in alerts
    )
    st.markdown(f'<div class="badge-row">{badges}</div>', unsafe_allow_html=True)


def render_threat_alerts(alerts: list[Any]) -> None:
    st.markdown('<div class="panel-title">Threat/Risk Alerts</div>', unsafe_allow_html=True)
    st.caption("Threat level means mission/platform risk estimate, not hostile intent or engagement authorization.")
    if not alerts:
        st.info("No high-risk tracks in selected output.")
        return
    for alert in alerts:
        payload = alert.model_dump(mode="json") if hasattr(alert, "model_dump") else dict(alert)
        level = str(payload.get("threat_level") or "unknown")
        card_class = "critical" if level == "critical" else ""
        actions = payload.get("recommended_actions") if isinstance(payload.get("recommended_actions"), list) else []
        action_badges = "".join(
            f'<span class="badge badge-blue">{html.escape(str(action))}</span>' for action in actions[:6]
        )
        st.markdown(
            f"""
            <div class="threat-alert-card {card_class}">
              <div class="threat-alert-title">
                {html.escape(str(payload.get("track_id", "unknown")))}
                / {html.escape(str(payload.get("object_class", "unknown")))}
                <span class="badge {threat_badge_class(level)}">{html.escape(level.upper())}</span>
                <span class="badge badge-gray">score {html.escape(str(payload.get("threat_score", "n/a")))}</span>
              </div>
              <div class="threat-alert-meta">
                frame={html.escape(str(payload.get("frame_id", "n/a")))}
                alert_id={html.escape(str(payload.get("alert_id", "n/a")))}
              </div>
              <div class="small-muted">{html.escape(str(payload.get("summary", "")))}</div>
              <div class="badge-row">{action_badges}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_map(
    active_tracks: list[dict[str, Any]],
    events: list[dict[str, Any]],
    mission_context: dict[str, Any],
    show_all_trails: bool = False,
) -> None:
    valid_tracks = [track for track in active_tracks if _valid_lat_lon(track.get("latest_lat"), track.get("latest_lon"))]
    if not valid_tracks:
        st.warning("No valid geo-track coordinates available for the selected output root.")
        return

    try:
        import folium
        from streamlit_folium import st_folium
    except Exception as exc:
        st.warning(f"Map unavailable: {exc}")
        return

    center_track = valid_tracks[0]
    fmap = folium.Map(
        location=[center_track["latest_lat"], center_track["latest_lon"]],
        zoom_start=16,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    if show_all_trails:
        for event in events:
            geo = extract_geo(event)
            if not _valid_lat_lon(geo["lat"], geo["lon"]):
                continue
            level = _confidence_level_from_event(event)
            folium.CircleMarker(
                location=[geo["lat"], geo["lon"]],
                radius=2,
                color=_marker_color(level),
                fill=True,
                fill_opacity=0.18,
                opacity=0.26,
                weight=1,
            ).add_to(fmap)

    mission = mission_context.get("display") or {}
    for track in valid_tracks:
        level = str(track.get("confidence_level") or "unknown")
        threat_level = str(track.get("threat_level") or "unknown")
        popup = _track_popup(track, mission)
        folium.Marker(
            location=[track["latest_lat"], track["latest_lon"]],
            popup=folium.Popup(popup, max_width=360),
            tooltip=f"{track.get('track_id')} {track.get('object_class')} conf={level} risk={threat_level}",
            icon=folium.Icon(color=_folium_icon_color(level, threat_level), icon="info-sign"),
        ).add_to(fmap)

    st_folium(fmap, use_container_width=True, height=470)


def render_status_panel(status_rows: list[dict[str, str]]) -> None:
    badges = []
    for row in status_rows:
        status = row.get("status", "unknown")
        level = "green" if status in {"ready", "present", "simulated"} else "amber" if status in {"absent", "empty"} else "red"
        badges.append(
            f'<span class="badge {alert_badge_class(level)}">{html.escape(row.get("item", ""))}: {html.escape(status)}</span>'
        )
    st.markdown(f'<div class="badge-row">{"".join(badges)}</div>', unsafe_allow_html=True)


def render_track_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.info("No active tracks match the current filters.")
        return
    frame = pd.DataFrame(rows)
    important = [
        "track_id",
        "object_class",
        "latest_frame",
        "age_frames",
        "latest_confidence",
        "confidence_level",
        "threat_score",
        "threat_level",
        "threat_category",
        "geo_uncertainty_m",
        "geo_lock_status",
        "latest_lat",
        "latest_lon",
        "last_decision",
        "human_review_required",
        "last_seen_timestamp_s",
        "evidence_available",
        "degraded_flags",
    ]
    frame = frame[[column for column in important if column in frame.columns]]
    if "latest_confidence" in frame.columns:
        frame["latest_confidence"] = frame["latest_confidence"].map(lambda value: f"{value:.2f}" if pd.notna(value) else "n/a")
    if "threat_score" in frame.columns:
        frame["threat_score"] = frame["threat_score"].map(lambda value: f"{value:.1f}" if pd.notna(value) else "n/a")
    st.dataframe(frame, use_container_width=True, hide_index=True)


def render_downloads(output_root: Path) -> None:
    events_path = output_root / "events.jsonl"
    summary_path = output_root / "summary.json"
    c1, c2 = st.columns(2)
    if events_path.exists():
        c1.download_button(
            "Download events.jsonl",
            events_path.read_bytes(),
            file_name=f"{output_root.name}_events.jsonl",
            mime="application/jsonl",
            use_container_width=True,
        )
    if summary_path.exists():
        c2.download_button(
            "Download summary.json",
            summary_path.read_bytes(),
            file_name=f"{output_root.name}_summary.json",
            mime="application/json",
            use_container_width=True,
        )


def render_important_frame_cards(frames: list[dict[str, Any]]) -> None:
    if not frames:
        st.warning("No evidence frames available. Run pipeline with evidence saving enabled.")
        return
    cols = st.columns(2)
    for idx in range(2):
        with cols[idx]:
            if idx >= len(frames):
                st.info("No second processed frame available for this output root.")
                continue
            frame = frames[idx]
            st.markdown('<div class="image-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="image-card-title">Processed Frame {idx + 1} / {html.escape(str(frame.get("source", "evidence")))}</div>',
                unsafe_allow_html=True,
            )
            st.image(str(frame["image_path"]), use_container_width=True)
            with st.expander(f"Show Processed Frame {idx + 1} event JSON", expanded=False):
                st.json(frame.get("event") or frame)
            st.markdown("</div>", unsafe_allow_html=True)


def render_badges(items: list[Any], level: str = "blue") -> None:
    if not items:
        st.caption("None recorded.")
        return
    css = alert_badge_class(level)
    badges = "".join(f'<span class="badge {css}">{html.escape(str(item))}</span>' for item in items)
    st.markdown(f'<div class="badge-row">{badges}</div>', unsafe_allow_html=True)


def render_human_review_pill(required: bool = True) -> None:
    if required:
        st.markdown('<span class="human-review-pill">HUMAN REVIEW REQUIRED</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-gray">Human review status unavailable</span>', unsafe_allow_html=True)


def _track_popup(track: dict[str, Any], mission: dict[str, Any]) -> str:
    values = {
        "track_id": track.get("track_id"),
        "object_class": track.get("object_class"),
        "final_confidence": track.get("latest_confidence"),
        "confidence_level": track.get("confidence_level"),
        "geo_uncertainty_m": track.get("geo_uncertainty_m"),
        "threat_level": track.get("threat_level"),
        "threat_score": track.get("threat_score"),
        "threat_category": track.get("threat_category"),
        "geo_lock_status": track.get("geo_lock_status"),
        "frame_id": track.get("latest_frame"),
        "latest_decision": track.get("last_decision"),
        "human_review_required": track.get("human_review_required", True),
        "mission_phase": mission.get("mission_phase", "Not specified"),
        "sector_id": mission.get("sector_id", "Not specified"),
    }
    rows = "".join(
        f"<tr><td><b>{html.escape(key)}</b></td><td>{html.escape(_stringify(value))}</td></tr>"
        for key, value in values.items()
    )
    return f"<table>{rows}</table>"


def _confidence_level_from_event(event: dict[str, Any]) -> str:
    confidence = event.get("confidence") if isinstance(event.get("confidence"), dict) else {}
    return str(confidence.get("level") or event.get("confidence_level") or "unknown")


def _marker_color(level: str) -> str:
    level = (level or "unknown").lower()
    if level == "high":
        return "#4ade80"
    if level == "medium":
        return "#f59e0b"
    if level == "low":
        return "#f87171"
    return "#9ca3af"


def _folium_icon_color(level: str, threat_level: str | None = None) -> str:
    risk = (threat_level or "unknown").lower()
    if risk == "critical":
        return "red"
    if risk == "high":
        return "orange"
    level = (level or "unknown").lower()
    if level == "high":
        return "green"
    if level == "medium":
        return "orange"
    if level == "low":
        return "red"
    return "gray"


def _valid_lat_lon(lat: Any, lon: Any) -> bool:
    try:
        return lat is not None and lon is not None and -90 <= float(lat) <= 90 and -180 <= float(lon) <= 180
    except (TypeError, ValueError):
        return False


def _stringify(value: Any) -> str:
    if value is None:
        return "Not specified"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "Not specified"
    if isinstance(value, dict):
        return ", ".join(f"{key}: {val}" for key, val in value.items()) if value else "Not specified"
    return str(value)

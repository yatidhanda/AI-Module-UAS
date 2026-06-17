from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from garuda_edge.dashboard.data_loader import extract_decision_label, extract_geo
from garuda_edge.dashboard.ownship import platform_telemetry_status, project_point, valid_lat_lon
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


def render_ownship_telemetry_panel(ownship_state: dict[str, Any] | None) -> None:
    status = platform_telemetry_status(ownship_state)
    status_badge = alert_badge_class(status.get("level"))
    if not ownship_state or not valid_lat_lon(ownship_state.get("lat"), ownship_state.get("lon")):
        st.markdown(
            f"""
            <div class="panel ownship-panel">
              <div class="ownship-title-row">
                <div class="panel-title">Ownship Telemetry</div>
                <span class="badge {status_badge}">{html.escape(status.get("status", "Telemetry unavailable"))}</span>
              </div>
              <div class="small-muted">No platform telemetry found in selected output. Re-run pipeline with telemetry enabled.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    fields = [
        ("Latitude", _format_lat_lon(ownship_state.get("lat")), "gray"),
        ("Longitude", _format_lat_lon(ownship_state.get("lon")), "gray"),
        ("Altitude", _format_meters(ownship_state.get("altitude_m")), "gray"),
        ("Heading", _format_degrees(ownship_state.get("heading_deg")), "gray"),
        ("Speed", _format_speed(ownship_state.get("speed_mps")), "gray"),
        ("Battery", _format_percent_value(ownship_state.get("battery_pct")), _field_level("battery", ownship_state)),
        ("GPS Quality", _stringify_or_na(ownship_state.get("gps_quality")), _field_level("gps", ownship_state)),
        ("Link Quality", _stringify_or_na(ownship_state.get("link_quality")), _field_level("link", ownship_state)),
        ("Packet Loss", _format_packet_loss(ownship_state.get("packet_loss")), _field_level("packet_loss", ownship_state)),
        ("Latency", _format_latency(ownship_state.get("latency_ms")), _field_level("latency", ownship_state)),
        ("Latest Frame", _stringify_or_na(ownship_state.get("frame_id")), "gray"),
        ("Timestamp", _format_seconds(ownship_state.get("timestamp_s")), "gray"),
    ]
    items = "\n".join(
        f"""
        <div class="ownship-item ownship-item-{html.escape(level)}">
          <div class="ownship-label">{html.escape(label)}</div>
          <div class="ownship-value">{html.escape(value)}</div>
        </div>
        """
        for label, value, level in fields
    )
    reasons = status.get("reasons") or []
    reason_text = f" / {'; '.join(str(item) for item in reasons)}" if reasons else ""
    st.markdown(
        f"""
        <div class="panel ownship-panel">
          <div class="ownship-title-row">
            <div class="panel-title">Ownship Telemetry</div>
            <span class="badge {status_badge}">{html.escape(status.get("status", "Telemetry nominal"))}</span>
          </div>
          <div class="ownship-grid">{items}</div>
          <div class="small-muted">Read-only platform state for operator awareness{html.escape(reason_text)}.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_map(
    active_tracks: list[dict[str, Any]],
    events: list[dict[str, Any]],
    mission_context: dict[str, Any],
    ownship_state: dict[str, Any] | None = None,
    ownship_trail: list[dict[str, Any]] | None = None,
    selected_track_id: str = "all",
    show_all_trails: bool = False,
    show_drone_trail: bool = True,
    show_selected_track_range_line: bool = True,
    map_center_mode: str = "Auto",
) -> None:
    valid_tracks = [track for track in active_tracks if _valid_lat_lon(track.get("latest_lat"), track.get("latest_lon"))]
    has_ownship = bool(ownship_state and valid_lat_lon(ownship_state.get("lat"), ownship_state.get("lon")))
    if not valid_tracks and not has_ownship:
        st.warning("No valid ownship or object track coordinates available for the selected output root.")
        return

    try:
        import folium
        from streamlit_folium import st_folium
    except Exception as exc:
        st.warning(f"Map unavailable: {exc}")
        return

    center = _map_center(ownship_state, valid_tracks, selected_track_id, map_center_mode)
    fmap = folium.Map(
        location=center,
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
    if show_drone_trail and ownship_trail:
        trail_points = [
            [point["lat"], point["lon"]]
            for point in ownship_trail
            if valid_lat_lon(point.get("lat"), point.get("lon"))
        ]
        if len(trail_points) >= 2:
            folium.PolyLine(
                trail_points,
                color="#38bdf8",
                weight=2,
                opacity=0.64,
                tooltip="Ownship trail",
            ).add_to(fmap)

    if has_ownship and ownship_state:
        ownship_location = [ownship_state["lat"], ownship_state["lon"]]
        ownship_popup = _ownship_popup(ownship_state, mission)
        folium.CircleMarker(
            location=ownship_location,
            radius=9,
            color="#67e8f9",
            fill=True,
            fill_color="#0ea5e9",
            fill_opacity=0.82,
            opacity=0.95,
            weight=2,
            popup=folium.Popup(ownship_popup, max_width=400),
            tooltip="DRONE / OWNSHIP",
        ).add_to(fmap)
        folium.Marker(
            location=ownship_location,
            icon=folium.DivIcon(html=_ownship_marker_html()),
            popup=folium.Popup(ownship_popup, max_width=400),
            tooltip="DRONE / OWNSHIP",
        ).add_to(fmap)
        heading = _float_or_none(ownship_state.get("heading_deg"))
        if heading is not None:
            heading_end = project_point(float(ownship_state["lat"]), float(ownship_state["lon"]), heading, 120.0)
            folium.PolyLine(
                [ownship_location, [heading_end[0], heading_end[1]]],
                color="#7dd3fc",
                weight=3,
                opacity=0.78,
                tooltip=f"Ownship heading {heading:.1f} deg",
            ).add_to(fmap)

    selected_track = _track_by_id(valid_tracks, selected_track_id)
    if show_selected_track_range_line and has_ownship and ownship_state and selected_track:
        folium.PolyLine(
            [
                [ownship_state["lat"], ownship_state["lon"]],
                [selected_track["latest_lat"], selected_track["latest_lon"]],
            ],
            color="#bae6fd",
            weight=2,
            opacity=0.54,
            dash_array="6,8",
            tooltip=f"Range line to {selected_track.get('track_id')}",
        ).add_to(fmap)

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

    render_map_legend()
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
        "latest_confidence",
        "confidence_level",
        "threat_score",
        "threat_level",
        "geo_lock_status",
        "range_from_drone_m",
        "bearing_from_drone_deg",
        "geo_uncertainty_m",
        "last_decision",
        "human_review_required",
    ]
    frame = frame[[column for column in important if column in frame.columns]]
    if "latest_confidence" in frame.columns:
        frame["latest_confidence"] = frame["latest_confidence"].map(lambda value: _format_table_number(value, 2))
    if "threat_score" in frame.columns:
        frame["threat_score"] = frame["threat_score"].map(lambda value: _format_table_number(value, 1))
    if "range_from_drone_m" in frame.columns:
        frame["range_from_drone_m"] = frame["range_from_drone_m"].map(lambda value: _format_table_number(value, 1))
    if "bearing_from_drone_deg" in frame.columns:
        frame["bearing_from_drone_deg"] = frame["bearing_from_drone_deg"].map(lambda value: _format_table_number(value, 1))
    if "geo_uncertainty_m" in frame.columns:
        frame["geo_uncertainty_m"] = frame["geo_uncertainty_m"].map(lambda value: _format_table_number(value, 1))
    st.dataframe(frame, use_container_width=True, hide_index=True)


def render_map_legend() -> None:
    st.markdown(
        """
        <div class="map-legend">
          <span><i class="legend-swatch legend-ownship"></i>Ownship / drone</span>
          <span><i class="legend-swatch legend-object"></i>Object track</span>
          <span><i class="legend-swatch legend-risk"></i>High / critical risk</span>
          <span><i class="legend-line legend-trail"></i>Drone trail</span>
          <span><i class="legend-line legend-range"></i>Selected range line</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        "range_from_drone_m": _format_meters(track.get("range_from_drone_m")),
        "bearing_from_drone_deg": _format_degrees(track.get("bearing_from_drone_deg")),
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


def _ownship_popup(ownship_state: dict[str, Any], mission: dict[str, Any]) -> str:
    values = {
        "label": "DRONE / OWNSHIP",
        "mission_id": mission.get("mission_id", "Not specified"),
        "frame_id": ownship_state.get("frame_id"),
        "timestamp_s": _format_seconds(ownship_state.get("timestamp_s")),
        "latitude": _format_lat_lon(ownship_state.get("lat")),
        "longitude": _format_lat_lon(ownship_state.get("lon")),
        "altitude": _format_meters(ownship_state.get("altitude_m")),
        "heading": _format_degrees(ownship_state.get("heading_deg")),
        "speed": _format_speed(ownship_state.get("speed_mps")),
        "battery": _format_percent_value(ownship_state.get("battery_pct")),
        "gps_quality": ownship_state.get("gps_quality"),
        "link_quality": ownship_state.get("link_quality"),
        "packet_loss": _format_packet_loss(ownship_state.get("packet_loss")),
        "latency": _format_latency(ownship_state.get("latency_ms")),
        "sensor_health": ownship_state.get("sensor_health"),
    }
    rows = "".join(
        f"<tr><td><b>{html.escape(key)}</b></td><td>{html.escape(_stringify_or_na(value))}</td></tr>"
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


def _map_center(
    ownship_state: dict[str, Any] | None,
    valid_tracks: list[dict[str, Any]],
    selected_track_id: str,
    mode: str,
) -> list[float]:
    mode_normalized = (mode or "Auto").strip().lower()
    selected_track = _track_by_id(valid_tracks, selected_track_id)
    has_ownship = bool(ownship_state and valid_lat_lon(ownship_state.get("lat"), ownship_state.get("lon")))

    if mode_normalized == "selected track" and selected_track:
        return [float(selected_track["latest_lat"]), float(selected_track["latest_lon"])]
    if mode_normalized == "drone / ownship" and has_ownship and ownship_state:
        return [float(ownship_state["lat"]), float(ownship_state["lon"])]
    if mode_normalized == "active tracks" and valid_tracks:
        return _average_track_center(valid_tracks)
    if has_ownship and ownship_state:
        return [float(ownship_state["lat"]), float(ownship_state["lon"])]
    if selected_track:
        return [float(selected_track["latest_lat"]), float(selected_track["latest_lon"])]
    return _average_track_center(valid_tracks)


def _average_track_center(valid_tracks: list[dict[str, Any]]) -> list[float]:
    if not valid_tracks:
        return [0.0, 0.0]
    return [
        sum(float(track["latest_lat"]) for track in valid_tracks) / len(valid_tracks),
        sum(float(track["latest_lon"]) for track in valid_tracks) / len(valid_tracks),
    ]


def _track_by_id(valid_tracks: list[dict[str, Any]], track_id: str | None) -> dict[str, Any] | None:
    if not track_id or str(track_id).lower() == "all":
        return None
    for track in valid_tracks:
        if str(track.get("track_id")) == str(track_id):
            return track
    return None


def _ownship_marker_html() -> str:
    return """
    <div style="transform: translate(-50%, -50%);">
      <div style="width: 78px; text-align: center; color: #e0f2fe; font-size: 10px; font-weight: 700; letter-spacing: 0.08em; text-shadow: 0 1px 3px #020617;">DRONE</div>
    </div>
    """


def _valid_lat_lon(lat: Any, lon: Any) -> bool:
    try:
        return lat is not None and lon is not None and -90 <= float(lat) <= 90 and -180 <= float(lon) <= 180
    except (TypeError, ValueError):
        return False


def _field_level(field: str, ownship_state: dict[str, Any]) -> str:
    if field == "battery":
        battery = _float_or_none(ownship_state.get("battery_pct"))
        if battery is not None and battery < 15:
            return "red"
        if battery is not None and battery < 25:
            return "amber"
    if field == "packet_loss":
        packet_loss = _float_or_none(ownship_state.get("packet_loss"))
        if packet_loss is not None and packet_loss > 0.25:
            return "red"
        if packet_loss is not None and packet_loss > 0.10:
            return "amber"
    if field == "latency":
        latency = _float_or_none(ownship_state.get("latency_ms"))
        if latency is not None and latency > 800:
            return "red"
        if latency is not None and latency > 300:
            return "amber"
    if field == "gps":
        gps = str(ownship_state.get("gps_quality") or "").lower()
        if gps == "lost":
            return "red"
        if gps in {"poor", "degraded"}:
            return "amber"
    if field == "link":
        link = str(ownship_state.get("link_quality") or "").lower()
        if link in {"lost", "down"}:
            return "red"
        if link in {"poor", "degraded"}:
            return "amber"
    return "green" if field in {"battery", "gps", "link", "packet_loss", "latency"} else "gray"


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_table_number(value: Any, precision: int) -> str:
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return "N/A"


def _format_lat_lon(value: Any) -> str:
    number = _float_or_none(value)
    return f"{number:.6f}" if number is not None else "N/A"


def _format_meters(value: Any) -> str:
    number = _float_or_none(value)
    return f"{number:.1f} m" if number is not None else "N/A"


def _format_degrees(value: Any) -> str:
    number = _float_or_none(value)
    return f"{number:.1f} deg" if number is not None else "N/A"


def _format_speed(value: Any) -> str:
    number = _float_or_none(value)
    return f"{number:.1f} m/s" if number is not None else "N/A"


def _format_percent_value(value: Any) -> str:
    number = _float_or_none(value)
    return f"{number:.0f}%" if number is not None else "N/A"


def _format_packet_loss(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return "N/A"
    percent = number * 100 if number <= 1 else number
    return f"{percent:.1f}%"


def _format_latency(value: Any) -> str:
    number = _float_or_none(value)
    return f"{number:.0f} ms" if number is not None else "N/A"


def _format_seconds(value: Any) -> str:
    number = _float_or_none(value)
    return f"{number:.3f} s" if number is not None else "N/A"


def _stringify_or_na(value: Any) -> str:
    text = _stringify(value)
    return "N/A" if text == "Not specified" else text


def _stringify(value: Any) -> str:
    if value is None:
        return "Not specified"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "Not specified"
    if isinstance(value, dict):
        return ", ".join(f"{key}: {val}" for key, val in value.items()) if value else "Not specified"
    return str(value)

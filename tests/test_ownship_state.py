from pathlib import Path

from garuda_edge.dashboard.data_loader import build_operator_action, extract_latest_track_state
from garuda_edge.dashboard.ownship import (
    compute_distance_bearing_between_ownship_and_tracks,
    extract_latest_ownship_state,
    extract_ownship_trail,
    extract_platform_state,
)


def test_extract_platform_state_from_platform_state():
    event = {
        "platform_state": {
            "lat": "28.6139",
            "lon": "77.2090",
            "altitude_m": "101.5",
            "heading_deg": "43.2",
            "speed_mps": "12.4",
            "battery_pct": "84",
            "packet_loss": "0.03",
            "latency_ms": "95",
            "link_quality": "nominal",
            "gps_quality": "good",
            "sensor_health": "nominal",
            "timestamp_s": "1.25",
            "frame_id": "40",
        }
    }

    state = extract_platform_state(event)

    assert state["lat"] == 28.6139
    assert state["lon"] == 77.209
    assert state["altitude_m"] == 101.5
    assert state["heading_deg"] == 43.2
    assert state["speed_mps"] == 12.4
    assert state["frame_id"] == 40


def test_extract_platform_state_from_telemetry_and_link():
    event = {
        "telemetry": {
            "latitude": 28.0,
            "longitude": 77.0,
            "alt_m": 120,
            "yaw_deg": 91,
            "groundspeed_mps": 15,
            "battery_pct": 76,
            "gps_quality": "poor",
            "sensor_health": "nominal",
        },
        "link": {"packet_loss": 0.12, "latency_ms": 340, "link_quality": "degraded"},
        "timestamp_s": 2.0,
        "frame_id": 60,
    }

    state = extract_platform_state(event)

    assert state["lat"] == 28.0
    assert state["lon"] == 77.0
    assert state["altitude_m"] == 120
    assert state["heading_deg"] == 91
    assert state["speed_mps"] == 15
    assert state["packet_loss"] == 0.12
    assert state["latency_ms"] == 340
    assert state["link_quality"] == "degraded"


def test_latest_ownship_state_selected_by_frame_id():
    events = [
        {"platform_state": {"lat": 1.0, "lon": 2.0, "frame_id": 10, "timestamp_s": 2.0}},
        {"platform_state": {"lat": 3.0, "lon": 4.0, "frame_id": 12, "timestamp_s": 1.0}},
        {"platform_state": {"lat": 5.0, "lon": 6.0, "frame_id": 11, "timestamp_s": 5.0}},
    ]

    latest = extract_latest_ownship_state(events)

    assert latest is not None
    assert latest["lat"] == 3.0
    assert latest["frame_id"] == 12


def test_ownship_trail_extraction_returns_valid_points():
    events = [
        {"platform_state": {"lat": None, "lon": None, "frame_id": 1}},
        {"platform_state": {"lat": 1.0, "lon": 2.0, "frame_id": 2}},
        {"platform_state": {"lat": 1.0, "lon": 2.0, "frame_id": 3}},
        {"platform_state": {"lat": 1.1, "lon": 2.1, "frame_id": 4}},
    ]

    trail = extract_ownship_trail(events, max_points=20)

    assert len(trail) == 2
    assert trail[0]["frame_id"] == 2
    assert trail[1]["frame_id"] == 4


def test_distance_bearing_between_ownship_and_track():
    ownship = {"lat": 0.0, "lon": 0.0}
    tracks = [{"track_id": "T1", "latest_lat": 0.0, "latest_lon": 0.001}]

    enriched = compute_distance_bearing_between_ownship_and_tracks(ownship, tracks)

    assert 100 <= enriched[0]["range_from_drone_m"] <= 120
    assert 89 <= enriched[0]["bearing_from_drone_deg"] <= 91


def test_missing_ownship_telemetry_does_not_crash():
    assert extract_latest_ownship_state([{"event_id": "E1"}]) is None
    assert extract_ownship_trail([{"event_id": "E1"}]) == []
    enriched = compute_distance_bearing_between_ownship_and_tracks(None, [{"track_id": "T1"}])
    assert enriched[0]["range_from_drone_m"] is None
    assert enriched[0]["bearing_from_drone_deg"] is None


def test_active_track_table_enrichment_adds_range_and_bearing_when_data_exists():
    events = [
        {
            "event_id": "E1",
            "track_id": "T1",
            "frame_id": 5,
            "timestamp_s": 0.5,
            "object_class": "car",
            "confidence": {"final_confidence": 0.81, "level": "high"},
            "geo_estimate": {"estimated_lat": 0.0, "estimated_lon": 0.001, "geo_uncertainty_m": 12},
        }
    ]
    active_tracks = extract_latest_track_state(events)

    enriched = compute_distance_bearing_between_ownship_and_tracks({"lat": 0.0, "lon": 0.0}, active_tracks)

    assert enriched[0]["track_id"] == "T1"
    assert enriched[0]["range_from_drone_m"] is not None
    assert enriched[0]["bearing_from_drone_deg"] is not None


def test_operator_action_schema_can_include_ownship_fields(tmp_path: Path):
    payload = build_operator_action(
        output_root=tmp_path,
        track_id="T1",
        action="continue_observation",
        ownship_lat=28.6139,
        ownship_lon=77.209,
        ownship_altitude_m=101.5,
        ownship_heading_deg=43.2,
        range_from_drone_m=118.7,
        bearing_from_drone_deg=91.4,
    )

    assert payload["ownship_lat"] == 28.6139
    assert payload["ownship_lon"] == 77.209
    assert payload["ownship_altitude_m"] == 101.5
    assert payload["ownship_heading_deg"] == 43.2
    assert payload["range_from_drone_m"] == 118.7
    assert payload["bearing_from_drone_deg"] == 91.4

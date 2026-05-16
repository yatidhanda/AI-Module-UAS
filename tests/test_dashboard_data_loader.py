from pathlib import Path

from garuda_edge.dashboard import data_loader
from garuda_edge.dashboard.data_loader import (
    append_operator_action,
    build_operator_action,
    extract_geo,
    extract_gemma_decision,
    extract_latest_track_state,
    find_existing_evidence_frames,
    load_jsonl,
    load_mission_context,
    load_operator_actions,
    select_important_frames,
)


def test_load_jsonl_and_latest_track_state(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"event_id":"E1","track_id":"T1","frame_id":1,"timestamp_s":0.1,"object_class":"car","confidence":{"final_confidence":0.4,"level":"low"},"geo_estimate":{"estimated_lat":1,"estimated_lon":2,"geo_uncertainty_m":30}}',
                '{"event_id":"E2","track_id":"T1","frame_id":5,"timestamp_s":0.5,"object_class":"car","confidence":{"final_confidence":0.8,"level":"high"},"geo_estimate":{"estimated_lat":3,"estimated_lon":4,"geo_uncertainty_m":10}}',
                '{"event_id":"E3","track_id":"T2","frame_id":2,"timestamp_s":0.2,"object_class":"person","confidence":{"final_confidence":0.6,"level":"medium"}}',
            ]
        ),
        encoding="utf-8",
    )

    events = load_jsonl(path)
    tracks = extract_latest_track_state(events)

    assert len(events) == 3
    assert len(tracks) == 2
    t1 = next(track for track in tracks if track["track_id"] == "T1")
    assert t1["latest_frame"] == 5
    assert t1["latest_confidence"] == 0.8
    assert t1["latest_lat"] == 3.0


def test_missing_gemma_and_geo_fields_are_safe():
    event = {"event_id": "E1", "frame_id": 1}

    assert extract_gemma_decision(event) == {}
    assert extract_geo(event) == {"lat": None, "lon": None, "geo_uncertainty_m": None}


def test_load_mission_context_with_missing_fields(tmp_path: Path):
    mission_path = tmp_path / "mission.yaml"
    mission_path.write_text("mission_id: demo\nsector_id: S-01\n", encoding="utf-8")

    mission = load_mission_context(mission_path)

    assert mission["loaded"] is True
    assert mission["display"]["mission_id"] == "demo"
    assert mission["display"]["mission_name"] == "Not specified"


def test_operator_action_append_format(tmp_path: Path):
    payload = append_operator_action(
        tmp_path,
        "T1",
        "return_to_base",
        notes="reviewed",
        event_id="E1",
        mission_id="M1",
        scenario_name="vehicle_heavy_road",
        preview_frame_id=42,
        preview_image_path=str(tmp_path / "dashboard_frames" / "selected_track_T1_42.jpg"),
        selected_object_class="car",
        selected_confidence=0.82,
        selected_threat_level="medium",
        selected_threat_score=44.0,
        selected_geo_lock_status="ready",
    )
    actions = load_operator_actions(tmp_path)

    assert payload["track_id"] == "T1"
    assert actions[0]["output_root"] == str(tmp_path)
    assert actions[0]["action"] == "return_to_base"
    assert actions[0]["action_label"] == "Return to Base"
    assert actions[0]["operator_mode"] == "human_review"
    assert actions[0]["mission_id"] == "M1"
    assert actions[0]["scenario_name"] == "vehicle_heavy_road"
    assert actions[0]["notes"] == "reviewed"
    assert actions[0]["preview_frame_id"] == 42
    assert actions[0]["selected_object_class"] == "car"
    assert actions[0]["selected_confidence"] == 0.82
    assert actions[0]["selected_threat_level"] == "medium"
    assert actions[0]["selected_threat_score"] == 44.0
    assert actions[0]["selected_geo_lock_status"] == "ready"


def test_full_mission_yaml_loads_required_fields():
    mission = load_mission_context("configs/mission.yaml")

    assert mission["loaded"] is True
    assert mission["display"]["mission_id"] == "garudaedge-visdrone-eval-001"
    assert mission["display"]["mission_name"] == "GarudaEdge VisDrone ISR Evaluation"
    assert mission["display"]["human_review_required"] is True
    assert "return_to_base" in mission["display"]["allowed_operator_actions"]
    assert mission["display"]["notes"]


def test_important_frame_selection_uses_existing_evidence(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    high = evidence_dir / "high.jpg"
    low = evidence_dir / "low.jpg"
    high.write_text("image", encoding="utf-8")
    low.write_text("image", encoding="utf-8")
    events = [
        {
            "event_id": "E1",
            "track_id": "T-low",
            "frame_id": 1,
            "timestamp_s": 1.0,
            "object_class": "person",
            "evidence_frame_path": str(low),
            "confidence": {"final_confidence": 0.2, "level": "low", "degraded_flags": ["low_track_stability"]},
            "geo_estimate": {"estimated_lat": 1, "estimated_lon": 2, "geo_uncertainty_m": 80},
            "policy_result": {"final_decision": {"decision": "request_sensor_confirmation", "human_review_required": True}},
        },
        {
            "event_id": "E2",
            "track_id": "T-high",
            "frame_id": 5,
            "timestamp_s": 5.0,
            "object_class": "car",
            "evidence_frame_path": str(high),
            "confidence": {"final_confidence": 0.9, "level": "high", "degraded_flags": []},
            "geo_estimate": {"estimated_lat": 3, "estimated_lon": 4, "geo_uncertainty_m": 10},
            "policy_result": {"final_decision": {"decision": "continue_observation", "human_review_required": True}},
        },
    ]

    frames = select_important_frames(events, tmp_path, limit=2)

    assert find_existing_evidence_frames(tmp_path) == [high, low]
    assert len(frames) == 2
    assert frames[0]["track_id"] == "T-high"
    assert frames[1]["track_id"] == "T-low"


def test_important_frame_selection_falls_back_to_video_keyframes(tmp_path: Path, monkeypatch):
    keyframe_dir = tmp_path / "dashboard_frames"
    keyframe_dir.mkdir()
    key1 = keyframe_dir / "dashboard_keyframe_25.jpg"
    key2 = keyframe_dir / "dashboard_keyframe_75.jpg"
    key1.write_text("frame", encoding="utf-8")
    key2.write_text("frame", encoding="utf-8")

    def fake_extract(video_path, output_dir, fractions=None):
        return [key1, key2]

    monkeypatch.setattr(data_loader, "extract_keyframes_from_video", fake_extract)

    frames = select_important_frames([], tmp_path, limit=2)

    assert [Path(frame["image_path"]) for frame in frames] == [key1, key2]
    assert frames[0]["source"] == "video_keyframe"


def test_build_operator_action_schema(tmp_path: Path):
    payload = build_operator_action(
        output_root=tmp_path,
        track_id="T1",
        action="hold_position",
        notes="maintain observation",
        mission_id="M1",
        scenario_name="moving_drifting_camera",
    )

    assert payload["action_label"] == "Hold Position"
    assert payload["operator_mode"] == "human_review"
    assert payload["output_root"] == str(tmp_path)

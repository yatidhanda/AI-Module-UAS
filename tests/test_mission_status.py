from garuda_edge.mission.mission_status import compute_mission_status


def make_event(index: int, track_id: str):
    return {
        "event_id": f"E{index}",
        "track_id": track_id,
        "object_class": "car",
        "frame_id": index,
        "timestamp_s": float(index),
        "track_age_frames": 40,
        "confidence": {"final_confidence": 0.8, "level": "high", "degraded_flags": []},
        "geo_estimate": {"geo_uncertainty_m": 30},
        "threat_assessment": {"score": 35, "level": "medium"},
    }


def test_mission_status_computes_completion_percentage(tmp_path):
    mission = {
        "geo_lock_objectives": {
            "enabled": True,
            "target_classes": ["car"],
            "min_confidence": 0.6,
            "max_geo_uncertainty_m": 60,
            "min_track_age_frames": 30,
        },
        "mission_completion": {
            "required_scenarios": ["vehicle_heavy_road"],
            "completion_rules": {"min_events_per_scenario": 50, "min_tracks_per_scenario": 3},
        },
    }
    events = [make_event(i, f"T{i % 3}") for i in range(50)]

    status = compute_mission_status(events, [], {}, mission, tmp_path / "vehicle_heavy_road")

    assert status["mission_completion_estimate"]["percent_complete"] == 100.0
    assert status["mission_completion_estimate"]["status"] == "complete"
    assert len(status["geo_lock_ready_tracks"]) == 3
    assert status["mission_health"]["status"] == "nominal"

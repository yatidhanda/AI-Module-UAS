from garuda_edge.mission.mission_status import evaluate_geo_lock_readiness


MISSION = {
    "geo_lock_objectives": {
        "enabled": True,
        "target_classes": ["truck", "car"],
        "min_confidence": 0.6,
        "max_geo_uncertainty_m": 60,
        "min_track_age_frames": 30,
    }
}


def event(**overrides):
    payload = {
        "track_id": "T1",
        "object_class": "truck",
        "track_age_frames": 35,
        "confidence": {"final_confidence": 0.8, "level": "high"},
        "geo_estimate": {"geo_uncertainty_m": 40},
    }
    payload.update(overrides)
    return payload


def test_geo_lock_ready_when_criteria_pass():
    readiness = evaluate_geo_lock_readiness(event(), MISSION)

    assert readiness["ready"] is True
    assert readiness["status"] == "ready"


def test_geo_lock_not_ready_when_uncertainty_high():
    readiness = evaluate_geo_lock_readiness(event(geo_estimate={"geo_uncertainty_m": 90}), MISSION)

    assert readiness["ready"] is False
    assert readiness["status"] == "not_ready"
    assert any("geo_uncertainty_above_threshold" in reason for reason in readiness["reasons"])

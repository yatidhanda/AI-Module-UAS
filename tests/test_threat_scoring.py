from garuda_edge.threat.threat_library import load_threat_library
from garuda_edge.threat.threat_scoring import score_threat_for_track


def base_event(**overrides):
    event = {
        "event_id": "E1",
        "track_id": "T1",
        "object_class": "car",
        "frame_id": 10,
        "timestamp_s": 1.0,
        "track_age_frames": 10,
        "detection_confidence": 0.7,
        "confidence": {"final_confidence": 0.7, "level": "medium", "degraded_flags": []},
        "geo_estimate": {"geo_uncertainty_m": 20},
        "platform": {"sensor_health": "nominal"},
        "link": {"packet_loss": 0.0, "latency_ms": 60},
    }
    event.update(overrides)
    return event


def test_jammer_scores_critical():
    library = load_threat_library("configs/threat_library.yaml")
    assessment = score_threat_for_track(
        base_event(object_class="jammer", confidence={"final_confidence": 0.9, "level": "high", "degraded_flags": []}),
        {},
        library,
    )

    assert assessment.level == "critical"
    assert assessment.score >= 80
    assert assessment.human_review_required is True


def test_normal_car_is_low_or_medium_with_mission_priority():
    library = load_threat_library("configs/threat_library.yaml")
    mission = {"priority_classes": ["car"], "geo_lock_objectives": {"enabled": True, "target_classes": ["car"]}}
    assessment = score_threat_for_track(base_event(object_class="car"), mission, library)

    assert assessment.level in {"low", "medium"}
    assert 20 <= assessment.score < 60


def test_mission_priority_class_bonus_applies():
    library = load_threat_library("configs/threat_library.yaml")
    plain = score_threat_for_track(base_event(object_class="truck"), {"priority_classes": []}, library)
    priority = score_threat_for_track(base_event(object_class="truck"), {"priority_classes": ["truck"]}, library)

    assert priority.score > plain.score
    assert "mission_priority_class_bonus:+10.0" in priority.contributing_factors


def test_degraded_link_increases_ew_risk():
    library = load_threat_library("configs/threat_library.yaml")
    nominal = score_threat_for_track(base_event(object_class="radar"), {}, library)
    degraded = score_threat_for_track(
        base_event(object_class="radar", link={"packet_loss": 0.2, "latency_ms": 400}),
        {},
        library,
    )

    assert degraded.score > nominal.score
    assert any("degraded_link_bonus" in factor for factor in degraded.contributing_factors)


def test_low_confidence_penalty_reduces_score():
    library = load_threat_library("configs/threat_library.yaml")
    high = score_threat_for_track(
        base_event(object_class="truck", confidence={"final_confidence": 0.9, "level": "high", "degraded_flags": []}),
        {},
        library,
    )
    low = score_threat_for_track(
        base_event(object_class="truck", confidence={"final_confidence": 0.2, "level": "low", "degraded_flags": []}),
        {},
        library,
    )

    assert low.score < high.score
    assert any("low_confidence_penalty" in factor for factor in low.contributing_factors)

from garuda_edge.confidence.scoring import score_confidence
from garuda_edge.ingestion.schemas import GeoEstimate, TelemetryPacket, Track


def tel(**kwargs):
    base = dict(
        timestamp_s=0,
        lat=1,
        lon=2,
        alt_m=80,
        roll_deg=0,
        pitch_deg=0,
        yaw_deg=0,
        gimbal_pan_deg=0,
        gimbal_tilt_deg=-45,
        battery_pct=90,
        packet_loss=0.01,
        latency_ms=100,
        vibration="low",
        gps_quality="good",
        mission_phase="area_search",
        sensor_health="nominal",
        compute_temp_c=55,
        dropped_frames=0,
    )
    base.update(kwargs)
    return TelemetryPacket(**base)


def track():
    return Track(
        track_id="T-0001",
        class_name="person",
        last_bbox_xyxy=[10, 10, 50, 50],
        last_confidence=0.9,
        track_age_frames=40,
        track_stability=0.95,
        last_seen_timestamp_s=0,
    )


def test_high_quality_packet_gives_high_confidence():
    result = score_confidence(track(), 0.95, tel(), GeoEstimate(estimated_lat=1, estimated_lon=2, geo_uncertainty_m=20, method="x"), 0)
    assert result.level == "high"


def test_bad_conditions_lower_confidence():
    result = score_confidence(
        track(),
        0.9,
        tel(gps_quality="poor", vibration="high", packet_loss=0.3, latency_ms=500, sensor_health="degraded", dropped_frames=3),
        GeoEstimate(estimated_lat=1, estimated_lon=2, geo_uncertainty_m=120, method="x"),
        1.0,
    )
    assert result.final_confidence < 0.6
    assert result.level in {"medium", "low"}


def test_score_between_zero_and_one():
    result = score_confidence(track(), 0.1, tel(packet_loss=1), GeoEstimate(estimated_lat=1, estimated_lon=2, geo_uncertainty_m=999, method="x"), 10)
    assert 0 <= result.final_confidence <= 1

import pytest

from garuda_edge.ingestion.schemas import TelemetryPacket
from garuda_edge.ingestion.sync import sync_frame_to_telemetry
from garuda_edge.ingestion.telemetry import TelemetryTimeline, get_nearest_telemetry


def packet(t):
    return TelemetryPacket(
        timestamp_s=t,
        lat=0,
        lon=0,
        alt_m=80,
        roll_deg=0,
        pitch_deg=0,
        yaw_deg=0,
        gimbal_pan_deg=0,
        gimbal_tilt_deg=-45,
        battery_pct=90,
        packet_loss=0,
        latency_ms=100,
        vibration="low",
        gps_quality="good",
        mission_phase="area_search",
        sensor_health="nominal",
    )


def test_nearest_telemetry_selected():
    assert sync_frame_to_telemetry(1.1, [packet(0), packet(1.0), packet(3)]).telemetry.timestamp_s == 1.0


def test_large_sync_delta_flagged():
    assert sync_frame_to_telemetry(10, [packet(0)], threshold_s=0.5).warning is not None


def test_interpolates_between_bracketing_samples():
    a = packet(0.0).model_copy(update={"lat": 10.0, "alt_m": 80.0, "yaw_deg": 350.0, "gps_quality": "good"})
    b = packet(0.1).model_copy(update={"lat": 10.001, "alt_m": 90.0, "yaw_deg": 10.0, "gps_quality": "poor"})
    sync = sync_frame_to_telemetry(0.075, [a, b], interpolate=True)
    assert sync.telemetry.timestamp_s == 0.075
    assert sync.telemetry.lat == pytest.approx(10.00075)
    assert sync.telemetry.alt_m == pytest.approx(87.5)
    assert sync.telemetry.yaw_deg == pytest.approx(5.0)  # shortest arc across north, not via 180
    assert sync.telemetry.gps_quality == "poor"  # categorical: nearer sample
    assert sync.sync_delta_s == pytest.approx(0.025)


def test_no_interpolation_across_large_gap_or_outside_log():
    a, b = packet(0.0).model_copy(update={"alt_m": 80.0}), packet(5.0).model_copy(update={"alt_m": 120.0})
    across_gap = sync_frame_to_telemetry(1.0, [a, b], interpolate=True)
    assert across_gap.telemetry.alt_m == 80.0 and across_gap.sync_delta_s == pytest.approx(1.0)
    beyond = sync_frame_to_telemetry(7.0, [a, b], interpolate=True)
    assert beyond.telemetry.alt_m == 120.0 and beyond.warning is not None


def test_timeline_nearest_matches_linear_scan():
    packets = [packet(t) for t in (0.0, 1.0, 2.0, 3.0)]
    timeline = TelemetryTimeline(list(reversed(packets)))
    for t in (-1.0, 0.4, 0.5, 1.5, 2.6, 9.0):
        assert timeline.nearest(t).timestamp_s == get_nearest_telemetry(t, packets).timestamp_s

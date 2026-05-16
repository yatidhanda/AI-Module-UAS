from garuda_edge.ingestion.schemas import TelemetryPacket
from garuda_edge.ingestion.sync import sync_frame_to_telemetry


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

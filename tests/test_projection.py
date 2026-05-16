from garuda_edge.geolocation.projection import project_bbox_to_ground
from garuda_edge.ingestion.schemas import TelemetryPacket


def tel(**kwargs):
    base = dict(
        timestamp_s=0,
        lat=28.6,
        lon=77.2,
        alt_m=80,
        roll_deg=0,
        pitch_deg=0,
        yaw_deg=45,
        gimbal_pan_deg=0,
        gimbal_tilt_deg=-45,
        battery_pct=90,
        packet_loss=0.01,
        latency_ms=100,
        vibration="low",
        gps_quality="good",
        mission_phase="area_search",
        sensor_health="nominal",
    )
    base.update(kwargs)
    return TelemetryPacket(**base)


def test_projection_returns_lat_lon_floats():
    geo = project_bbox_to_ground([10, 10, 50, 50], 640, 480, tel())
    assert isinstance(geo.estimated_lat, float)
    assert isinstance(geo.estimated_lon, float)


def test_uncertainty_increases_with_bad_telemetry():
    good = project_bbox_to_ground([10, 10, 50, 50], 640, 480, tel())
    bad = project_bbox_to_ground([10, 10, 50, 50], 640, 480, tel(gps_quality="poor", vibration="high", packet_loss=0.4), sync_delta_s=2)
    assert bad.geo_uncertainty_m > good.geo_uncertainty_m

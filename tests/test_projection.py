import math

import pytest

from garuda_edge.geolocation.camera_model import CameraModel
from garuda_edge.geolocation.projection import METERS_PER_DEG_LAT, project_bbox_to_ground
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


W, H = 640, 480


def _ground_box(x, y):
    """Small box whose bottom-centre (the projected ground point) is at pixel (x, y)."""
    return [x - 1, y - 2, x + 1, y]


def _offset_m(geo, telemetry):
    east = (geo.estimated_lon - telemetry.lon) * METERS_PER_DEG_LAT * math.cos(math.radians(telemetry.lat))
    north = (geo.estimated_lat - telemetry.lat) * METERS_PER_DEG_LAT
    return east, north


def _project(x, y, camera=None, **telemetry_kwargs):
    t = tel(**{"alt_m": 100, "yaw_deg": 0, "gimbal_tilt_deg": -90, **telemetry_kwargs})
    return _offset_m(project_bbox_to_ground(_ground_box(x, y), W, H, t, camera), t)


def test_nadir_image_centre_is_directly_below_drone():
    assert _project(320, 240) == pytest.approx((0.0, 0.0), abs=0.01)


def test_nadir_image_edges_follow_pinhole_field_of_view():
    assert _project(640, 240) == pytest.approx((100 * math.tan(math.radians(35)), 0.0), abs=0.05)
    assert _project(320, 0) == pytest.approx((0.0, 100 * math.tan(math.radians(21.5))), abs=0.05)


@pytest.mark.parametrize(
    "yaw, pan, expected",
    [(0, 0, (0.0, 100.0)), (90, 0, (100.0, 0.0)), (90, 90, (0.0, -100.0))],
)
def test_oblique_boresight_follows_yaw_and_pan(yaw, pan, expected):
    assert _project(320, 240, gimbal_tilt_deg=-45, yaw_deg=yaw, gimbal_pan_deg=pan) == pytest.approx(expected, abs=0.01)


def test_ground_contact_point_is_box_bottom():
    t = tel(alt_m=100, yaw_deg=0, gimbal_tilt_deg=-45)
    tall_box = _offset_m(project_bbox_to_ground([300, 100, 340, 240], W, H, t), t)
    centre_box = _offset_m(project_bbox_to_ground([300, 220, 340, 240], W, H, t), t)
    assert tall_box == pytest.approx(centre_box, abs=0.01)  # same bottom edge, same ground point


def test_gimbal_roll_rotates_the_image_offset():
    east, north = _project(640, 240, gimbal_roll_deg=90)
    assert east == pytest.approx(0.0, abs=0.05)
    assert abs(north) == pytest.approx(100 * math.tan(math.radians(35)), abs=0.05)


def test_uncertainty_grows_as_view_becomes_shallower():
    def uncertainty(tilt):
        return project_bbox_to_ground(_ground_box(320, 240), W, H, tel(alt_m=100, gimbal_tilt_deg=tilt)).geo_uncertainty_m

    assert uncertainty(-90) < uncertainty(-45) < uncertainty(-20) < uncertainty(-8)


def test_ray_near_horizon_is_clamped_and_flagged():
    geo = project_bbox_to_ground(_ground_box(320, 240), W, H, tel(alt_m=100, gimbal_tilt_deg=0))
    assert geo.method == "pinhole_ray_ground_plane_clamped"
    assert geo.geo_uncertainty_m > 100


def test_zero_distortion_matches_pinhole_and_distortion_moves_edge_points():
    assert _project(600, 400, camera=CameraModel(k1=0.0, k2=0.0)) == pytest.approx(_project(600, 400), abs=1e-9)
    assert _project(600, 400, camera=CameraModel(k1=-0.2)) != pytest.approx(_project(600, 400), abs=0.1)

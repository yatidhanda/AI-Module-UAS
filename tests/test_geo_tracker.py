import math
import random

import pytest

from garuda_edge.geolocation.geo_tracker import GeoTrackFilter
from garuda_edge.ingestion.schemas import GeoEstimate

LAT0, LON0 = 28.6, 77.2
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = M_PER_DEG_LAT * math.cos(math.radians(LAT0))


def _raw(east, north, uncertainty=5.0):
    return GeoEstimate(
        estimated_lat=LAT0 + north / M_PER_DEG_LAT,
        estimated_lon=LON0 + east / M_PER_DEG_LON,
        geo_uncertainty_m=uncertainty,
        method="pinhole_ray_ground_plane",
    )


def _local(geo):
    return (geo.estimated_lon - LON0) * M_PER_DEG_LON, (geo.estimated_lat - LAT0) * M_PER_DEG_LAT


def _run(velocity_en, seconds=10.0, hz=10, noise=5.0, seed=0):
    rng = random.Random(seed)
    f = GeoTrackFilter()
    out = None
    for k in range(int(seconds * hz)):
        t = k / hz
        out = f.update("T", _raw(velocity_en[0] * t + rng.gauss(0, noise), velocity_en[1] * t + rng.gauss(0, noise)), t, 2.5)
    return out, t


def test_moving_car_speed_and_heading():
    out, _ = _run((0.0, 10.0))  # 10 m/s due north, 5 m noise
    assert out.speed_mps == pytest.approx(10.0, abs=1.5)
    assert min(out.heading_deg, 360 - out.heading_deg) < 10


def test_heading_follows_direction_of_travel():
    out, _ = _run((10.0, 0.0), seed=1)  # due east
    assert out.heading_deg == pytest.approx(90.0, abs=10)


def test_parked_car_gets_no_heading_and_stays_put():
    for seed in range(10):
        out, _ = _run((0.0, 0.0), seed=seed)
        assert out.heading_deg is None
        assert math.hypot(*_local(out)) < 5.0  # filtered fix beats the 5 m raw noise


def test_speed_withheld_until_enough_updates():
    f = GeoTrackFilter(min_updates_for_speed=5)
    for k in range(4):
        out = f.update("T", _raw(0, k), k / 10, 2.5)
    assert out.speed_mps is None
    assert f.update("T", _raw(0, 4), 0.4, 2.5).speed_mps is not None


def test_single_outlier_rejected_and_repeated_jump_restarts():
    f = GeoTrackFilter()
    for k in range(20):
        f.update("T", _raw(0, 0), k / 10, 2.5)
    after_outlier = f.update("T", _raw(200, 0), 2.0, 2.5)
    assert math.hypot(*_local(after_outlier)) < 5.0

    for k in range(1, 3):
        out = f.update("T", _raw(200, 0), 2.0 + k / 10, 2.5)
    assert _local(out)[0] == pytest.approx(200.0, abs=0.01)  # third rejection in a row restarts at the new fix


def test_uncertainty_floored_at_gps_sigma():
    f = GeoTrackFilter()
    for k in range(300):
        out = f.update("T", _raw(0, 0, uncertainty=3.0), k / 30, 2.5)
    assert out.geo_uncertainty_m == pytest.approx(2.5)


def test_prune_drops_deleted_tracks():
    f = GeoTrackFilter()
    f.update("A", _raw(0, 0), 0.0, 2.5)
    f.update("B", _raw(10, 0), 0.0, 2.5)
    f.prune(["B"])
    assert "A" not in f and "B" in f

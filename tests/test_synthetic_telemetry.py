import math

from garuda_edge.ingestion.synthetic_telemetry import FIELDS, generate_rows, write_telemetry_csv
from garuda_edge.ingestion.telemetry import load_telemetry_csv

M_PER_DEG_LAT = 111_320.0


def _local_m(row, origin):
    north = (row["lat"] - origin["lat"]) * M_PER_DEG_LAT
    east = (row["lon"] - origin["lon"]) * M_PER_DEG_LAT * math.cos(math.radians(origin["lat"]))
    return east, north


def test_continuous_fields_change_smoothly():
    rows = generate_rows(1000, 30, seed=1)
    for key, max_step in (("alt_m", 0.2), ("gimbal_tilt_deg", 0.3), ("yaw_deg", 1.0)):
        steps = [abs(b[key] - a[key]) for a, b in zip(rows, rows[1:])]
        assert max(steps) < max_step, key


def test_hover_holds_position():
    rows = generate_rows(1000, 30, profile="hover", seed=2)
    centre = {"lat": sum(r["lat"] for r in rows) / len(rows), "lon": sum(r["lon"] for r in rows) / len(rows)}
    assert max(math.hypot(*_local_m(row, centre)) for row in rows) < 2.0


def test_transect_flies_at_cruise_speed():
    rows = generate_rows(30 * 30, 30, profile="transect", seed=3)  # 30 s, before the first turn at 150 m
    east, north = _local_m(rows[-1], rows[0])
    assert 108 < math.hypot(east, north) < 132  # 4 m/s * 30 s = 120 m


def test_same_seed_is_reproducible_and_columns_are_stable(tmp_path):
    assert generate_rows(200, 15, seed=5) == generate_rows(200, 15, seed=5)
    assert generate_rows(200, 15, seed=5) != generate_rows(200, 15, seed=6)
    path = write_telemetry_csv(generate_rows(50, 15, seed=5), tmp_path / "t.csv")
    assert path.read_text().splitlines()[0].split(",") == FIELDS
    assert len(load_telemetry_csv(path)) == 50

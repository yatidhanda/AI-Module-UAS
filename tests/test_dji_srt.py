from pathlib import Path

import pytest

from garuda_edge.ingestion.dji_srt import parse_dji_srt
from garuda_edge.ingestion.telemetry import load_telemetry

FIXTURES = Path(__file__).parent / "fixtures"


def test_consumer_format_with_longtitude_typo_and_course_over_ground():
    packets = parse_dji_srt(FIXTURES / "dji_srt_consumer.srt")
    assert [p.timestamp_s for p in packets] == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert packets[0].lat == pytest.approx(28.6139)
    assert packets[-1].lon == pytest.approx(77.2090 + 4 * 0.00002558)
    assert all(p.alt_m == 80.0 for p in packets)  # rel_alt preferred over abs_alt
    # No gimbal yaw: heading comes from motion (due east), including the first frame.
    assert all(p.yaw_deg == pytest.approx(90.0, abs=1.0) for p in packets)
    assert all(p.gimbal_tilt_deg == -90.0 and p.gimbal_pan_deg == 0.0 for p in packets)
    assert packets[0].battery_pct is None and packets[0].packet_loss == 0.0


def test_enterprise_format_uses_absolute_gimbal_angles():
    packets = parse_dji_srt(FIXTURES / "dji_srt_enterprise.srt", gimbal_tilt_deg=-90.0)
    first, second = packets
    assert first.timestamp_s == 0.0 and second.timestamp_s == pytest.approx(0.033)
    assert first.alt_m == 120.0
    assert first.yaw_deg + first.gimbal_pan_deg == pytest.approx(92.5)
    assert first.gimbal_tilt_deg == -45.0  # from gb_pitch, not the fallback
    assert second.gimbal_roll_deg == 1.5


def test_legacy_format_reads_lon_first_and_skips_frames_without_fix():
    packets = parse_dji_srt(FIXTURES / "dji_srt_legacy.srt", yaw_deg=30.0)
    assert [p.timestamp_s for p in packets] == [1.0, 2.0]  # GPS(0,0) frame dropped
    assert packets[0].lon == pytest.approx(77.2091) and packets[0].lat == pytest.approx(28.6140)
    assert packets[1].alt_m == pytest.approx(85.6)
    assert all(p.yaw_deg == 30.0 for p in packets)  # stationary, no gimbal yaw: caller's heading


def test_mavic_air_style_height_field(tmp_path):
    srt = tmp_path / "air.SRT"
    srt.write_text("1\n00:00:00,000 --> 00:00:00,033\nF/2.8, SS 1000, ISO 100, EV 0, GPS (77.2092, 28.6141, 18), D 12.34m, H 23.45m, H.S 0.00m/s, V.S 0.00m/s\n")
    (packet,) = load_telemetry(srt)  # dispatch by extension, case-insensitive
    assert (packet.lat, packet.lon, packet.alt_m) == (pytest.approx(28.6141), pytest.approx(77.2092), 23.45)


def test_file_without_any_fix_is_rejected(tmp_path):
    srt = tmp_path / "empty.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:00,033\n[latitude: 0.0] [longitude: 0.0] [rel_alt: 0.0]\n")
    with pytest.raises(ValueError):
        parse_dji_srt(srt)

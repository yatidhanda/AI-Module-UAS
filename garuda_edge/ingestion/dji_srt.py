"""DJI flight-subtitle (.SRT) telemetry parser.

DJI drones can record an .SRT file next to each video with one subtitle per frame.
Field layout varies by model and firmware, so parsing is tolerant:

- Bracketed key/value style (Mini, Air, Mavic 3 and Enterprise models):
  ``[latitude: 28.6139] [longitude: 77.2090] [rel_alt: 80.0 abs_alt: 300.0]``, with
  ``[gb_yaw: .. gb_pitch: .. gb_roll: ..]`` gimbal angles on Enterprise models.
  Some firmware spells longitude as "longtitude".
- Legacy style (Phantom 4, Mavic Pro, Mavic Air, Spark): ``GPS(lon,lat,sats)`` with
  longitude first, and altitude as ``BAROMETER:x`` or ``H 23.4m``.

Anything an SRT file does not record (link quality, body attitude, battery) is filled
with neutral values; gimbal tilt and yaw fall back to caller-supplied values, which
determine the projected coordinates, so a warning is logged when they are used.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)

from garuda_edge.ingestion.schemas import TelemetryPacket

_TIME_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->")
_PAIR_RE = re.compile(r"([A-Za-z_]+)\s*:\s*(-?\d+(?:\.\d+)?)")
_LEGACY_GPS_RE = re.compile(r"GPS\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")
_LEGACY_HEIGHT_RE = re.compile(r"\bH\s+(-?\d+(?:\.\d+)?)\s*m\b")
_TAG_RE = re.compile(r"<[^>]+>")

_LAT_KEYS = ("latitude", "lat")
_LON_KEYS = ("longitude", "longtitude", "lon")
_RELATIVE_ALT_KEYS = ("rel_alt", "barometer")
_OTHER_ALT_KEYS = ("altitude", "abs_alt")

MIN_SPEED_FOR_COURSE_MPS = 1.0
COURSE_WINDOW_S = 1.0
_METERS_PER_DEG_LAT = 111_320.0


def parse_dji_srt(path: str | Path, gimbal_tilt_deg: float = -90.0, yaw_deg: float | None = None) -> list[TelemetryPacket]:
    records = [rec for rec in (_parse_block(block) for block in _blocks(Path(path))) if rec is not None]
    if not records:
        raise ValueError(f"No GPS fixes found in SRT file: {path}")

    if not any("gb_pitch" in rec for rec in records):
        logger.warning(
            f"SRT has no gimbal angles; assuming gimbal tilt {gimbal_tilt_deg:g} deg for every frame. "
            "Projected coordinates depend on this: pass the real tilt (--srt-gimbal-tilt-deg)."
        )
    if any(rec["alt_source"] == "other" for rec in records):
        logger.warning(
            "SRT altitude is not marked relative to takeoff (rel_alt/BAROMETER); using it as height above "
            "ground. If it is above sea level, projected coordinates will be badly wrong."
        )

    yaws, defaulted = _resolve_yaws(records, yaw_deg)
    if defaulted:
        logger.warning(
            f"Drone heading unknown for {defaulted} of {len(records)} SRT frames (no gb_yaw and not moving); "
            f"using {yaw_deg if yaw_deg is not None else 0.0:g} deg. Pass --srt-yaw-deg if the camera heading is known."
        )

    packets: list[TelemetryPacket] = []
    for rec, yaw in zip(records, yaws):
        packets.append(
            TelemetryPacket(
                timestamp_s=rec["timestamp_s"],
                lat=rec["lat"],
                lon=rec["lon"],
                alt_m=rec["alt_m"],
                roll_deg=0.0,
                pitch_deg=0.0,
                yaw_deg=yaw,
                gimbal_pan_deg=_wrap180(rec["gb_yaw"] - yaw) if "gb_yaw" in rec else 0.0,
                gimbal_tilt_deg=rec.get("gb_pitch", gimbal_tilt_deg),
                gimbal_roll_deg=rec.get("gb_roll"),
                battery_pct=None,
                packet_loss=0.0,  # recorded on board: no radio link involved
                latency_ms=0.0,
                vibration="unknown",
                gps_quality="unknown",
                mission_phase="recorded_flight",
                sensor_health="nominal",
            )
        )
    return packets


def _blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
    return [block for block in re.split(r"\n\s*\n", text) if block.strip()]


def _parse_block(block: str) -> dict[str, Any] | None:
    time_match = _TIME_RE.search(block)
    if time_match is None:
        return None
    h, m, s, frac = time_match.groups()
    record: dict[str, Any] = {"timestamp_s": int(h) * 3600 + int(m) * 60 + int(s) + int(frac) / 10 ** len(frac)}

    body = _TAG_RE.sub(" ", block[time_match.end() :])
    values = {key.lower(): float(value) for key, value in _PAIR_RE.findall(body)}
    lat = _first(values, _LAT_KEYS)
    lon = _first(values, _LON_KEYS)
    legacy_gps = _LEGACY_GPS_RE.search(body)
    if (lat is None or lon is None) and legacy_gps:
        lon, lat = float(legacy_gps.group(1)), float(legacy_gps.group(2))
    if lat is None or lon is None or (lat == 0.0 and lon == 0.0):  # no fix yet
        return None

    alt = _first(values, _RELATIVE_ALT_KEYS)
    alt_source = "relative"
    if alt is None:
        legacy_height = _LEGACY_HEIGHT_RE.search(body)
        alt = float(legacy_height.group(1)) if legacy_height else _first(values, _OTHER_ALT_KEYS)
        alt_source = "relative" if legacy_height else "other"
    if alt is None:
        return None

    record.update(lat=lat, lon=lon, alt_m=alt, alt_source=alt_source)
    for key in ("gb_yaw", "gb_pitch", "gb_roll"):
        if key in values:
            record[key] = values[key]
    return record


def _resolve_yaws(records: list[dict[str, Any]], fallback: float | None) -> tuple[list[float], int]:
    """Yaw per frame: gimbal yaw, else course over ground while moving, else the nearest known yaw, else fallback."""
    known: list[float | None] = []
    last_known: float | None = None
    start = 0
    for rec in records:
        while records[start]["timestamp_s"] < rec["timestamp_s"] - COURSE_WINDOW_S:
            start += 1
        yaw = rec.get("gb_yaw")
        if yaw is None:
            yaw = _course_deg(records[start], rec)
        if yaw is None:
            yaw = last_known  # hovering or slow: keep the previous heading
        last_known = yaw
        known.append(yaw)

    first_known = next((yaw for yaw in known if yaw is not None), None)
    if first_known is None:
        return [fallback if fallback is not None else 0.0] * len(records), len(records)
    # Frames before the first heading fix (e.g. before take-off movement) take that first heading.
    return [float(yaw if yaw is not None else first_known) for yaw in known], 0


def _course_deg(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    dt = b["timestamp_s"] - a["timestamp_s"]
    if dt < 0.2:
        return None
    north = (b["lat"] - a["lat"]) * _METERS_PER_DEG_LAT
    east = (b["lon"] - a["lon"]) * _METERS_PER_DEG_LAT * math.cos(math.radians(a["lat"]))
    if math.hypot(east, north) / dt < MIN_SPEED_FOR_COURSE_MPS:
        return None
    return math.degrees(math.atan2(east, north)) % 360.0


def _first(values: dict[str, float], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in values:
            return values[key]
    return None


def _wrap180(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0

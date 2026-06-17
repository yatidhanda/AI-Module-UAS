from __future__ import annotations

import math
from typing import Any


NORMALIZED_PLATFORM_FIELDS = {
    "lat": None,
    "lon": None,
    "altitude_m": None,
    "heading_deg": None,
    "roll_deg": None,
    "pitch_deg": None,
    "yaw_deg": None,
    "speed_mps": None,
    "battery_pct": None,
    "packet_loss": None,
    "latency_ms": None,
    "link_quality": None,
    "gps_quality": None,
    "sensor_health": None,
    "timestamp_s": None,
    "frame_id": None,
}


def extract_platform_state(event: dict[str, Any]) -> dict[str, Any]:
    """Normalize platform/ownship telemetry from old and new event schemas."""
    if not isinstance(event, dict):
        return dict(NORMALIZED_PLATFORM_FIELDS)

    sources = _platform_sources(event)
    link = event.get("link") if isinstance(event.get("link"), dict) else {}

    state = {
        "lat": _float_or_none(_first_value(sources, "lat", "latitude")),
        "lon": _float_or_none(_first_value(sources, "lon", "longitude", "lng")),
        "altitude_m": _float_or_none(_first_value(sources, "altitude_m", "alt_m", "altitude")),
        "heading_deg": _float_or_none(_first_value(sources, "heading_deg", "yaw_deg", "yaw")),
        "roll_deg": _float_or_none(_first_value(sources, "roll_deg", "roll")),
        "pitch_deg": _float_or_none(_first_value(sources, "pitch_deg", "pitch")),
        "yaw_deg": _float_or_none(_first_value(sources, "yaw_deg", "yaw", "heading_deg")),
        "speed_mps": _float_or_none(_first_value(sources, "speed_mps", "airspeed_mps", "groundspeed_mps")),
        "battery_pct": _float_or_none(_first_value(sources, "battery_pct", "battery_percent")),
        "packet_loss": _float_or_none(_first_value([*sources, link], "packet_loss")),
        "latency_ms": _float_or_none(_first_value([*sources, link], "latency_ms")),
        "link_quality": _str_or_none(_first_value([*sources, link], "link_quality")),
        "gps_quality": _str_or_none(_first_value(sources, "gps_quality")),
        "sensor_health": _str_or_none(_first_value(sources, "sensor_health")),
        "timestamp_s": _float_or_none(_first_value(sources, "timestamp_s")),
        "frame_id": _int_or_none(_first_value(sources, "frame_id")),
    }
    return state


def extract_latest_ownship_state(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid_states = [state for state in (extract_platform_state(event) for event in events) if _valid_lat_lon(state)]
    if not valid_states:
        return None
    return max(valid_states, key=_state_sort_key)


def extract_ownship_trail(events: list[dict[str, Any]], max_points: int = 200) -> list[dict[str, Any]]:
    max_points = max(1, int(max_points))
    states = sorted(
        [state for state in (extract_platform_state(event) for event in events) if _valid_lat_lon(state)],
        key=_state_sort_key,
    )
    trail: list[dict[str, Any]] = []
    for state in states:
        if trail and state["lat"] == trail[-1]["lat"] and state["lon"] == trail[-1]["lon"]:
            continue
        trail.append(state)
    return trail[-max_points:]


def compute_distance_bearing_between_ownship_and_tracks(
    ownship_state: dict[str, Any] | None,
    active_tracks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ownship_valid = _valid_lat_lon(ownship_state or {})
    ownship_lat = float(ownship_state["lat"]) if ownship_valid and ownship_state else None
    ownship_lon = float(ownship_state["lon"]) if ownship_valid and ownship_state else None

    for track in active_tracks:
        row = dict(track)
        row["range_from_drone_m"] = None
        row["bearing_from_drone_deg"] = None
        track_lat, track_lon = _extract_track_lat_lon(track)
        if ownship_lat is not None and ownship_lon is not None and _valid_coordinate_pair(track_lat, track_lon):
            row["range_from_drone_m"] = round(_haversine_m(ownship_lat, ownship_lon, float(track_lat), float(track_lon)), 1)
            row["bearing_from_drone_deg"] = round(
                _bearing_deg(ownship_lat, ownship_lon, float(track_lat), float(track_lon)),
                1,
            )
        rows.append(row)
    return rows


def platform_telemetry_status(ownship_state: dict[str, Any] | None) -> dict[str, Any]:
    if not ownship_state or not _valid_lat_lon(ownship_state):
        return {
            "status": "No platform telemetry found",
            "level": "red",
            "reasons": ["missing lat/lon"],
        }

    categories: set[str] = set()
    reasons: list[str] = []
    level = "green"

    battery = _float_or_none(ownship_state.get("battery_pct"))
    if battery is not None and battery < 25:
        categories.add("battery")
        reasons.append(f"battery {battery:.0f}%")
        level = "red" if battery < 15 else _max_level(level, "amber")

    packet_loss = _float_or_none(ownship_state.get("packet_loss"))
    latency = _float_or_none(ownship_state.get("latency_ms"))
    if packet_loss is not None and packet_loss > 0.10:
        categories.add("link")
        reasons.append(f"packet loss {packet_loss:.2f}")
        level = "red" if packet_loss > 0.25 else _max_level(level, "amber")
    if latency is not None and latency > 300:
        categories.add("link")
        reasons.append(f"latency {latency:.0f} ms")
        level = "red" if latency > 800 else _max_level(level, "amber")

    gps_quality = str(ownship_state.get("gps_quality") or "").lower()
    if gps_quality in {"poor", "lost", "degraded"}:
        categories.add("gps")
        reasons.append(f"gps {gps_quality}")
        level = "red" if gps_quality == "lost" else _max_level(level, "amber")

    if not categories:
        return {"status": "Telemetry nominal", "level": "green", "reasons": []}
    if len(categories) > 1:
        status = "Telemetry degraded"
    elif "gps" in categories:
        status = "GPS degraded"
    elif "link" in categories:
        status = "Link degraded"
    else:
        status = "Battery low"
    return {"status": status, "level": level, "reasons": reasons}


def project_point(lat: float, lon: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    radius_m = 6_371_000.0
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    angular_distance = distance_m / radius_m
    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), ((math.degrees(lon2) + 540.0) % 360.0) - 180.0


def valid_lat_lon(lat: Any, lon: Any) -> bool:
    return _valid_coordinate_pair(lat, lon)


def _platform_sources(event: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for key in ("platform_state", "telemetry", "platform", "drone_state"):
        value = event.get(key)
        if isinstance(value, dict):
            sources.append(value)
    sources.append(event)
    return sources


def _first_value(sources: list[dict[str, Any]], *keys: str) -> Any:
    for source in sources:
        for key in keys:
            if key not in source:
                continue
            value = source.get(key)
            if value is not None and value != "":
                return value
    return None


def _extract_track_lat_lon(track: dict[str, Any]) -> tuple[Any, Any]:
    geo = track.get("geo_estimate") if isinstance(track.get("geo_estimate"), dict) else {}
    lat = _first_value([track, geo], "latest_lat", "lat", "estimated_lat", "latitude")
    lon = _first_value([track, geo], "latest_lon", "lon", "estimated_lon", "longitude", "lng")
    return lat, lon


def _valid_lat_lon(state: dict[str, Any]) -> bool:
    return _valid_coordinate_pair(state.get("lat"), state.get("lon"))


def _valid_coordinate_pair(lat: Any, lon: Any) -> bool:
    try:
        lat_f = float(lat)
        lon_f = float(lon)
        return math.isfinite(lat_f) and math.isfinite(lon_f) and -90 <= lat_f <= 90 and -180 <= lon_f <= 180
    except (TypeError, ValueError):
        return False


def _state_sort_key(state: dict[str, Any]) -> tuple[float, float]:
    frame = _float_or_none(state.get("frame_id"))
    timestamp = _float_or_none(state.get("timestamp_s"))
    return (frame if frame is not None else -1.0, timestamp if timestamp is not None else -1.0)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return radius_m * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    x = math.sin(delta_lambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    number = _float_or_none(value)
    return int(number) if number is not None else None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _max_level(left: str, right: str) -> str:
    order = {"green": 0, "amber": 1, "red": 2}
    return right if order.get(right, 0) > order.get(left, 0) else left

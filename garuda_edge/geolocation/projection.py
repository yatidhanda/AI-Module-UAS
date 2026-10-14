from __future__ import annotations

import math

from garuda_edge.geolocation.camera_model import CameraModel
from garuda_edge.ingestion.schemas import GeoEstimate, TelemetryPacket

METERS_PER_DEG_LAT = 111_320.0
GPS_SIGMA_M = {"good": 2.5, "medium": 5.0, "poor": 15.0}
DEFAULT_GPS_SIGMA_M = 5.0
VIBRATION_EXTRA_ATTITUDE_DEG = {"medium": 1.0, "high": 3.0}
MIN_DOWNWARD_DEG = 5.0
HORIZON_CLAMP_PENALTY_M = 100.0
# A 1-sigma geo uncertainty above this is treated as poor (good conditions give ~2.5-8 m).
HIGH_GEO_UNCERTAINTY_M = 15.0
_UNDISTORT_ITERATIONS = 8

Vec3 = tuple[float, float, float]


def bbox_center(bbox_xyxy: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_ground_point(bbox_xyxy: list[float]) -> tuple[float, float]:
    """Bottom-centre of the box: where a ground vehicle touches the road."""
    x1, _, x2, y2 = bbox_xyxy
    return (x1 + x2) / 2.0, y2


def gps_sigma_m(gps_quality: str | None) -> float:
    return GPS_SIGMA_M.get(str(gps_quality or "").lower(), DEFAULT_GPS_SIGMA_M)


def project_bbox_to_ground(
    bbox_xyxy: list[float],
    image_width: int,
    image_height: int,
    telemetry: TelemetryPacket,
    camera: CameraModel | None = None,
    sync_delta_s: float = 0.0,
) -> GeoEstimate:
    """Pinhole camera ray from the box's ground-contact point to a flat ground plane.

    Works in a local east/north/up frame centred on the drone, with the ground at
    -alt_m. geo_uncertainty_m is an approximate 1-sigma horizontal error propagated
    from drone GPS, altitude and camera attitude errors, plus link/sync penalties.
    Flat ground is assumed: terrain relief adds error that is not modelled here.
    """

    camera = camera or CameraModel()
    gx, gy = bbox_ground_point(bbox_xyxy)
    xn = (gx - image_width / 2.0) / max(1.0, image_width / 2.0)
    yn = (gy - image_height / 2.0) / max(1.0, image_height / 2.0)
    xf, yf = _undistort(
        xn * math.tan(math.radians(camera.horizontal_fov_deg / 2.0)),
        yn * math.tan(math.radians(camera.vertical_fov_deg / 2.0)),
        camera.k1,
        camera.k2,
    )

    azimuth = telemetry.yaw_deg + telemetry.gimbal_pan_deg
    elevation = telemetry.gimbal_tilt_deg + telemetry.pitch_deg
    roll = telemetry.gimbal_roll_deg or 0.0
    alt = max(1.0, telemetry.alt_m)

    east, north, clamped = _ground_offset(xf, yf, azimuth, elevation, roll, alt)

    attitude_sigma = math.hypot(camera.attitude_sigma_deg, VIBRATION_EXTRA_ATTITUDE_DEG.get(telemetry.vibration, 0.0))
    spread_sq = gps_sigma_m(telemetry.gps_quality) ** 2
    for d_az, d_el, d_alt in (
        (attitude_sigma, 0.0, 0.0),
        (0.0, attitude_sigma, 0.0),
        (0.0, 0.0, camera.altitude_sigma_m),
    ):
        spread_sq += max(
            math.dist((east, north), _ground_offset(xf, yf, azimuth + s * d_az, elevation + s * d_el, roll, max(1.0, alt + s * d_alt))[:2])
            for s in (1.0, -1.0)
        ) ** 2
    uncertainty = math.sqrt(spread_sq)
    if telemetry.packet_loss > 0.1:
        uncertainty += telemetry.packet_loss * 80.0
    if sync_delta_s > 0.5:
        uncertainty += sync_delta_s * 20.0
    if clamped:
        uncertainty += HORIZON_CLAMP_PENALTY_M

    meters_per_lon = max(1.0, METERS_PER_DEG_LAT * math.cos(math.radians(telemetry.lat)))
    return GeoEstimate(
        estimated_lat=float(telemetry.lat + north / METERS_PER_DEG_LAT),
        estimated_lon=float(telemetry.lon + east / meters_per_lon),
        geo_uncertainty_m=float(round(uncertainty, 2)),
        method="pinhole_ray_ground_plane_clamped" if clamped else "pinhole_ray_ground_plane",
    )


def _undistort(xd: float, yd: float, k1: float, k2: float) -> tuple[float, float]:
    """Invert the radial model x_d = x_u * (1 + k1 r^2 + k2 r^4) by fixed-point iteration."""
    if k1 == 0.0 and k2 == 0.0:
        return xd, yd
    xu, yu = xd, yd
    for _ in range(_UNDISTORT_ITERATIONS):
        r2 = xu * xu + yu * yu
        scale = 1.0 + k1 * r2 + k2 * r2 * r2
        xu, yu = xd / scale, yd / scale
    return xu, yu


def _camera_ray(xf: float, yf: float, azimuth_deg: float, elevation_deg: float, roll_deg: float) -> Vec3:
    """Ray direction (east, north, up) through focal-plane point (xf right, yf down)."""
    az, el, rho = math.radians(azimuth_deg), math.radians(elevation_deg), math.radians(roll_deg)
    boresight = (math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el))
    right = (math.cos(az), -math.sin(az), 0.0)
    up = _cross(right, boresight)
    rolled_right = tuple(r * math.cos(rho) + u * math.sin(rho) for r, u in zip(right, up))
    rolled_up = tuple(u * math.cos(rho) - r * math.sin(rho) for r, u in zip(right, up))
    return tuple(b + xf * r - yf * u for b, r, u in zip(boresight, rolled_right, rolled_up))  # type: ignore[return-value]


def _ground_offset(
    xf: float, yf: float, azimuth_deg: float, elevation_deg: float, roll_deg: float, alt_m: float
) -> tuple[float, float, bool]:
    dx, dy, dz = _camera_ray(xf, yf, azimuth_deg, elevation_deg, roll_deg)
    horizontal = math.hypot(dx, dy)
    clamped = False
    max_dz = -math.tan(math.radians(MIN_DOWNWARD_DEG)) * horizontal
    if dz > max_dz:
        dz, clamped = max_dz, True
    scale = alt_m / -dz
    return dx * scale, dy * scale, clamped


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

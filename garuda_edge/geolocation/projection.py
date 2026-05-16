from __future__ import annotations

import math

from garuda_edge.geolocation.camera_model import CameraModel
from garuda_edge.ingestion.schemas import GeoEstimate, TelemetryPacket


def bbox_center(bbox_xyxy: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def project_bbox_to_ground(
    bbox_xyxy: list[float],
    image_width: int,
    image_height: int,
    telemetry: TelemetryPacket,
    camera: CameraModel | None = None,
    sync_delta_s: float = 0.0,
) -> GeoEstimate:
    """Approximate camera-ray to flat-ground projection.

    This is an honest ISR cueing estimate, not survey-grade geolocation.
    It treats the target as lying on a local tangent ground plane.
    """

    camera = camera or CameraModel()
    cx, cy = bbox_center(bbox_xyxy)
    nx = (cx - image_width / 2.0) / max(1.0, image_width / 2.0)
    ny = (cy - image_height / 2.0) / max(1.0, image_height / 2.0)

    bearing_deg = telemetry.yaw_deg + telemetry.gimbal_pan_deg + nx * (camera.horizontal_fov_deg / 2.0)
    tilt_deg = telemetry.gimbal_tilt_deg + telemetry.pitch_deg - ny * (camera.vertical_fov_deg / 2.0)
    downward_deg = max(5.0, min(89.0, abs(tilt_deg)))
    ground_range_m = telemetry.alt_m / max(0.087, math.tan(math.radians(downward_deg)))

    bearing = math.radians(bearing_deg)
    north_m = ground_range_m * math.cos(bearing)
    east_m = ground_range_m * math.sin(bearing)

    meters_per_lat = 111_320.0
    meters_per_lon = max(1.0, 111_320.0 * math.cos(math.radians(telemetry.lat)))
    est_lat = telemetry.lat + north_m / meters_per_lat
    est_lon = telemetry.lon + east_m / meters_per_lon

    uncertainty = 8.0 + ground_range_m * 0.08 + abs(telemetry.roll_deg) * 0.5 + abs(telemetry.pitch_deg) * 0.5
    if telemetry.gps_quality == "medium":
        uncertainty += 12.0
    elif telemetry.gps_quality == "poor":
        uncertainty += 28.0
    if telemetry.vibration == "medium":
        uncertainty += 8.0
    elif telemetry.vibration == "high":
        uncertainty += 22.0
    if telemetry.alt_m > 100:
        uncertainty += (telemetry.alt_m - 100) * 0.2
    if downward_deg < 25:
        uncertainty += 25.0
    if telemetry.packet_loss > 0.1:
        uncertainty += telemetry.packet_loss * 80.0
    if sync_delta_s > 0.5:
        uncertainty += sync_delta_s * 20.0

    return GeoEstimate(
        estimated_lat=float(est_lat),
        estimated_lon=float(est_lon),
        geo_uncertainty_m=float(round(uncertainty, 2)),
        method="approx_camera_ray_ground_plane",
    )

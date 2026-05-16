from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraModel:
    horizontal_fov_deg: float = 70.0
    vertical_fov_deg: float = 43.0
    default_alt_m: float = 80.0
    default_gimbal_tilt_deg: float = -45.0

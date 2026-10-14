from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera on a stabilised gimbal.

    Angle convention: boresight azimuth = yaw + gimbal_pan (clockwise from north),
    elevation = gimbal_tilt + pitch (negative is below the horizon), and camera roll
    about the boresight = gimbal_roll (0 when not reported). k1/k2 are radial
    distortion coefficients in normalised focal-plane units (0 = ideal pinhole).
    """

    horizontal_fov_deg: float = 70.0
    vertical_fov_deg: float = 43.0
    default_alt_m: float = 80.0
    default_gimbal_tilt_deg: float = -45.0
    k1: float = 0.0
    k2: float = 0.0
    attitude_sigma_deg: float = 1.0
    altitude_sigma_m: float = 2.0

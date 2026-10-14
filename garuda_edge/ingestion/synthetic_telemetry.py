"""Physically consistent synthetic drone telemetry for demos and tests.

Continuous fields follow mean-reverting random walks (Ornstein-Uhlenbeck) so values
drift smoothly frame to frame; degraded conditions arrive as persistent episodes rather
than independent per-row draws. The telemetry is not derived from any video: drone
motion here makes stationary ground objects appear to move on the map.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any

FIELDS = [
    "timestamp_s",
    "lat",
    "lon",
    "alt_m",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "gimbal_pan_deg",
    "gimbal_tilt_deg",
    "battery_pct",
    "packet_loss",
    "latency_ms",
    "vibration",
    "gps_quality",
    "mission_phase",
    "sensor_health",
    "compute_temp_c",
    "dropped_frames",
]
PROFILES = ("hover", "transect")

_START_LAT, _START_LON = 28.6139, 77.2090
_METERS_PER_DEG_LAT = 111_320.0
_TRANSECT_SPEED_MPS = 4.0
_TRANSECT_LEG_M = 150.0
_TURN_RATE_DEG_S = 30.0  # 90 degree turn over 3 s


class _OU:
    """Mean-reverting random walk with stationary standard deviation `std` and time constant `tau_s`."""

    def __init__(self, rng: random.Random, mean: float, std: float, tau_s: float):
        self.rng, self.mean, self.std, self.tau_s = rng, mean, std, tau_s
        self.value = mean + rng.gauss(0.0, std)

    def step(self, dt: float) -> float:
        decay = math.exp(-dt / self.tau_s)
        self.value = self.mean + (self.value - self.mean) * decay + self.rng.gauss(0.0, self.std * math.sqrt(1 - decay**2))
        return self.value


class _Episodes:
    """Condition that starts at `rate_per_s` and lasts a uniform random duration."""

    def __init__(self, rng: random.Random, rate_per_s: float, duration_s: tuple[float, float]):
        self.rng, self.rate_per_s, self.duration_s = rng, rate_per_s, duration_s
        self.remaining_s = 0.0

    def step(self, dt: float) -> bool:
        if self.remaining_s > 0:
            self.remaining_s -= dt
        elif self.rng.random() < self.rate_per_s * dt:
            self.remaining_s = self.rng.uniform(*self.duration_s)
        return self.remaining_s > 0


def generate_rows(
    frames: int,
    fps: float,
    profile: str = "hover",
    seed: int | None = 7,
    mission_phase: str = "area_search",
) -> list[dict[str, Any]]:
    if profile not in PROFILES:
        raise ValueError(f"Unknown telemetry profile {profile!r}; expected one of {PROFILES}")
    rng = random.Random(seed)
    dt = 1.0 / fps

    east_jitter, north_jitter = _OU(rng, 0.0, 0.5, 10.0), _OU(rng, 0.0, 0.5, 10.0)
    alt = _OU(rng, 90.0, 0.3, 8.0)
    yaw_jitter = _OU(rng, 0.0, 1.0, 6.0)
    roll, pitch = _OU(rng, 0.0, 1.0, 2.0), _OU(rng, 0.0, 1.0, 2.0)
    gimbal_tilt, gimbal_pan = _OU(rng, -60.0, 0.3, 5.0), _OU(rng, 0.0, 0.5, 5.0)
    base_loss, base_latency = _OU(rng, 0.02, 0.01, 5.0), _OU(rng, 110.0, 15.0, 5.0)
    temp = _OU(rng, 62.0, 4.0, 30.0)

    link_spike = _Episodes(rng, 1 / 20, (0.5, 2.0))
    gps_medium, gps_poor = _Episodes(rng, 1 / 60, (3.0, 10.0)), _Episodes(rng, 1 / 300, (2.0, 5.0))
    vib_medium, vib_high = _Episodes(rng, 1 / 40, (2.0, 6.0)), _Episodes(rng, 1 / 400, (1.0, 3.0))
    sensor_degraded = _Episodes(rng, 1 / 120, (2.0, 8.0))

    east, north, heading = 0.0, 0.0, 45.0
    leg_m, turn_remaining_deg = 0.0, 0.0
    meters_per_deg_lon = _METERS_PER_DEG_LAT * math.cos(math.radians(_START_LAT))

    rows: list[dict[str, Any]] = []
    for frame in range(frames):
        t = frame / fps
        if profile == "transect" and frame > 0:
            if turn_remaining_deg > 0:
                turn = min(turn_remaining_deg, _TURN_RATE_DEG_S * dt)
                heading = (heading + turn) % 360
                turn_remaining_deg -= turn
            east += _TRANSECT_SPEED_MPS * dt * math.sin(math.radians(heading))
            north += _TRANSECT_SPEED_MPS * dt * math.cos(math.radians(heading))
            leg_m += _TRANSECT_SPEED_MPS * dt
            if leg_m >= _TRANSECT_LEG_M:
                leg_m, turn_remaining_deg = 0.0, 90.0

        spike = link_spike.step(dt)
        loss_base, latency_base = base_loss.step(dt), base_latency.step(dt)
        medium, poor = gps_medium.step(dt), gps_poor.step(dt)
        vib_m, vib_h = vib_medium.step(dt), vib_high.step(dt)
        rows.append(
            {
                "timestamp_s": round(t, 3),
                "lat": round(_START_LAT + (north + north_jitter.step(dt)) / _METERS_PER_DEG_LAT, 8),
                "lon": round(_START_LON + (east + east_jitter.step(dt)) / meters_per_deg_lon, 8),
                "alt_m": round(alt.step(dt), 2),
                "roll_deg": round(roll.step(dt), 2),
                "pitch_deg": round(pitch.step(dt), 2),
                "yaw_deg": round((heading + yaw_jitter.step(dt)) % 360, 2),
                "gimbal_pan_deg": round(gimbal_pan.step(dt), 2),
                "gimbal_tilt_deg": round(gimbal_tilt.step(dt), 2),
                "battery_pct": round(max(15.0, 100 - t * 0.02), 2),
                "packet_loss": round(rng.uniform(0.12, 0.35) if spike else min(0.06, max(0.0, loss_base)), 3),
                "latency_ms": round(rng.uniform(300, 650) if spike else max(40.0, latency_base), 1),
                "vibration": "high" if vib_h else "medium" if vib_m else "low",
                "gps_quality": "poor" if poor else "medium" if medium else "good",
                "mission_phase": mission_phase,
                "sensor_health": "degraded" if sensor_degraded.step(dt) else "nominal",
                "compute_temp_c": round(temp.step(dt), 1),
                "dropped_frames": rng.choice([0, 1, 2, 3]) if spike else 0,
            }
        )
    return rows


def write_telemetry_csv(rows: list[dict[str, Any]], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from garuda_edge.geolocation.projection import METERS_PER_DEG_LAT
from garuda_edge.ingestion.schemas import GeoEstimate
from garuda_edge.tracking.kalman import ConstantVelocityKalman

GATE_CHI2_2DOF_999 = 13.8


@dataclass
class _GeoState:
    origin_lat: float
    origin_lon: float
    kf: ConstantVelocityKalman
    updates: int = 1
    consecutive_rejects: int = 0


class GeoTrackFilter:
    """Per-track constant-velocity Kalman filter on projected ground positions.

    Each track is filtered in local east/north metres around its first fix. Raw fixes
    farther than the chi-square gate from the prediction are rejected; after
    `max_consecutive_rejects` in a row the track restarts at the new fix. Reported
    uncertainty never drops below the drone's GPS sigma, because that error is a bias
    shared by consecutive fixes and does not average out.
    """

    def __init__(
        self,
        accel_std_mps2: float = 3.0,
        initial_velocity_std_mps: float = 15.0,
        gate_chi2: float = GATE_CHI2_2DOF_999,
        max_consecutive_rejects: int = 3,
        min_updates_for_speed: int = 5,
        min_speed_for_heading_mps: float = 0.5,
    ):
        self.accel_std_mps2 = accel_std_mps2
        self.initial_velocity_std_mps = initial_velocity_std_mps
        self.gate_chi2 = gate_chi2
        self.max_consecutive_rejects = max_consecutive_rejects
        self.min_updates_for_speed = min_updates_for_speed
        self.min_speed_for_heading_mps = min_speed_for_heading_mps
        self._tracks: dict[str, _GeoState] = {}

    def update(self, track_id: str, raw: GeoEstimate, timestamp_s: float, gps_sigma_m: float) -> GeoEstimate:
        state = self._tracks.get(track_id)
        if state is None:
            state = self._start(track_id, raw, timestamp_s)
        else:
            state.kf.predict(timestamp_s)
            z = self._to_local(state, raw.estimated_lat, raw.estimated_lon)
            R = np.eye(2) * raw.geo_uncertainty_m**2
            if state.kf.mahalanobis_sq(z, R) > self.gate_chi2:
                state.consecutive_rejects += 1
                if state.consecutive_rejects >= self.max_consecutive_rejects:
                    state = self._start(track_id, raw, timestamp_s)
            else:
                state.kf.update(z, R)
                state.updates += 1
                state.consecutive_rejects = 0
        return self._estimate(state, gps_sigma_m)

    def prune(self, live_track_ids: Iterable[str]) -> None:
        live = set(live_track_ids)
        for track_id in [tid for tid in self._tracks if tid not in live]:
            del self._tracks[track_id]

    def __contains__(self, track_id: str) -> bool:
        return track_id in self._tracks

    def _start(self, track_id: str, raw: GeoEstimate, timestamp_s: float) -> _GeoState:
        accel = np.full(2, self.accel_std_mps2)
        kf = ConstantVelocityKalman(
            np.zeros(2), raw.geo_uncertainty_m, self.initial_velocity_std_mps, timestamp_s, lambda _x: accel
        )
        state = _GeoState(raw.estimated_lat, raw.estimated_lon, kf)
        self._tracks[track_id] = state
        return state

    def _estimate(self, state: _GeoState, gps_sigma_m: float) -> GeoEstimate:
        east, north = (float(v) for v in state.kf.position)
        lat = state.origin_lat + north / METERS_PER_DEG_LAT
        lon = state.origin_lon + east / _meters_per_deg_lon(state.origin_lat)
        sigma = math.sqrt(float(np.linalg.eigvalsh(state.kf.P[:2, :2]).max()))

        speed = heading = None
        if state.updates >= self.min_updates_for_speed:
            ve, vn = (float(v) for v in state.kf.velocity)
            speed = round(math.hypot(ve, vn), 2)
            # Noisy fixes give a parked car a small non-zero speed; only report a heading
            # when the motion is clearly larger than the velocity uncertainty.
            velocity_sigma = math.sqrt(float(np.linalg.eigvalsh(state.kf.P[2:, 2:]).max()))
            if speed >= max(self.min_speed_for_heading_mps, 2.0 * velocity_sigma):
                heading = round(math.degrees(math.atan2(ve, vn)) % 360.0, 1)
        return GeoEstimate(
            estimated_lat=lat,
            estimated_lon=lon,
            geo_uncertainty_m=round(max(sigma, gps_sigma_m), 2),
            method="kalman_geo_track",
            speed_mps=speed,
            heading_deg=heading,
        )

    @staticmethod
    def _to_local(state: _GeoState, lat: float, lon: float) -> np.ndarray:
        return np.array(
            [
                (lon - state.origin_lon) * _meters_per_deg_lon(state.origin_lat),
                (lat - state.origin_lat) * METERS_PER_DEG_LAT,
            ]
        )


def _meters_per_deg_lon(lat: float) -> float:
    return max(1.0, METERS_PER_DEG_LAT * math.cos(math.radians(lat)))

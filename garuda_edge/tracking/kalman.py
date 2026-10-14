from __future__ import annotations

from typing import Callable

import numpy as np

# Noise is scaled by box height (DeepSORT convention) so small distant objects and large
# close ones get comparable relative uncertainty. Units are pixels and seconds, so the same
# filter can step one video frame (~33 ms) or one 1000 Hz output tick (1 ms).
_MEAS_STD_PER_H = 1.0 / 20.0
_INIT_POS_STD_PER_H = 1.0 / 10.0
_INIT_VEL_STD_PER_H = 5.0
_ACCEL_STD_PER_H = 4.0  # px/s^2 per px of height, centre position axes
_SIZE_ACCEL_STD_PER_H = 1.0  # px/s^2 per px of height, width/height axes
_MIN_SIZE_PX = 1.0


class ConstantVelocityKalman:
    """Constant-velocity Kalman filter over `dim` position axes, with time steps in seconds.

    State is [p_1..p_dim, v_1..v_dim]; the measurement is the position vector.
    `accel_std` maps the propagated state to per-axis white-noise acceleration std.
    """

    def __init__(
        self,
        position: np.ndarray,
        position_std: np.ndarray | float,
        velocity_std: np.ndarray | float,
        timestamp_s: float,
        accel_std: Callable[[np.ndarray], np.ndarray],
    ):
        self.dim = len(position)
        self.x = np.concatenate([np.asarray(position, dtype=float), np.zeros(self.dim)])
        pos_var = np.broadcast_to(np.asarray(position_std, dtype=float) ** 2, (self.dim,))
        vel_var = np.broadcast_to(np.asarray(velocity_std, dtype=float) ** 2, (self.dim,))
        self.P = np.diag(np.concatenate([pos_var, vel_var]))
        self.t = float(timestamp_s)
        self._accel_std = accel_std
        self._H = np.hstack([np.eye(self.dim), np.zeros((self.dim, self.dim))])

    @property
    def position(self) -> np.ndarray:
        return self.x[: self.dim]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[self.dim :]

    def predict(self, timestamp_s: float) -> None:
        """Advance the state in place to timestamp_s."""
        dt = float(timestamp_s) - self.t
        if dt <= 0.0:
            return
        F = self._transition(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._process_noise(dt)
        self.t = float(timestamp_s)

    def peek(self, timestamp_s: float) -> np.ndarray:
        """Predicted position at timestamp_s without changing the filter state."""
        dt = float(timestamp_s) - self.t
        if dt <= 0.0:
            return self.position.copy()
        return (self._transition(dt) @ self.x)[: self.dim]

    def mahalanobis_sq(self, z: np.ndarray, R: np.ndarray) -> float:
        y = np.asarray(z, dtype=float) - self._H @ self.x
        S = self._H @ self.P @ self._H.T + R
        return float(y @ np.linalg.solve(S, y))

    def update(self, z: np.ndarray, R: np.ndarray) -> None:
        y = np.asarray(z, dtype=float) - self._H @ self.x
        S = self._H @ self.P @ self._H.T + R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2 * self.dim) - K @ self._H) @ self.P

    def _transition(self, dt: float) -> np.ndarray:
        F = np.eye(2 * self.dim)
        F[: self.dim, self.dim :] = np.eye(self.dim) * dt
        return F

    def _process_noise(self, dt: float) -> np.ndarray:
        # White-noise acceleration: G = [dt^2/2, dt] per axis.
        q = np.asarray(self._accel_std(self.x), dtype=float) ** 2
        d = self.dim
        Q = np.zeros((2 * d, 2 * d))
        Q[:d, :d] = np.diag(q * dt**4 / 4.0)
        Q[:d, d:] = np.diag(q * dt**3 / 2.0)
        Q[d:, :d] = np.diag(q * dt**3 / 2.0)
        Q[d:, d:] = np.diag(q * dt**2)
        return Q


def _xyxy_to_cxcywh(bbox_xyxy: list[float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    w = max(_MIN_SIZE_PX, x2 - x1)
    h = max(_MIN_SIZE_PX, y2 - y1)
    return np.array([x1 + (x2 - x1) / 2.0, y1 + (y2 - y1) / 2.0, w, h], dtype=float)


def _cxcywh_to_xyxy(state: np.ndarray) -> list[float]:
    cx, cy = float(state[0]), float(state[1])
    w = max(_MIN_SIZE_PX, float(state[2]))
    h = max(_MIN_SIZE_PX, float(state[3]))
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def _box_accel_std(x: np.ndarray) -> np.ndarray:
    h = max(_MIN_SIZE_PX, float(x[3]))
    return np.array([_ACCEL_STD_PER_H, _ACCEL_STD_PER_H, _SIZE_ACCEL_STD_PER_H, _SIZE_ACCEL_STD_PER_H]) * h


class BoxKalmanFilter:
    """Constant-velocity filter on an image box: state [cx, cy, w, h, vx, vy, vw, vh] in px and px/s."""

    def __init__(self, bbox_xyxy: list[float], timestamp_s: float):
        z = _xyxy_to_cxcywh(bbox_xyxy)
        h = z[3]
        self._kf = ConstantVelocityKalman(z, _INIT_POS_STD_PER_H * h, _INIT_VEL_STD_PER_H * h, timestamp_s, _box_accel_std)

    @property
    def x(self) -> np.ndarray:
        return self._kf.x

    @property
    def P(self) -> np.ndarray:
        return self._kf.P

    @property
    def t(self) -> float:
        return self._kf.t

    def predict(self, timestamp_s: float) -> None:
        self._kf.predict(timestamp_s)

    def peek(self, timestamp_s: float) -> list[float]:
        """Predicted box at timestamp_s without changing the filter state."""
        return _cxcywh_to_xyxy(self._kf.peek(timestamp_s))

    def update(self, bbox_xyxy: list[float]) -> None:
        meas_std = max(_MIN_SIZE_PX, _MEAS_STD_PER_H * max(_MIN_SIZE_PX, float(self._kf.x[3])))
        self._kf.update(_xyxy_to_cxcywh(bbox_xyxy), np.eye(4) * meas_std**2)

    def bbox_xyxy(self) -> list[float]:
        return _cxcywh_to_xyxy(self._kf.x)

    def velocity_px_s(self) -> list[float]:
        return [float(self._kf.x[4]), float(self._kf.x[5])]

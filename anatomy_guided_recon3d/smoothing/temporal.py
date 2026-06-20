from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OneEuroConfig:
    """One Euro filter parameters.

    Input shape/unit/frame: cutoff frequencies are Hz; beta is unitless.
    Return type: immutable config.
    """

    min_cutoff: float = 1.0
    beta: float = 0.02
    d_cutoff: float = 1.0


def _alpha(cutoff: float, dt_s: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt_s)


class OneEuroFilter3D:
    """Low-latency One Euro filter for 3D world points."""

    def __init__(self, config: OneEuroConfig | None = None) -> None:
        """Create a 3D One Euro filter.

        Input shape/unit/frame: observations are 3-vectors in world metres with
        nanosecond timestamps.
        Return type: ``OneEuroFilter3D``.
        """
        self.config = config or OneEuroConfig()
        self._last_t_ns: int | None = None
        self._last_x: np.ndarray | None = None
        self._last_dx: np.ndarray | None = None

    def reset(self) -> None:
        """Reset filter state.

        Input shape/unit/frame: no input.
        Return type: ``None``.
        """
        self._last_t_ns = None
        self._last_x = None
        self._last_dx = None

    def filter(self, x: np.ndarray, timestamp_ns: int) -> np.ndarray:
        """Filter one 3D world point.

        Input shape/unit/frame: x is 3-vector in world metres; timestamp_ns is
        monotonic nanoseconds.
        Return type: filtered 3-vector ndarray in world metres.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (3,):
            raise ValueError("x must be shape (3,)")
        if self._last_t_ns is None or self._last_x is None:
            self._last_t_ns = timestamp_ns
            self._last_x = x.copy()
            self._last_dx = np.zeros(3, dtype=np.float64)
            return x.copy()
        dt_s = max((timestamp_ns - self._last_t_ns) / 1e9, 1e-6)
        dx = (x - self._last_x) / dt_s
        a_d = _alpha(self.config.d_cutoff, dt_s)
        dx_hat = a_d * dx + (1.0 - a_d) * (self._last_dx if self._last_dx is not None else dx)
        cutoff = self.config.min_cutoff + self.config.beta * np.linalg.norm(dx_hat)
        a = _alpha(float(cutoff), dt_s)
        x_hat = a * x + (1.0 - a) * self._last_x
        self._last_t_ns = timestamp_ns
        self._last_x = x_hat.copy()
        self._last_dx = dx_hat.copy()
        return x_hat


class ConstantVelocityKalmanFilter3D:
    """Optional constant-velocity Kalman filter for 3D world points."""

    def __init__(self, process_noise: float = 1e-4) -> None:
        """Create a 3D constant-velocity Kalman filter.

        Input shape/unit/frame: process_noise is metres squared per second model
        noise; observations are world metres.
        Return type: ``ConstantVelocityKalmanFilter3D``.
        """
        self.process_noise = process_noise
        self._state: np.ndarray | None = None
        self._cov: np.ndarray | None = None
        self._last_t_ns: int | None = None

    def update(self, z: np.ndarray, R: np.ndarray, timestamp_ns: int) -> tuple[np.ndarray, np.ndarray]:
        """Update filter with one 3D observation.

        Input shape/unit/frame: z is world metres; R is 3x3 observation covariance
        in metre squared; timestamp is nanoseconds.
        Return type: ``(position, covariance)``.
        """
        z = np.asarray(z, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)
        if z.shape != (3,) or R.shape != (3, 3):
            raise ValueError("z must be (3,) and R must be (3,3)")
        if self._state is None:
            self._state = np.concatenate([z, np.zeros(3, dtype=np.float64)])
            self._cov = np.eye(6, dtype=np.float64) * 1e-3
            self._last_t_ns = timestamp_ns
            return z.copy(), R.copy()
        dt = max((timestamp_ns - (self._last_t_ns or timestamp_ns)) / 1e9, 1e-6)
        F = np.eye(6, dtype=np.float64)
        F[:3, 3:] = np.eye(3) * dt
        Q = np.eye(6, dtype=np.float64) * self.process_noise
        state_pred = F @ self._state
        cov_pred = F @ (self._cov if self._cov is not None else np.eye(6)) @ F.T + Q
        H = np.zeros((3, 6), dtype=np.float64)
        H[:3, :3] = np.eye(3)
        S = H @ cov_pred @ H.T + R
        K = cov_pred @ H.T @ np.linalg.inv(S)
        self._state = state_pred + K @ (z - H @ state_pred)
        self._cov = (np.eye(6) - K @ H) @ cov_pred
        self._last_t_ns = timestamp_ns
        return self._state[:3].copy(), self._cov[:3, :3].copy()


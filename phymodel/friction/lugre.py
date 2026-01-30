from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class LugreFriction:
    fc: Sequence[float]
    fs: Sequence[float]
    vs: Sequence[float]
    sigma0: Sequence[float]
    sigma1: Sequence[float]
    sigma2: Sequence[float]
    v_eps: float = 1e-6
    g_eps: float = 1e-9

    def __post_init__(self) -> None:
        fc = np.asarray(self.fc, dtype=float)
        fs = np.asarray(self.fs, dtype=float)
        vs = np.asarray(self.vs, dtype=float)
        s0 = np.asarray(self.sigma0, dtype=float)
        s1 = np.asarray(self.sigma1, dtype=float)
        s2 = np.asarray(self.sigma2, dtype=float)
        shapes = {fc.shape, fs.shape, vs.shape, s0.shape, s1.shape, s2.shape}
        if len(shapes) != 1:
            raise ValueError("All LuGre params must have the same shape.")
        self._n = fc.size
        self._z = np.zeros_like(fc)

    @property
    def z(self) -> np.ndarray:
        return self._z

    def reset(self) -> None:
        self._z[:] = 0.0

    def _g(self, v: np.ndarray) -> np.ndarray:
        fc = np.asarray(self.fc, dtype=float)
        fs = np.asarray(self.fs, dtype=float)
        vs = np.asarray(self.vs, dtype=float)

        v = np.asarray(v, dtype=float)
        vv = v / np.where(np.abs(vs) < self.v_eps, self.v_eps, vs)
        return fc + (fs - fc) * np.exp(-(vv * vv))

    def step(self, v: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance internal state and return (tau, z, z_dot)."""
        v = np.asarray(v, dtype=float)
        if v.size != self._n:
            raise ValueError("v shape mismatch with LuGre state.")
        if dt <= 0:
            raise ValueError("dt must be positive.")

        g = self._g(v)
        g = np.maximum(g, self.g_eps)

        z = self._z
        z_dot = v - (np.abs(v) / g) * z
        z_next = z + dt * z_dot
        self._z = z_next

        sigma0 = np.asarray(self.sigma0, dtype=float)
        sigma1 = np.asarray(self.sigma1, dtype=float)
        sigma2 = np.asarray(self.sigma2, dtype=float)
        tau = sigma0 * z_next + sigma1 * z_dot + sigma2 * v
        return tau, z_next, z_dot


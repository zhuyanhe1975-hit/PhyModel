from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class ViscousCoulombFriction:
    coulomb: Sequence[float]
    viscous: Sequence[float]
    v_eps: float = 1e-4

    def torque(self, qd: np.ndarray) -> np.ndarray:
        qd = np.asarray(qd, dtype=float)
        coulomb = np.asarray(self.coulomb, dtype=float)
        viscous = np.asarray(self.viscous, dtype=float)
        if coulomb.shape != qd.shape or viscous.shape != qd.shape:
            raise ValueError("Friction params must match qd shape.")

        sign = np.where(np.abs(qd) < self.v_eps, 0.0, np.sign(qd))
        return coulomb * sign + viscous * qd


from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class JointFrictionParams:
    """Baseline viscous + Coulomb friction.

    tau = Fc * sign(v) + Fv * v
    """

    coulomb: Sequence[float]
    viscous: Sequence[float]


@dataclass(frozen=True)
class LugreParams:
    """LuGre friction params (per joint).

    g(v) = Fc + (Fs - Fc) * exp(-(v/vs)^2)
    z_dot = v - |v|/g(v) * z
    tau = sigma0*z + sigma1*z_dot + sigma2*v
    """

    fc: Sequence[float]
    fs: Sequence[float]
    vs: Sequence[float]
    sigma0: Sequence[float]
    sigma1: Sequence[float]
    sigma2: Sequence[float]


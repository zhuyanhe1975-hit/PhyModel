from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np


@dataclass
class SimLog:
    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    tau_cmd: np.ndarray
    tau_fric: np.ndarray

    def to_npz_dict(self) -> Dict[str, np.ndarray]:
        return {
            "t": self.t,
            "q": self.q,
            "qd": self.qd,
            "tau_cmd": self.tau_cmd,
            "tau_fric": self.tau_fric,
        }


def run_torque_demo(
    mjm: "object",
    mjd: "object",
    duration: float,
    tau_cmd_fn: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    tau_fric_fn: Optional[Callable[[float, np.ndarray, np.ndarray, float], np.ndarray]] = None,
    qpos0: Optional[np.ndarray] = None,
) -> SimLog:
    """Run a simple torque-driven simulation with optional external friction torque.

    The friction torque is applied via `data.qfrc_applied` (added to the system).
    Convention: tau_fric is the torque contributed by the friction model (typically opposing motion).
    """
    import mujoco

    dt = float(mjm.opt.timestep)
    if dt <= 0:
        raise ValueError("Model timestep must be positive.")
    steps = int(np.ceil(duration / dt))
    if steps <= 0:
        raise ValueError("duration too small.")

    if qpos0 is not None:
        mjd.qpos[:] = qpos0
        mujoco.mj_forward(mjm, mjd)

    nq = int(mjm.nq)
    t = np.zeros(steps, dtype=float)
    q = np.zeros((steps, nq), dtype=float)
    qd = np.zeros((steps, nq), dtype=float)
    tau_cmd = np.zeros((steps, nq), dtype=float)
    tau_fric = np.zeros((steps, nq), dtype=float)

    for k in range(steps):
        now = float(mjd.time)
        t[k] = now
        q[k] = mjd.qpos.copy()
        qd[k] = mjd.qvel.copy()

        cmd = np.asarray(tau_cmd_fn(now, mjd.qpos, mjd.qvel), dtype=float)
        if cmd.shape != (nq,):
            raise ValueError("tau_cmd_fn must return shape (nq,).")
        tau_cmd[k] = cmd

        fric = np.zeros(nq, dtype=float)
        if tau_fric_fn is not None:
            fric = np.asarray(tau_fric_fn(now, mjd.qpos, mjd.qvel, dt), dtype=float)
            if fric.shape != (nq,):
                raise ValueError("tau_fric_fn must return shape (nq,).")
        tau_fric[k] = fric

        mjd.qfrc_applied[:] = 0.0
        mjd.qfrc_applied[:] = cmd + fric
        mujoco.mj_step(mjm, mjd)

    return SimLog(t=t, q=q, qd=qd, tau_cmd=tau_cmd, tau_fric=tau_fric)


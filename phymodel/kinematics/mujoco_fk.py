"""MuJoCo forward kinematics helper for TCP reference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class MujocoFK:
    mjm: "object"
    mjd: "object"
    body_name: str = "link_6"

    def tcp_from_q(self, q: List[float]) -> List[float]:
        import mujoco

        if len(q) != self.mjm.nq:
            return [0.0, 0.0, 0.0]
        self.mjd.qpos[:] = q
        mujoco.mj_forward(self.mjm, self.mjd)
        try:
            body_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        except Exception:
            return [0.0, 0.0, 0.0]
        return self.mjd.xpos[body_id].tolist()

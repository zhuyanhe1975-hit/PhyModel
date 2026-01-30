#!/usr/bin/env python3
"""MuJoCo viewer friction demo (interactive).

Notes:
- Requires a GUI-capable environment.
- Applies friction torque via qfrc_applied each step, same as run_friction_demo.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", default="models/er15-1400.mjcf.xml")
    ap.add_argument("--joint", type=int, default=1, help="1-based joint index for excitation")
    ap.add_argument("--amp", type=float, default=30.0)
    ap.add_argument("--freq", type=float, default=0.5)
    ap.add_argument("--model", choices=["vc", "lugre"], default="lugre")
    ap.add_argument("--params", default="params/er15-1400.params.json", help="Central params file (friction, etc.)")
    ap.add_argument("--timestep", type=float, default=None, help="Override MuJoCo timestep [s] (takes precedence)")
    ap.add_argument("--no-friction", action="store_true")
    args = ap.parse_args()

    import mujoco
    import mujoco.viewer

    mjm = mujoco.MjModel.from_xml_path(str(Path(args.mjcf)))
    mjd = mujoco.MjData(mjm)

    nq = int(mjm.nq)
    j = int(args.joint) - 1
    if j < 0 or j >= nq:
        raise SystemExit(f"--joint out of range: 1..{nq}")

    payload = {}
    try:
        from phymodel.params.io import load_params

        payload = load_params(args.params)
    except FileNotFoundError:
        print(f"warning: params file not found: {args.params} (using built-in defaults)")

    timestep_override = None
    if isinstance(payload, dict):
        tunable = payload.get("tunable")
        if isinstance(tunable, dict):
            sim = tunable.get("sim")
            if isinstance(sim, dict) and sim.get("timestep_override") is not None:
                timestep_override = float(sim["timestep_override"])
    if args.timestep is not None:
        timestep_override = float(args.timestep)
    if timestep_override is not None:
        mjm.opt.timestep = timestep_override

    from phymodel.params.friction import friction_params_from_payload

    tau_fric_fn = None
    if not args.no_friction:
        model_params, _ = friction_params_from_payload(payload, args.model)
        if args.model == "vc":
            from phymodel.friction.viscous_coulomb import ViscousCoulombFriction

            fric_model = ViscousCoulombFriction(
                coulomb=model_params.get("coulomb", [2.0] * nq),
                viscous=model_params.get("viscous", [0.8] * nq),
                v_eps=float(model_params.get("v_eps", 1e-4)),
            )

            def tau_fric_fn(_: float, qd: np.ndarray) -> np.ndarray:
                return -fric_model.torque(qd)

        else:
            from phymodel.friction.lugre import LugreFriction

            fric_model = LugreFriction(
                fc=model_params.get("fc", [2.0] * nq),
                fs=model_params.get("fs", [6.0] * nq),
                vs=model_params.get("vs", [0.05] * nq),
                sigma0=model_params.get("sigma0", [200.0] * nq),
                sigma1=model_params.get("sigma1", [2.0] * nq),
                sigma2=model_params.get("sigma2", [0.8] * nq),
                v_eps=float(model_params.get("v_eps", 1e-6)),
                g_eps=float(model_params.get("g_eps", 1e-9)),
            )

            def tau_fric_fn(_: float, qd: np.ndarray) -> np.ndarray:
                tau, _, _ = fric_model.step(qd, dt=float(mjm.opt.timestep))
                return -tau

    def tau_cmd(t: float) -> np.ndarray:
        tau = np.zeros(nq, dtype=float)
        tau[j] = float(args.amp) * np.sin(2.0 * np.pi * float(args.freq) * t)
        return tau

    with mujoco.viewer.launch_passive(mjm, mjd) as v:
        while v.is_running():
            cmd = tau_cmd(float(mjd.time))
            fric = np.zeros(nq, dtype=float)
            if tau_fric_fn is not None:
                fric = tau_fric_fn(float(mjd.time), mjd.qvel)

            mjd.qfrc_applied[:] = 0.0
            mjd.qfrc_applied[:] = cmd + fric
            mujoco.mj_step(mjm, mjd)
            v.sync()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

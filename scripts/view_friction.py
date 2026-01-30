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
    ap.add_argument(
        "--lock-others",
        action="store_true",
        help="Lock all other joints (q=0, qd=0) to isolate the selected joint.",
    )
    ap.add_argument(
        "--plot-torque",
        action="store_true",
        help="Draw realtime applied torque curve in the viewer overlay.",
    )
    ap.add_argument("--plot-window", type=float, default=2.0, help="Torque plot time window [s].")
    ap.add_argument("--no-friction", action="store_true")
    try:
        import argcomplete  # type: ignore

        argcomplete.autocomplete(ap)
    except Exception:
        pass
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

    lock_idx = []
    if args.lock_others:
        lock_idx = [ii for ii in range(nq) if ii != j]

    fig = None
    rect = None
    if args.plot_torque:
        fig = mujoco.MjvFigure()
        mujoco.mjv_defaultFigure(fig)
        fig.flg_extend = 0
        fig.flg_barplot = 0
        fig.flg_selection = 0
        fig.title = "Applied torque"
        fig.xlabel = "t [s]"
        fig.ylabel = "tau [N*m]"
        # try set line name
        try:
            fig.linename[0] = f"tau_applied[{args.joint}]"
        except Exception:
            try:
                fig.linename[0] = f"tau_applied[{args.joint}]".encode()
            except Exception:
                pass
        fig.linepnt[0] = 0
        maxpt = int(fig.linedata.shape[1] // 2)
        tbuf = np.full((maxpt,), np.nan, dtype=float)
        ybuf = np.full((maxpt,), np.nan, dtype=float)

    with mujoco.viewer.launch_passive(mjm, mjd) as v:
        while v.is_running():
            if lock_idx:
                mjd.qpos[lock_idx] = 0.0
                mjd.qvel[lock_idx] = 0.0
                mjd.qacc[lock_idx] = 0.0
                mujoco.mj_forward(mjm, mjd)

            cmd = tau_cmd(float(mjd.time))
            fric = np.zeros(nq, dtype=float)
            if tau_fric_fn is not None:
                fric = tau_fric_fn(float(mjd.time), mjd.qvel)

            mjd.qfrc_applied[:] = 0.0
            mjd.qfrc_applied[:] = cmd + fric
            mujoco.mj_step(mjm, mjd)

            if lock_idx:
                mjd.qpos[lock_idx] = 0.0
                mjd.qvel[lock_idx] = 0.0
                mjd.qacc[lock_idx] = 0.0
                mujoco.mj_forward(mjm, mjd)

            if fig is not None:
                # update plot buffers
                now = float(mjd.time)
                tau_applied = float(mjd.qfrc_applied[j])
                tbuf[:-1] = tbuf[1:]
                ybuf[:-1] = ybuf[1:]
                tbuf[-1] = now
                ybuf[-1] = tau_applied

                valid = np.isfinite(tbuf) & np.isfinite(ybuf)
                tv = tbuf[valid]
                yv = ybuf[valid]
                n = int(tv.size)
                if n > 0:
                    # time window
                    tmax = float(tv[-1])
                    tw = max(float(args.plot_window), 1e-3)
                    tmin = max(float(tv[0]), tmax - tw)
                    mask = tv >= tmin
                    tvw = tv[mask]
                    yvw = yv[mask]
                    if tvw.size > 0:
                        # pack into linedata (x,y interleaved)
                        fig.linepnt[0] = int(tvw.size)
                        fig.linedata[0, : 2 * tvw.size : 2] = tvw
                        fig.linedata[0, 1 : 2 * tvw.size : 2] = yvw
                        # set ranges with margins
                        ymin = float(np.min(yvw))
                        ymax = float(np.max(yvw))
                        if abs(ymax - ymin) < 1e-9:
                            ymin -= 1.0
                            ymax += 1.0
                        margin = 0.1 * (ymax - ymin)
                        fig.range[0, 0] = float(tmin)
                        fig.range[0, 1] = float(tmax)
                        fig.range[1, 0] = float(ymin - margin)
                        fig.range[1, 1] = float(ymax + margin)

                        vp = v.viewport
                        rect = mujoco.MjrRect(
                            left=0,
                            bottom=0,
                            width=max(1, int(vp.width * 0.55)),
                            height=max(1, int(vp.height * 0.25)),
                        )
                        v.set_figures((rect, fig))

            v.sync()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

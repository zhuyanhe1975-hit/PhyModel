#!/usr/bin/env python3
"""Run a simple friction demo and export plots/data.

This script is intended to make friction effects visible early:
- apply a sinusoidal torque on one joint
- compare responses with/without friction
- export curves for q, qd, tau_cmd, tau_fric
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _plot(out_png: Path, t: np.ndarray, curves: dict[str, np.ndarray], title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(curves), 1, figsize=(10, 2.6 * len(curves)), sharex=True)
    if len(curves) == 1:
        axes = [axes]
    for ax, (name, y) in zip(axes, curves.items(), strict=True):
        ax.plot(t, y, linewidth=1.2)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t [s]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", default="models/er15-1400.mjcf.xml")
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--joint", type=int, default=1, help="1-based joint index for excitation")
    ap.add_argument("--amp", type=float, default=30.0, help="sinusoidal torque amplitude [N*m]")
    ap.add_argument("--freq", type=float, default=0.5, help="sinusoidal frequency [Hz]")
    ap.add_argument("--model", choices=["vc", "lugre"], default="lugre")
    ap.add_argument("--no-friction", action="store_true")
    ap.add_argument("--outdir", default="artifacts/friction_demo")
    args = ap.parse_args()

    import mujoco

    mjcf_path = Path(args.mjcf)
    mjm = mujoco.MjModel.from_xml_path(str(mjcf_path))
    mjd = mujoco.MjData(mjm)

    nq = int(mjm.nq)
    j = int(args.joint) - 1
    if j < 0 or j >= nq:
        raise SystemExit(f"--joint out of range: 1..{nq}")

    from phymodel.friction.lugre import LugreFriction
    from phymodel.friction.viscous_coulomb import ViscousCoulombFriction
    from phymodel.sim.run import run_torque_demo

    # Default params: placeholders for demonstration (must be calibrated later).
    if args.model == "vc":
        fric_model = ViscousCoulombFriction(
            coulomb=[2.0] * nq,
            viscous=[0.8] * nq,
        )

        def tau_fric_fn(_: float, __: np.ndarray, qd: np.ndarray, ___: float) -> np.ndarray:
            tau = -fric_model.torque(qd)
            tau[np.isnan(tau)] = 0.0
            return tau

        fric_meta = {"type": "viscous_coulomb", **asdict(fric_model)}
    else:
        fric_model = LugreFriction(
            fc=[2.0] * nq,
            fs=[6.0] * nq,
            vs=[0.05] * nq,
            sigma0=[200.0] * nq,
            sigma1=[2.0] * nq,
            sigma2=[0.8] * nq,
        )

        def tau_fric_fn(_: float, __: np.ndarray, qd: np.ndarray, dt: float) -> np.ndarray:
            tau, _, _ = fric_model.step(qd, dt=dt)
            tau[np.isnan(tau)] = 0.0
            return -tau

        fric_meta = {
            "type": "lugre",
            "fc": list(fric_model.fc),
            "fs": list(fric_model.fs),
            "vs": list(fric_model.vs),
            "sigma0": list(fric_model.sigma0),
            "sigma1": list(fric_model.sigma1),
            "sigma2": list(fric_model.sigma2),
            "v_eps": fric_model.v_eps,
        }

    if args.no_friction:
        tau_fric_fn_use = None
        fric_meta = {"type": "none"}
    else:
        tau_fric_fn_use = tau_fric_fn

    def tau_cmd_fn(t: float, _q: np.ndarray, _qd: np.ndarray) -> np.ndarray:
        tau = np.zeros(nq, dtype=float)
        tau[j] = float(args.amp) * np.sin(2.0 * np.pi * float(args.freq) * t)
        return tau

    log = run_torque_demo(
        mjm=mjm,
        mjd=mjd,
        duration=float(args.duration),
        tau_cmd_fn=tau_cmd_fn,
        tau_fric_fn=tau_fric_fn_use,
    )

    outdir = Path(args.outdir)
    _mkdir(outdir)

    out_npz = outdir / f"demo_{args.model}_{'nofric' if args.no_friction else 'fric'}.npz"
    np.savez(out_npz, **log.to_npz_dict(), meta=np.array([str(fric_meta)], dtype=object))

    t = log.t
    _plot(
        out_png=outdir / f"joint{args.joint}_q.png",
        t=t,
        curves={f"q[{args.joint}] [rad]": log.q[:, j]},
        title="Joint position",
    )
    _plot(
        out_png=outdir / f"joint{args.joint}_qd.png",
        t=t,
        curves={f"qd[{args.joint}] [rad/s]": log.qd[:, j]},
        title="Joint velocity",
    )
    _plot(
        out_png=outdir / f"joint{args.joint}_tau.png",
        t=t,
        curves={
            "tau_cmd [N*m]": log.tau_cmd[:, j],
            "tau_fric [N*m]": log.tau_fric[:, j],
            "tau_applied [N*m]": (log.tau_cmd[:, j] + log.tau_fric[:, j]),
        },
        title="Torques",
    )

    print(f"wrote: {out_npz}")
    print(f"wrote: {outdir / f'joint{args.joint}_q.png'}")
    print(f"wrote: {outdir / f'joint{args.joint}_qd.png'}")
    print(f"wrote: {outdir / f'joint{args.joint}_tau.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


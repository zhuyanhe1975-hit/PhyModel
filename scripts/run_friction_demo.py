#!/usr/bin/env python3
"""Run a simple friction demo and export plots/data.

This script is intended to make friction effects visible early:
- apply controlled excitations on one joint
- optionally compare responses with/without friction
- export curves for q, qd, tau_cmd, tau_fric (+ optional v_ref)
"""

from __future__ import annotations

import argparse
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


def _plot_xy(out_png: Path, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.2))
    ax.plot(x, y, linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_xy_compare(
    out_png: Path,
    x_a: np.ndarray,
    y_a: np.ndarray,
    x_b: np.ndarray,
    y_b: np.ndarray,
    xlabel: str,
    ylabel: str,
    label_a: str,
    label_b: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.2))
    ax.plot(x_a, y_a, linewidth=1.2, label=label_a)
    ax.plot(x_b, y_b, linewidth=1.2, label=label_b)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_compare(
    out_png: Path,
    t: np.ndarray,
    curves_a: dict[str, np.ndarray],
    curves_b: dict[str, np.ndarray],
    label_a: str,
    label_b: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if curves_a.keys() != curves_b.keys():
        raise ValueError("Compare plots require same curve keys.")

    fig, axes = plt.subplots(len(curves_a), 1, figsize=(10, 2.8 * len(curves_a)), sharex=True)
    if len(curves_a) == 1:
        axes = [axes]
    for ax, name in zip(axes, curves_a.keys(), strict=True):
        ax.plot(t, curves_a[name], linewidth=1.4, label=label_a)
        ax.plot(t, curves_b[name], linewidth=1.4, label=label_b)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("t [s]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _triangle_wave(phase: float) -> float:
    """Triangle wave in [-1, 1], phase in [0,1)."""
    phase = float(phase % 1.0)
    return float(4.0 * abs(phase - 0.5) - 1.0)


def _ramp_bidir_tau(t: float, amp: float, period: float, hold: float) -> float:
    """0->+amp (ramp), hold, +amp->0, 0->-amp, hold, -amp->0 within one period."""
    if period <= 0:
        raise ValueError("period must be positive")
    if hold < 0:
        raise ValueError("hold must be non-negative")
    seg_ramp = (period - 2.0 * hold) / 4.0
    if seg_ramp <= 0:
        raise ValueError("period too small for requested hold")

    tt = float(t % period)
    # boundaries
    b0 = 0.0
    b1 = b0 + seg_ramp
    b2 = b1 + hold
    b3 = b2 + seg_ramp
    b4 = b3 + seg_ramp
    b5 = b4 + hold
    b6 = b5 + seg_ramp

    if b0 <= tt < b1:  # 0 -> +amp
        u = (tt - b0) / seg_ramp
        return amp * u
    if b1 <= tt < b2:  # hold +amp
        return amp
    if b2 <= tt < b3:  # +amp -> 0
        u = (tt - b2) / seg_ramp
        return amp * (1.0 - u)
    if b3 <= tt < b4:  # 0 -> -amp
        u = (tt - b3) / seg_ramp
        return -amp * u
    if b4 <= tt < b5:  # hold -amp
        return -amp
    if b5 <= tt < b6:  # -amp -> 0
        u = (tt - b5) / seg_ramp
        return -amp * (1.0 - u)
    return 0.0


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", default="models/er15-1400.mjcf.xml")
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--joint", type=int, default=1, help="1-based joint index for excitation")
    ap.add_argument(
        "--excitation",
        choices=["sin_torque", "ramp_torque", "tri_vel", "vel_sweep"],
        default="sin_torque",
        help="Excitation type for friction visualization.",
    )
    ap.add_argument("--amp", type=float, default=30.0, help="Torque amplitude for *_torque modes [N*m]")
    ap.add_argument("--freq", type=float, default=0.5, help="Frequency for sin/triangle modes [Hz]")
    ap.add_argument("--period", type=float, default=4.0, help="Period for ramp_torque [s]")
    ap.add_argument("--hold", type=float, default=0.5, help="Hold time for ramp_torque [s]")
    ap.add_argument("--vmax", type=float, default=0.6, help="Max velocity for tri_vel/vel_sweep [rad/s]")
    ap.add_argument("--kp", type=float, default=80.0, help="Velocity-loop gain for tri_vel/vel_sweep [N*m per rad/s]")
    ap.add_argument("--tau_max", type=float, default=120.0, help="Torque clamp for velocity-loop modes [N*m]")
    ap.add_argument("--model", choices=["vc", "lugre"], default="lugre")
    ap.add_argument("--params", default="params/er15-1400.params.json", help="Central params file (friction, etc.)")
    ap.add_argument("--timestep", type=float, default=None, help="Override MuJoCo timestep [s] (takes precedence)")
    ap.add_argument(
        "--lock-others",
        action="store_true",
        help="Lock all other joints (q=0, qd=0) to isolate the selected joint.",
    )
    ap.add_argument("--no-friction", action="store_true")
    ap.add_argument(
        "--compare",
        action="store_true",
        help="Run both with/without friction and export comparison plots.",
    )
    ap.add_argument("--outdir", default="artifacts/friction_demo")
    try:
        import argcomplete  # type: ignore

        argcomplete.autocomplete(ap)
    except Exception:
        pass
    args = ap.parse_args()

    import mujoco

    mjcf_path = Path(args.mjcf)
    mjm = mujoco.MjModel.from_xml_path(str(mjcf_path))

    nq = int(mjm.nq)
    j = int(args.joint) - 1
    if j < 0 or j >= nq:
        raise SystemExit(f"--joint out of range: 1..{nq}")

    from phymodel.friction.lugre import LugreFriction
    from phymodel.friction.viscous_coulomb import ViscousCoulombFriction
    from phymodel.params.friction import friction_params_from_payload
    from phymodel.params.io import load_params
    from phymodel.sim.run import run_torque_demo

    payload = {}
    try:
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

    def build_tau_fric_fn(model_name: str):
        model_params, _ = friction_params_from_payload(payload, model_name)
        # Default params: placeholders for demonstration (must be calibrated later).
        if model_name == "vc":
            coulomb = model_params.get("coulomb", [2.0] * nq)
            viscous = model_params.get("viscous", [0.8] * nq)
            v_eps = float(model_params.get("v_eps", 1e-4))
            fric_model = ViscousCoulombFriction(
                coulomb=coulomb,
                viscous=viscous,
                v_eps=v_eps,
            )

            def tau_fric_fn(_: float, __: np.ndarray, qd: np.ndarray, ___: float) -> np.ndarray:
                tau = -fric_model.torque(qd)
                tau[np.isnan(tau)] = 0.0
                return tau

            fric_meta = {
                "type": "viscous_coulomb",
                "coulomb": list(fric_model.coulomb),
                "viscous": list(fric_model.viscous),
                "v_eps": fric_model.v_eps,
            }
            return tau_fric_fn, fric_meta

        fc = model_params.get("fc", [2.0] * nq)
        fs = model_params.get("fs", [6.0] * nq)
        vs = model_params.get("vs", [0.05] * nq)
        sigma0 = model_params.get("sigma0", [200.0] * nq)
        sigma1 = model_params.get("sigma1", [2.0] * nq)
        sigma2 = model_params.get("sigma2", [0.8] * nq)
        v_eps = float(model_params.get("v_eps", 1e-6))
        g_eps = float(model_params.get("g_eps", 1e-9))

        fric_model = LugreFriction(
            fc=fc,
            fs=fs,
            vs=vs,
            sigma0=sigma0,
            sigma1=sigma1,
            sigma2=sigma2,
            v_eps=v_eps,
            g_eps=g_eps,
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
        return tau_fric_fn, fric_meta

    def v_ref_at(t: float) -> float:
        if args.excitation == "tri_vel":
            phase = (float(t) * float(args.freq)) % 1.0
            return float(args.vmax) * _triangle_wave(phase)
        if args.excitation == "vel_sweep":
            T = max(float(args.duration), 1e-6)
            u = float(np.clip(float(t) / T, 0.0, 1.0))
            if u < 0.2:
                s = u / 0.2
            elif u < 0.4:
                s = 1.0
            elif u < 0.6:
                s = 1.0 - (u - 0.4) / 0.2
            elif u < 0.8:
                s = -(u - 0.6) / 0.2
            else:
                s = -1.0 + (u - 0.8) / 0.2
            return float(args.vmax) * float(s)
        return 0.0

    def tau_cmd_fn(t: float, _q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        tau = np.zeros(nq, dtype=float)
        if args.excitation == "sin_torque":
            tau[j] = float(args.amp) * np.sin(2.0 * np.pi * float(args.freq) * float(t))
            return tau
        if args.excitation == "ramp_torque":
            tau[j] = _ramp_bidir_tau(float(t), amp=float(args.amp), period=float(args.period), hold=float(args.hold))
            return tau
        if args.excitation in ("tri_vel", "vel_sweep"):
            v_ref = v_ref_at(float(t))
            err = v_ref - float(qd[j])
            u_cmd = float(args.kp) * err
            u_cmd = float(np.clip(u_cmd, -float(args.tau_max), float(args.tau_max)))
            tau[j] = u_cmd
            return tau
        return tau

    outdir = Path(args.outdir)
    _mkdir(outdir)

    lock_idx = []
    lock_val = []
    if args.lock_others:
        lock_idx = [ii for ii in range(nq) if ii != j]
        lock_val = [0.0 for _ in lock_idx]

    if args.compare:
        tau_fric_fn, fric_meta = build_tau_fric_fn(args.model)

        log_fric = run_torque_demo(
            mjm=mjm,
            mjd=mujoco.MjData(mjm),
            duration=float(args.duration),
            tau_cmd_fn=tau_cmd_fn,
            tau_fric_fn=tau_fric_fn,
            lock_qpos_idx=lock_idx,
            lock_qpos_val=lock_val,
        )
        log_nofric = run_torque_demo(
            mjm=mjm,
            mjd=mujoco.MjData(mjm),
            duration=float(args.duration),
            tau_cmd_fn=tau_cmd_fn,
            tau_fric_fn=None,
            lock_qpos_idx=lock_idx,
            lock_qpos_val=lock_val,
        )

        tag = f"{args.excitation}_{args.model}_joint{args.joint}"
        out_npz = outdir / f"compare_{tag}.npz"
        vref = np.array([v_ref_at(tt) for tt in log_fric.t], dtype=float)
        np.savez(
            out_npz,
            fric=log_fric.to_npz_dict(),
            nofric=log_nofric.to_npz_dict(),
            v_ref=vref,
            meta=np.array(
                [
                    str(
                        {
                            "excitation": args.excitation,
                            "joint": args.joint,
                            "mjcf": str(mjcf_path.as_posix()),
                            "timestep": float(mjm.opt.timestep),
                            "lock_others": bool(args.lock_others),
                            "friction": fric_meta,
                        }
                    )
                ],
                dtype=object,
            ),
        )

        t = log_fric.t
        vref = np.array([v_ref_at(tt) for tt in t], dtype=float)
        _plot_compare(
            out_png=outdir / f"{tag}_q_compare.png",
            t=t,
            curves_a={f"q[{args.joint}] [rad]": log_fric.q[:, j]},
            curves_b={f"q[{args.joint}] [rad]": log_nofric.q[:, j]},
            label_a="friction",
            label_b="no friction",
            title="Joint position (compare)",
        )
        _plot_compare(
            out_png=outdir / f"{tag}_qd_compare.png",
            t=t,
            curves_a={f"qd[{args.joint}] [rad/s]": log_fric.qd[:, j]},
            curves_b={f"qd[{args.joint}] [rad/s]": log_nofric.qd[:, j]},
            label_a="friction",
            label_b="no friction",
            title="Joint velocity (compare)",
        )
        if args.excitation in ("tri_vel", "vel_sweep"):
            _plot(
                out_png=outdir / f"{tag}_vref.png",
                t=t,
                curves={f"v_ref[{args.joint}] [rad/s]": vref},
                title="Velocity reference",
            )
        _plot(
            out_png=outdir / f"{tag}_tau_compare.png",
            t=t,
            curves={
                "tau_cmd [N*m]": log_fric.tau_cmd[:, j],
                "tau_applied (fric) [N*m]": (log_fric.tau_cmd[:, j] + log_fric.tau_fric[:, j]),
                "tau_applied (no fric) [N*m]": (log_nofric.tau_cmd[:, j] + log_nofric.tau_fric[:, j]),
                "tau_fric [N*m]": log_fric.tau_fric[:, j],
            },
            title="Torques (compare)",
        )
        _plot_xy_compare(
            out_png=outdir / f"{tag}_taufric_vs_qd.png",
            x_a=log_fric.qd[:, j],
            y_a=log_fric.tau_fric[:, j],
            x_b=log_nofric.qd[:, j],
            y_b=log_nofric.tau_fric[:, j],
            xlabel="qd [rad/s]",
            ylabel="tau_fric [N*m]",
            label_a="friction",
            label_b="no friction",
            title="Friction hysteresis (tau_fric vs qd)",
        )

        if args.excitation == "ramp_torque":
            qd_thresh = 1e-3
            idx = int(np.argmax(np.abs(log_fric.qd[:, j]) > qd_thresh))
            if np.abs(log_fric.qd[idx, j]) > qd_thresh:
                print(f"breakaway_tau_est [N*m] (friction): {log_fric.tau_cmd[idx, j]:.6f}")

        print(f"wrote: {out_npz}")
        print(f"wrote: {outdir / f'{tag}_q_compare.png'}")
        print(f"wrote: {outdir / f'{tag}_qd_compare.png'}")
        print(f"wrote: {outdir / f'{tag}_tau_compare.png'}")
        if args.excitation in ("tri_vel", "vel_sweep"):
            print(f"wrote: {outdir / f'{tag}_vref.png'}")
        print(f"wrote: {outdir / f'{tag}_taufric_vs_qd.png'}")
        return 0

    if args.no_friction:
        tau_fric_fn_use = None
        fric_meta = {"type": "none"}
    else:
        tau_fric_fn_use, fric_meta = build_tau_fric_fn(args.model)

    log = run_torque_demo(
        mjm=mjm,
        mjd=mujoco.MjData(mjm),
        duration=float(args.duration),
        tau_cmd_fn=tau_cmd_fn,
        tau_fric_fn=tau_fric_fn_use,
        lock_qpos_idx=lock_idx,
        lock_qpos_val=lock_val,
    )

    tag = f"{args.excitation}_{args.model}_joint{args.joint}_{'nofric' if args.no_friction else 'fric'}"
    out_npz = outdir / f"demo_{tag}.npz"
    vref = np.array([v_ref_at(tt) for tt in log.t], dtype=float)
    np.savez(
        out_npz,
        **log.to_npz_dict(),
        v_ref=vref,
        meta=np.array(
            [
                str(
                    {
                        "excitation": args.excitation,
                        "joint": args.joint,
                        "mjcf": str(mjcf_path.as_posix()),
                        "timestep": float(mjm.opt.timestep),
                        "lock_others": bool(args.lock_others),
                        "friction": fric_meta,
                    }
                )
            ],
            dtype=object,
        ),
    )

    t = log.t
    _plot(
        out_png=outdir / f"{tag}_q.png",
        t=t,
        curves={f"q[{args.joint}] [rad]": log.q[:, j]},
        title="Joint position",
    )
    _plot(
        out_png=outdir / f"{tag}_qd.png",
        t=t,
        curves={f"qd[{args.joint}] [rad/s]": log.qd[:, j]},
        title="Joint velocity",
    )
    if args.excitation in ("tri_vel", "vel_sweep"):
        _plot(
            out_png=outdir / f"{tag}_vref.png",
            t=t,
            curves={f"v_ref[{args.joint}] [rad/s]": vref},
            title="Velocity reference",
        )
    _plot(
        out_png=outdir / f"{tag}_tau.png",
        t=t,
        curves={
            "tau_cmd [N*m]": log.tau_cmd[:, j],
            "tau_fric [N*m]": log.tau_fric[:, j],
            "tau_applied [N*m]": (log.tau_cmd[:, j] + log.tau_fric[:, j]),
        },
        title="Torques",
    )
    _plot_xy(
        out_png=outdir / f"{tag}_taufric_vs_qd.png",
        x=log.qd[:, j],
        y=log.tau_fric[:, j],
        xlabel="qd [rad/s]",
        ylabel="tau_fric [N*m]",
        title="Friction hysteresis (tau_fric vs qd)",
    )

    print(f"wrote: {out_npz}")
    print(f"wrote: {outdir / f'{tag}_q.png'}")
    print(f"wrote: {outdir / f'{tag}_qd.png'}")
    print(f"wrote: {outdir / f'{tag}_tau.png'}")
    if args.excitation in ("tri_vel", "vel_sweep"):
        print(f"wrote: {outdir / f'{tag}_vref.png'}")
    print(f"wrote: {outdir / f'{tag}_taufric_vs_qd.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

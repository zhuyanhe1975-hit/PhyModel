#!/usr/bin/env python3
"""Headless forward-kinematics check (no viewer, no long simulation).

Loads an MJCF with MuJoCo, evaluates TCP position at a few joint poses via
``phymodel.kinematics.mujoco_fk.MujocoFK``, and prints a JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _hinge_limits(mjm: object) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    nq = int(mjm.nq)
    lo = np.full(nq, -np.inf)
    hi = np.full(nq, np.inf)
    for i in range(int(mjm.njnt)):
        if int(mjm.jnt_type[i]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        if int(mjm.jnt_limited[i]) == 0:
            continue
        adr = int(mjm.jnt_qposadr[i])
        if adr < 0 or adr >= nq:
            continue
        lo[adr] = float(mjm.jnt_range[i, 0])
        hi[adr] = float(mjm.jnt_range[i, 1])
    return lo, hi


def _sample_poses(mjm: object) -> list[tuple[str, np.ndarray]]:
    nq = int(mjm.nq)
    lo, hi = _hinge_limits(mjm)
    q0 = np.clip(np.zeros(nq), lo, hi)

    lo_use = np.where(np.isfinite(lo), lo, 0.0)
    hi_use = np.where(np.isfinite(hi), hi, 0.0)
    q_mid = np.clip(0.5 * (lo_use + hi_use), lo, hi)

    q_off = q0.copy()
    if nq > 0:
        span = hi[0] - lo[0]
        delta = 0.15 * span if np.isfinite(span) and span > 0 else 0.3
        q_off[0] = float(np.clip(q0[0] + delta, lo[0], hi[0]))
        if abs(float(q_off[0]) - float(q0[0])) < 1e-9:
            q_off[0] = float(np.clip(q0[0] - delta, lo[0], hi[0]))
    return [("zero", q0), ("mid_range", q_mid), ("joint1_offset", q_off)]


def _write_plot(path: Path, poses: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(p["name"]) for p in poses]
    xyz = np.asarray([p["tcp_xyz"] for p in poses], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = np.arange(len(labels))
    for i, name in enumerate(("x", "y", "z")):
        ax.plot(xs, xyz[:, i], marker="o", linewidth=1.4, label=name)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("TCP position [m]")
    ax.set_xlabel("pose")
    ax.set_title("ER15-1400 forward kinematics")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_check(mjcf_path: str, body_name: str) -> dict:
    errors: list[str] = []
    report: dict = {
        "ok": False,
        "errors": errors,
        "body": body_name,
        "model": {},
        "poses": [],
    }

    path = Path(mjcf_path)
    if not path.exists():
        errors.append(f"MJCF not found: {path}")
        return report

    try:
        import mujoco
    except Exception as exc:
        errors.append(f"mujoco import failed: {exc}")
        return report

    try:
        mjm = mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:
        errors.append(f"MuJoCo failed to load MJCF: {exc}")
        return report

    mjd = mujoco.MjData(mjm)
    joint_names = []
    for i in range(int(mjm.njnt)):
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_names.append(name or f"joint_{i}")
    report["model"] = {
        "nq": int(mjm.nq),
        "nv": int(mjm.nv),
        "nbody": int(mjm.nbody),
        "njnt": int(mjm.njnt),
        "timestep": float(mjm.opt.timestep),
        "gravity": [float(v) for v in mjm.opt.gravity],
        "joint_names": joint_names,
    }

    if int(mjm.nq) <= 0:
        errors.append("Model has no generalized coordinates (nq=0).")
        return report

    body_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        errors.append(f"Body not found: {body_name}")
        return report

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from phymodel.kinematics.mujoco_fk import MujocoFK
    except Exception as exc:
        errors.append(f"Failed to import MujocoFK: {exc}")
        return report

    fk = MujocoFK(mjm, mjd, body_name=body_name)
    poses = []
    for name, q in _sample_poses(mjm):
        tcp = fk.tcp_from_q([float(v) for v in q])
        poses.append(
            {
                "name": name,
                "q": [float(v) for v in q],
                "tcp_xyz": [float(v) for v in tcp],
            }
        )
    report["poses"] = poses

    pts = np.asarray([p["tcp_xyz"] for p in poses], dtype=float)
    if pts.size == 0 or not np.all(np.isfinite(pts)):
        errors.append("TCP position is not finite.")
    else:
        spread = float(np.max(np.linalg.norm(pts - pts[0], axis=1)))
        if spread < 1e-6:
            errors.append("TCP did not move across sampled poses; FK result looks degenerate.")

    report["ok"] = not errors
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Headless MuJoCo FK check for an MJCF model.")
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--body", default="link_6", help="Body used as the TCP reference.")
    ap.add_argument("--out", default=None, help="Write the JSON report to this path.")
    ap.add_argument("--plot", default=None, help="Write a TCP xyz plot (PNG) to this path.")
    try:
        import argcomplete  # type: ignore

        argcomplete.autocomplete(ap)
    except Exception:
        pass
    args = ap.parse_args()

    report = run_check(args.mjcf, args.body)
    if args.plot and report.get("poses"):
        plot_path = Path(args.plot)
        try:
            _write_plot(plot_path, report["poses"])
            report["plot"] = str(plot_path)
        except Exception as exc:
            report["errors"].append(f"Failed to write plot: {exc}")
            report["ok"] = False

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

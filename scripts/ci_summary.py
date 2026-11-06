#!/usr/bin/env python3
"""Render a short Markdown job summary from MJCF and FK validation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str | None) -> tuple[dict | None, str | None]:
    if not path:
        return None, None
    file = Path(path)
    if not file.exists():
        return None, f"未找到 `{file.as_posix()}`。"
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"`{file.as_posix()}` 不是合法 JSON（{exc}）。"
    if not isinstance(payload, dict):
        return None, f"`{file.as_posix()}` 不是 JSON 对象。"
    return payload, None


def _status(ok: object) -> str:
    return "通过" if ok else "未通过"


def _fmt(value: object) -> str:
    number = float(value)
    if abs(number) < 5e-7:
        number = 0.0
    return f"{number:.6f}"


def build_summary(mjcf: dict | None, mjcf_note: str | None, fk: dict | None, fk_note: str | None) -> str:
    lines = ["## ER15-1400 校验", ""]

    lines.append("### MJCF")
    if mjcf is None:
        lines.append(f"- {mjcf_note or '没有 MJCF 报告。'}")
    else:
        summary = mjcf.get("summary") if isinstance(mjcf.get("summary"), dict) else {}
        sync = mjcf.get("sync") if isinstance(mjcf.get("sync"), dict) else {}
        lines.append(f"- 结果：**{_status(mjcf.get('ok'))}**")
        lines.append(
            "- 计数：mesh {mesh}，body {body}，joint {joint}".format(
                mesh=summary.get("mesh_count", "?"),
                body=summary.get("body_count", "?"),
                joint=summary.get("joint_count", "?"),
            )
        )
        if sync:
            lines.append(f"- 参数同步：**{_status(sync.get('ok'))}**")
        for label, key in (("错误", "errors"), ("警告", "warnings")):
            items = mjcf.get(key) if isinstance(mjcf.get(key), list) else []
            for item in items:
                lines.append(f"- {label}：{item}")
        sync_errors = sync.get("errors") if isinstance(sync.get("errors"), list) else []
        for item in sync_errors:
            lines.append(f"- 参数同步错误：{item}")
    lines.append("")

    lines.append("### 正解 FK")
    if fk is None:
        lines.append(f"- {fk_note or '没有 FK 报告。'}")
    else:
        model = fk.get("model") if isinstance(fk.get("model"), dict) else {}
        body = fk.get("body", "tcp")
        lines.append(f"- 结果：**{_status(fk.get('ok'))}**（body `{body}`）")
        if model:
            gravity = model.get("gravity")
            gravity_txt = ", ".join(_fmt(v) for v in gravity) if isinstance(gravity, list) else "?"
            lines.append(
                "- 已加载模型：nq={nq}，nv={nv}，timestep={dt} s，gravity=[{g}]".format(
                    nq=model.get("nq", "?"),
                    nv=model.get("nv", "?"),
                    dt=_fmt(model["timestep"]) if isinstance(model.get("timestep"), (int, float)) else "?",
                    g=gravity_txt,
                )
            )
        poses = fk.get("poses") if isinstance(fk.get("poses"), list) else []
        if poses:
            lines.append("")
            lines.append("| 姿态 | x (m) | y (m) | z (m) |")
            lines.append("| --- | ---: | ---: | ---: |")
            for pose in poses:
                if not isinstance(pose, dict):
                    continue
                xyz = pose.get("tcp_xyz") if isinstance(pose.get("tcp_xyz"), list) else []
                cells = [_fmt(xyz[i]) if i < len(xyz) else "?" for i in range(3)]
                lines.append(f"| {pose.get('name', '?')} | {cells[0]} | {cells[1]} | {cells[2]} |")
        for item in fk.get("errors") if isinstance(fk.get("errors"), list) else []:
            lines.append(f"- 错误：{item}")
        if fk.get("plot"):
            plot_name = Path(str(fk["plot"])).name
            lines.append("")
            lines.append(f"TCP 曲线图：`{plot_name}`（artifact `er15-validation`）。")
    lines.append("")
    lines.append("日志与图在该次运行的 artifact **er15-validation** 中。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Write the GitHub Actions job summary for ER15 validation.")
    ap.add_argument("--mjcf", default=None, help="validate_mjcf.py JSON report.")
    ap.add_argument("--fk", default=None, help="check_fk.py JSON report.")
    ap.add_argument("--out", default=None, help="Also write the markdown to this path.")
    args = ap.parse_args()

    mjcf, mjcf_note = _load(args.mjcf)
    fk, fk_note = _load(args.fk)
    text = build_summary(mjcf, mjcf_note, fk, fk_note)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

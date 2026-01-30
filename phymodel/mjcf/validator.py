"""Minimal MJCF validation for model integration checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import xml.etree.ElementTree as ET


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, int]


def _findall(parent: ET.Element, tag: str) -> List[ET.Element]:
    return list(parent.iter(tag)) if parent is not None else []


def validate_mjcf(mjcf_path: str | Path) -> ValidationResult:
    path = Path(mjcf_path)
    errors: List[str] = []
    warnings: List[str] = []

    if not path.exists():
        return ValidationResult(False, [f"MJCF not found: {path}"], [], {})

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return ValidationResult(False, [f"MJCF parse error: {exc}"], [], {})

    compiler = root.find("compiler")
    if compiler is None:
        warnings.append("<compiler> missing; expected angle='radian'.")
    else:
        if compiler.get("angle") != "radian":
            warnings.append("<compiler angle> is not 'radian'.")

    option = root.find("option")
    if option is None:
        warnings.append("<option> missing; timestep/gravity not set explicitly.")
    else:
        if option.get("timestep") is None:
            warnings.append("<option timestep> missing.")
        if option.get("gravity") is None:
            warnings.append("<option gravity> missing.")

    assets = root.find("asset")
    mesh_elems = _findall(assets, "mesh")
    mesh_files = [m.get("file") for m in mesh_elems if m.get("file")]
    for mf in mesh_files:
        if not (path.parent / mf).exists():
            errors.append(f"Mesh file missing: {mf}")

    worldbody = root.find("worldbody")
    body_elems = _findall(worldbody, "body")
    joint_elems = _findall(worldbody, "joint")

    if not joint_elems:
        errors.append("No <joint> found in worldbody.")

    seen_joint_names = set()
    for j in joint_elems:
        name = j.get("name")
        if not name:
            errors.append("Joint without name.")
            continue
        if name in seen_joint_names:
            errors.append(f"Duplicate joint name: {name}")
        seen_joint_names.add(name)

        rng = j.get("range")
        if rng:
            parts = rng.split()
            if len(parts) == 2:
                try:
                    lo = float(parts[0])
                    hi = float(parts[1])
                    if lo >= hi:
                        errors.append(f"Invalid joint range for {name}: {rng}")
                except ValueError:
                    errors.append(f"Non-numeric joint range for {name}: {rng}")
            else:
                errors.append(f"Malformed joint range for {name}: {rng}")
        else:
            warnings.append(f"Joint range missing for {name}.")

    # Inertial checks on bodies that have a joint
    for body in body_elems:
        has_joint = body.find("joint") is not None
        if has_joint and body.find("inertial") is None:
            bname = body.get("name", "<unnamed>")
            warnings.append(f"Body {bname} has joint but no <inertial>.")

    summary = {
        "mesh_count": len(mesh_elems),
        "body_count": len(body_elems),
        "joint_count": len(joint_elems),
    }

    ok = not errors
    return ValidationResult(ok, errors, warnings, summary)

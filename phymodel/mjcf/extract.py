"""Extract a structured snapshot from an MJCF file.

Used to keep a single source-of-truth parameter file in sync with MJCF edits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import xml.etree.ElementTree as ET


def _as_floats(s: Optional[str]) -> Optional[List[float]]:
    if not s:
        return None
    try:
        return [float(x) for x in s.split()]
    except ValueError:
        return None


def _attrs(elem: Optional[ET.Element], keys: List[str]) -> Dict[str, Optional[str]]:
    if elem is None:
        return {k: None for k in keys}
    return {k: elem.get(k) for k in keys}


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_mjcf_snapshot(mjcf_path: str | Path) -> Dict[str, Any]:
    path = Path(mjcf_path)
    root = ET.parse(path).getroot()

    compiler = root.find("compiler")
    option = root.find("option")
    contact = root.find("contact")
    assets = root.find("asset")
    worldbody = root.find("worldbody")

    meshes = []
    if assets is not None:
        for m in assets.iter("mesh"):
            meshes.append({"name": m.get("name"), "file": m.get("file")})

    excludes = []
    if contact is not None:
        for ex in contact.iter("exclude"):
            excludes.append({"body1": ex.get("body1"), "body2": ex.get("body2")})

    bodies: List[Dict[str, Any]] = []
    joints_flat: List[Dict[str, Any]] = []

    def visit_body(elem: ET.Element, parent: Optional[str]) -> None:
        bname = elem.get("name")
        bpos = _as_floats(elem.get("pos"))
        bquat = _as_floats(elem.get("quat"))

        inertial = elem.find("inertial")
        inertial_dict = None
        if inertial is not None:
            inertial_dict = {
                "pos": _as_floats(inertial.get("pos")),
                "quat": _as_floats(inertial.get("quat")),
                "mass": float(inertial.get("mass")) if inertial.get("mass") else None,
                "diaginertia": _as_floats(inertial.get("diaginertia")),
            }

        joints = []
        for j in elem.findall("joint"):
            jd = {
                "name": j.get("name"),
                "pos": _as_floats(j.get("pos")),
                "axis": _as_floats(j.get("axis")),
                "range": _as_floats(j.get("range")),
                "type": j.get("type"),
            }
            joints.append(jd)
            joints_flat.append(jd)

        geoms = []
        for g in elem.findall("geom"):
            geoms.append(
                {
                    "name": g.get("name"),
                    "type": g.get("type"),
                    "mesh": g.get("mesh"),
                    "pos": _as_floats(g.get("pos")),
                    "quat": _as_floats(g.get("quat")),
                    "contype": g.get("contype"),
                    "conaffinity": g.get("conaffinity"),
                    "rgba": _as_floats(g.get("rgba")),
                }
            )

        bodies.append(
            {
                "name": bname,
                "parent": parent,
                "pos": bpos,
                "quat": bquat,
                "inertial": inertial_dict,
                "joints": joints,
                "geoms": geoms,
            }
        )

        for child in elem.findall("body"):
            visit_body(child, parent=bname)

    if worldbody is not None:
        for b in worldbody.findall("body"):
            visit_body(b, parent=None)

    world_geoms = []
    if worldbody is not None:
        for g in worldbody.findall("geom"):
            world_geoms.append(
                {
                    "name": g.get("name"),
                    "type": g.get("type"),
                    "mesh": g.get("mesh"),
                    "pos": _as_floats(g.get("pos")),
                    "quat": _as_floats(g.get("quat")),
                    "contype": g.get("contype"),
                    "conaffinity": g.get("conaffinity"),
                    "rgba": _as_floats(g.get("rgba")),
                }
            )

    snapshot: Dict[str, Any] = {
        "mjcf_path": str(path.as_posix()),
        "model": root.get("model"),
        "compiler": _attrs(compiler, ["angle"]),
        "option": _attrs(option, ["timestep", "gravity", "iterations", "integrator"]),
        "contact": {"excludes": excludes},
        "asset": {"meshes": meshes},
        "worldbody": {
            "geoms": world_geoms,
            "bodies": bodies,
        },
        "summary": {
            "mesh_count": len(meshes),
            "body_count": len(bodies),
            "joint_count": len(joints_flat),
        },
        "joint_names": [j.get("name") for j in joints_flat],
        "body_names": [b.get("name") for b in bodies],
    }
    return snapshot


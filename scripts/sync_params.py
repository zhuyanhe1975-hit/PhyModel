#!/usr/bin/env python3
"""Sync a single parameter file from an MJCF source.

Policy:
- `mjcf_snapshot` and `sources.*` are regenerated from MJCF.
- `tunable` is preserved if the params file already exists.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys


def default_tunable(nq: int) -> dict:
    return {
        "sim": {
            "timestep_override": 0.001,
        },
        "friction": {
            "model": "lugre",
            "vc": {"coulomb": [2.0] * nq, "viscous": [0.8] * nq, "v_eps": 1e-4},
            "lugre": {
                "fc": [2.0] * nq,
                "fs": [6.0] * nq,
                "vs": [0.05] * nq,
                "sigma0": [200.0] * nq,
                "sigma1": [2.0] * nq,
                "sigma2": [0.8] * nq,
                "v_eps": 1e-6,
                "g_eps": 1e-9,
            },
        }
    }

def _merge_defaults(existing: object, defaults: object) -> object:
    if isinstance(existing, dict) and isinstance(defaults, dict):
        merged = dict(existing)
        for k, v in defaults.items():
            if k not in merged:
                merged[k] = v
            else:
                merged[k] = _merge_defaults(merged[k], v)
        return merged
    return existing


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", default="models/er15-1400.mjcf.xml")
    ap.add_argument("--out", default="params/er15-1400.params.json")
    args = ap.parse_args()

    from phymodel.mjcf.extract import extract_mjcf_snapshot, sha256_file
    from phymodel.params.io import load_params, save_params

    mjcf_path = Path(args.mjcf)
    out_path = Path(args.out)

    snapshot = extract_mjcf_snapshot(mjcf_path)
    nq = int(snapshot["summary"]["joint_count"])

    existing = None
    if out_path.exists():
        existing = load_params(out_path)

    tunable_existing = existing.get("tunable") if isinstance(existing, dict) else None
    tunable = default_tunable(nq) if not isinstance(tunable_existing, dict) else _merge_defaults(tunable_existing, default_tunable(nq))

    payload = {
        "version": 1,
        "sources": {
            "mjcf_path": str(mjcf_path.as_posix()),
            "mjcf_sha256": sha256_file(mjcf_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "mjcf_snapshot": snapshot,
        "tunable": tunable,
    }

    save_params(out_path, payload)
    print(f"synced: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

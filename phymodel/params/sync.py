from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io import load_params
from phymodel.mjcf.extract import sha256_file


@dataclass
class ParamsSyncResult:
    ok: bool
    errors: List[str]
    warnings: List[str]


def check_params_sync(mjcf_path: str | Path, params_path: str | Path) -> ParamsSyncResult:
    mjcf_path = Path(mjcf_path)
    params_path = Path(params_path)

    errors: List[str] = []
    warnings: List[str] = []

    if not params_path.exists():
        errors.append(f"Params file not found: {params_path}")
        return ParamsSyncResult(False, errors, warnings)

    payload: Dict[str, Any] = load_params(params_path)
    sources: Optional[Dict[str, Any]] = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(sources, dict):
        errors.append("Params missing required key: sources")
        return ParamsSyncResult(False, errors, warnings)

    expected_mjcf = sources.get("mjcf_path")
    if expected_mjcf and str(Path(expected_mjcf).as_posix()) != str(mjcf_path.as_posix()):
        warnings.append(f"Params sources.mjcf_path != requested mjcf: {expected_mjcf} vs {mjcf_path}")

    expected_sha = sources.get("mjcf_sha256")
    actual_sha = sha256_file(mjcf_path)
    if expected_sha != actual_sha:
        errors.append(
            "Params out of sync with MJCF: mjcf_sha256 mismatch. "
            "Run: python3 scripts/sync_params.py --mjcf "
            f"{mjcf_path.as_posix()} --out {params_path.as_posix()}"
        )

    ok = not errors
    return ParamsSyncResult(ok, errors, warnings)


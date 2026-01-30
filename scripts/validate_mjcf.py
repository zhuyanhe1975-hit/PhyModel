#!/usr/bin/env python3
"""Validate and summarize an MJCF model (repo-local, no installation required)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    # Allow running from repo without installing the package.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", required=True)
    args = ap.parse_args()

    from phymodel.mjcf.validator import validate_mjcf

    rep = validate_mjcf(args.mjcf)
    payload = {
        "ok": rep.ok,
        "errors": rep.errors,
        "warnings": rep.warnings,
        "summary": rep.summary,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if rep.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

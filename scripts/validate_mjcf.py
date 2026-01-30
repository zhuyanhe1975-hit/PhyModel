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
    ap.add_argument("--params", default=None, help="Optional params file to check sync (sha256).")
    try:
        import argcomplete  # type: ignore

        argcomplete.autocomplete(ap)
    except Exception:
        pass
    args = ap.parse_args()

    from phymodel.mjcf.validator import validate_mjcf

    rep = validate_mjcf(args.mjcf)
    sync_errors = []
    sync_warnings = []
    if args.params:
        from phymodel.params.sync import check_params_sync

        sync = check_params_sync(args.mjcf, args.params)
        sync_errors = sync.errors
        sync_warnings = sync.warnings

    payload = {
        "ok": rep.ok,
        "errors": rep.errors,
        "warnings": rep.warnings,
        "summary": rep.summary,
        "sync": {
            "ok": len(sync_errors) == 0,
            "errors": sync_errors,
            "warnings": sync_warnings,
        }
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    ok = rep.ok and (len(sync_errors) == 0)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

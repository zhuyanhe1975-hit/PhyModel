#!/usr/bin/env bash
set -euo pipefail

# Enable TAB completion for repo scripts that use argparse + argcomplete.
#
# Usage:
#   source scripts/enable_tab_completion.bash
#
# Notes:
# - Requires: argcomplete installed in the active Python environment.
# - Works best in bash/zsh.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v register-python-argcomplete >/dev/null 2>&1; then
  echo "register-python-argcomplete not found. Install argcomplete first:"
  echo "  pip install argcomplete"
  return 1 2>/dev/null || exit 1
fi

eval "$(register-python-argcomplete "${repo_root}/scripts/run_friction_demo.py")"
eval "$(register-python-argcomplete "${repo_root}/scripts/view_friction.py")"
eval "$(register-python-argcomplete "${repo_root}/scripts/validate_mjcf.py")"
eval "$(register-python-argcomplete "${repo_root}/scripts/sync_params.py")"

echo "TAB completion enabled for PhyModel scripts."


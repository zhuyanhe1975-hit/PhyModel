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

# IMPORTANT:
# Bash completion matches the *exact command word* you type.
# If you run scripts via `python3 scripts/foo.py ...`, completion will NOT trigger.
# To make completion reliable, provide stable wrapper commands (shell functions) and register those.

fricdemo() { python3 "${repo_root}/scripts/run_friction_demo.py" "$@"; }
phymodel-view() { python3 "${repo_root}/scripts/view_friction.py" "$@"; }
phymodel-validate() { python3 "${repo_root}/scripts/validate_mjcf.py" "$@"; }
phymodel-sync() { python3 "${repo_root}/scripts/sync_params.py" "$@"; }

export -f fricdemo phymodel-view phymodel-validate phymodel-sync

eval "$(register-python-argcomplete fricdemo)"
eval "$(register-python-argcomplete phymodel-view)"
eval "$(register-python-argcomplete phymodel-validate)"
eval "$(register-python-argcomplete phymodel-sync)"

echo "TAB completion enabled: fricdemo / phymodel-view / phymodel-validate / phymodel-sync"

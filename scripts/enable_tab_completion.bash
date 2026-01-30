#!/usr/bin/env bash

# This file is meant to be SOURCED:
#   source scripts/enable_tab_completion.bash
#
# Do NOT use `set -euo pipefail` here because it would leak shell options into
# the caller and can cause the terminal to exit unexpectedly.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced (not executed):"
  echo "  source scripts/enable_tab_completion.bash"
  exit 1
fi

__phymodel_saved_opts="$(set +o)"

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
  echo "Or activate your conda env first (e.g. mjwarp_env)."
  eval "${__phymodel_saved_opts}"
  return 0
fi

# Make `run_friction_demo.py ...` work without typing `./scripts/...`.
export PATH="${repo_root}/scripts:${PATH}"

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

# Also register completion for direct script invocations.
eval "$(register-python-argcomplete "${repo_root}/scripts/run_friction_demo.py")"
eval "$(register-python-argcomplete "${repo_root}/scripts/view_friction.py")"
eval "$(register-python-argcomplete "${repo_root}/scripts/validate_mjcf.py")"
eval "$(register-python-argcomplete "${repo_root}/scripts/sync_params.py")"

# And for PATH-based names (works because we prepended repo_root/scripts to PATH above).
eval "$(register-python-argcomplete run_friction_demo.py)"
eval "$(register-python-argcomplete view_friction.py)"
eval "$(register-python-argcomplete validate_mjcf.py)"
eval "$(register-python-argcomplete sync_params.py)"

echo "TAB completion enabled: run_friction_demo.py / fricdemo / phymodel-*"

eval "${__phymodel_saved_opts}"

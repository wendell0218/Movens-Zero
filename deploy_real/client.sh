#!/usr/bin/env bash
set -Eeuo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$D/.." && pwd)"
if [[ -f "$ROOT/.deploy_real.env" ]]; then
    set -a; source "$ROOT/.deploy_real.env"; set +a
fi
export CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-/usr/local}"
export LD_LIBRARY_PATH="${CYCLONEDDS_LIB:-$CYCLONEDDS_HOME/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MOVENS_DEPLOY_PYTHON="${MOVENS_DEPLOY_PYTHON:-$ROOT/.venv-deploy/bin/python}"
export UNITREE_SDK_ROOT="${UNITREE_SDK_ROOT:-${HOME}/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python}"
export PYTHONPATH="$UNITREE_SDK_ROOT:$D${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$D/bin:$(dirname "$MOVENS_DEPLOY_PYTHON"):$PATH"
export PYTHONUNBUFFERED=1
export MOVENS_HOST="${MOVENS_HOST:-127.0.0.1}"
export MOVENS_PORT="${MOVENS_PORT:-8014}"
export G1_NETWORK_INTERFACE="${G1_NETWORK_INTERFACE:-enP8p1s0}"
cd "$ROOT"
exec "$MOVENS_DEPLOY_PYTHON" -u "$D/client.py" "$@"

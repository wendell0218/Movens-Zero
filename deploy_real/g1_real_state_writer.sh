#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
PSI_DEPLOY_PYTHON="${PSI_DEPLOY_PYTHON:-$REPO_ROOT/.venv-deploy/bin/python}"
if [[ ! -x "$PSI_DEPLOY_PYTHON" ]]; then
    PSI_DEPLOY_PYTHON="${PSI_PYTHON:-python}"
fi
if ! "$PSI_DEPLOY_PYTHON" -c 'import numpy' >/dev/null 2>&1; then
    echo "[g1-state] ERROR: invalid Python environment: $PSI_DEPLOY_PYTHON" >&2
    exit 1
fi

export CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-/usr/local}"
export LD_LIBRARY_PATH="${CYCLONEDDS_LIB:-/usr/local/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export UNITREE_SDK_ROOT="${UNITREE_SDK_ROOT:-${HOME}/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python}"
export PYTHONPATH="$UNITREE_SDK_ROOT:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

LIVE_STATE_NPY="${LIVE_STATE_NPY:-/tmp/psi0_sonic_live_state.npy}"
G1_STATE_PERIOD="${G1_STATE_PERIOD:-0.02}"
G1_STATE_STATUS_PERIOD="${G1_STATE_STATUS_PERIOD:-0}"
G1_ROOT_Z_FALLBACK="${G1_ROOT_Z_FALLBACK:-0.78}"
G1_QUAT_ORDER="${G1_QUAT_ORDER:-wxyz}"

ARGS=()
[[ "${G1_REQUIRE_ODOM:-0}" == "1" ]] && ARGS+=(--require-odom)

if [[ -z "${CYCLONEDDS_URI:-}" ]]; then
    echo "[g1-state] CYCLONEDDS_URI is not set; using current Unitree SDK default"
fi

exec "$PSI_DEPLOY_PYTHON" -u "$SCRIPT_DIR/g1_real_state_writer.py" \
    --live-state-npy "$LIVE_STATE_NPY" \
    --period "$G1_STATE_PERIOD" \
    --status-period "$G1_STATE_STATUS_PERIOD" \
    --root-z-fallback "$G1_ROOT_Z_FALLBACK" \
    --quat-order "$G1_QUAT_ORDER" \
    "${ARGS[@]}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
MOVENS_DEPLOY_PYTHON="$PWD/.venv-deploy/bin/python"
UNITREE_SDK_ROOT="$HOME/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python"
LIVE_STATE_NPY="/tmp/movens_wbc_live_state.npy"
if [ -f .sim2real.env ]; then
    set -a
    source .sim2real.env
    set +a
fi
export PYTHONPATH="$UNITREE_SDK_ROOT:$PWD:$PYTHONPATH"
"$MOVENS_DEPLOY_PYTHON" -u sim2real/state_reader.py --live-state-npy "$LIVE_STATE_NPY" "$@"

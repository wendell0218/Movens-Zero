cd "$(dirname "${BASH_SOURCE[0]}")/.."
MOVENS_DEPLOY_PYTHON="$PWD/.venv-deploy/bin/python"
UNITREE_SDK_ROOT="$HOME/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python"
CYCLONEDDS_HOME="/usr/local"
CYCLONEDDS_LIB="/usr/local/lib"
if [ -f .sim2real.env ]; then
    set -a
    source .sim2real.env
    set +a
fi
export CYCLONEDDS_HOME
export LD_LIBRARY_PATH="$CYCLONEDDS_LIB:$LD_LIBRARY_PATH"
export MOVENS_DEPLOY_PYTHON UNITREE_SDK_ROOT
export PYTHONPATH="$UNITREE_SDK_ROOT:$PWD/sim2real:$PYTHONPATH"
export PYTHONUNBUFFERED=1
"$MOVENS_DEPLOY_PYTHON" -u sim2real/client.py "$@"

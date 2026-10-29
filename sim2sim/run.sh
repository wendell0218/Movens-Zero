cd "$(dirname "${BASH_SOURCE[0]}")/.."
SONIC_ROOT="$HOME/GR00T-WholeBodyControl"
MOVENS_HOST="127.0.0.1"
MOVENS_PORT="8014"
if [ -f .sim2sim.env ]; then
    set -a
    source .sim2sim.env
    set +a
fi
export SONIC_ROOT MOVENS_HOST MOVENS_PORT
export PYTHONPATH="$SONIC_ROOT:$SONIC_ROOT/external_dependencies/unitree_sdk2_python:$PYTHONPATH"
source "$SONIC_ROOT/.venv_sim/bin/activate"
python sim2sim/run.py "$@"

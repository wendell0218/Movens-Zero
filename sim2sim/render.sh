cd "$(dirname "${BASH_SOURCE[0]}")/.."
SONIC_ROOT="$HOME/GR00T-WholeBodyControl"
if [ -f .sim2sim.env ]; then
    source .sim2sim.env
fi
source "$SONIC_ROOT/.venv_sim/bin/activate"
python sim2sim/render.py "$@"

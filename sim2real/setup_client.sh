cd "$(dirname "${BASH_SOURCE[0]}")/.."
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
python3 -m venv --system-site-packages .venv-deploy
.venv-deploy/bin/python -m pip install -r sim2real/requirements-client.txt
PYTHONPATH="$UNITREE_SDK_ROOT:$PWD/sim2real" .venv-deploy/bin/python -c 'import numpy, pandas, websocket, zmq; from unitree_sdk2py.core.channel import ChannelFactoryInitialize; print("CLIENT_ENV_OK DDS_NOT_INITIALIZED")'

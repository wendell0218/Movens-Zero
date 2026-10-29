cd "$(dirname "${BASH_SOURCE[0]}")/.."
MOVENS_SERVER_PYTHON="python"
CUDA_VISIBLE_DEVICES="0"
if [ -f .sim2real.env ]; then
    set -a
    source .sim2real.env
    set +a
fi
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
python_bin="$MOVENS_SERVER_PYTHON"
"$python_bin" -u sim2real/server.py "$@"

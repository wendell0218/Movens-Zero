#!/usr/bin/env bash
# Installs only lightweight client packages. Does not initialize DDS or start WBC.
set -Eeuo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$D/.." && pwd)"
# Local settings are not part of the public deployment tree.
if [[ -f "$ROOT/.deploy_real.env" ]]; then
    set -a; source "$ROOT/.deploy_real.env"; set +a
fi
# Match the existing Unitree SDK runtime instead of the distribution DDS library.
export CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-/usr/local}"
export LD_LIBRARY_PATH="${CYCLONEDDS_LIB:-$CYCLONEDDS_HOME/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
VENV="${MOVENS_CLIENT_VENV:-$ROOT/.venv-deploy}"
SYSTEM_PYTHON="${SYSTEM_PYTHON:-python3}"
"$SYSTEM_PYTHON" - "$VENV" <<'PY'
import sys, venv
# Some Jetson images omit ensurepip. Install through the system pip --target below.
venv.EnvBuilder(system_site_packages=True,with_pip=False).create(sys.argv[1])
PY
SITE="$($VENV/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
"$SYSTEM_PYTHON" -m pip install --upgrade --target "$SITE" \
    'numpy==1.26.4' 'pandas==2.2.3' 'websocket-client>=1.8,<2' 'pyzmq>=26,<28' 'imageio-ffmpeg>=0.6,<0.7'
# Old/new Jetson images differ: inherit DDS only when the import actually works.
if ! "$VENV/bin/python" -c 'import cyclonedds' >/dev/null 2>&1; then
    CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-/usr/local}" "$SYSTEM_PYTHON" -m pip install \
        --target "$SITE" 'cyclonedds==0.10.2'
fi
SDK="${UNITREE_SDK_ROOT:-${HOME}/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python}"
PYTHONPATH="$SDK:$D" "$VENV/bin/python" -c 'import numpy,pandas,websocket,zmq; from unitree_sdk2py.core.channel import ChannelFactoryInitialize; print("CLIENT_ENV_OK DDS_NOT_INITIALIZED")'

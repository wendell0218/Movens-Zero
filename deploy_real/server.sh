#!/usr/bin/env bash
set -Eeuo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$D/.." && pwd)"
# Local settings are not part of the public deployment tree.
if [[ -f "$ROOT/.deploy_real.env" ]]; then
    set -a; source "$ROOT/.deploy_real.env"; set +a
fi
PY="${MOVENS_SERVER_PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PSI_PRETRAIN_BACKBONE_PATH="${PSI_PRETRAIN_BACKBONE_PATH:-Qwen/Qwen2.5-Omni-3B}"
export PSI_SKIP_VLM_MODEL=0
export PSI_FFMPEG_THREADS=1
cd "$ROOT"
exec "$PY" -u "$D/server.py" "$@"

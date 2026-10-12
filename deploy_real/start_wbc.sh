#!/usr/bin/env bash
set -Ee

WBC_ROOT="${WBC_ROOT:-${HOME}/GR00T-WholeBodyControl/gear_sonic_deploy}"
cd "$WBC_ROOT"

if [[ "${MOVENS_REAL_CONFIRMED:-0}" != "1" ]]; then
    echo "[wbc-real] WARNING: this starts the REAL G1 WBC controller."
    echo "[wbc-real] Make sure the robot is supported/hung or in a safe test posture."
    read -r -p "[wbc-real] Proceed? [Y/n]: " CONFIRM
    if [[ -n "$CONFIRM" && ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        echo "[wbc-real] cancelled"
        exit 1
    fi
fi

if [[ -f scripts/setup_env.sh ]]; then
    source scripts/setup_env.sh
fi

CHECKPOINT="${WBC_CHECKPOINT:-policy/release/model}"
OBS_CONFIG="${WBC_OBS_CONFIG:-policy/release/observation_config.yaml}"
PLANNER="${WBC_PLANNER:-planner/target_vel/V2/planner_sonic.onnx}"
MOTION_DATA="${WBC_MOTION_DATA:-reference/example/}"
INPUT_TYPE="${WBC_INPUT_TYPE:-manager}"
OUTPUT_TYPE="${WBC_OUTPUT_TYPE:-all}"
ZMQ_HOST="${WBC_ZMQ_HOST:-localhost}"
ZMQ_PORT="${WBC_ZMQ_PORT:-5556}"
WBC_ZMQ_MODE="${WBC_ZMQ_MODE:-queue}"
ZMQ_CONFLATE="${WBC_ZMQ_CONFLATE:-0}"

case "$WBC_ZMQ_MODE" in
    queue)
        [[ "$ZMQ_CONFLATE" == "0" ]] || {
            echo "[wbc-real] queue mode requires WBC_ZMQ_CONFLATE=0" >&2
            exit 1
        }
        ;;
    latest)
        [[ "$ZMQ_CONFLATE" == "1" ]] || {
            echo "[wbc-real] latest mode requires WBC_ZMQ_CONFLATE=1" >&2
            exit 1
        }
        ;;
    *)
        echo "[wbc-real] unsupported WBC_ZMQ_MODE=$WBC_ZMQ_MODE (queue/latest)" >&2
        exit 1
        ;;
esac

if [[ -n "${WBC_TARGET:-}" ]]; then
    TARGET="$WBC_TARGET"
else
    TARGET="$(ip -4 addr show 2>/dev/null | awk '/inet 192\.168\.123\./ {print prev} /^[0-9]+:/ {prev=$2; sub(/:$/, "", prev)}' | head -1)"
    if [[ -z "$TARGET" ]]; then
        TARGET="enP8p1s0"
        echo "[wbc-real] could not auto-detect 192.168.123.x interface; using $TARGET"
    fi
fi

if [[ ! -x ./target/release/g1_deploy_onnx_ref ]]; then
    echo "[wbc-real] target binary missing, building once"
    just build
fi

echo "[wbc-real] target: $TARGET"
echo "[wbc-real] input type: $INPUT_TYPE"
echo "[wbc-real] zmq host: $ZMQ_HOST"
echo "[wbc-real] wbc zmq mode: $WBC_ZMQ_MODE"
echo "[wbc-real] zmq conflate: $ZMQ_CONFLATE"
echo "[wbc-real] planner: $PLANNER"
echo "[wbc-real] motion data: $MOTION_DATA"

ARGS=(
    "$TARGET"
    "${CHECKPOINT}_decoder.onnx"
    "$MOTION_DATA"
    --obs-config "$OBS_CONFIG"
    --encoder-file "${CHECKPOINT}_encoder.onnx"
    --planner-file "$PLANNER"
    --input-type "$INPUT_TYPE"
    --output-type "$OUTPUT_TYPE"
    --zmq-host "$ZMQ_HOST"
    --zmq-port "$ZMQ_PORT"
)

if [[ "$ZMQ_CONFLATE" == "1" ]]; then
    ARGS+=(--zmq-conflate)
fi

exec ./target/release/g1_deploy_onnx_ref "${ARGS[@]}"

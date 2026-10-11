#!/usr/bin/env bash
# Match the G1-authoritative v18 wrapper here. Sonic's setup_env.sh is written
# for an interactive shell and reads PS1 directly, so nounset would abort the
# launcher before the controller binary is executed.
set -Ee

SONIC_ROOT="${SONIC_ROOT:-${HOME}/GR00T-WholeBodyControl/gear_sonic_deploy}"
cd "$SONIC_ROOT"

if [[ "${PSI0_REAL_CONFIRMED:-0}" != "1" ]]; then
    echo "[sonic-real] WARNING: this starts the REAL G1 Sonic controller."
    echo "[sonic-real] Make sure the robot is supported/hung or in a safe test posture."
    read -r -p "[sonic-real] Proceed? [Y/n]: " CONFIRM
    if [[ -n "$CONFIRM" && ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        echo "[sonic-real] cancelled"
        exit 1
    fi
fi

if [[ -f scripts/setup_env.sh ]]; then
    # shellcheck disable=SC1091
    source scripts/setup_env.sh
fi

CHECKPOINT="${SONIC_CHECKPOINT:-policy/release/model}"
OBS_CONFIG="${SONIC_OBS_CONFIG:-policy/release/observation_config.yaml}"
PLANNER="${SONIC_PLANNER:-planner/target_vel/V2/planner_sonic.onnx}"
MOTION_DATA="${SONIC_MOTION_DATA:-reference/example/}"
INPUT_TYPE="${SONIC_INPUT_TYPE:-manager}"
OUTPUT_TYPE="${SONIC_OUTPUT_TYPE:-all}"
ZMQ_HOST="${SONIC_ZMQ_HOST:-localhost}"
ZMQ_PORT="${SONIC_ZMQ_PORT:-5556}"
WBC_ZMQ_MODE="${SONIC_WBC_ZMQ_MODE:-queue}"
ZMQ_CONFLATE="${SONIC_ZMQ_CONFLATE:-0}"

case "$WBC_ZMQ_MODE" in
    queue)
        [[ "$ZMQ_CONFLATE" == "0" ]] || {
            echo "[sonic-real] queue mode requires SONIC_ZMQ_CONFLATE=0" >&2
            exit 1
        }
        ;;
    latest)
        [[ "$ZMQ_CONFLATE" == "1" ]] || {
            echo "[sonic-real] latest mode requires SONIC_ZMQ_CONFLATE=1" >&2
            exit 1
        }
        ;;
    *)
        echo "[sonic-real] unsupported SONIC_WBC_ZMQ_MODE=$WBC_ZMQ_MODE (queue/latest)" >&2
        exit 1
        ;;
esac

if [[ -n "${SONIC_TARGET:-}" ]]; then
    TARGET="$SONIC_TARGET"
else
    TARGET="$(ip -4 addr show 2>/dev/null | awk '/inet 192\.168\.123\./ {print prev} /^[0-9]+:/ {prev=$2; sub(/:$/, "", prev)}' | head -1)"
    if [[ -z "$TARGET" ]]; then
        TARGET="enP8p1s0"
        echo "[sonic-real] could not auto-detect 192.168.123.x interface; using $TARGET"
    fi
fi

if [[ ! -x ./target/release/g1_deploy_onnx_ref ]]; then
    echo "[sonic-real] target binary missing, building once"
    just build
fi

echo "[sonic-real] target: $TARGET"
echo "[sonic-real] input type: $INPUT_TYPE"
echo "[sonic-real] zmq host: $ZMQ_HOST"
echo "[sonic-real] wbc zmq mode: $WBC_ZMQ_MODE"
echo "[sonic-real] zmq conflate: $ZMQ_CONFLATE"
echo "[sonic-real] planner: $PLANNER"
echo "[sonic-real] motion data: $MOTION_DATA"

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

cd "$(dirname "${BASH_SOURCE[0]}")/.."
WBC_ROOT="$HOME/GR00T-WholeBodyControl/gear_sonic_deploy"
WBC_TARGET="enP8p1s0"
WBC_CHECKPOINT="policy/release/model"
WBC_OBS_CONFIG="policy/release/observation_config.yaml"
WBC_PLANNER="planner/target_vel/V2/planner_sonic.onnx"
WBC_MOTION_DATA="reference/example/"
WBC_ZMQ_HOST="localhost"
WBC_ZMQ_PORT="5556"
WBC_ZMQ_MODE="queue"
if [ -f .sim2real.env ]; then
    set -a
    source .sim2real.env
    set +a
fi
if [ "$WBC_ZMQ_CONFLATE" = "1" ]; then
    WBC_ZMQ_MODE="latest"
fi
cd "$WBC_ROOT"
source scripts/setup_env.sh
ARGS=("$WBC_TARGET" "${WBC_CHECKPOINT}_decoder.onnx" "$WBC_MOTION_DATA"
    --obs-config "$WBC_OBS_CONFIG" --encoder-file "${WBC_CHECKPOINT}_encoder.onnx"
    --planner-file "$WBC_PLANNER" --input-type manager --output-type all
    --zmq-host "$WBC_ZMQ_HOST" --zmq-port "$WBC_ZMQ_PORT")
if [ "$WBC_ZMQ_MODE" = "latest" ]; then
    ARGS+=(--zmq-conflate)
fi
./target/release/g1_deploy_onnx_ref "${ARGS[@]}"

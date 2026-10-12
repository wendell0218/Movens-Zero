# Dependencies

## Included

`root40_runtime/` contains the Movens model and motion/media helpers. `config/`
contains the root40 configuration, normalization, expected checkpoint hash and
static probe pose. No training repository or evaluation directory is required.

## GPU Workstation

The tested environment uses Python 3.10, RTX 4090, PyTorch 2.7.0/CUDA 12.6, BF16
and FlashAttention 2.7.4.post1. Install `requirements-server.txt`, FlashAttention,
FFmpeg and FFprobe. Provide the full Qwen2.5-Omni-3B backbone and matching Movens
root40 checkpoint. Weights are not included in this repository.

## G1 Onboard Computer

Install the full Unitree SDK Python source and a matching CycloneDDS C runtime.
`setup_client.sh` installs lightweight Python dependencies without initializing DDS.
Do not replace the robot's DDS runtime simply to satisfy an import: set
`CYCLONEDDS_HOME` and `CYCLONEDDS_LIB` to its existing compatible installation.

The whole-body controller is an external dependency from
[GR00T WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl).
The tested revision is `3d3e593369bc1384a8f5d1f22830bf20cb3a6047`.
Follow the upstream installation instructions for that revision; a newer
controller or policy is not assumed to preserve playback behavior.

Set `WBC_ROOT` to its deployment directory. Required files beneath that directory:

```text
target/release/g1_deploy_onnx_ref
policy/release/model_decoder.onnx
policy/release/model_encoder.onnx
policy/release/observation_config.yaml
planner/target_vel/V2/planner_sonic.onnx
reference/example/
```

The launcher sources `scripts/setup_env.sh` if present and uses the manager input
interface. If the executable is missing, it invokes `just build`; prepare and
verify the controller installation before operating the robot.

Controller binaries, ONNX policies and initial reference motion are not
redistributed. Upstream software and model licenses continue to apply. Names in
external asset paths identify actual dependency files, not Movens modules.

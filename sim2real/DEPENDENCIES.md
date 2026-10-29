# Dependencies

## Included

The server imports the installed `movens_zero` package from this repository.
The same public root40 model loader, trained Qwen weights, codec, and normalization
are used for offline inference and online serving. `root40_runtime/` and
`backend_legacy.py` retain the historical deployment source for reference; they
are not used by the default server.

## GPU Workstation

Install the repository and `requirements-server.txt`, FFmpeg and FFprobe. Provide
one public S/M/L/XL root40 checkpoint plus Qwen2.5-Omni-3B configuration/processor
assets. CUDA BF16 is required. SDPA is the default; FlashAttention 2 is optional.
The former 99,000-step deployment was tested on RTX 4090 with FlashAttention 2;
this is not a performance guarantee for the larger public checkpoints. The
180 ms inference deadline remains enforced.

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

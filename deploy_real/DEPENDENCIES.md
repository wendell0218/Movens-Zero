# External dependency inventory

The package is independent of the surrounding Movens training/inference source.
The 39D repository runtime is deliberately not imported by this root40 service.

| Component | Included? | Used by |
|---|---|---|
| Root40 model/codec/progress/text/video runtime | Yes, `root40_runtime/` | GPU server |
| Root40 model config, statistics, expected checkpoint hash | Yes, `config/` | GPU server |
| Static probe pose, joint-order map, motion bounds | Yes | Probe/client |
| Python dependencies | Requirements/setup scripts only | Both |
| FFmpeg + FFprobe | No | Server and client media input |
| ckpt_99000 safetensors | No; hash pinned in `config/release_manifest.json` | Server |
| Qwen2.5-Omni-3B backbone | No | Server |
| Unitree SDK source and CycloneDDS runtime | No | G1 client |
| Sonic C++ executable + encoder/decoder/planner ONNX + observation config | No | G1 client |
| Sonic initialization motion reference | No; `reference/example/` | G1 controller |

Sonic reference revision: `3d3e593369bc1384a8f5d1f22830bf20cb3a6047`
of [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl).
Recorded hashes from the tested installation (not redistributed):

```text
1a3a3492167bef9c65cf38581bbe51c3bc7c431be22e044cff6d55a7622d5b03  target/release/g1_deploy_onnx_ref
c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed  policy/release/model_decoder.onnx
013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3  policy/release/model_encoder.onnx
39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea  planner/target_vel/V2/planner_sonic.onnx
```

A locally rebuilt executable can have a different binary hash; revalidate its
protocol and playback behavior rather than treating compilation as equivalence.
Source/dependency licenses remain those of their respective projects. This
package does not introduce a new blanket license for third-party code or models.

The deployment wrappers and lightweight Sonic packet/state helpers derive from
the project's earlier real-G1 deployment. `root40_runtime/extraction_manifest.json`
records the source-module hashes used for extraction; training datasets, loaders,
trainer/optimizer entry points, and private paths are excluded. Model checkpoint
keys and inference functions retain the tested runtime semantics.

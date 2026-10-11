# Real G1 deployment — root40

A two-machine deployment: **GPU server → WebSocket → G1 client → local Sonic WBC**.
The client sends live robot observations; the server returns 30-frame absolute
motion chunks with a 20-frame execution stride at 50 Hz.

## Compatibility: read this first

This directory packages the **root40_v1 / ckpt_99000** runtime used for this
real-robot deployment. It is **not** the repository's current 39D offline runtime.
Do not feed the 39D checkpoint, `configs/inference.json`, or
`assets/normalization.json` into this server. The root40 model predicts 40D
features; the wire format contains 36D absolute poses (29 IsaacLab-order joints,
root xyz, quaternion wxyz). Heading-aware RTC rebasing preserves the 10-frame
unexecuted prefix against the latest observation.

The Python runtime, normalization, probe pose, and deployment config needed by
this directory are included here. Neither the private training repository nor
its `psi`, `release`, `visualization`, or evaluation folders are required.
`root40_runtime/` contains the extracted model, root40 codec, prompt/progress and
video helpers; its extraction manifest records source hashes. Model parameter
names and retained inference computations are preserved. The original 39D
repository entry points are not changed.

**Not included:** checkpoint/backbone weights, Sonic binaries/ONNX policies,
Unitree SDK, CycloneDDS C library, user media, credentials, logs, or a robot image.
Those remain external dependencies. This directory alone is not a complete
robot operating environment.

## 1. GPU server

Tested with Python 3.10, RTX 4090, PyTorch 2.7.0/CUDA 12.6, BF16 and
FlashAttention 2.7.4.post1. Use a separate environment from the 39D offline package:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r deploy_real/requirements-server.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
# Install ffmpeg and ffprobe using your OS package manager.
cp deploy_real/config/server.env.example .deploy_real.env
```

Edit `.deploy_real.env` with absolute paths to this environment, the complete
**Qwen2.5-Omni-3B** backbone, and the **matching root40 checkpoint**. The example
Python path must be adjusted to the environment you created. Shell launchers load
this file automatically; it is machine-local and must not be committed.

The required checkpoint SHA-256 is:

```text
5ca25d8114ee9e0d55ba303d8da18e19b5829a432943f2e7e2fe651cf3789b10
```

The checkpoint is distributed separately; this folder does not grant access to
private model storage or provide a public weight download. Startup hashes the
entire checkpoint and validates the root40 configuration/statistics before
loading. Do not replace the expected digest merely to accept unrelated weights.

```bash
bash deploy_real/server.sh
# Optional explicit bind/port:
bash deploy_real/server.sh --host 0.0.0.0 --port 8014
curl http://127.0.0.1:8014/health
```

`/health` must report `backend=checkpoint`, `checkpoint_verified=true`,
`model_action_dim=40`, `action_dim=36`, and the expected digest. This is a
single-client service. Use a trusted LAN or tunnel: the service does **not**
implement authentication or TLS; do not expose port 8014 to the public Internet.

## 2. G1 client dependencies

Use the same `deploy_real/` revision on both machines. The G1 does not need Torch,
CUDA or either model checkpoint.

Install/configure these dependencies separately:

- [GR00T WholeBodyControl / Sonic](https://github.com/NVlabs/GR00T-WholeBodyControl).
  The deployed reference checkout was commit
  `3d3e593369bc1384a8f5d1f22830bf20cb3a6047`; do not assume a newer
  controller has identical queue/lookahead behavior. See `DEPENDENCIES.md`.
- The full `unitree_sdk2_python` source tree, including its `unitree_sdk2py`
  namespace, and the matching CycloneDDS C runtime. A partial wheel may omit
  namespaces used by the SDK. Configure `CYCLONEDDS_HOME`/`CYCLONEDDS_LIB` to match
  your robot's installation rather than replacing its system DDS libraries.
- Python 3.10, NumPy, pandas, websocket-client, pyzmq, and FFmpeg/FFprobe.

Required files beneath `SONIC_ROOT`:

```text
target/release/g1_deploy_onnx_ref
policy/release/model_decoder.onnx
policy/release/model_encoder.onnx
policy/release/observation_config.yaml
planner/target_vel/V2/planner_sonic.onnx
reference/example/                 # controller initialization motion data
scripts/setup_env.sh               # if required by your Sonic installation
```

The wrapper uses the manager input interface and queued, non-conflated local ZMQ
at port 5556. It can invoke `just build` when the controller executable is absent;
prepare and verify the controller installation before operating the robot.
Upstream SDK/controller/policy licenses and installation instructions apply;
none of their binaries or weights are redistributed here.

```bash
cp deploy_real/config/client.env.example .deploy_real.env
# Edit PSI_HOST to the server LAN address and set SDK/Sonic paths + robot DDS NIC.
bash deploy_real/setup_client.sh
bash deploy_real/client.sh --check-only --modality text --text 'A person stands still.'
bash deploy_real/client.sh --dry-run --modality text --text 'A person stands still.'
```

`--check-only` checks the endpoint, imports and controller files, without DDS
initialization or WBC launch. `--dry-run` requests two real inference chunks from
a bundled static pose and never initializes DDS or publishes motor commands.
These checks do not demonstrate physical tracking quality.

## 3. Operate / 终端操作

Only operate a healthy, supported robot with a trained operator and an available
physical emergency stop. Launching the normal client initializes the deployment
and controller processes; there is no initial `YES` prompt. `s` and the subsequent
operator controls are retained.

```bash
# Start the server first, then run on the G1:
bash deploy_real/client.sh
```

1. **`s`**：启动 Sonic planner / 站立。
2. **Enter**：退出 planner 输入，进入 action/ZMQ 任务界面。
3. **`1` 文本 / `2` video / `3` speech / `4` music**。
4. 输入文字或 **G1 本地文件路径**，回车确认。路径可带引号；不存在、不可读或
   格式不支持时会提示并响铃，不会覆盖此前有效任务。
5. **`]`**：准备任务并执行。任务运行时不接受替换输入。
6. **`h`**：查看本次启动内的历史，选择编号复用；媒体路径会重新检查。
   历史仅在当前进程内保留，重启后清空。
7. **Enter / `!`** 返回 planner，**`o`** 停止，**Ctrl+C** 退出并清理子进程。
   编辑文字/路径期间，`o` 是普通字符；Esc 取消编辑，Ctrl+U 清空输入。

You can also preselect a task (the same standing/execution keys still apply):

```bash
bash deploy_real/client.sh --modality text --text 'A person waves their right hand.' --seconds 6
bash deploy_real/client.sh --modality video --video /path/on/g1/reference.mp4
bash deploy_real/client.sh --modality speech --speech /path/on/g1/instruction.wav --seconds 6
bash deploy_real/client.sh --modality music --music /path/on/g1/music.wav
```

Text and speech default to 6 seconds; `--seconds` changes their target duration.
Video/music follow media duration. Speech conditions on the complete audio;
music uses a moving 0.6-second window. Video uses 8 fps, a 0.6-second window and
50176-pixel frames, with a bounded 512 MiB decoded session cache. Media files are
uploaded by the client and verified by SHA-256; server-side matching paths are
not required. The server caches at most eight media items until restart.

## 4. Tests, synchronization and limitations

On the GPU server environment, without loading model weights or operating motors:

```bash
python -m unittest discover -s deploy_real -p 'test_*.py'
(cd deploy_real && sha256sum -c SHA256SUMS)
```

Run the hash check on **both** machines. `SHA256SUMS` covers all published files
except itself; local `.deploy_real.env` belongs in the project root, outside that
shared tree. Its host/IP/device paths may differ; code, config and model contract
must not. The shared first-frame jump default is 0.35 rad. Machine-local overrides
are not a recommended way to work around a broken motor or a rejected motion.

Non-actuating four-modality tests of this packaged runtime used the verified
checkpoint over the G1→GPU WebSocket link. They validate protocol, finite output,
chunk shape and overlap, **not** a new physical motion-quality acceptance test.

Known behavior:

- Fixed monotonic chunk deadlines avoid accumulating a one-off send delay. A
  lateness over 180 ms stops the segment instead of fast-forwarding it. Stale
  observation, quaternion, joint-bound and mechanical-velocity checks remain.
- Sonic's tested encoder uses a 46-frame lookahead, larger than the first
  30-frame model chunk. Initial waiting/playback lag is **not eliminated** by
  the deadline correction; do not change the trained encoder horizon casually.
- A short previous replay exposed a possible final-packet/EOF race. Full final
  frame delivery remains a separate acceptance item, not a guaranteed property.
- Logs normally go to `/tmp/movens_real_logs`. Bridge logs/generated CSVs can be
  overwritten by another task in the same session; archive them before rerunning.
  Rejection snapshots capture inputs for diagnosis, but are not public datasets.
- One controller/client at a time; shared `/tmp` state/flag paths are not designed
  for concurrent robot sessions. Keep joint ordering, 50 Hz, 30/20/10 chunk sizes,
  normalization and checkpoint identity consistent.

To update, stop the client normally, save both deployment folders and local env
files, copy the same new directory to both hosts, run hashes/tests, then restart
the server. To roll back, restore both saved directories and each host's own env
file before restarting. Do not update files underneath an active robot session.

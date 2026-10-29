# Sim2Real

Run Movens-Zero on a GPU workstation and stream generated motion to a Unitree G1 through its onboard client and whole-body controller. Text, video, speech, and music use the same deployment service. Run the commands below from the repository root on the indicated host.

## 1. Requirements

| GPU workstation | G1 onboard computer |
| --- | --- |
| Linux, Python 3.10, CUDA GPU with BF16 support | Python 3.10, Unitree SDK, and matching CycloneDDS |
| Movens-Zero checkpoint and Qwen2.5-Omni-3B processor assets | [GR00T WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl), including its executable and policy assets |
| FFmpeg and FFprobe | FFmpeg and FFprobe |

Both hosts need this repository and a reachable network connection. The G1 does not need Torch, CUDA, or Movens-Zero weights. See [Dependencies](DEPENDENCIES.md) for the controller revision and required assets.

## 2. Set Up the Server

On the GPU workstation, install the server environment and download the model:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r sim2real/requirements-server.txt
python scripts/download_models.py --model-size XL
cp sim2real/config/server.env.example .sim2real.env
```

Edit `.sim2real.env` to select this environment and checkpoint:

```bash
MOVENS_SERVER_PYTHON="/absolute/path/to/Movens-Zero/.venv/bin/python"
MOVENS_CHECKPOINT="checkpoints/Movens-Zero-XL/model.safetensors"
MOVENS_PRETRAIN_BACKBONE_PATH="Qwen/Qwen2.5-Omni-3B"
CUDA_VISIBLE_DEVICES=0
MOVENS_PORT=8014
```

S, M, and L checkpoints are also supported; their architecture is detected from the weights. The backbone setting accepts the model ID above or a local model directory. SDPA is the default attention implementation.

The server enforces a **180 ms inference deadline** for each chunk. The selected model and workstation must meet this limit; missed deadlines stop the task.

## 3. Set Up the Client

On the G1, install the controller and Unitree SDK using [Dependencies](DEPENDENCIES.md), then create the client configuration:

```bash
cp sim2real/config/client.env.example .sim2real.env
```

Edit these settings for the G1 installation:

| Variable | Value |
| --- | --- |
| `MOVENS_HOST` | GPU workstation's reachable LAN address |
| `MOVENS_PORT` | Same port as the server; default `8014` |
| `WBC_ROOT` | Controller deployment directory, such as `~/GR00T-WholeBodyControl/gear_sonic_deploy` |
| `UNITREE_SDK_ROOT` | Installed `unitree_sdk2_python` directory |
| `G1_NETWORK_INTERFACE` | Robot network interface, such as `enP8p1s0` |
| `CYCLONEDDS_HOME`, `CYCLONEDDS_LIB` | Existing compatible DDS installation and library directory |

Install the lightweight client environment:

```bash
source sim2real/setup_client.sh
```

This creates `.venv-deploy/`. Each host has its own `.sim2real.env`; both launchers read the file from the repository root.

## 4. Start Deployment

On the GPU workstation:

```bash
source sim2real/server.sh
```

On the G1, check the client installation and service, then request a non-actuating inference:

```bash
source sim2real/client.sh --check-only --modality text --text 'A person stands still.'
source sim2real/client.sh --dry-run --modality text --text 'A person stands still.'
```

`--check-only` validates dependencies and service availability. `--dry-run` generates two motion chunks from a static pose. Neither initializes DDS nor starts the controller.

Support the robot and keep the physical emergency stop available during live operation. Start the interactive client:

```bash
source sim2real/client.sh
```

The client starts the state reader and controller processes. Use the keyboard to activate standing and select a task:

| Key | Action |
| --- | --- |
| `s` | Start standing in planner mode |
| Enter while idle | Switch between planner and task mode |
| `1` / `2` / `3` / `4` | Select text / video / speech / music in task mode |
| Enter while editing | Confirm the description or local media path |
| `]` | Prepare and run the selected task |
| `h` | Select an input from this session's history |
| `!` | Return to planner mode |
| `o` outside editing / Ctrl+C | Stop / exit |

Input changes are disabled during motion. While entering a description or path, Esc cancels editing and `o` is ordinary text.

## 5. Task Input

To preselect an input when launching the client:

```bash
source sim2real/client.sh --modality text --text 'wave your hand.' --seconds 6
source sim2real/client.sh --modality video --video /path/on/g1/reference.mp4
source sim2real/client.sh --modality speech --speech /path/on/g1/instruction.wav --seconds 6
source sim2real/client.sh --modality music --music /path/on/g1/music.wav
```

These commands use the same activation keys. Text and speech default to six seconds; `--seconds` changes their duration. Video and music follow the media duration. Media files must exist on the G1, which uploads them to the server.

## Logs

| Path on the G1 | Content |
| --- | --- |
| `/tmp/movens_real_logs/g1_state.log` | Live state reader output |
| `/tmp/movens_real_logs/wbc.log` | Whole-body controller output |
| `/tmp/movens_real_logs/bridge.log` | Motion streaming and task diagnostics |
| `sim2real/logs/generated_joint_pos_real.csv` | Generated joint reference positions |

Preparing another task can overwrite the bridge log and generated CSV. See [Operations](OPERATIONS.md) for troubleshooting and configuration details.

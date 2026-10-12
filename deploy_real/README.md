# Movens-Zero Real-Robot Deployment

Run text-, video-, speech-, and music-conditioned motion on a Unitree G1.

```text
GPU workstation                    G1 onboard computer
Movens inference server <--WS--> Movens client --> whole-body controller
                                     ^
                                  live state
```

## 1. Requirements

| GPU workstation | G1 onboard computer |
| --- | --- |
| Linux, Python 3.10, CUDA-capable GPU with BF16 | Python 3.10, Unitree SDK and matching CycloneDDS |
| Movens root40 checkpoint and Qwen2.5-Omni-3B | Installed whole-body controller and its policy assets |
| FFmpeg and FFprobe | FFmpeg and FFprobe |

The G1 does not need CUDA, Torch, or Movens weights. See
[Dependencies](DEPENDENCIES.md) for the controller installation and required files.

## 2. Set Up the Server

Run from the repository root on the GPU workstation:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r deploy_real/requirements-server.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
cp deploy_real/config/server.env.example .deploy_real.env
```

Edit `.deploy_real.env`:

```bash
MOVENS_SERVER_PYTHON="/absolute/path/to/Movens-Zero/.venv/bin/python"
MOVENS_CHECKPOINT="/absolute/path/to/ckpt_99000/model.safetensors"
MOVENS_PRETRAIN_BACKBONE_PATH="/absolute/path/to/Qwen2.5-Omni-3B"
CUDA_VISIBLE_DEVICES=0
MOVENS_PORT=8014
```

Weights are distributed separately. The server checks the checkpoint against
`config/release_manifest.json` and uses the bundled root40 normalization.

## 3. Set Up the Client

Run from the repository root on the G1:

```bash
cp deploy_real/config/client.env.example .deploy_real.env
```

Set `MOVENS_HOST` to the GPU workstation's reachable LAN address. Set `WBC_ROOT`,
`UNITREE_SDK_ROOT`, the DDS library paths, and `G1_NETWORK_INTERFACE` for this robot.
Then install the lightweight client environment:

```bash
bash deploy_real/setup_client.sh
```

Each host has its own `.deploy_real.env`. Keep it outside `deploy_real/` and do not
commit it. See [Configuration and upgrades](OPERATIONS.md#configuration-and-upgrades)
when updating an existing installation.

## 4. Start Deployment

**Terminal 1 — GPU workstation:**

```bash
bash deploy_real/server.sh
```

**Terminal 2 — G1:** first check the installation and request a non-actuating inference:

```bash
bash deploy_real/client.sh --check-only --modality text --text 'A person stands still.'
bash deploy_real/client.sh --dry-run --modality text --text 'A person stands still.'
```

`--dry-run` requests two model chunks without initializing DDS or starting the
controller. Before live operation, support the robot and keep the physical
emergency stop available. Then start the interactive client:

```bash
bash deploy_real/client.sh
```

Launching initializes the deployment processes; there is no initial confirmation
prompt. Motion activation remains under the following keyboard controls.

| Key | Action |
| --- | --- |
| `s` | Start standing in planner mode |
| Enter | Switch from planner to task mode |
| `1` / `2` / `3` / `4` | Select text / video / speech / music |
| Enter while editing | Confirm a description or local media path |
| `]` | Prepare and run the selected task |
| `h` | Reuse an input from this session's history |
| Enter / `!` in task mode | Return to planner |
| `o` / Ctrl+C | Stop / exit |

Missing or unreadable media paths produce an error without replacing the selected
task. History lasts until the client exits. While editing, Esc cancels, Backspace
edits, Ctrl+U clears the line, and `o` is ordinary text. Input changes are blocked
during motion.

## 5. Select a Task from the Command Line

These commands preselect the task; the same activation keys still apply.

```bash
bash deploy_real/client.sh --modality text --text 'wave your hand.' --seconds 6
bash deploy_real/client.sh --modality video --video /path/on/g1/reference.mp4
bash deploy_real/client.sh --modality speech --speech /path/on/g1/instruction.wav --seconds 6
bash deploy_real/client.sh --modality music --music /path/on/g1/music.wav
```

Text and speech default to six seconds; video and music follow media duration.
Media paths refer to files on the G1, which uploads them to the server.

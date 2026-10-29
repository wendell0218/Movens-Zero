# Sim2Sim

Run Movens-Zero with the SONIC whole-body controller in MuJoCo. The simulation feeds its measured state back to the model and continues running during inference, so the closed loop includes actual inference latency. Run the commands below from the repository root.

## 1. Setup

Use the repository's inference environment, a CUDA GPU with BF16 support, and FFmpeg/FFprobe. Install the server dependencies and download a checkpoint:

```bash
conda activate movens-zero
pip install -r sim2real/requirements-server.txt
source scripts/download_models.sh --model-size XL
```

Install [GR00T WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl), including the SONIC controller and policy assets, Unitree SDK, and CycloneDDS. The required controller files are listed in [Dependencies](../sim2real/DEPENDENCIES.md).

For a fresh upstream checkout, its MuJoCo installer creates the separate `.venv_sim` environment. Install the bridge dependencies in that environment:

```bash
SONIC_ROOT="/path/to/GR00T-WholeBodyControl"
bash "$SONIC_ROOT/install_scripts/install_mujoco_sim.sh"
source "$SONIC_ROOT/.venv_sim/bin/activate"
pip install 'pandas>=2.2,<3' 'websocket-client>=1.8,<2' 'pyzmq>=26,<28'
```

If `.venv_sim` is already installed, activate it and install only the bridge dependencies.

## 2. Configure

Create the local server and simulation settings:

```bash
cp sim2real/config/server.env.example .sim2real.env
cp sim2sim/config.env.example .sim2sim.env
```

Set `SONIC_ROOT` in `.sim2sim.env` to the installed upstream checkout:

```bash
SONIC_ROOT="/path/to/GR00T-WholeBodyControl"
```

The server defaults to the downloaded XL checkpoint on GPU 0. Edit `.sim2real.env` to change the checkpoint or GPU. If the server runs on another machine, set `MOVENS_HOST` and `MOVENS_PORT` in `.sim2sim.env` to match it.

## 3. Start the server

In terminal 1, activate the inference environment and start the model server:

```bash
conda activate movens-zero
source sim2real/server.sh
```

The server enforces a **180 ms inference deadline**. The selected model, GPU, and attention backend must meet that deadline; otherwise the run stops with an error.

## 4. Check and run

In terminal 2, check the simulation dependencies and live server:

```bash
source sim2sim/run.sh --check-only
```

Then run a text-conditioned motion:

```bash
source sim2sim/run.sh \
  --mode text --prompt "A person walks slowly forward." \
  --seconds 6 --output-dir outputs/sim2sim_walk
```

For a video input:

```bash
source sim2sim/run.sh \
  --mode video --input-path /path/to/video.mp4 \
  --seconds 6 --output-dir outputs/sim2sim_video
```

Use `--mode music` or `--mode speech` with an audio `--input-path` for the other modalities. `--seconds` sets the motion duration for every modality. Each run needs an empty output directory. Ctrl+C stops the processes launched for that run.

## 5. Render

Render the recorded simulation trajectory:

```bash
source sim2sim/render.sh outputs/sim2sim_walk
```

The video is saved as `outputs/sim2sim_walk/simulation.mp4`. Keep the upstream robot assets available when rendering.

## Outputs

Files are saved under the selected `--output-dir`:

| Path | Content |
| --- | --- |
| `simulated_states.npz` | Measured robot motion, MuJoCo poses, and timestamps |
| `simulation.mp4` | Rendered simulation trajectory |
| `generated_joint_pos.csv` | Generated joint references |
| `latency.jsonl` | Inference latency and received chunk timestamps |
| `run_config.json` | Run settings and model identity |
| `run_status.json` | Completion status or failure |
| `simulation_status.json` | Simulation timing and recorded frame count |
| `simulator.log`, `wbc.log`, `bridge.log` | Simulator, controller, and inference bridge logs |

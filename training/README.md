# Training

Continue training a released Movens-Zero checkpoint with text, speech, music, or video. Run the commands below from the repository root.

## 1. Setup

Install the training dependencies and download the checkpoint:

```bash
pip install -e ".[train]"
source scripts/download_models.sh --model-size XL
```

## 2. Prepare data

This example prepares text and video pairs from the public 1,000-sample [Movens-5M subset](https://huggingface.co/datasets/wendell0218/Movens-5M):

```bash
hf download wendell0218/Movens-5M --repo-type dataset --local-dir data/Movens-5M
python training/prepare_data.py \
  --source data/Movens-5M \
  --output data/movens_demo \
  --target g1_reference
```

The command extracts videos, converts and resamples reference motions to 50 Hz, and writes `data/movens_demo/train.jsonl`. To use executed tracking targets, choose `--target g1_tracking` and provide `--robot-xml` for the matching robot model.

### Custom datasets

Create a JSONL manifest with one motion/condition pair per line:

```jsonl
{"id":"walk_001","mode":"text","motion":"motion/walk_001.npy","fps":50,"text":"A person walks forward."}
{"id":"video_001","mode":"video","motion":"motion/video_001.npy","fps":50,"media":"video/video_001.mp4"}
{"id":"music_001","mode":"music","motion":"motion/music_001.npy","fps":50,"media":"audio/music_001.wav"}
{"id":"speech_001","mode":"speech","motion":"motion/speech_001.npy","fps":50,"media":"audio/speech_001.wav"}
```

Paths are relative to the manifest. Each motion is a 50 Hz NumPy array of shape `(T, 36)`: **29 IsaacLab-ordered joint angles, root xyz in meters, and root quaternion wxyz**. The loader constructs the normalized 40D representation from these absolute poses.

Video and music must align with the motion timeline; use `media_offset_seconds` when the motion starts later in the media file. Speech uses the full utterance. Split training and validation by source video.

## 3. Train

```bash
python training/train.py \
  --config configs/inference_xl.json \
  --manifest data/movens_demo/train.jsonl \
  --checkpoint checkpoints/Movens-Zero-XL/model.safetensors \
  --batch-size 4 --accumulation 4 \
  --head-lr 1e-4 --backbone-lr 1e-5 \
  --max-steps 30000 --save-every 1000 --keep-last 3 \
  --output runs/movens_zero
```

Use a config and checkpoint of the same model size. `--batch-size` is per GPU; the effective batch is `batch-size × accumulation × GPU count`. For multiple GPUs, replace `python` with `torchrun --standalone --nproc-per-node=2` and set the GPU count as needed.

The action head and Qwen backbone use separate learning rates. Add `--freeze-backbone` to train only the action head. CFG condition dropout defaults to `0.1`.

## 4. Resume

Resume a saved training run, including its optimizer, scheduler, step, and random-number state:

```bash
python training/train.py \
  --config configs/inference_xl.json \
  --manifest data/movens_demo/train.jsonl \
  --resume runs/movens_zero/ckpt_030000 \
  --batch-size 4 --accumulation 4 \
  --head-lr 1e-4 --backbone-lr 1e-5 \
  --max-steps 40000 --save-every 1000 --keep-last 3 \
  --output runs/movens_zero
```

Keep the same GPU count, batch settings, data manifest, and training recipe. `--max-steps` is the total target step, so this example adds 10,000 steps after step 30,000. For a distributed run, use the same `torchrun` launcher when resuming.

To change the recipe or continue from released weights, use `--checkpoint /path/to/model.safetensors` instead of `--resume`; this starts a new optimizer and step count.

## Outputs

| Path | Content |
| --- | --- |
| `data/movens_demo/train.jsonl` | Prepared training manifest |
| `runs/movens_zero/train.jsonl` | Loss, learning rates, effective batch, step time, and gradient norm |
| `runs/movens_zero/run_config.json` | Training parameters |
| `runs/movens_zero/ckpt_XXXXXX/model.safetensors` | Model weights |
| `runs/movens_zero/ckpt_XXXXXX/training_state.pt` | Optimizer, scheduler, step, and random-number state |
| `runs/movens_zero/ckpt_XXXXXX/` | Run config, inference config, and normalization snapshot |

The example saves every 1,000 steps and retains the latest three checkpoints. Copy checkpoints you want to keep outside the run directory.

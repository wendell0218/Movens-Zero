# Movens-Intent Evaluation

Evaluate a Movens-Zero checkpoint on text, speech, music, or video. Run the commands below from the repository root.

## 1. Setup

Install the evaluation dependencies and download the model and evaluation assets:

```bash
pip install -e ".[eval]"
source scripts/download_models.sh --model-size XL
python evaluation/setup.py
```

## 2. Prepare data

Choose a track: `text`, `speech`, `music`, or `video`. Use the same `TRACK` for the following steps.

```bash
TRACK="text"
python evaluation/prepare.py \
  --track "$TRACK" \
  --output-dir "data/intent/$TRACK"
```

## 3. Generate motion

```bash
python evaluation/generate.py \
  --track "$TRACK" \
  --checkpoint checkpoints/Movens-Zero-XL/model.safetensors \
  --data-dir "data/intent/$TRACK" \
  --output-dir "outputs/intent/$TRACK"
```

Use `--checkpoint /path/to/model.safetensors` to evaluate another checkpoint. Rerun the same command to continue interrupted generation.

## 4. Compute metrics

```bash
python evaluation/score.py \
  --track "$TRACK" \
  --data-dir "data/intent/$TRACK" \
  --generation-dir "outputs/intent/$TRACK" \
  --output-dir "outputs/intent/$TRACK/metrics"
```

## Outputs

| Path | Content |
| --- | --- |
| `data/intent/<track>/` | Prepared conditions and reference motions |
| `outputs/intent/<track>/generated_qpos.npz` | Generated trajectories |
| `outputs/intent/<track>/metrics/` | Evaluation results |

## Dependencies

Setup downloads pinned [metric utilities](https://github.com/Tsinghua-MARS-Lab/OMG/tree/61e196010b332acbace223b2c449e25f454ea0a3), the [G1 robot model](https://github.com/YanjieZe/GMR), the [Movens-Intent evaluator](https://huggingface.co/wendell0218/Movens-Intent-Evaluator), and the [tracking policy](https://huggingface.co/HorizonRobotics/HoloMotion_models). Upstream licenses are retained with the downloaded dependencies.

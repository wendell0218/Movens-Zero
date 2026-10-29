import json
from pathlib import Path
import random
import shutil

import numpy as np
from safetensors.torch import save_file
import torch


def capture_rng(device=None):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
        torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device)
        if device is not None and device.type == "cuda" else None)


def restore_rng(state, device=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def save_checkpoint(folder, model, optimizer, scheduler, step, rank_rng, run,
                    config_path, normalization_path):
    folder = Path(folder)
    staging = folder.parent / ("." + folder.name + ".incomplete")
    staging.mkdir(parents=True, exist_ok=False)
    weights = {}
    for key, value in model.state_dict().items():
        public_key = ("action_header." + key.removeprefix("action_expert.")) if key.startswith("action_expert.") else ("vlm_model." + key.removeprefix("backbone."))
        weights[public_key] = value.detach().cpu().contiguous().clone()
    save_file(weights, str(staging / "model.safetensors"))
    del weights
    torch.save(dict(step=step, optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), rank_rng=rank_rng), staging / "training_state.pt")
    (staging / "run_config.json").write_text(json.dumps(run, indent=2) + "\n")
    shutil.copy2(config_path, staging / "inference_config.json")
    shutil.copy2(normalization_path, staging / "normalization.json")
    staging.rename(folder)

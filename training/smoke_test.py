import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from safetensors.torch import load_file
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from movens_zero.config import load_config
from movens_zero.model import _ActionExpert
from movens_zero.normalization import MotionNormalizer
from training.checkpoint import capture_rng, restore_rng, save_checkpoint
from training.data import MotionBatches
from training.objective import FlowObjective, flow_random_batch


class TinyRuntime(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Embedding(32, 16)
        self.action_expert = _ActionExpert(dict(action_dim=40, action_horizon=30,
            state_dim=40, hidden_dim=64, num_blocks=2, num_heads=4,
            qk_norm=None, view_feature_dim=16, use_future_progress=True,
            future_progress_dim=5))
        self.train_diffusion_steps = 1000

    def _extract_context(self, inputs):
        return self.backbone(inputs["input_ids"]).unsqueeze(1)

    def _autocast(self):
        return contextlib.nullcontext()


def update(objective, optimizer, scheduler, batch, index, rank=0, world=1):
    actions, states, progress, valid = [batch[key] for key in ("actions", "states", "progress", "valid")]
    sigma, delay, noise = flow_random_batch(actions, 123, index, rank, world)
    inputs = dict(input_ids=torch.tensor([[1, 2, 3, 0]] * len(actions)),
                  attention_mask=torch.tensor([[1, 1, 1, 0]] * len(actions)))
    optimizer.zero_grad(set_to_none=True)
    loss = objective(inputs, actions, states, progress, valid, sigma, delay, noise)
    loss.backward()
    assert torch.isfinite(loss), "Non-finite training loss"
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ddp", action="store_true", help="Run under torchrun --nproc-per-node=2")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if args.ddp:
        dist.init_process_group("gloo")
    elif world != 1:
        raise ValueError("Use --ddp under torchrun")
    random.seed(17 + rank)
    np.random.seed(17 + rank)
    torch.manual_seed(17)
    config = load_config("configs/inference_l.json")
    normalizer = MotionNormalizer.from_file(config.normalization)
    with tempfile.TemporaryDirectory(prefix="movens_training_smoke_") as temporary:
        directory = Path(temporary)
        motion = np.zeros((101, 36), dtype=np.float32)
        motion[:, 29] = np.linspace(0, 1, 101)
        motion[:, 31], motion[:, 32] = 0.8, 1.0
        np.save(directory / "motion.npy", motion)
        manifest = directory / "train.jsonl"
        manifest.write_text(json.dumps(dict(mode="text", motion="motion.npy", text="A person walks forward.", fps=50)) + "\n")
        dataset = MotionBatches(manifest, config, normalizer, 2, 4, rank=rank, seed=23)
        batch0, batch1 = dataset[0], dataset[1]
        resumed = MotionBatches(manifest, config, normalizer, 2, 4, start_microstep=1, rank=rank, seed=23)[0]
        for key in ("actions", "states", "progress", "valid"):
            assert torch.equal(batch1[key], resumed[key]), f"Resume changed {key}"
        assert batch1["prompts"] == resumed["prompts"], "Resume changed text prompts"
        model = TinyRuntime()
        objective = FlowObjective(model, cfg_dropout=0.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        if args.ddp:
            runner = DistributedDataParallel(objective, find_unused_parameters=True)
            optimizer.zero_grad(set_to_none=True)
            loss = 0.0
            for index, batch in enumerate((batch0, batch1)):
                sigma, delay, noise = flow_random_batch(batch["actions"], 123, index, rank, world)
                inputs = dict(input_ids=torch.tensor([[1, 2, 3, 0]] * 2),
                              attention_mask=torch.tensor([[1, 1, 1, 0]] * 2))
                sync = runner.no_sync() if index == 0 else contextlib.nullcontext()
                with sync:
                    value = runner(inputs, batch["actions"], batch["states"],
                        batch["progress"], batch["valid"], sigma, delay, noise)
                    (value / 2).backward()
                    loss += float(value.detach()) / 2
            optimizer.step()
            scheduler.step()
            gathered = [None] * world
            dist.all_gather_object(gathered, model.action_expert.action_proj_out.linear.weight.detach())
            assert all(torch.equal(gathered[0], tensor) for tensor in gathered), "DDP ranks diverged"
            if rank == 0:
                print(json.dumps(dict(test="cpu_ddp", world_size=world, loss=loss,
                    accumulation=2, synchronized=True)))
            dist.destroy_process_group()
            return
        initial = model.action_expert.action_proj_out.linear.weight.detach().clone()
        first_loss = update(objective, optimizer, scheduler, batch0, 0)
        assert not torch.equal(initial, model.action_expert.action_proj_out.linear.weight), "Optimizer did not update real head"
        rng = capture_rng()
        checkpoint = directory / "ckpt_000001"
        save_checkpoint(checkpoint, model, optimizer, scheduler, 1, [rng], {},
                        "configs/inference_l.json", config.normalization)
        expected_loss = update(objective, optimizer, scheduler, batch1, 1)
        expected = copy.deepcopy(model.state_dict())
        restored_model = TinyRuntime()
        public = load_file(str(checkpoint / "model.safetensors"))
        state = {key.replace("action_header.", "action_expert.", 1).replace("vlm_model.", "backbone.", 1): value
                 for key, value in public.items()}
        restored_model.load_state_dict(state, strict=True)
        restored_objective = FlowObjective(restored_model, cfg_dropout=0.5)
        restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
        restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
        saved = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        restored_optimizer.load_state_dict(saved["optimizer"])
        restored_scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rank_rng"][0])
        actual_loss = update(restored_objective, restored_optimizer, restored_scheduler, resumed, 1)
        assert expected_loss == actual_loss, (expected_loss, actual_loss)
        assert all(torch.equal(expected[key], value) for key, value in restored_model.state_dict().items()), "Restored optimizer trajectory changed"
        reference = torch.zeros((4, 30, 40))
        full = flow_random_batch(reference, 0, 0, 0, 1)
        left = flow_random_batch(reference[:2], 0, 0, 0, 2)
        right = flow_random_batch(reference[:2], 0, 0, 1, 2)
        assert torch.equal(full[0], torch.cat([left[0], right[0]])), "Sigma coverage changed across ranks"
        assert torch.equal(full[1], torch.cat([left[1], right[1]])), "Delay coverage changed across ranks"
        assert not torch.equal(left[2], right[2]), "Ranks share action noise"
        print(json.dumps(dict(test="cpu_training_resume", first_loss=first_loss,
            resumed_loss=actual_loss, exact_resume=True, data_cursor=True,
            distributed_random_coverage=True, checkpoint_keys=len(public))))


if __name__ == "__main__":
    main()

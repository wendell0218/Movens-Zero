import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from movens_zero.config import load_config
from movens_zero.model import MovensZeroModel
from movens_zero.normalization import MotionNormalizer
from training.data import MotionBatches
from training.objective import FlowObjective, flow_random_batch
from training.checkpoint import capture_rng, restore_rng, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inference_l.json")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", help="model.safetensors: weights-only continuation")
    parser.add_argument("--resume", help="checkpoint directory: restores all training state")
    parser.add_argument("--fresh", action="store_true", help="Initialize a new head on pretrained Qwen")
    parser.add_argument("--backbone", default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--output", default="runs/movens_zero")
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-rank microbatch")
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--cfg-dropout", type=float, default=0.1)
    parser.add_argument("--group-weights", default="training/loss_weights.json")
    parser.add_argument("--task-weights", help='JSON object, e.g. {"text":0.5,"video":0.5}')
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--keep-last", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "flash_attention_2"])
    args = parser.parse_args()
    if sum([bool(args.checkpoint), bool(args.resume), args.fresh]) != 1:
        parser.error("Choose exactly one of --checkpoint, --resume or --fresh")
    if min(args.max_steps, args.batch_size, args.accumulation, args.save_every, args.keep_last) <= 0:
        parser.error("Step, batch, accumulation, save and retention values must be positive")
    if not 0 <= args.cfg_dropout < 1 or args.workers < 0:
        parser.error("Invalid CFG dropout or worker count")
    if not torch.cuda.is_available():
        raise RuntimeError("The full multimodal training model requires CUDA")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    config = load_config(args.config)
    normalizer = MotionNormalizer.from_file(config.normalization)
    run = vars(args).copy()
    run["world_size"] = world
    run["manifest_sha256"] = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    run["config_sha256"] = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    run["normalization_sha256"] = hashlib.sha256(Path(config.normalization).read_bytes()).hexdigest()
    run["group_weights"] = json.loads(Path(args.group_weights).read_text())
    run["task_weights"] = json.loads(args.task_weights) if args.task_weights else None
    resume_state = None
    start = 0
    if args.resume:
        folder = Path(args.resume).resolve(strict=True)
        saved = json.loads((folder / "run_config.json").read_text())
        for key in ("world_size", "batch_size", "accumulation", "seed", "manifest_sha256",
                    "cfg_dropout", "group_weights", "task_weights", "freeze_backbone",
                    "head_lr", "backbone_lr", "weight_decay", "warmup_steps",
                    "config_sha256", "normalization_sha256"):
            if saved[key] != run[key]:
                raise ValueError(f"Full resume cannot change {key}; use --checkpoint for a new recipe")
        resume_state = torch.load(folder / "training_state.pt", map_location="cpu", weights_only=False)
        start = int(resume_state["step"])
        checkpoint = folder / "model.safetensors"
    else:
        checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if start >= args.max_steps:
        raise ValueError("--max-steps must exceed the saved training step")
    if checkpoint:
        model = MovensZeroModel.from_pretrained(checkpoint, args.backbone, config,
            device=device, torch_dtype=torch.bfloat16, attn_implementation=args.attn_implementation)
    else:
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
        backbone = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(args.backbone,
            dtype=torch.bfloat16, attn_implementation=args.attn_implementation)
        model = MovensZeroModel(backbone, Qwen2_5OmniProcessor.from_pretrained(args.backbone), config)
        model.runtime_device, model.backbone_dtype = device, torch.bfloat16
        model.to(device)
    model.action_expert.float()
    model.backbone.requires_grad_(not args.freeze_backbone)
    if not args.freeze_backbone:
        model.backbone.float()
    model.backbone.config.use_cache = False
    if args.gradient_checkpointing and not args.freeze_backbone:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    if args.freeze_backbone:
        model.backbone.eval()
    objective = FlowObjective(model, args.cfg_dropout, run["group_weights"]).to(device)
    groups = [{"params": [p for p in model.action_expert.parameters() if p.requires_grad], "lr": args.head_lr}]
    if not args.freeze_backbone:
        groups.append({"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps)))
    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
    runner = DistributedDataParallel(objective, device_ids=[local_rank], find_unused_parameters=True) if world > 1 else objective
    dataset = MotionBatches(args.manifest, config, normalizer, args.batch_size,
        args.max_steps * args.accumulation, start * args.accumulation, rank, args.seed,
        run["task_weights"])
    loader = DataLoader(dataset, batch_size=None, num_workers=args.workers,
        pin_memory=True, **({"prefetch_factor": 2, "multiprocessing_context": "spawn"} if args.workers else {}))
    iterator = iter(loader)
    if resume_state:
        restore_rng(resume_state["rank_rng"][rank], device)
        del resume_state
    output = Path(args.output).resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_config.json").write_text(json.dumps(run, indent=2) + "\n")
    trainable = [p for p in objective.parameters() if p.requires_grad]
    for step in range(start, args.max_steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        for micro in range(args.accumulation):
            batch = next(iterator)
            actions, states, progress, valid = [batch[key].to(device, non_blocking=True)
                for key in ("actions", "states", "progress", "valid")]
            inputs = model._prepare_inputs(batch["prompts"], batch["mode"],
                batch["media"] if batch["mode"] in ("music", "speech") else None,
                batch["media"] if batch["mode"] == "video" else None,
                config.modalities.video.sample_fps, config.modalities.video.size_pixels)
            sigma, delays, noise = flow_random_batch(actions, args.seed,
                int(batch["microstep"]), rank, world, int(config.model.rtc_max_delay))
            delays = torch.minimum(delays, valid.sum(1) - 1)
            sync = runner.no_sync() if world > 1 and micro + 1 < args.accumulation else contextlib.nullcontext()
            with sync:
                loss = runner(inputs, actions, states, progress, valid, sigma, delays, noise)
                (loss / args.accumulation).backward()
            total_loss += loss.detach() / args.accumulation
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if world > 1:
            dist.all_reduce(total_loss)
            total_loss /= world
        torch.cuda.synchronize(device)
        row = dict(step=step + 1, loss=float(total_loss), grad_norm=float(grad_norm),
            seconds=time.perf_counter() - started, lr=[group["lr"] for group in optimizer.param_groups],
            effective_batch=args.batch_size * args.accumulation * world, mode=batch["mode"])
        if rank == 0:
            print(json.dumps(row), flush=True)
            with (output / "train.jsonl").open("a") as log:
                log.write(json.dumps(row) + "\n")
        if (step + 1) % args.save_every == 0 or step + 1 == args.max_steps:
            state = capture_rng(device)
            rank_rng = [None] * world
            if world > 1:
                dist.all_gather_object(rank_rng, state)
            else:
                rank_rng[0] = state
            if rank == 0:
                folder = output / f"ckpt_{step + 1:06d}"
                save_checkpoint(folder, model, optimizer, scheduler, step + 1,
                    rank_rng, run, args.config, config.normalization)
                for old in sorted(output.glob("ckpt_[0-9]*"))[:-args.keep_last]:
                    shutil.rmtree(old)
            if world > 1:
                dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from movens_zero.config import load_config
from movens_zero.export import export_motion
from movens_zero.normalization import MotionNormalizer, load_initial_pose
from movens_zero.prompts import compose_motion_prompt
from movens_zero.rollout import generate_motion


MODES = ("text", "music", "speech", "video")
PACKAGE_ASSETS = Path(__file__).resolve().parent / "assets"


def parse_args(default_mode: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Movens-Zero inference")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=default_mode,
        required=default_mode is None,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ASSETS / "inference.json",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--backbone", default="")
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--initial-pose", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--name", default="motion")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-output-frames", type=int, default=300)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main(default_mode: str | None = None) -> None:
    args = parse_args(default_mode)
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {config_path}")
    config = load_config(config_path)
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path.cwd() / config.checkpoint
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if args.num_output_frames <= 0:
        raise ValueError("--num-output-frames must be positive")
    if not args.prompt.strip():
        raise ValueError("--prompt cannot be empty")
    if args.mode == "text" and args.input_path is not None:
        raise ValueError("--input-path is not valid for text mode")
    if args.mode != "text" and args.input_path is None:
        raise ValueError(f"--input-path is required for {args.mode} mode")

    input_path = None
    if args.input_path is not None:
        input_path = args.input_path.expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Input does not exist: {input_path}")

    backbone = args.backbone or config.model.backbone
    inference_steps = (
        args.num_inference_steps
        if args.num_inference_steps is not None
        else int(config.default_inference_steps)
    )
    if inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive")
    seed = args.seed if args.seed is not None else int(config.seed)

    initial_pose_path = args.initial_pose
    if initial_pose_path is None:
        initial_pose_path = PACKAGE_ASSETS / Path(config.initial_pose).name
    initial_pose_path = initial_pose_path.expanduser().resolve()
    normalization_path = PACKAGE_ASSETS / Path(config.normalization).name
    initial_pose = load_initial_pose(initial_pose_path)
    normalizer = MotionNormalizer.from_file(normalization_path)

    preview = {
        "mode": args.mode,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "backbone": backbone,
        "input": str(input_path) if input_path else None,
        "initial_pose": str(initial_pose_path),
        "output": str(output_dir),
        "frames": args.num_output_frames,
        "steps": inference_steps,
        "device": args.device,
        "attention": args.attention_implementation,
        "seed": seed,
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return

    from movens_zero.model import MovensZeroModel

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    prompt = args.prompt.strip()
    if args.mode == "text":
        prompt = compose_motion_prompt(prompt, random.Random(seed))

    model = MovensZeroModel.from_pretrained(
        checkpoint_path=checkpoint_path,
        backbone_path=backbone,
        config=config,
        device=args.device,
        attn_implementation=args.attention_implementation,
    )
    model.to(args.device)
    model.eval()
    poses = generate_motion(
        model=model,
        config=config,
        normalizer=normalizer,
        mode=args.mode,
        prompt=prompt,
        input_path=input_path,
        initial_pose=initial_pose,
        output_frames=args.num_output_frames,
        inference_steps=inference_steps,
        device=args.device,
    )
    export_motion(
        output_dir,
        poses,
        int(config.motion.fps),
        {
            "mode": args.mode,
            "prompt": args.prompt.strip(),
            "name": args.name,
            "checkpoint": checkpoint_path.name,
            "backbone": backbone,
            "inference_steps": inference_steps,
        },
    )
    print(f"Motion written to {output_dir}")


if __name__ == "__main__":
    main()

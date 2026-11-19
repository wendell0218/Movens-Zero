from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from movens_zero.config import load_config
from movens_zero.conditioning import DEFAULT_PROMPTS, build_prompt
from movens_zero.export import export_motion
from movens_zero.motion_format import (
    continuous_quaternion_signs,
    motion36_to_qpos,
    qpos_to_motion36,
    sample_motion36,
)
from movens_zero.normalization import MotionNormalizer, load_initial_pose
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
        default=None,
    )
    parser.add_argument("--model-size", choices=("S", "M", "L", "XL"), default="XL")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--backbone", default="")
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--initial-pose", type=Path)
    parser.add_argument("--prompt", help="Override the default task instruction for any modality")
    parser.add_argument("--description", help="Required for text; describes the motion to generate")
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
    parser.add_argument(
        "--postprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cfg-scale", type=float)
    parser.add_argument("--cfg-interval", nargs=2, type=float)
    parser.add_argument("--auto-zoom", action=argparse.BooleanOptionalAction, default=True,
                        help="Video only: zoom in on the main person when they appear small (default: on)")
    args = parser.parse_args()
    if args.prompt is None:
        args.prompt = DEFAULT_PROMPTS[args.mode]
    if args.description is not None:
        args.description = args.description.strip()
    try:
        args.instruction = build_prompt(args.mode, args.prompt, args.description)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(default_mode: str | None = None) -> None:
    args = parse_args(default_mode)
    config_path = (args.config or PACKAGE_ASSETS / f"inference_{args.model_size.lower()}.json").expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {config_path}")
    config = load_config(config_path)
    if args.cfg_scale is not None:
        config.cfg_scale = args.cfg_scale
    if args.cfg_interval is not None:
        config.cfg_interval = args.cfg_interval
    if not np.isfinite(config.cfg_scale) or config.cfg_scale < 0:
        raise ValueError("CFG scale must be finite and nonnegative")
    low, high = config.cfg_interval
    if not 0 <= low < high <= 1:
        raise ValueError("CFG interval must satisfy 0 <= low < high <= 1")
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        candidates = [Path.cwd() / config.checkpoint, Path(__file__).resolve().parents[1] / config.checkpoint]
        checkpoint_path = next((path for path in candidates if path.is_file()), candidates[0])
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    from movens_zero.checkpoints import checkpoint_model_config
    architecture = checkpoint_model_config(checkpoint_path)
    if architecture["action_dim"] != int(config.model.action_dim):
        raise ValueError("Checkpoint representation does not match the selected configuration")
    if args.num_output_frames <= 0:
        raise ValueError("--num-output-frames must be positive")
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

    metadata = {
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
        "cfg_scale": config.cfg_scale,
        "cfg_interval": config.cfg_interval,
        "motion_representation": config.motion.representation,
        "checkpoint_architecture": architecture,
        "auto_zoom": False,
        "postprocess": args.postprocess,
        "postprocess_time_seconds": 0.0,
        "postprocess_clamped_50hz_frames": 0,
        "description": args.description,
        "effective_prompt": args.instruction,
    }

    from movens_zero.model import MovensZeroModel

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    prompt = args.instruction

    video_decoder = None
    if args.mode == "video" and args.auto_zoom:
        from movens_zero.auto_zoom import PersonZoom
        zoom = PersonZoom.from_video(input_path, int(config.modalities.video.size_pixels), args.device)
        video_decoder = zoom.decode
        metadata["auto_zoom"] = dict(enabled=zoom.enabled, person_pixels=round(zoom.person_pixels, 1))

    model = MovensZeroModel.from_pretrained(
        checkpoint_path=checkpoint_path,
        backbone_path=backbone,
        config=config,
        device=args.device,
        attn_implementation=args.attention_implementation,
    )
    model.to(args.device)
    model.eval()
    device_type = torch.device(args.device).type
    with torch.inference_mode(), torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
        future = generate_motion(
            model=model,
            config=config,
            normalizer=normalizer,
            mode=args.mode,
            prompt=prompt,
            input_path=input_path,
            initial_pose=initial_pose,
            output_frames=max(1, args.num_output_frames - 1),
            inference_steps=inference_steps,
            device=args.device,
            video_decoder=video_decoder,
        )
    poses = np.concatenate([initial_pose[None].astype(np.float32), future])[: args.num_output_frames]
    if args.postprocess:
        from movens_zero.jerk import postprocess_qpos

        started = time.perf_counter()
        output_dtype = poses.dtype
        output_frames = len(poses)
        motion_fps = int(config.motion.fps)
        frames_30hz = ((output_frames - 1) * 30) // motion_fps + 1
        sample_times_30hz = np.arange(frames_30hz, dtype=np.float64) / 30.0
        poses_50hz = poses
        poses_30hz = sample_motion36(
            continuous_quaternion_signs(poses_50hz),
            source_fps=motion_fps,
            sample_times=sample_times_30hz,
        )
        benchmark_30hz = sample_motion36(
            poses_50hz,
            source_fps=motion_fps,
            sample_times=sample_times_30hz,
        )
        poses_30hz[:, 32:36] = benchmark_30hz[:, 32:36]
        qpos_30hz = motion36_to_qpos(poses_30hz)
        qpos_30hz, _ = postprocess_qpos(qpos_30hz, fps=30.0)
        poses_30hz = continuous_quaternion_signs(qpos_to_motion36(qpos_30hz))
        sample_times_50hz = np.arange(output_frames, dtype=np.float64) / motion_fps
        last_time_30hz = (frames_30hz - 1) / 30.0
        clamped_frames = int(np.count_nonzero(sample_times_50hz > last_time_30hz + 1e-9))
        sample_times_50hz = np.minimum(sample_times_50hz, last_time_30hz)
        poses = sample_motion36(
            poses_30hz,
            source_fps=30.0,
            sample_times=sample_times_50hz,
        ).astype(output_dtype, copy=False)
        metadata["postprocess_time_seconds"] = time.perf_counter() - started
        metadata["postprocess_clamped_50hz_frames"] = clamped_frames
    export_motion(
        output_dir,
        poses,
        int(config.motion.fps),
        {
            **metadata,
            "mode": args.mode,
            "prompt": args.prompt.strip(),
            "checkpoint": checkpoint_path.name,
            "backbone": backbone,
            "inference_steps": inference_steps,
            "seed": seed,
            "cfg_scale": config.cfg_scale,
            "cfg_interval": config.cfg_interval,
            "initial_pose": str(initial_pose_path),
            "motion_representation": config.motion.representation,
        },
    )
    print(f"Motion written to {output_dir}")


if __name__ == "__main__":
    main()

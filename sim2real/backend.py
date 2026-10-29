from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
DEFAULT_WEIGHT = str(PROJECT / "checkpoints/Movens-Zero-XL/model.safetensors")
PROTOCOL = "movens-root40-v1"
INFERENCE_PROTOCOL = "root40_live_feedback_rtc_v1"


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_preflight(project=PROJECT, weight=None, expected_sha256=None):
    from movens_zero.checkpoints import checkpoint_model_config
    from movens_zero.config import load_config
    from movens_zero.normalization import MotionNormalizer

    project = Path(project)
    weight = Path(weight or os.environ.get("MOVENS_CHECKPOINT", DEFAULT_WEIGHT)).expanduser().resolve(strict=True)
    config_path = project / "configs/inference.json"
    config = load_config(config_path)
    architecture = checkpoint_model_config(weight)
    if architecture["action_dim"] != 40 or architecture["state_dim"] != 40 or architecture["action_horizon"] != 30:
        raise ValueError("Deployment requires a root40 checkpoint with state/action dimension 40 and horizon 30")
    if config.motion.representation != "root40_v1" or config.model.action_exec_horizon != 20 or config.model.rtc_max_delay != 11:
        raise ValueError("Deployment requires root40_v1 with execute=20 and RTC max_delay=11")
    normalizer_path = project / "assets/normalization.json"
    normalizer = MotionNormalizer.from_file(normalizer_path)
    if normalizer.action_min.shape != (40,) or normalizer.state_min.shape != (40,):
        raise ValueError("Deployment normalization must have 40 dimensions")
    if not np.isfinite(np.concatenate([normalizer.action_min, normalizer.action_max, normalizer.state_min, normalizer.state_max])).all():
        raise ValueError("Nonfinite deployment normalization")
    digest = hash_file(weight)
    expected = expected_sha256 or os.environ.get("MOVENS_CHECKPOINT_SHA256", "")
    if expected and digest != expected:
        raise ValueError(f"Checkpoint SHA256 mismatch: expected={expected} actual={digest} path={weight}")
    return dict(weight=str(weight), sha256=digest, hash_pinned=bool(expected),
                architecture=architecture, config=config, normalization=str(normalizer_path))


class Root40Policy:
    def __init__(self, verified, device="cuda:0"):
        import torch
        from movens_zero.model import MovensZeroModel
        from movens_zero.normalization import MotionNormalizer

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Real-time serving requires working CUDA")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 CUDA support is required")
        self.config = verified["config"]
        self.normalizer = MotionNormalizer.from_file(verified["normalization"])
        self.contact = self.config.motion.contact_protocol
        backbone = os.environ.get("MOVENS_PRETRAIN_BACKBONE_PATH", self.config.model.backbone)
        self.model = MovensZeroModel.from_pretrained(
            verified["weight"], backbone, self.config, device=self.device,
            torch_dtype=torch.bfloat16,
            attn_implementation=os.environ.get("MOVENS_ATTENTION", "sdpa"))
        self.model.requires_grad_(False)
        self.sound = None
        self.video_cache = {}

    def infer(self, state, mode, prompt, media, frame, target, previous=None, seed=0):
        from movens_zero.media import decode_video_segment, sound_segment
        from movens_zero.progress import build_future_progress
        from movens_zero.root40 import (choose_heading_anchor, build_root40_state,
            recover_motion_abs_from_root40_action, reencode_root40_prefix)

        if mode == "text" and not prompt.strip():
            raise ValueError("Text prompt cannot be empty")
        torch = self.torch
        current = np.asarray(state, dtype=np.float32).reshape(1, 36)
        anchor = choose_heading_anchor(current)
        states = build_root40_state(current, anchor, contact_protocol=self.contact)
        states = np.clip(self.normalizer.normalize_state(states), -1, 1)
        if previous is None:
            torch.manual_seed(seed)
            self.video_cache.clear()
        progress_frame, progress_total = (0, 31) if mode == "video" else (frame, target + 1)
        prompts = dict(text=prompt.strip() if mode == "text" else "",
            music="Generate a dance motion that follows the rhythm and energy of this music.",
            speech="Listen for the spoken task in the audio, then carry out that task as body motion.",
            video="The clip is the reference. Imitate the person's body movement as closely as possible.")
        arguments = dict(states=torch.from_numpy(states[None]).to(self.device),
            prompt=prompts[mode], mode=mode, num_inference_steps=10,
            future_progress=torch.from_numpy(build_future_progress(progress_frame, progress_total, 30)[None]).to(self.device),
            cfg_scale=float(getattr(self.config, "cfg_scale", 1.5)),
            cfg_interval=getattr(self.config, "cfg_interval", (0.2, 0.8)))
        if mode == "speech":
            arguments["sound"] = media["audio"]
        elif mode == "music":
            settings = self.config.modalities.music
            arguments["sound"] = sound_segment(media["audio"], settings.sampling_rate,
                frame / 50 - settings.left_context_seconds,
                settings.window_seconds + settings.left_context_seconds)
        elif mode == "video":
            if frame not in self.video_cache:
                settings = self.config.modalities.video
                self.video_cache = {frame: decode_video_segment(media["path"],
                    frame / 50, 0.6, settings.sample_fps, settings.size_pixels)[0]}
            arguments["video"] = self.video_cache[frame]
        with torch.inference_mode():
            if previous is None:
                predicted = self.model.predict_action(**arguments)
            else:
                prefix = reencode_root40_prefix(previous["raw"], previous["current"], previous["anchor"],
                    current, anchor, start_index=20, length=10, contact_protocol=self.contact)
                normalized = np.zeros((1, 30, 40), dtype=np.float32)
                normalized[0, :10] = self.normalizer.normalize_action(prefix)
                predicted = self.model.predict_action_rtc(**arguments,
                    previous_actions=torch.from_numpy(normalized).to(self.device), inference_delay=10, max_delay=11)
        normalized = predicted.detach().float().cpu().numpy().reshape(30, 40)
        raw = self.normalizer.denormalize_action(normalized)
        absolute = recover_motion_abs_from_root40_action(raw, current, anchor)
        if absolute.shape != (30, 36) or not np.isfinite(absolute).all():
            raise ValueError("Invalid model output")
        return absolute, dict(raw=raw, current=current.copy(), anchor=anchor.copy())

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from functools import partial
import random
import subprocess
import tempfile
import time
import numpy as np

PROTOCOL = "movens-root40-v1"
INFERENCE_PROTOCOL = (
    "root40_current_heading_fp32_video_local_progress_projection_fp32_v4"
)
PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHT = str(PROJECT / "checkpoints/root40/ckpt_99000/model.safetensors")


def checkpoint_preflight(project=PROJECT, weight=None):
    project = Path(project)
    weight = (
        Path(weight or os.environ.get("MOVENS_CHECKPOINT", DEFAULT_WEIGHT))
        .expanduser()
        .resolve(strict=True)
    )
    release = Path(__file__).resolve().parent / "config"
    config = json.loads((release / "run_config.json").read_text())
    statistics = json.loads((release / "dataset_statistics.json").read_text())
    expected_hash = json.loads((release / "release_manifest.json").read_text())[
        "checkpoint_sha256"
    ]
    digest = hash_file(weight)
    if digest != expected_hash:
        raise ValueError(
            f"Checkpoint SHA256 mismatch: expected={expected_hash} actual={digest} path={weight}"
        )
    expected = dict(
        motion_representation="root40_v1",
        action_dim=40,
        action_chunk_size=30,
        action_exec_horizon=20,
        observation_horizon=1,
        max_delay=11,
        noise_scheduler="flow",
        train_diffusion_steps=1000,
        eval_diffusion_steps=10,
    )
    for key, value in expected.items():
        if config["model"].get(key) != value:
            raise ValueError(f"Checkpoint contract mismatch: {key} expected={value}")
    field = config["data"]["transform"]["field"]
    for kind in ("min", "max"):
        bounds = np.asarray(field["action_" + kind], dtype=np.float32)
        if bounds.shape != (40,) or not np.isfinite(bounds).all():
            raise ValueError("Invalid root40 normalization bounds")
        for other in (
            field["state_" + kind],
            statistics["action"][kind],
            statistics["states"][kind],
        ):
            np.testing.assert_array_equal(bounds, np.asarray(other, dtype=np.float32))
    if np.any(np.asarray(field["action_max"]) <= np.asarray(field["action_min"])):
        raise ValueError("Nonpositive normalization range")
    return dict(weight=str(weight), sha256=digest, config=config, release=str(release))


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


class Root40Policy:
    def __init__(self, verified, device="cuda:0"):
        import torch
        from root40_runtime.model import MovensModel
        from precision import (
            configure_projection_fp32,
            step_flow_fp32,
            InferenceProcessor,
        )

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "Real-time serving requires working CUDA; CUDA is unavailable"
            )
        torch.cuda.set_device(self.device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 CUDA support is required")
        self.config = json.loads(
            json.dumps(verified["config"]), object_hook=lambda d: SimpleNamespace(**d)
        )
        self.config.data.transform.field.stat_path = str(
            Path(verified["release"]) / "dataset_statistics.json"
        )
        self.low = np.asarray(
            self.config.data.transform.field.action_min, dtype=np.float32
        )
        self.width = (
            np.asarray(self.config.data.transform.field.action_max, dtype=np.float32)
            - self.low
        )
        self.contact = self.config.data.transform.repack.root40_contact_protocol
        self.runtime = tempfile.TemporaryDirectory(prefix="movens-real-run-")
        run = Path(self.runtime.name)
        (run / "checkpoints/ckpt_99000").mkdir(parents=True)
        (run / "checkpoints/ckpt_99000/model.safetensors").symlink_to(
            verified["weight"]
        )
        for name in ("run_config.json", "dataset_statistics.json"):
            (run / name).symlink_to(Path(verified["release"]) / name)
        self.model = (
            MovensModel.from_pretrained(
                str(run), 99000, self.config, self.device, allow_root40=True
            )
            .to(self.device)
            .eval()
        )
        self.model.requires_grad_(False)
        configure_projection_fp32(self.model)
        if (
            next(self.model.vlm_model.parameters()).dtype != torch.bfloat16
            or self.model.infer_autocast_dtype != torch.bfloat16
        ):
            raise RuntimeError(
                "Expected BF16 VLM inference, with FP32 action projection and flow samples"
            )
        self.model.noise_scheduler.step = partial(
            step_flow_fp32, self.model.noise_scheduler.step
        )
        self.model.vlm_processor = InferenceProcessor(
            self.model.vlm_processor, self.device.type
        )
        self.hidden_cache = {}
        self.video_cache = None

        def feature_forward(**kwargs):
            key = (
                "constant"
                if self.model.text_only
                or (self.model.audio_only and self.active_mode == "speech")
                else None
            )
            if key is not None and key in self.hidden_cache:
                hidden = self.hidden_cache[key]
            else:
                hidden = self.model._omni_hidden_states_without_lm_head(kwargs)
                if key is not None:
                    self.hidden_cache[key] = hidden
            return SimpleNamespace(hidden_states=(hidden,))

        self.model.vlm_model.forward = feature_forward

    def infer(
        self, state, mode, prompt, media, frame, target, previous=None, seed=20260930
    ):
        from root40_runtime.motion_repr40 import (
            choose_heading_anchor,
            build_root40_state,
            recover_motion_abs_from_root40_action,
            reencode_root40_prefix,
        )
        from root40_runtime.text_prompt import compose_motion_prompt
        from root40_runtime.video import load_video_clip_array_ffmpeg
        from precision import build_case_progress

        torch, model = (self.torch, self.model)
        self.active_mode = mode
        if previous is None:
            self.hidden_cache.clear()
            self.video_cache = None
        current = np.asarray(state, dtype=np.float32).reshape(1, 36)
        anchor = choose_heading_anchor(current)
        raw_state = build_root40_state(current, anchor, contact_protocol=self.contact)
        states = np.clip((raw_state - self.low) / self.width * 2 - 1, -1, 1)
        model.text_only, model.audio_only, model.video_only = (
            mode == "text",
            mode in ("music", "speech"),
            mode == "video",
        )
        if previous is None:
            torch.manual_seed(seed)
        prompts = dict(
            text=compose_motion_prompt(prompt, rng=random.Random(seed)),
            music="Generate a dance motion that follows the rhythm and energy of this music.",
            speech="Listen for the spoken task in the audio, then carry out that task as body motion.",
            video="The clip is the reference. Imitate the person's body movement as closely as possible.",
        )
        progress, future = build_case_progress(mode, frame, target + 1, 30)
        kwargs = dict(
            observations=None,
            states=torch.from_numpy(states[None].astype(np.float32)).to(self.device),
            instructions=[prompts[mode]],
            num_inference_steps=10,
            traj2ds=None,
            progress=torch.from_numpy(progress[None]).to(self.device),
            future_progress=torch.from_numpy(future[None]).to(self.device),
        )
        if mode in ("music", "speech"):
            audio = media["audio"]
            if mode == "speech":
                segment = audio
            else:
                start = round(frame / 50 * 16000)
                segment = np.zeros(9600, dtype=np.float32)
                available = audio[start : start + 9600]
                segment[: len(available)] = available
            kwargs["audios"] = [segment]
        elif mode == "video":
            from root40_runtime.video import _probe_video

            if previous is None or self.video_cache is None:
                width, height, source_fps, duration = _probe_video(media["path"])
                if not np.isfinite(duration) or duration <= 0:
                    raise ValueError("Video duration must be finite and positive")
                scale = np.sqrt(50176 / max(1, width * height))
                w = max(2, round(width * scale / 2) * 2)
                h = max(2, round(height * scale / 2) * 2)
                if int(np.ceil(duration * 8)) * w * h * 3 > 512 * 1024 * 1024:
                    raise ValueError("Decoded video exceeds the 512 MiB session cache")
                full, _, _ = load_video_clip_array_ffmpeg(
                    media["path"],
                    0,
                    duration,
                    video_sample_fps=8,
                    video_size_pixels=50176,
                )
                self.video_cache = (full, duration)
            full, duration = self.video_cache
            clip_start = max(0.0, min(frame / 50, duration - 0.6))
            start_index = int(np.floor(clip_start * 8 + 0.5))
            video = full[start_index : start_index + 5].copy()
            if len(video) == 0:
                video = full[-1:].copy()
            if len(video) < 5:
                video = np.concatenate(
                    [video, np.repeat(video[-1:], 5 - len(video), axis=0)]
                )
            kwargs.update(
                videos=[video],
                video_paths=[str(media["path"])],
                video_sample_fps=8,
                video_size_pixels=50176,
            )
        with (
            torch.inference_mode(),
            torch.autocast(self.device.type, dtype=torch.bfloat16),
        ):
            if previous is None:
                normalized = model.predict_action(**kwargs)[0].float().cpu().numpy()
            else:
                prefix = reencode_root40_prefix(
                    previous["raw"],
                    previous["current"],
                    previous["anchor"],
                    current,
                    anchor,
                    start_index=20,
                    length=10,
                    contact_protocol=self.contact,
                )
                prev = np.zeros((1, 30, 40), dtype=np.float32)
                prev[0, :10] = (prefix - self.low) / self.width * 2 - 1
                normalized = (
                    model.predict_action_with_training_rtc_flow(
                        **kwargs,
                        prev_actions=torch.from_numpy(prev).to(self.device),
                        inference_delay=10,
                        max_delay=11,
                    )[0]
                    .float()
                    .cpu()
                    .numpy()
                )
        raw = 0.5 * (normalized + 1) * self.width + self.low
        absolute = recover_motion_abs_from_root40_action(raw, current, anchor)
        if absolute.shape != (30, 36) or not np.isfinite(absolute).all():
            raise ValueError("Invalid model output")
        return (absolute, dict(raw=raw, current=current.copy(), anchor=anchor.copy()))

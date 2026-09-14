from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from movens_zero.media import decode_sound, decode_video_segment, sound_segment
from movens_zero.motion import (
    MOTION_DIM,
    build_state_history,
    normalize_quaternion,
    recover_poses,
)
from movens_zero.normalization import MotionNormalizer
from movens_zero.progress import build_future_progress


def generate_motion(
    model,
    config: SimpleNamespace,
    normalizer: MotionNormalizer,
    mode: str,
    prompt: str,
    input_path: Path | None,
    initial_pose: np.ndarray,
    output_frames: int,
    inference_steps: int,
    device: str,
) -> np.ndarray:
    horizon = int(config.model.action_horizon)
    execute = int(config.model.action_exec_horizon)
    overlap = horizon - execute
    if overlap <= 0 or overlap >= int(config.model.rtc_max_delay):
        raise ValueError(
            f"Invalid RTC configuration: horizon={horizon}, execute={execute}, max_delay={config.model.rtc_max_delay}"
        )
    if output_frames <= 0:
        raise ValueError("Output frames must be positive")

    initial_pose = np.asarray(initial_pose, dtype=np.float32).copy()
    initial_pose[32:36] = normalize_quaternion(initial_pose[32:36])
    state_count = int(config.motion.past_frames) + 1
    state_history = [initial_pose.copy() for _ in range(state_count)]
    current_pose = initial_pose.copy()
    previous_actions = None
    generated = []
    remaining = int(output_frames)
    chunks = int(math.ceil(output_frames / execute))
    sound = None

    if mode in ("music", "speech"):
        settings = getattr(config.modalities, mode)
        sound = decode_sound(input_path, int(settings.sampling_rate))

    if hasattr(model, "set_mode"):
        model.set_mode(mode)

    for chunk_index in range(chunks):
        frame = output_frames - remaining
        states = build_state_history(
            state_history[-state_count:], int(config.model.state_dim)
        )
        if normalizer.normalize_state_enabled:
            states = normalizer.normalize_state(states)
        states_tensor = torch.from_numpy(states[None]).to(device)
        future_progress = None
        if bool(config.model.use_future_progress):
            if mode == "video":
                progress_frame = 0
                progress_total = horizon + 1
            else:
                progress_frame = frame
                progress_total = output_frames
            future_progress = torch.from_numpy(
                build_future_progress(progress_frame, progress_total, horizon)[None]
            ).to(device)

        sound_chunk = None
        video_chunk = None
        if mode == "music":
            settings = config.modalities.music
            start = frame / float(config.motion.fps) - float(
                settings.left_context_seconds
            )
            duration = float(settings.window_seconds) + float(
                settings.left_context_seconds
            )
            sound_chunk = sound_segment(
                sound, int(settings.sampling_rate), start, duration
            )
        elif mode == "speech":
            sound_chunk = sound
        elif mode == "video":
            settings = config.modalities.video
            video_chunk, _, _ = decode_video_segment(
                input_path,
                start_seconds=chunk_index * execute / float(config.motion.fps),
                duration_seconds=horizon / float(config.motion.fps),
                sample_fps=float(settings.sample_fps),
                size_pixels=int(settings.size_pixels),
            )

        arguments = {
            "states": states_tensor,
            "prompt": prompt,
            "mode": mode,
            "num_inference_steps": inference_steps,
            "future_progress": future_progress,
            "sound": sound_chunk,
            "video": video_chunk,
        }
        with torch.inference_mode():
            if previous_actions is None:
                prediction = model.predict_action(**arguments)
            else:
                prefix = np.concatenate(
                    [
                        previous_actions[execute:],
                        np.zeros((execute, MOTION_DIM), dtype=np.float32),
                    ]
                )
                prediction = model.predict_action_rtc(
                    **arguments,
                    previous_actions=torch.from_numpy(prefix[None]).to(device),
                    inference_delay=overlap,
                    max_delay=int(config.model.rtc_max_delay),
                )

        if isinstance(prediction, torch.Tensor):
            prediction = prediction.float().cpu().numpy()
        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.ndim == 3:
            prediction = prediction[0]
        prediction = np.clip(prediction, -1.0, 1.0)
        features = normalizer.denormalize_action(prediction)
        poses = recover_poses(features, current_pose)
        previous_actions = prediction
        take = min(execute, remaining)
        executed = poses[:take]
        generated.append(executed)
        current_pose = executed[-1].copy()
        state_history.extend(executed)
        state_history = state_history[-state_count:]
        remaining -= take

    return np.concatenate(generated)[:output_frames].astype(np.float32)

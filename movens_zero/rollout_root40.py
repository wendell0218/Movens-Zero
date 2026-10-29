from pathlib import Path

import numpy as np
import torch

from movens_zero.media import decode_sound, decode_video_segment, sound_segment
from movens_zero.progress import build_future_progress
from movens_zero.root40 import (
    build_root40_state,
    choose_heading_anchor,
    recover_motion_abs_from_root40_action,
    reencode_root40_prefix,
)


def generate_root40_motion(model, config, normalizer, mode, prompt, input_path,
                           initial_pose, output_frames, inference_steps, device):
    horizon, execute = int(config.model.action_horizon), int(config.model.action_exec_horizon)
    overlap = horizon - execute
    if output_frames <= 0 or not 0 < overlap < int(config.model.rtc_max_delay):
        raise ValueError("Invalid output length or RTC overlap")
    current = np.asarray(initial_pose, dtype=np.float32).reshape(1, 36)
    contact = config.motion.contact_protocol
    previous = None
    sound = None
    generated = []
    if mode in ("speech", "music"):
        sound = decode_sound(Path(input_path), int(getattr(config.modalities, mode).sampling_rate))
    for frame in range(0, output_frames, execute):
        anchor = choose_heading_anchor(current)
        state = build_root40_state(current, anchor, contact_protocol=contact)
        state = np.clip(normalizer.normalize_state(state), -1, 1)
        progress_frame, progress_total = (0, horizon + 1) if mode == "video" else (frame, output_frames + 1)
        progress = build_future_progress(progress_frame, progress_total, horizon)
        arguments = dict(states=torch.from_numpy(state[None]).to(device), prompt=prompt,
                         mode=mode, num_inference_steps=inference_steps,
                         future_progress=torch.from_numpy(progress[None]).to(device),
                         cfg_scale=float(getattr(config, "cfg_scale", 1.0)),
                         cfg_interval=getattr(config, "cfg_interval", None))
        if mode == "speech":
            arguments["sound"] = sound
        elif mode == "music":
            settings = config.modalities.music
            arguments["sound"] = sound_segment(sound, int(settings.sampling_rate),
                frame / config.motion.fps - settings.left_context_seconds,
                settings.window_seconds + settings.left_context_seconds)
        elif mode == "video":
            settings = config.modalities.video
            arguments["video"], _, _ = decode_video_segment(Path(input_path),
                frame / config.motion.fps, horizon / config.motion.fps,
                settings.sample_fps, settings.size_pixels)
        if previous is None:
            prediction = model.predict_action(**arguments)
        else:
            prefix = reencode_root40_prefix(previous["raw"], previous["state"],
                previous["anchor"], current, anchor, start_index=execute,
                length=overlap, contact_protocol=contact)
            normalized = np.zeros((horizon, 40), dtype=np.float32)
            normalized[:overlap] = normalizer.normalize_action(prefix)
            prediction = model.predict_action_rtc(**arguments,
                previous_actions=torch.from_numpy(normalized[None]).to(device),
                inference_delay=overlap, max_delay=int(config.model.rtc_max_delay))
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().float().cpu().numpy()
        prediction = np.asarray(prediction, dtype=np.float32).reshape(horizon, 40)
        raw = normalizer.denormalize_action(prediction)
        poses = recover_motion_abs_from_root40_action(raw, current, anchor)
        previous = dict(raw=raw, state=current.copy(), anchor=anchor.copy())
        take = min(execute, output_frames - frame)
        generated.append(poses[:take])
        current = poses[take - 1:take].copy()
    return np.concatenate(generated).astype(np.float32)

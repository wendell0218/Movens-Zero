from functools import lru_cache
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from movens_zero.media import decode_sound, decode_video_segment, sound_segment
from movens_zero.progress import build_future_progress
from movens_zero.root40 import encode_root40


@lru_cache(maxsize=8)
def load_motion(path):
    path = Path(path)
    if path.is_dir():
        values = np.concatenate([
            np.loadtxt(path / name, delimiter=",", skiprows=1, ndmin=2)
            for name in ("joint_pos.csv", "body_pos.csv", "body_quat.csv")
        ], axis=1)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            values = data["motion_abs"]
    else:
        values = np.load(path, allow_pickle=False)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 36 or len(values) < 2:
        raise ValueError(f"{path}: expected at least two (joints29,xyz,qwxyz) frames")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite motion")
    norm = np.linalg.norm(values[:, 32:36], axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError(f"{path}: invalid root quaternion")
    values = values.copy()
    values[:, 32:36] /= norm
    return values


class MotionBatches(Dataset):
    def __init__(self, manifest, config, normalizer, batch_size, microsteps,
                 start_microstep=0, rank=0, seed=0, task_weights=None):
        self.path = Path(manifest).resolve(strict=True)
        self.groups = {}
        self.config, self.normalizer = config, normalizer
        self.batch_size, self.microsteps = batch_size, microsteps
        self.start_microstep, self.rank, self.seed = start_microstep, rank, seed
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            mode = row["mode"]
            if mode not in ("text", "music", "speech", "video"):
                raise ValueError(f"Unknown modality: {mode}")
            if float(row.get("fps", 50)) != float(config.motion.fps):
                raise ValueError("Resample motion to the configured 50 Hz before training")
            for key in ("motion", "media"):
                if row.get(key):
                    row[key] = str((self.path.parent / row[key]).resolve(strict=True))
            if mode != "text" and not row.get("media"):
                raise ValueError(f"{mode} requires a media path")
            if mode == "text" and not row.get("text", "").strip():
                raise ValueError("Text records require a non-empty description")
            self.groups.setdefault(mode, []).append(row)
        if not self.groups:
            raise ValueError("Manifest is empty")
        self.modes = sorted(self.groups)
        self.weights = [float(task_weights.get(mode, 0)) if task_weights else
                        len(self.groups[mode]) for mode in self.modes]
        if any(weight < 0 for weight in self.weights) or sum(self.weights) <= 0:
            raise ValueError("Task weights must be nonnegative with a positive total")

    def __len__(self):
        return self.microsteps - self.start_microstep

    def __getitem__(self, index):
        microstep = self.start_microstep + index
        mode = random.Random(self.seed + microstep * 104729).choices(
            self.modes, weights=self.weights, k=1)[0]
        rng = random.Random(self.seed + microstep * 1000003 + self.rank * 10007)
        states, actions, masks, progress, prompts, media = [], [], [], [], [], []
        horizon = int(self.config.model.action_horizon)
        fps = float(self.config.motion.fps)
        for _ in range(self.batch_size):
            row = rng.choice(self.groups[mode])
            motion = load_motion(row["motion"])
            frame = rng.randrange(max(1, len(motion) - horizon))
            history_start = max(0, frame - int(self.config.motion.past_frames))
            history = motion[history_start:frame + 1]
            history = np.concatenate([np.repeat(history[:1],
                int(self.config.motion.past_frames) + 1 - len(history), axis=0), history])
            future = motion[frame + 1:frame + horizon + 1]
            valid = len(future)
            future = np.concatenate([future, np.repeat(future[-1:], horizon - valid, axis=0)])
            action, state, _ = encode_root40(future, history,
                contact_protocol=self.config.motion.contact_protocol)
            actions.append(np.clip(self.normalizer.normalize_action(action), -1, 1))
            states.append(np.clip(self.normalizer.normalize_state(state), -1, 1))
            masks.append(np.arange(horizon) < valid)
            progress.append(build_future_progress(0, horizon + 1, horizon) if mode == "video"
                else build_future_progress(frame, len(motion), horizon))
            text = row.get("text", "")
            prompts.append(text)
            offset = float(row.get("media_offset_seconds", 0))
            if mode == "video":
                settings = self.config.modalities.video
                video, _, _ = decode_video_segment(Path(row["media"]), offset + frame / fps,
                    horizon / fps, settings.sample_fps, settings.size_pixels)
                media.append(video)
            elif mode in ("music", "speech"):
                settings = getattr(self.config.modalities, mode)
                sound = decode_sound(Path(row["media"]), int(settings.sampling_rate))
                if mode == "music":
                    sound = sound_segment(sound, int(settings.sampling_rate),
                        offset + frame / fps - settings.left_context_seconds,
                        settings.window_seconds + settings.left_context_seconds)
                media.append(sound)
        return dict(mode=mode, prompts=prompts, media=media,
            states=torch.from_numpy(np.stack(states)), actions=torch.from_numpy(np.stack(actions)),
            valid=torch.from_numpy(np.stack(masks)), progress=torch.from_numpy(np.stack(progress)),
            microstep=microstep)

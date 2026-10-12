from __future__ import annotations
import math
from typing import Tuple
import numpy as np


def build_progress_conditions(
    *,
    frame_id: int,
    total_frames: int,
    action_horizon: int,
    step_stride: int = 1,
    use_delta_actions: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    total_frames = int(total_frames)
    action_horizon = int(action_horizon)
    step_stride = int(step_stride)
    progress_den = float(max(total_frames - 1, 1))
    max_frame = max(total_frames - 1, 0)
    start_frame = min(int(frame_id), max_frame)
    start_progress = float(start_frame) / progress_den
    end_frame = min(int(frame_id) + action_horizon * step_stride, max_frame)
    end_progress = float(end_frame) / progress_den
    remaining_progress = max(0.0, 1.0 - start_progress)
    progress = np.asarray(
        [
            start_progress,
            end_progress,
            remaining_progress,
            math.sin(2.0 * math.pi * start_progress),
            math.cos(2.0 * math.pi * start_progress),
        ],
        dtype=np.float32,
    )
    rows = []
    for step in range(action_horizon):
        if use_delta_actions:
            future_frame = int(frame_id) + step * step_stride
        else:
            future_frame = int(frame_id) + (step + 1) * step_stride
        future_frame = min(future_frame, max_frame)
        future_progress = float(future_frame) / progress_den
        future_remaining = max(0.0, 1.0 - future_progress)
        rows.append(
            [
                future_progress,
                future_remaining,
                start_progress,
                math.sin(2.0 * math.pi * future_progress),
                math.cos(2.0 * math.pi * future_progress),
            ]
        )
    return (progress, np.asarray(rows, dtype=np.float32))

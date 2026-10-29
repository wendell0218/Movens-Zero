from __future__ import annotations

import math

import numpy as np


def build_future_progress(
    frame: int, total_frames: int, action_horizon: int
) -> np.ndarray:
    denominator = float(max(total_frames - 1, 1))
    maximum = max(total_frames - 1, 0)
    start = float(min(frame, maximum)) / denominator
    rows = []
    for step in range(action_horizon):
        future = float(min(frame + step + 1, maximum)) / denominator
        rows.append(
            [
                future,
                max(0.0, 1.0 - future),
                start,
                math.sin(2.0 * math.pi * future),
                math.cos(2.0 * math.pi * future),
            ]
        )
    return np.asarray(rows, dtype=np.float32)

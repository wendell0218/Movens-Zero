from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MotionNormalizer:
    action_min: np.ndarray
    action_max: np.ndarray
    state_min: np.ndarray
    state_max: np.ndarray
    normalize_state_enabled: bool = True
    use_norm_mask: bool = False
    action_norm_masks: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.action_min = np.asarray(self.action_min, dtype=np.float32)
        self.action_max = np.asarray(self.action_max, dtype=np.float32)
        self.state_min = np.asarray(self.state_min, dtype=np.float32)
        self.state_max = np.asarray(self.state_max, dtype=np.float32)
        if self.action_min.shape != self.action_max.shape:
            raise ValueError("Action normalization bounds must have matching shapes")
        if self.state_min.shape != self.state_max.shape:
            raise ValueError("State normalization bounds must have matching shapes")
        if self.action_norm_masks is None:
            self.action_norm_masks = np.ones(self.action_min.shape, dtype=bool)
        else:
            self.action_norm_masks = np.asarray(self.action_norm_masks, dtype=bool)
        if self.action_norm_masks.shape != self.action_min.shape:
            raise ValueError(
                "Action normalization mask must match the action dimension"
            )

    @classmethod
    def from_file(cls, path: str | Path) -> MotionNormalizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            action_min=data["action"]["q01"],
            action_max=data["action"]["q99"],
            state_min=data["state"]["q01"],
            state_max=data["state"]["q99"],
            normalize_state_enabled=bool(data.get("normalize_state", True)),
            use_norm_mask=bool(data.get("use_norm_mask", False)),
            action_norm_masks=data.get("action_norm_masks"),
        )

    @staticmethod
    def _normalize(
        array: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> np.ndarray:
        values = np.asarray(array, dtype=np.float32)
        if values.shape[-1] != lower.shape[0]:
            raise ValueError("Input dimension does not match normalization bounds")
        width = upper - lower
        stable = np.abs(width) >= 1e-8
        safe_width = np.where(stable, width, 1.0)
        normalized = (values - lower) / safe_width * 2.0 - 1.0
        return np.where(stable, normalized, values).astype(np.float32)

    @staticmethod
    def _denormalize(
        array: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> np.ndarray:
        values = np.asarray(array, dtype=np.float32)
        if values.shape[-1] != lower.shape[0]:
            raise ValueError("Input dimension does not match normalization bounds")
        width = upper - lower
        stable = np.abs(width) >= 1e-8
        restored = (values + 1.0) * 0.5 * width + lower
        return np.where(stable, restored, values).astype(np.float32)

    def normalize_state(self, array: np.ndarray) -> np.ndarray:
        values = np.asarray(array, dtype=np.float32)
        if not self.normalize_state_enabled:
            return values.copy()
        return self._normalize(values, self.state_min, self.state_max)

    def normalize_action(self, array: np.ndarray) -> np.ndarray:
        values = np.asarray(array, dtype=np.float32)
        normalized = self._normalize(values, self.action_min, self.action_max)
        if self.use_norm_mask:
            return np.where(self.action_norm_masks, normalized, values).astype(
                np.float32
            )
        return normalized

    def denormalize_action(self, array: np.ndarray) -> np.ndarray:
        values = np.asarray(array, dtype=np.float32)
        restored = self._denormalize(values, self.action_min, self.action_max)
        if self.use_norm_mask:
            return np.where(self.action_norm_masks, restored, values).astype(np.float32)
        return restored


def load_initial_pose(path: str | Path) -> np.ndarray:
    pose = np.load(Path(path), allow_pickle=False).astype(np.float32)
    if pose.shape != (36,):
        raise ValueError("Initial pose must have shape (36,)")
    if not np.isfinite(pose).all():
        raise ValueError("Initial pose must contain finite values")
    return pose

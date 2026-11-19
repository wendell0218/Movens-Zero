from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

G1_MUJOCO_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

MUJOCO_TO_ISAACLAB_DOF = np.asarray(
    [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ],
    dtype=np.int64,
)

ISAACLAB_TO_MUJOCO_DOF = np.argsort(MUJOCO_TO_ISAACLAB_DOF)

G1_ISAACLAB_JOINT_NAMES: tuple[str, ...] = tuple(
    G1_MUJOCO_JOINT_NAMES[index] for index in MUJOCO_TO_ISAACLAB_DOF
)

G1_JOINT_NAMES = G1_MUJOCO_JOINT_NAMES

def _as_motion36(motion: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(motion, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 36:
        raise ValueError(f"Expected motion shape (F, 36), got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("Motion must contain at least one frame")
    if not np.isfinite(arr).all():
        raise ValueError("Motion contains NaN or infinite values")
    quat_norm = np.linalg.norm(arr[:, 32:36], axis=1)
    if np.any(quat_norm < 1e-8):
        raise ValueError("Motion contains a zero-length root quaternion")
    arr = arr.copy()
    arr[:, 32:36] /= quat_norm[:, None]
    return arr

def _reorder_dof(
    values: np.ndarray | Iterable[float],
    indices: np.ndarray,
    *,
    source_order: str,
    target_order: str,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim < 1 or arr.shape[-1] != 29:
        raise ValueError(
            f"Expected {source_order} DoFs with final dimension 29, got {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"{source_order} DoFs contain NaN or infinite values")
    result = arr[..., indices]
    if result.shape[-1] != 29:
        raise RuntimeError(f"Invalid {source_order}-to-{target_order} permutation")
    return result

def dof_mujoco_to_isaaclab(values: np.ndarray | Iterable[float]) -> np.ndarray:

    return _reorder_dof(
        values,
        MUJOCO_TO_ISAACLAB_DOF,
        source_order="MuJoCo",
        target_order="IsaacLab",
    )

def dof_isaaclab_to_mujoco(values: np.ndarray | Iterable[float]) -> np.ndarray:

    return _reorder_dof(
        values,
        ISAACLAB_TO_MUJOCO_DOF,
        source_order="IsaacLab",
        target_order="MuJoCo",
    )

def motion36_to_qpos(motion: np.ndarray) -> np.ndarray:

    arr = _as_motion36(motion)
    dof_mujoco = dof_isaaclab_to_mujoco(arr[:, :29])
    return np.concatenate([arr[:, 29:32], arr[:, 32:36], dof_mujoco], axis=1)

def qpos_to_motion36(qpos: np.ndarray) -> np.ndarray:

    arr = np.asarray(qpos, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 36:
        raise ValueError(f"Expected qpos shape (F, 36), got {arr.shape}")
    dof_isaaclab = dof_mujoco_to_isaaclab(arr[:, 7:36])
    result = np.concatenate([dof_isaaclab, arr[:, :3], arr[:, 3:7]], axis=1)
    return _as_motion36(result)

def sample_motion36(
    motion: np.ndarray,
    *,
    source_fps: float,
    sample_times: np.ndarray,
) -> np.ndarray:

    arr = _as_motion36(motion)
    source_fps = float(source_fps)
    if source_fps <= 0:
        raise ValueError(f"source_fps must be positive, got {source_fps}")
    source_times = np.arange(len(arr), dtype=np.float64) / source_fps
    times = np.asarray(sample_times, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("sample_times must be a finite 1D array")
    if np.any(times < source_times[0] - 1e-9) or np.any(times > source_times[-1] + 1e-9):
        raise ValueError(
            f"Sample times [{times.min()}, {times.max()}] exceed motion interval "
            f"[{source_times[0]}, {source_times[-1]}]"
        )
    times = np.clip(times, source_times[0], source_times[-1])
    linear = np.empty((len(times), 32), dtype=np.float64)
    source_linear = arr[:, :32]
    for dim in range(source_linear.shape[1]):
        linear[:, dim] = np.interp(times, source_times, source_linear[:, dim])

    if len(arr) == 1:
        quat_wxyz = np.repeat(arr[:, 32:36], len(times), axis=0)
    else:
        quat_xyzw = arr[:, 32:36][:, [1, 2, 3, 0]]
        quat_out_xyzw = Slerp(source_times, Rotation.from_quat(quat_xyzw))(times).as_quat()
        quat_wxyz = quat_out_xyzw[:, [3, 0, 1, 2]]
    return _as_motion36(np.concatenate([linear, quat_wxyz], axis=1))



def continuous_quaternion_signs(motion: np.ndarray) -> np.ndarray:
    arr = _as_motion36(motion)
    for index in range(1, len(arr)):
        if np.dot(arr[index - 1, 32:36], arr[index, 32:36]) < 0:
            arr[index, 32:36] *= -1
    return arr

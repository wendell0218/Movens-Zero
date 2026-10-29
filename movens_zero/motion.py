from __future__ import annotations

import numpy as np


JOINT_DIM = 29
POSE_DIM = 36
MOTION_DIM = 39
JOINTS = slice(0, 29)
ROOT_DELTA_XY = slice(29, 31)
ROOT_ROTATION = slice(31, 36)
ROOT_HEIGHT = slice(36, 37)


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)
    quaternion = quaternion / norm
    return np.where(quaternion[..., :1] < 0.0, -quaternion, quaternion).astype(
        np.float32
    )


def multiply_quaternions(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    result = np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )
    return normalize_quaternion(result)


def inverse_quaternion(quaternion: np.ndarray) -> np.ndarray:
    result = normalize_quaternion(quaternion).copy()
    result[..., 1:] *= -1.0
    return result


def quaternion_yaw(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(normalize_quaternion(quaternion), -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).astype(
        np.float32
    )


def quaternion_rpy(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(normalize_quaternion(quaternion), -1, 0)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return np.stack([roll, pitch, quaternion_yaw(quaternion)], axis=-1).astype(
        np.float32
    )


def rpy_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return normalize_quaternion(
        np.asarray(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float32,
        )
    )


def local_delta(
    previous_position: np.ndarray,
    current_position: np.ndarray,
    previous_quaternion: np.ndarray,
) -> np.ndarray:
    delta = np.asarray(current_position) - np.asarray(previous_position)
    yaw = quaternion_yaw(previous_quaternion)
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    result = np.empty_like(delta, dtype=np.float32)
    result[..., 0] = cosine * delta[..., 0] + sine * delta[..., 1]
    result[..., 1] = -sine * delta[..., 0] + cosine * delta[..., 1]
    result[..., 2] = delta[..., 2]
    return result


def world_delta(previous_quaternion: np.ndarray, delta_local: np.ndarray) -> np.ndarray:
    delta_local = np.asarray(delta_local, dtype=np.float32)
    yaw = quaternion_yaw(previous_quaternion)
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    result = np.empty_like(delta_local, dtype=np.float32)
    result[..., 0] = cosine * delta_local[..., 0] - sine * delta_local[..., 1]
    result[..., 1] = sine * delta_local[..., 0] + cosine * delta_local[..., 1]
    result[..., 2] = delta_local[..., 2]
    return result


def rotation_features(
    previous_quaternion: np.ndarray, current_quaternion: np.ndarray
) -> np.ndarray:
    previous = quaternion_rpy(previous_quaternion)
    current = quaternion_rpy(current_quaternion)
    yaw_delta = (current[..., 2] - previous[..., 2] + np.pi) % (2.0 * np.pi) - np.pi
    return np.stack(
        [
            np.sin(current[..., 0]),
            np.cos(current[..., 0]),
            np.sin(current[..., 1]),
            np.cos(current[..., 1]),
            yaw_delta,
        ],
        axis=-1,
    ).astype(np.float32)


def foot_contacts(joint_delta: np.ndarray, root_delta: np.ndarray) -> np.ndarray:
    left = np.logical_and(
        np.linalg.norm(joint_delta[..., 0:6], axis=-1) < 0.035,
        np.abs(root_delta[..., 2]) < 0.025,
    )
    right = np.logical_and(
        np.linalg.norm(joint_delta[..., 6:12], axis=-1) < 0.035,
        np.abs(root_delta[..., 2]) < 0.025,
    )
    return np.stack([left, right], axis=-1).astype(np.float32)


def build_motion_state(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    if poses.shape[-1] != POSE_DIM:
        raise ValueError(f"Expected {POSE_DIM}D poses, got {poses.shape[-1]}")
    if poses.ndim == 1:
        poses = poses[None]
    previous = np.concatenate([poses[:1], poses[:-1]], axis=0)
    joint_delta = poses[:, :JOINT_DIM] - previous[:, :JOINT_DIM]
    root_delta = local_delta(previous[:, 29:32], poses[:, 29:32], previous[:, 32:36])
    return np.concatenate(
        [
            poses[:, :JOINT_DIM],
            root_delta[:, :2],
            rotation_features(previous[:, 32:36], poses[:, 32:36]),
            poses[:, 31:32],
            foot_contacts(joint_delta, root_delta),
        ],
        axis=-1,
    ).astype(np.float32)


def recover_poses(features: np.ndarray, initial_pose: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    previous = np.asarray(initial_pose, dtype=np.float32).copy()
    if features.shape[-1] != MOTION_DIM:
        raise ValueError(f"Expected {MOTION_DIM}D motion, got {features.shape[-1]}")
    if previous.shape != (POSE_DIM,):
        raise ValueError(
            f"Expected initial pose shape ({POSE_DIM},), got {previous.shape}"
        )
    frames = []
    for feature in features:
        delta_xy = feature[ROOT_DELTA_XY]
        delta = np.asarray([delta_xy[0], delta_xy[1], 0.0], dtype=np.float32)
        position = previous[29:32] + world_delta(previous[32:36], delta)
        position[2] = feature[ROOT_HEIGHT][0]
        rotation = feature[ROOT_ROTATION]
        roll = float(np.arctan2(rotation[0], rotation[1]))
        pitch = float(np.arctan2(rotation[2], rotation[3]))
        yaw = float(quaternion_yaw(previous[32:36]) + rotation[4])
        quaternion = rpy_quaternion(roll, pitch, yaw)
        frame = np.concatenate([feature[JOINTS], position, quaternion]).astype(
            np.float32
        )
        frames.append(frame)
        previous = frame
    return np.stack(frames).astype(np.float32)


def build_state_history(poses: list[np.ndarray], state_dim: int) -> np.ndarray:
    states = build_motion_state(np.stack(poses).astype(np.float32))
    if states.shape[-1] != state_dim:
        raise ValueError(
            f"State dimension mismatch: built {states.shape[-1]}, expected {state_dim}"
        )
    return states

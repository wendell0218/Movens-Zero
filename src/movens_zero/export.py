from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from movens_zero.motion import inverse_quaternion, multiply_quaternions


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._")
    return name or "motion"


def finite_difference(values: np.ndarray, step: float) -> np.ndarray:
    if values.shape[0] < 2:
        return np.zeros_like(values, dtype=np.float64)
    order = 2 if values.shape[0] > 2 else 1
    return np.gradient(values, step, axis=0, edge_order=order)


def quaternion_to_rotation_vector(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / max(np.linalg.norm(quaternion), 1e-12)
    if quaternion[0] < 0:
        quaternion = -quaternion
    vector = quaternion[1:]
    sine = np.linalg.norm(vector)
    if sine < 1e-10:
        return np.zeros(3, dtype=np.float64)
    return vector / sine * (2.0 * np.arctan2(sine, quaternion[0]))


def angular_velocity(quaternions: np.ndarray, step: float) -> np.ndarray:
    result = np.zeros((quaternions.shape[0], 3), dtype=np.float64)
    for index in range(quaternions.shape[0] - 1):
        delta = multiply_quaternions(
            quaternions[index + 1], inverse_quaternion(quaternions[index])
        )
        result[index] = quaternion_to_rotation_vector(delta) / step
    if quaternions.shape[0] > 1:
        result[-1] = result[-2]
    return result


def write_csv(path: Path, values: np.ndarray, columns: list[str]) -> None:
    np.savetxt(
        path,
        np.asarray(values, dtype=np.float64),
        delimiter=",",
        header=",".join(columns),
        comments="",
        fmt="%.6f",
    )


def export_motion(
    output_dir: Path,
    poses: np.ndarray,
    fps: int,
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joint_position = poses[:, :29]
    body_position = poses[:, 29:32]
    body_quaternion = poses[:, 32:36]
    step = 1.0 / float(fps)
    joint_velocity = finite_difference(joint_position, step)
    body_linear_velocity = finite_difference(body_position, step)
    body_angular_velocity = angular_velocity(body_quaternion, step)
    write_csv(
        output_dir / "joint_pos.csv",
        joint_position,
        [f"joint_{index}" for index in range(29)],
    )
    write_csv(
        output_dir / "joint_vel.csv",
        joint_velocity,
        [f"joint_vel_{index}" for index in range(29)],
    )
    write_csv(
        output_dir / "body_pos.csv",
        body_position,
        ["body_0_x", "body_0_y", "body_0_z"],
    )
    write_csv(
        output_dir / "body_quat.csv",
        body_quaternion,
        ["body_0_w", "body_0_x", "body_0_y", "body_0_z"],
    )
    write_csv(
        output_dir / "body_lin_vel.csv",
        body_linear_velocity,
        ["body_0_vel_x", "body_0_vel_y", "body_0_vel_z"],
    )
    write_csv(
        output_dir / "body_ang_vel.csv",
        body_angular_velocity,
        ["body_0_angvel_x", "body_0_angvel_y", "body_0_angvel_z"],
    )
    np.save(output_dir / "motion.npy", poses)
    metadata = dict(metadata)
    metadata["frames"] = int(poses.shape[0])
    metadata["fps"] = int(fps)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

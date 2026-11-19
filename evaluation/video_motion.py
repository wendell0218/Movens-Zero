from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from movens_zero.motion_format import (
    G1_ISAACLAB_JOINT_NAMES,
    G1_JOINT_NAMES,
    G1_MUJOCO_JOINT_NAMES,
    ISAACLAB_TO_MUJOCO_DOF,
    MUJOCO_TO_ISAACLAB_DOF,
    _as_motion36,
    _reorder_dof,
    dof_isaaclab_to_mujoco,
    dof_mujoco_to_isaaclab,
    motion36_to_qpos,
    qpos_to_motion36,
    sample_motion36,
)


G1_24_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)

G1_24_BODIES: tuple[str, ...] = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_knee_link",
    "right_knee_link",
    "waist_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_toe_link",
    "right_toe_link",
    "imu_in_torso",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "head_mocap",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_rubber_hand",
    "right_rubber_hand",
)

G1_SOLE_OFFSETS: dict[str, tuple[tuple[float, float, float], ...]] = {
    "left_ankle_roll_link": (
        (-0.05, 0.025, -0.03),
        (-0.05, -0.025, -0.03),
        (0.12, 0.03, -0.03),
        (0.12, -0.03, -0.03),
    ),
    "right_ankle_roll_link": (
        (-0.05, 0.025, -0.03),
        (-0.05, -0.025, -0.03),
        (0.12, 0.03, -0.03),
        (0.12, -0.03, -0.03),
    ),
}


def load_motion36(path: str | Path) -> np.ndarray:

    path = Path(path)
    if path.is_file():
        if path.suffix != ".npy":
            raise ValueError(f"Only .npy motion files are supported, got {path}")
        return _as_motion36(np.load(path))
    if not path.is_dir():
        raise FileNotFoundError(path)
    joint = np.loadtxt(path / "joint_pos.csv", delimiter=",", skiprows=1)
    root_pos = np.loadtxt(path / "body_pos.csv", delimiter=",", skiprows=1)
    root_quat = np.loadtxt(path / "body_quat.csv", delimiter=",", skiprows=1)
    if joint.ndim == 1:
        joint = joint[None]
        root_pos = root_pos[None]
        root_quat = root_quat[None]
    lengths = {len(joint), len(root_pos), len(root_quat)}
    if len(lengths) != 1:
        raise ValueError(f"CSV frame-count mismatch in {path}")
    return _as_motion36(np.concatenate([joint, root_pos, root_quat], axis=1))

def _write_csv(path: Path, values: np.ndarray, header: Iterable[str]) -> None:
    np.savetxt(path, values, delimiter=",", header=",".join(header), comments="")

def save_motion36(path: str | Path, motion: np.ndarray, *, fps: float) -> None:

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    arr = _as_motion36(motion).astype(np.float32)
    np.save(path / "motion.npy", arr)
    _write_csv(path / "joint_pos.csv", arr[:, :29], (f"joint_{i}" for i in range(29)))
    _write_csv(path / "body_pos.csv", arr[:, 29:32], ("body_0_x", "body_0_y", "body_0_z"))
    _write_csv(
        path / "body_quat.csv",
        arr[:, 32:36],
        ("body_0_w", "body_0_x", "body_0_y", "body_0_z"),
    )
    (path / "metadata.txt").write_text(
        f"motion_fps: {float(fps):.8g}\nframes: {len(arr)}\nroot_quaternion: wxyz\n",
        encoding="utf-8",
    )


def resample_motion36(motion: np.ndarray, *, source_fps: float, target_fps: float) -> np.ndarray:

    arr = _as_motion36(motion)
    target_fps = float(target_fps)
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    duration = (len(arr) - 1) / float(source_fps)
    target_frames = int(round(duration * target_fps)) + 1
    target_times = np.arange(target_frames, dtype=np.float64) / target_fps
    target_times[-1] = duration
    return sample_motion36(arr, source_fps=source_fps, sample_times=target_times)

@dataclass(frozen=True)
class G1KinematicOutput:
    joints24: np.ndarray
    sole_points: np.ndarray

class G1Kinematics:

    def __init__(self, xml_path: str | Path):
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError(
                "G1Kinematics requires MuJoCo. Run with the GMR environment or install "
                "the 'kinematics' extra."
            ) from exc
        self._mj = mujoco
        self.xml_path = Path(xml_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        if self.model.nq != 36 or self.model.nu != 29:
            raise ValueError(
                f"Expected G1 29-DoF model with nq=36, nu=29; got nq={self.model.nq}, "
                f"nu={self.model.nu} from {self.xml_path}"
            )
        actuator_names = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        )
        if actuator_names != G1_MUJOCO_JOINT_NAMES:
            raise ValueError(f"Unexpected G1 actuator order: {actuator_names}")
        self._body_ids = {
            name: int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name))
            for name in set(G1_24_BODIES) | set(G1_SOLE_OFFSETS)
        }
        missing = [name for name, idx in self._body_ids.items() if idx < 0]
        if missing:
            raise ValueError(f"G1 XML is missing required bodies: {missing}")

    def forward(self, motion: np.ndarray) -> G1KinematicOutput:
        qpos = motion36_to_qpos(motion)
        joints = np.empty((len(qpos), len(G1_24_BODIES), 3), dtype=np.float64)
        sole = np.empty((len(qpos), 8, 3), dtype=np.float64)
        for frame_idx, frame_qpos in enumerate(qpos):
            self.data.qpos[:] = frame_qpos
            self._mj.mj_forward(self.model, self.data)
            joints[frame_idx] = np.stack(
                [self.data.xpos[self._body_ids[name]].copy() for name in G1_24_BODIES]
            )
            sole_frame: list[np.ndarray] = []
            for body_name, offsets in G1_SOLE_OFFSETS.items():
                body_id = self._body_ids[body_name]
                rotation = self.data.xmat[body_id].reshape(3, 3)
                origin = self.data.xpos[body_id]
                sole_frame.extend(origin + rotation @ np.asarray(offset) for offset in offsets)
            sole[frame_idx] = np.stack(sole_frame)
        return G1KinematicOutput(joints24=joints, sole_points=sole)

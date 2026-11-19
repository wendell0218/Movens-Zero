


from __future__ import annotations

import functools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.signal import find_peaks
import torch
import torch.nn as nn
import torch.nn.functional as F

ASSETS = Path(__file__).resolve().parent.parent/'assets'


def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def matrix_to_quaternion(matrix):
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    indices = q_abs.argmax(dim=-1, keepdim=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = indices.unsqueeze(-1).expand(expand_dims)
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return standardize_quaternion(out)


def _axis_angle_rotation(axis: str, angle):
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles, convention: str):
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)


def standardize_quaternion(quaternions):
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def axis_angle_to_matrix(axis_angle):
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def axis_angle_to_quaternion(axis_angle):
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )

    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def _resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else ASSETS/path.name


def _resolve_asset_path(base_path: Path, asset_path: str | Path) -> Path:
    path = Path(asset_path)
    if path.is_absolute():
        return path
    relative_to_base = (base_path.parent / path).resolve()
    if relative_to_base.exists():
        return relative_to_base
    return _resolve_repo_path(path)


def _view_for_batch(x: torch.Tensor, batch_ndim: int) -> torch.Tensor:
    return x.view(*([1] * batch_ndim), *x.shape)


def _axis_aligned_rotation_matrix(
    angle: torch.Tensor,
    axis_dim: int,
    axis_sign: float,
) -> torch.Tensor:
    angle = angle.squeeze(-1) * axis_sign
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    zero = torch.zeros_like(angle)
    one = torch.ones_like(angle)
    rot = torch.empty(*angle.shape, 3, 3, device=angle.device, dtype=angle.dtype)

    if axis_dim == 0:
        rot[..., 0, 0] = one
        rot[..., 0, 1] = zero
        rot[..., 0, 2] = zero
        rot[..., 1, 0] = zero
        rot[..., 1, 1] = cos
        rot[..., 1, 2] = -sin
        rot[..., 2, 0] = zero
        rot[..., 2, 1] = sin
        rot[..., 2, 2] = cos
    elif axis_dim == 1:
        rot[..., 0, 0] = cos
        rot[..., 0, 1] = zero
        rot[..., 0, 2] = sin
        rot[..., 1, 0] = zero
        rot[..., 1, 1] = one
        rot[..., 1, 2] = zero
        rot[..., 2, 0] = -sin
        rot[..., 2, 1] = zero
        rot[..., 2, 2] = cos
    elif axis_dim == 2:
        rot[..., 0, 0] = cos
        rot[..., 0, 1] = -sin
        rot[..., 0, 2] = zero
        rot[..., 1, 0] = sin
        rot[..., 1, 1] = cos
        rot[..., 1, 2] = zero
        rot[..., 2, 0] = zero
        rot[..., 2, 1] = zero
        rot[..., 2, 2] = one
    else:
        raise ValueError(f"axis_dim must be 0, 1, or 2, got {axis_dim}")
    return rot


def _cardinal_axis_dim_sign(axis: list[float]) -> tuple[int, float] | None:
    abs_axis = [abs(float(v)) for v in axis]
    axis_dim = max(range(3), key=lambda idx: abs_axis[idx])
    if abs(abs_axis[axis_dim] - 1.0) > 1e-6:
        return None
    if any(abs_axis[idx] > 1e-6 for idx in range(3) if idx != axis_dim):
        return None
    return axis_dim, 1.0 if float(axis[axis_dim]) >= 0.0 else -1.0


class G1Kinematics(nn.Module):
    def __init__(self, kinematics_path: str | Path = "assets/robots/g1/g1_kinematics.json"):
        super().__init__()

        kinematics_path = _resolve_repo_path(kinematics_path)
        with kinematics_path.open("r", encoding="utf-8") as f:
            spec = json.load(f)

        self.kinematics_path = str(kinematics_path)
        self.robot_name = spec["robot_name"]
        self.root_link = spec["root_link"]
        self.body_order = tuple(spec["body_order"])
        self.joint_order = tuple(spec["joint_order"])
        self.body_name_to_index = dict(spec["body_name_to_index"])
        self.joint_name_to_qpos_index = dict(spec["joint_name_to_qpos_index"])
        urdf_path = _resolve_asset_path(kinematics_path, spec.get("source_urdf", "g1_29dof.urdf"))
        self.urdf_path = str(urdf_path)
        self._parent_body_indices_py = tuple(int(v) for v in spec["parent_body_indices"])
        self._child_body_indices_py = tuple(int(v) for v in spec["child_body_indices"])
        axis_dim_signs = [_cardinal_axis_dim_sign(axis) for axis in spec["joint_axes"]]
        self._joint_axes_are_cardinal = all(item is not None for item in axis_dim_signs)
        self._joint_axis_dims_py = tuple(-1 if item is None else item[0] for item in axis_dim_signs)
        self._joint_axis_signs_py = tuple(1.0 if item is None else item[1] for item in axis_dim_signs)

        self.register_buffer(
            "parent_body_indices",
            torch.tensor(spec["parent_body_indices"], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "child_body_indices",
            torch.tensor(spec["child_body_indices"], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "joint_axes",
            torch.tensor(spec["joint_axes"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "joint_origin_xyz",
            torch.tensor(spec["joint_origin_xyz"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "joint_origin_rpy",
            torch.tensor(spec["joint_origin_rpy"], dtype=torch.float32),
            persistent=False,
        )

        joint_origin_rot = euler_angles_to_matrix(
            self.joint_origin_rpy[..., [2, 1, 0]], convention="ZYX"
        )
        self.register_buffer("joint_origin_rot", joint_origin_rot, persistent=False)

        joint_lower_limits, joint_upper_limits = self._load_joint_limits(urdf_path)
        self.register_buffer("joint_lower_limits", joint_lower_limits, persistent=False)
        self.register_buffer("joint_upper_limits", joint_upper_limits, persistent=False)

        sole_proxy_body_indices, sole_proxy_local_positions, sole_proxy_radii, sole_proxy_foot_ids = (
            self._load_sole_proxy_spec(urdf_path)
        )
        self.register_buffer("sole_proxy_body_indices", sole_proxy_body_indices, persistent=False)
        self.register_buffer("sole_proxy_local_positions", sole_proxy_local_positions, persistent=False)
        self.register_buffer("sole_proxy_radii", sole_proxy_radii, persistent=False)
        self.register_buffer("sole_proxy_foot_ids", sole_proxy_foot_ids, persistent=False)

    def _load_joint_limits(self, urdf_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        root = ET.parse(urdf_path).getroot()
        lower_limits = []
        upper_limits = []
        for joint_name in self.joint_order:
            joint_node = root.find(f"./joint[@name='{joint_name}']")
            if joint_node is None:
                raise ValueError(f"Missing joint `{joint_name}` in URDF `{urdf_path}`")
            limit_node = joint_node.find("limit")
            if limit_node is None:
                lower_limits.append(float("-inf"))
                upper_limits.append(float("inf"))
                continue
            lower_limits.append(float(limit_node.get("lower", "-inf")))
            upper_limits.append(float(limit_node.get("upper", "inf")))
        return (
            torch.tensor(lower_limits, dtype=torch.float32),
            torch.tensor(upper_limits, dtype=torch.float32),
        )

    def _load_sole_proxy_spec(
        self, urdf_path: Path
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        root = ET.parse(urdf_path).getroot()
        foot_links = (
            ("left_ankle_roll_link", 0),
            ("right_ankle_roll_link", 1),
        )
        body_indices: list[int] = []
        local_positions: list[list[float]] = []
        radii: list[float] = []
        foot_ids: list[int] = []
        for link_name, foot_id in foot_links:
            link_node = root.find(f"./link[@name='{link_name}']")
            if link_node is None:
                raise ValueError(f"Missing link `{link_name}` in URDF `{urdf_path}`")
            body_idx = self.body_name_to_index[link_name]
            for collision in link_node.findall("collision"):
                geometry = collision.find("geometry")
                sphere = None if geometry is None else geometry.find("sphere")
                if sphere is None:
                    continue
                origin = collision.find("origin")
                xyz_str = "0 0 0" if origin is None else origin.get("xyz", "0 0 0")
                xyz = [float(v) for v in xyz_str.split()]
                radius = float(sphere.get("radius"))
                body_indices.append(body_idx)
                local_positions.append(xyz)
                radii.append(radius)
                foot_ids.append(foot_id)
        if not body_indices:
            raise ValueError(f"No sole proxy spheres found in URDF `{urdf_path}`")
        return (
            torch.tensor(body_indices, dtype=torch.long),
            torch.tensor(local_positions, dtype=torch.float32),
            torch.tensor(radii, dtype=torch.float32),
            torch.tensor(foot_ids, dtype=torch.long),
        )

    @property
    def num_joints(self) -> int:
        return len(self.joint_order)

    @property
    def num_bodies(self) -> int:
        return len(self.body_order)

    @staticmethod
    def body_quat_wxyz_to_matrix(body_quat_wxyz: torch.Tensor) -> torch.Tensor:
        body_quat_wxyz = F.normalize(body_quat_wxyz, dim=-1)
        return quaternion_to_matrix(body_quat_wxyz)

    @staticmethod
    def matrix_to_body_quat_wxyz(matrix: torch.Tensor) -> torch.Tensor:
        quat = matrix_to_quaternion(matrix)
        quat = F.normalize(quat, dim=-1)
        return standardize_quaternion(quat)

    def get_sole_proxy_points(
        self,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        body_dim = body_pos_w.ndim - 2
        proxy_body_indices = self.sole_proxy_body_indices.to(device=body_pos_w.device)
        proxy_offsets = self.sole_proxy_local_positions.to(
            device=body_pos_w.device,
            dtype=body_pos_w.dtype,
        )
        proxy_radii = self.sole_proxy_radii.to(device=body_pos_w.device, dtype=body_pos_w.dtype)

        proxy_body_pos = body_pos_w.index_select(dim=body_dim, index=proxy_body_indices)
        proxy_body_quat = body_quat_w.index_select(dim=body_dim, index=proxy_body_indices)
        proxy_body_rot = quaternion_to_matrix(proxy_body_quat)
        proxy_offsets = proxy_offsets.view(*([1] * (body_pos_w.ndim - 2)), *proxy_offsets.shape)
        proxy_points = proxy_body_pos + torch.matmul(
            proxy_body_rot, proxy_offsets.unsqueeze(-1)
        ).squeeze(-1)
        return proxy_points, proxy_radii


    def _split_qpos(self, qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if qpos.shape[-1] != 36:
            raise ValueError(f"Expected qpos_36 last dim 36, got {qpos.shape}")
        root_pos = qpos[..., :3]
        root_quat = F.normalize(qpos[..., 3:7], dim=-1)
        root_quat = standardize_quaternion(root_quat)
        root_rot = quaternion_to_matrix(root_quat)
        joint_pos = qpos[..., 7:]
        return root_pos, root_quat, root_rot, joint_pos

    def _joint_rotation_matrix(
        self,
        joint_pos: torch.Tensor,
        joint_idx: int,
        batch_ndim: int,
        joint_axes: torch.Tensor,
    ) -> torch.Tensor:
        angle = joint_pos[..., joint_idx : joint_idx + 1]
        if self._joint_axes_are_cardinal:
            return _axis_aligned_rotation_matrix(
                angle,
                self._joint_axis_dims_py[joint_idx],
                self._joint_axis_signs_py[joint_idx],
            )
        axis = joint_axes[joint_idx]
        axis_view = _view_for_batch(axis, batch_ndim)
        return axis_angle_to_matrix(axis_view * angle)

    def _forward_body_pos_rot(self, qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos, _, root_rot, joint_pos = self._split_qpos(qpos)
        batch_ndim = len(qpos.shape[:-1])
        device = qpos.device
        dtype = qpos.dtype
        joint_axes = self.joint_axes.to(device=device, dtype=dtype)
        joint_origin_rot = self.joint_origin_rot.to(device=device, dtype=dtype)
        joint_origin_xyz = self.joint_origin_xyz.to(device=device, dtype=dtype)

        body_pos_by_index: list[torch.Tensor | None] = [None] * self.num_bodies
        body_rot_by_index: list[torch.Tensor | None] = [None] * self.num_bodies
        body_pos_by_index[0] = root_pos
        body_rot_by_index[0] = root_rot

        for joint_idx in range(self.num_joints):
            parent_idx = self._parent_body_indices_py[joint_idx]
            child_idx = self._child_body_indices_py[joint_idx]

            parent_pos = body_pos_by_index[parent_idx]
            parent_rot = body_rot_by_index[parent_idx]
            if parent_pos is None or parent_rot is None:
                raise RuntimeError(
                    f"Kinematic chain is not topologically ordered for joint {joint_idx}: "
                    f"parent body {parent_idx} is unresolved"
                )

            joint_rot = self._joint_rotation_matrix(joint_pos, joint_idx, batch_ndim, joint_axes)
            origin_rot = _view_for_batch(joint_origin_rot[joint_idx], batch_ndim)
            origin_xyz = _view_for_batch(joint_origin_xyz[joint_idx], batch_ndim).unsqueeze(-1)

            child_pos = parent_pos + (parent_rot @ origin_xyz).squeeze(-1)
            child_rot = parent_rot @ (origin_rot @ joint_rot)
            body_pos_by_index[child_idx] = child_pos
            body_rot_by_index[child_idx] = child_rot

        if any(x is None for x in body_pos_by_index) or any(x is None for x in body_rot_by_index):
            raise RuntimeError("Forward kinematics did not resolve all body states")
        body_pos = torch.stack([x for x in body_pos_by_index if x is not None], dim=-2)
        body_rot = torch.stack([x for x in body_rot_by_index if x is not None], dim=-3)
        return body_pos, body_rot

    def forward_body_positions(self, qpos: torch.Tensor) -> torch.Tensor:
        body_pos, _ = self._forward_body_pos_rot(qpos)
        return body_pos


    def forward_kinematics(self, qpos: torch.Tensor) -> dict[str, torch.Tensor]:
        body_pos, body_rot = self._forward_body_pos_rot(qpos)
        return {
            "body_pos_w": body_pos,
            "body_quat_w": self.matrix_to_body_quat_wxyz(body_rot),
        }

FK_DOUBLE = G1Kinematics(ASSETS/'g1_kinematics.json').double().cpu().eval()
FK_FLOAT = G1Kinematics(ASSETS/'g1_kinematics.json').float().cpu().eval()
urdf = ET.parse(FK_DOUBLE.urdf_path).getroot()
LIMITS = np.array([[float(node.attrib['lower']), float(node.attrib['upper'])]
                   for name in FK_DOUBLE.joint_order
                   for node in [urdf.find(f"./joint[@name='{name}']/limit")]], dtype=np.float64)


def original_mask(valid, length):
    if valid is None:
        return np.ones(length, bool)
    if np.ndim(valid) == 0:
        if isinstance(valid, (bool,np.bool_)) or not np.isfinite(valid) or int(valid) != valid or not 0 <= valid <= length:
            raise ValueError("Original valid prefix must be an integer in[0,T]")
        return np.arange(length) < int(valid)
    mask = np.asarray(valid)
    if mask.dtype != bool or mask.shape != (length,):
        raise ValueError("Original validity must be a boolean[T] mask")
    return mask


def predicted_valleys(qpos, distance):
    with torch.inference_mode():
        body = FK_FLOAT.forward_kinematics(torch.from_numpy(qpos.astype(np.float32)).unsqueeze(0))["body_pos_w"][0].numpy()
    if not np.isfinite(body).all():
        raise ValueError("Nonfinite float32 G1 positions")
    speed = np.linalg.norm(np.diff(body, axis=0), axis=-1).mean(axis=-1)
    return find_peaks(-speed, distance=distance)[0]


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=value.device, dtype=value.dtype)
    return (value * mask).sum() / mask.expand_as(value).sum().clamp_min(1.0)


def _fps_vector(fps: torch.Tensor | float, *, batch_size: int, like: torch.Tensor) -> torch.Tensor:
    fps_tensor = torch.as_tensor(fps, device=like.device, dtype=like.dtype)
    if fps_tensor.ndim == 0 or fps_tensor.numel() == 1:
        return fps_tensor.reshape(1).expand(batch_size)
    fps_tensor = fps_tensor.reshape(-1)
    if fps_tensor.numel() != batch_size:
        raise ValueError(f"fps must be scalar or have one value per batch item, got {tuple(fps_tensor.shape)}")
    return fps_tensor


def body_jerk_mean(
    *,
    body_pos_w: torch.Tensor,
    valid: torch.Tensor,
    fps: torch.Tensor | float,
) -> torch.Tensor:

    if body_pos_w.ndim != 4 or body_pos_w.shape[-1] != 3:
        raise ValueError(f"body_pos_w must have shape (B, T, J, 3), got {tuple(body_pos_w.shape)}")
    if valid.shape != body_pos_w.shape[:2]:
        raise ValueError(f"valid must have shape {tuple(body_pos_w.shape[:2])}, got {tuple(valid.shape)}")
    if body_pos_w.shape[1] < 4:
        raise ValueError("body_jerk_mean requires at least 4 frames")

    fps_per_batch = _fps_vector(fps, batch_size=body_pos_w.shape[0], like=body_pos_w)
    jerk = body_pos_w[:, 3:] - 3.0 * body_pos_w[:, 2:-1] + 3.0 * body_pos_w[:, 1:-2] - body_pos_w[:, :-3]
    jerk = jerk.norm(dim=-1) * fps_per_batch.pow(3).view(-1, 1, 1)
    valid_jerk = valid[:, 3:] & valid[:, 2:-1] & valid[:, 1:-2] & valid[:, :-3]
    return _masked_mean(jerk, valid_jerk)


def physical(values):
    with torch.inference_mode():
        body = FK_FLOAT.forward_body_positions(torch.from_numpy(values.astype(np.float32)))
    position = body.numpy()
    speed = np.linalg.norm(np.diff(position, axis=0), axis=-1).mean(axis=-1)
    events = find_peaks(-speed, distance=8)[0]
    jerk = float(body_jerk_mean(body_pos_w=body[None], valid=torch.ones((1, len(values)), dtype=torch.bool), fps=30.))
    return jerk, speed, events

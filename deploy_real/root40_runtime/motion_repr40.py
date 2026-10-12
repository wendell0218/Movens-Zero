from __future__ import annotations
import numpy as np
import torch
from .motion_repr import estimate_two_foot_contacts, matrix_to_quat_wxyz

ROOT40_ID = "root40_v1"
ROOT40_ACTION_DIM = 40
ROOT40_STATE_DIM = 40
MOTION_ABS_DIM = 36
ROOT40_JOINT_ABS = slice(0, 29)
ROOT40_ROOT_DELTA_XY = slice(29, 31)
ROOT40_ROOT_ROT6D = slice(31, 37)
ROOT40_ROOT_HEIGHT = slice(37, 38)
ROOT40_FOOT_CONTACT = slice(38, 40)
HEADING_MIN_PROJECTION = 0.25
ROTATION_MIN_BASIS = 0.0001
ROOT40_CONTACT_LEGACY = "legacy_joint_delta_v1"
ROOT40_CONTACT_ISAAC = "isaac_leg_delta_v2"
ROOT40_CONTACT_PROTOCOLS = (ROOT40_CONTACT_LEGACY, ROOT40_CONTACT_ISAAC)
ROOT40_ISAAC_LEFT_LEG = (0, 3, 6, 9, 13, 17)
ROOT40_ISAAC_RIGHT_LEG = (1, 4, 7, 10, 14, 18)
ROOT40_REPRESENTATION = {
    "id": ROOT40_ID,
    "action_dim": ROOT40_ACTION_DIM,
    "state_dim": ROOT40_STATE_DIM,
    "absolute_pose_dim": MOTION_ABS_DIM,
    "joint_order": "isaaclab_29",
    "slices": {
        "joint_abs": [0, 29],
        "root_delta_anchor_xy": [29, 31],
        "root_rotation_anchor_6d": [31, 37],
        "root_world_height": [37, 38],
        "foot_contact": [38, 40],
    },
    "rotation_6d_order": ["r00", "r10", "r20", "r01", "r11", "r21"],
    "rotation": "absolute body-to-anchor rotation, not relative rotation increments",
    "translation": "frame displacement in one fixed gravity-aligned chunk anchor",
    "anchor": "observed history only; reuse previous anchor when supplied",
    "heading_min_forward_projection": HEADING_MIN_PROJECTION,
    "heading_fallback": "latest observed stable forward axis, otherwise latest lateral axis",
    "projection_min_basis": ROTATION_MIN_BASIS,
    "state_first_delta": "zero",
    "contacts": "unchanged legacy joint-delta/world-z heuristic",
    "rtc_contacts": "preserve selected predicted channels when rebasing",
}


def _absolute_sequence(
    value: np.ndarray, name: str, allow_empty: bool = False
) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2 or value.shape[-1] != MOTION_ABS_DIM:
        raise ValueError(f"{name} must have shape (T,36) or (36,), got {value.shape}")
    if not allow_empty and (not len(value)):
        raise ValueError(f"{name} must contain an observed pose")
    if not np.isfinite(value).all() or np.any(np.abs(value) > np.finfo(np.float32).max):
        raise ValueError(f"{name} must contain finite float32-representable values")
    norm = np.linalg.norm(value[:, 32:36], axis=-1)
    if np.any(norm < 1e-08):
        raise ValueError(f"{name} contains a zero or near-zero root quaternion")
    return value


def _anchor_matrix(anchor: np.ndarray) -> np.ndarray:
    anchor = np.asarray(anchor, dtype=np.float64)
    if anchor.shape != (3, 3) or not np.isfinite(anchor).all():
        raise ValueError("anchor must be a finite 3x3 yaw-only rotation")
    if (
        not np.allclose(anchor.T @ anchor, np.eye(3), atol=1e-05, rtol=0)
        or not np.isclose(np.linalg.det(anchor), 1.0, atol=1e-05, rtol=0)
        or (not np.allclose(anchor[:, 2], [0, 0, 1], atol=1e-05, rtol=0))
        or (not np.allclose(anchor[2, :], [0, 0, 1], atol=1e-05, rtol=0))
    ):
        raise ValueError("anchor must be a proper gravity-aligned yaw rotation")
    return anchor


def _quaternion_matrices(quaternions: np.ndarray) -> np.ndarray:
    quaternions = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(quaternions, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(quaternions.shape[:-1] + (3, 3))


def choose_heading_anchor(
    states_abs: np.ndarray, previous_anchor: np.ndarray | None = None
) -> np.ndarray:
    states_abs = _absolute_sequence(states_abs, "states_abs")
    if previous_anchor is not None:
        return _anchor_matrix(previous_anchor).astype(np.float32)
    rotations = _quaternion_matrices(states_abs[:, 32:36])
    projected_forward = rotations[:, :2, 0]
    norms = np.linalg.norm(projected_forward, axis=-1)
    stable = np.flatnonzero(norms >= HEADING_MIN_PROJECTION)
    if len(stable):
        forward = projected_forward[stable[-1]]
    else:
        lateral = rotations[-1, :2, 1]
        forward = np.array([lateral[1], -lateral[0]], dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    x, y = forward
    return np.array([[x, -y, 0], [y, x, 0], [0, 0, 1]], dtype=np.float32)


def estimate_root40_foot_contacts(
    joint_delta: np.ndarray,
    root_delta_world: np.ndarray,
    *,
    contact_protocol: str = ROOT40_CONTACT_LEGACY,
) -> np.ndarray:
    if contact_protocol == ROOT40_CONTACT_LEGACY:
        return estimate_two_foot_contacts(joint_delta, root_delta_world)
    if contact_protocol != ROOT40_CONTACT_ISAAC:
        raise ValueError(f"Unknown root40 contact protocol: {contact_protocol}")
    joint_delta = np.asarray(joint_delta)
    if joint_delta.shape[-1:] != (29,):
        raise ValueError("isaac_leg_delta_v2 requires 29 IsaacLab-ordered joint deltas")
    legs = joint_delta[..., ROOT40_ISAAC_LEFT_LEG + ROOT40_ISAAC_RIGHT_LEG]
    return estimate_two_foot_contacts(legs, root_delta_world)


def _encode(
    poses: np.ndarray,
    previous: np.ndarray,
    anchor: np.ndarray,
    contact_protocol: str = ROOT40_CONTACT_LEGACY,
) -> np.ndarray:
    delta_world = poses[:, 29:32] - previous[:, 29:32]
    delta_anchor = delta_world @ anchor
    local_rotation = anchor.T @ _quaternion_matrices(poses[:, 32:36])
    rotation6 = np.concatenate(
        [local_rotation[:, :, 0], local_rotation[:, :, 1]], axis=-1
    )
    poses32, previous32 = (poses.astype(np.float32), previous.astype(np.float32))
    contacts = estimate_root40_foot_contacts(
        poses32[:, :29] - previous32[:, :29],
        poses32[:, 29:32] - previous32[:, 29:32],
        contact_protocol=contact_protocol,
    )
    return np.concatenate(
        [poses[:, :29], delta_anchor[:, :2], rotation6, poses[:, 31:32], contacts],
        axis=-1,
    ).astype(np.float32)


def encode_root40(
    actions_abs: np.ndarray,
    states_abs: np.ndarray,
    anchor: np.ndarray | None = None,
    previous_anchor: np.ndarray | None = None,
    *,
    contact_protocol: str = ROOT40_CONTACT_LEGACY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions_abs = _absolute_sequence(actions_abs, "actions_abs", allow_empty=True)
    states_abs = _absolute_sequence(states_abs, "states_abs")
    if anchor is not None and previous_anchor is not None:
        raise ValueError("Provide anchor or previous_anchor, not both")
    anchor = (
        choose_heading_anchor(states_abs, previous_anchor)
        if anchor is None
        else _anchor_matrix(anchor)
    )
    previous_actions = np.concatenate([states_abs[-1:], actions_abs[:-1]], axis=0)[
        : len(actions_abs)
    ]
    previous_states = np.concatenate([states_abs[:1], states_abs[:-1]], axis=0)
    return (
        _encode(actions_abs, previous_actions, anchor, contact_protocol),
        _encode(states_abs, previous_states, anchor, contact_protocol),
        anchor.astype(np.float32),
    )


def build_root40_action(
    actions_abs: np.ndarray,
    states_abs: np.ndarray,
    anchor: np.ndarray | None = None,
    *,
    contact_protocol: str = ROOT40_CONTACT_LEGACY,
) -> np.ndarray:
    return encode_root40(
        actions_abs, states_abs, anchor, contact_protocol=contact_protocol
    )[0]


def build_root40_state(
    states_abs: np.ndarray,
    anchor: np.ndarray | None = None,
    *,
    contact_protocol: str = ROOT40_CONTACT_LEGACY,
) -> np.ndarray:
    return encode_root40(
        np.empty((0, MOTION_ABS_DIM)),
        states_abs,
        anchor,
        contact_protocol=contact_protocol,
    )[1]


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    values = np.asarray(rot6d, dtype=np.float64)
    if values.shape[-1:] != (6,) or not np.isfinite(values).all():
        raise ValueError("rotation features must be finite with last dimension 6")
    first, second = (values[..., :3], values[..., 3:])
    first = first / np.maximum(np.max(np.abs(first), axis=-1, keepdims=True), 1.0)
    second = second / np.maximum(np.max(np.abs(second), axis=-1, keepdims=True), 1.0)
    norm_first = np.linalg.norm(first, axis=-1, keepdims=True)
    first = np.where(norm_first >= ROTATION_MIN_BASIS, first, [1.0, 0.0, 0.0])
    first = first / np.linalg.norm(first, axis=-1, keepdims=True)
    orthogonal = second - np.sum(first * second, axis=-1, keepdims=True) * first
    fallback = np.eye(3)[np.argmin(np.abs(first), axis=-1)]
    fallback = fallback - np.sum(first * fallback, axis=-1, keepdims=True) * first
    orthogonal = np.where(
        np.linalg.norm(orthogonal, axis=-1, keepdims=True) >= ROTATION_MIN_BASIS,
        orthogonal,
        fallback,
    )
    second = orthogonal / np.linalg.norm(orthogonal, axis=-1, keepdims=True)
    third = np.cross(first, second)
    third = third / np.linalg.norm(third, axis=-1, keepdims=True)
    second = np.cross(third, first)
    return np.stack([first, second, third], axis=-1).astype(np.float32)


def recover_motion_abs_from_root40_action(
    features: np.ndarray, current_abs: np.ndarray, anchor: np.ndarray | None = None
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    current_abs = _absolute_sequence(current_abs, "current_abs")
    if (
        features.ndim != 2
        or features.shape[-1] != ROOT40_ACTION_DIM
        or (not np.isfinite(features).all())
    ):
        raise ValueError("features must have finite shape (T,40)")
    anchor = (
        choose_heading_anchor(current_abs) if anchor is None else _anchor_matrix(anchor)
    )
    if not len(features):
        return np.empty((0, MOTION_ABS_DIM), dtype=np.float32)
    world_delta_xy = features[:, ROOT40_ROOT_DELTA_XY] @ anchor[:2, :2].T
    root_xy = current_abs[-1, 29:31] + np.cumsum(world_delta_xy, axis=0)
    rotations = anchor @ rot6d_to_matrix(features[:, ROOT40_ROOT_ROT6D])
    return np.concatenate(
        [
            features[:, :29],
            root_xy,
            features[:, ROOT40_ROOT_HEIGHT],
            matrix_to_quat_wxyz(rotations),
        ],
        axis=-1,
    ).astype(np.float32)


def reencode_root40_prefix(
    features: np.ndarray,
    old_current_abs: np.ndarray,
    old_anchor: np.ndarray,
    new_current_abs: np.ndarray,
    new_anchor: np.ndarray,
    start_index: int = 0,
    length: int | None = None,
    *,
    contact_protocol: str = ROOT40_CONTACT_LEGACY,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if (
        not isinstance(start_index, (int, np.integer))
        or start_index < 0
        or start_index > len(features)
    ):
        raise ValueError(
            "start_index must identify a position within the old action chunk"
        )
    if length is not None and (not isinstance(length, (int, np.integer)) or length < 0):
        raise ValueError("length must be a nonnegative integer or None")
    stop = len(features) if length is None else start_index + length
    if stop > len(features):
        raise ValueError("Requested prefix extends past the old action chunk")
    absolute = recover_motion_abs_from_root40_action(
        features, old_current_abs, old_anchor
    )
    rebased = build_root40_action(
        absolute[start_index:stop],
        new_current_abs,
        new_anchor,
        contact_protocol=contact_protocol,
    )
    rebased[:, ROOT40_FOOT_CONTACT] = features[start_index:stop, ROOT40_FOOT_CONTACT]
    return rebased


def _torch_float(values: torch.Tensor, last_dim: int) -> torch.Tensor:
    if values.shape[-1:] != (last_dim,) or not torch.is_floating_point(values):
        raise ValueError(f"Expected floating tensor with last dimension {last_dim}")
    return values if values.dtype == torch.float64 else values.float()


def rot6d_to_matrix_torch(rot6d: torch.Tensor) -> torch.Tensor:
    values = _torch_float(rot6d, 6)
    first, second = (values[..., :3], values[..., 3:])
    first = first / first.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    second = second / second.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    norm_first = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    first = torch.where(
        norm_first >= ROTATION_MIN_BASIS, first, first.new_tensor([1.0, 0.0, 0.0])
    )
    first = first / torch.linalg.vector_norm(first, dim=-1, keepdim=True).clamp_min(
        ROTATION_MIN_BASIS
    )
    orthogonal = second - (first * second).sum(dim=-1, keepdim=True) * first
    fallback = torch.eye(3, device=values.device, dtype=values.dtype)[
        first.abs().argmin(dim=-1)
    ]
    fallback = fallback - (first * fallback).sum(dim=-1, keepdim=True) * first
    norm_second = torch.linalg.vector_norm(orthogonal, dim=-1, keepdim=True)
    orthogonal = torch.where(norm_second >= ROTATION_MIN_BASIS, orthogonal, fallback)
    second = orthogonal / torch.linalg.vector_norm(
        orthogonal, dim=-1, keepdim=True
    ).clamp_min(ROTATION_MIN_BASIS)
    third = torch.linalg.cross(first, second, dim=-1)
    third = third / torch.linalg.vector_norm(third, dim=-1, keepdim=True).clamp_min(
        ROTATION_MIN_BASIS
    )
    second = torch.linalg.cross(third, first, dim=-1)
    return torch.stack([first, second, third], dim=-1)


def matrix_to_quat_wxyz_torch(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3) or not torch.is_floating_point(matrix):
        raise ValueError("Expected a floating (...,3,3) rotation matrix")
    matrix = matrix if matrix.dtype == torch.float64 else matrix.float()
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix.flatten(-2).unbind(-1)
    trace = m00 + m11 + m22
    squared = torch.stack(
        [1 + trace, 1 + 2 * m00 - trace, 1 + 2 * m11 - trace, 1 + 2 * m22 - trace],
        dim=-1,
    )
    index = squared.argmax(dim=-1)
    rows = torch.stack(
        [
            torch.stack([squared[..., 0], m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, squared[..., 1], m01 + m10, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m01 + m10, squared[..., 2], m12 + m21], dim=-1),
            torch.stack([m10 - m01, m02 + m20, m12 + m21, squared[..., 3]], dim=-1),
        ],
        dim=-2,
    )
    selected = rows.gather(
        -2, index[..., None, None].expand(index.shape + (1, 4))
    ).squeeze(-2)
    denominator = 2 * squared.gather(-1, index[..., None]).clamp_min(1e-08).sqrt()
    quaternion = selected / denominator
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1e-08)
    return torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)


def root40_to_motion_abs_torch(
    features: torch.Tensor,
    current_abs: torch.Tensor | None = None,
    anchor: torch.Tensor | None = None,
) -> torch.Tensor:
    features = _torch_float(features, ROOT40_ACTION_DIM)
    if features.ndim < 2:
        raise ValueError("features must have shape (...,T,40)")
    local_delta = features[..., ROOT40_ROOT_DELTA_XY]
    rotations = rot6d_to_matrix_torch(features[..., ROOT40_ROOT_ROT6D])
    if anchor is not None:
        anchor = torch.as_tensor(anchor, device=features.device, dtype=features.dtype)
        if anchor.shape[-2:] != (3, 3):
            raise ValueError("anchor must have shape (...,3,3)")
        local_delta = torch.matmul(local_delta, anchor[..., :2, :2].transpose(-1, -2))
        rotations = torch.matmul(anchor.unsqueeze(-3), rotations)
    root_xy = local_delta.cumsum(dim=-2)
    if current_abs is not None:
        current_abs = torch.as_tensor(
            current_abs, device=features.device, dtype=features.dtype
        )
        if current_abs.shape[-1:] != (MOTION_ABS_DIM,):
            raise ValueError("current_abs must have shape (...,36)")
        root_xy = root_xy + current_abs[..., None, 29:31]
    root_quaternion = matrix_to_quat_wxyz_torch(rotations)
    return torch.cat(
        [
            features[..., :29],
            root_xy,
            features[..., ROOT40_ROOT_HEIGHT],
            root_quaternion,
        ],
        dim=-1,
    )

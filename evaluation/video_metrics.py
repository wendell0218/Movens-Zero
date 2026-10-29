from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

MetricArrays = dict[str, np.ndarray]

def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:

    arr = _positions(points, "points")
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape == (4, 4):
        return np.einsum("ij,tkj->tki", matrix[:3, :3], arr) + matrix[:3, 3]
    if matrix.shape == (len(arr), 4, 4):
        return np.einsum("tij,tkj->tki", matrix[:, :3, :3], arr) + matrix[:, None, :3, 3]
    raise ValueError(
        f"transform must have shape (4,4) or ({len(arr)},4,4), got {matrix.shape}"
    )

def _positions(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (F, J, 3), got {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one frame and one marker")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr

def _paired_positions(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_arr = _positions(pred, "pred")
    target_arr = _positions(target, "target")
    if pred_arr.shape != target_arr.shape:
        raise ValueError(f"pred/target shape mismatch: {pred_arr.shape} vs {target_arr.shape}")
    return pred_arr, target_arr

def _apply_mask(arrays: Sequence[np.ndarray], mask: np.ndarray | None) -> list[np.ndarray]:
    if mask is None:
        return [np.asarray(x) for x in arrays]
    mask_arr = np.asarray(mask, dtype=bool)
    frames = arrays[0].shape[0]
    if mask_arr.shape != (frames,):
        raise ValueError(f"mask must have shape ({frames},), got {mask_arr.shape}")
    if not mask_arr.any():
        raise ValueError("mask selects no valid frames")
    return [np.asarray(x)[mask_arr].copy() for x in arrays]

def align_similarity(
    target: np.ndarray,
    pred: np.ndarray,
    *,
    fixed_scale: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:

    y = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    x = np.asarray(pred, dtype=np.float64).reshape(-1, 3)
    if x.shape != y.shape or len(x) == 0:
        raise ValueError(f"Alignment inputs must have equal non-empty Nx3 shapes: {x.shape}, {y.shape}")
    mean_y = y.mean(axis=0)
    mean_x = x.mean(axis=0)
    y0 = y - mean_y
    x0 = x - mean_x
    covariance = y0.T @ x0 / len(x)
    u, singular_values, vh = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vh.T) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vh
    if fixed_scale:
        scale = 1.0
    else:
        variance = np.square(x0).sum() / len(x)
        if variance <= np.finfo(np.float64).eps:
            raise ValueError("Cannot estimate similarity scale from a degenerate prediction")
        scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = mean_y - scale * (rotation @ mean_x)
    return scale, rotation, translation

def apply_similarity(
    points: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    return scale * np.einsum("ij,...j->...i", rotation, arr) + translation

def _joint_error_mm(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - target, axis=-1).mean(axis=-1) * 1000.0

def compute_local_metrics(
    pred_joints: np.ndarray,
    target_joints: np.ndarray,
    *,
    pelvis_indices: tuple[int, ...] = (1, 2),
    fps: float = 30.0,
    mask: np.ndarray | None = None,
) -> MetricArrays:

    pred, target = _paired_positions(pred_joints, target_joints)
    pred, target = _apply_mask([pred, target], mask)
    if not pelvis_indices:
        raise ValueError("pelvis_indices cannot be empty")
    if min(pelvis_indices) < 0 or max(pelvis_indices) >= pred.shape[1]:
        raise ValueError(f"Invalid pelvis indices {pelvis_indices} for {pred.shape[1]} markers")
    pred_root = pred[:, pelvis_indices].mean(axis=1, keepdims=True)
    target_root = target[:, pelvis_indices].mean(axis=1, keepdims=True)
    pred_rel = pred - pred_root
    target_rel = target - target_root

    pa_pred = np.empty_like(pred_rel)
    for frame_idx in range(len(pred_rel)):
        transform = align_similarity(target_rel[frame_idx], pred_rel[frame_idx])
        pa_pred[frame_idx] = apply_similarity(pred_rel[frame_idx], *transform)

    if len(pred_rel) >= 3:
        pred_accel = (pred_rel[:-2] - 2 * pred_rel[1:-1] + pred_rel[2:]) * float(fps) ** 2
        target_accel = (target_rel[:-2] - 2 * target_rel[1:-1] + target_rel[2:]) * float(fps) ** 2
        accel = np.linalg.norm(pred_accel - target_accel, axis=-1).mean(axis=-1)
    else:
        accel = np.empty(0, dtype=np.float64)

    return {
        "pa_mpjpe_mm": _joint_error_mm(pa_pred, target_rel),
        "mpjpe_mm": _joint_error_mm(pred_rel, target_rel),
        "accel_m_per_s2": accel,
    }

def compute_rte_percent(
    target_root: np.ndarray,
    pred_root: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    allow_undefined: bool = False,
) -> np.ndarray:

    target = np.asarray(target_root, dtype=np.float64)
    pred = np.asarray(pred_root, dtype=np.float64)
    if target.shape != pred.shape or target.ndim != 2 or target.shape[1] != 3:
        raise ValueError(f"Root trajectories must have equal (F,3) shapes: {target.shape}, {pred.shape}")

    if valid is None:
        mask = np.ones(len(target), dtype=bool)
    else:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != (len(target),):
            raise ValueError(f"RTE valid mask must have shape ({len(target)},)")
    adjacent = mask[:-1] & mask[1:]
    path_length = np.linalg.norm(np.diff(target, axis=0), axis=-1)[adjacent].sum()
    if path_length <= np.finfo(np.float64).eps:
        if allow_undefined:
            return np.empty(0, dtype=np.float64)
        raise ValueError("RTE is undefined for a target root trajectory with zero valid path length")
    target = target[mask]
    pred = pred[mask]
    scale, rotation, translation = align_similarity(target, pred, fixed_scale=True)
    pred_aligned = apply_similarity(pred, scale, rotation, translation)

    return np.linalg.norm(target - pred_aligned, axis=-1) / path_length * 100.0

def compute_jitter(joints: np.ndarray, *, fps: float = 30.0) -> np.ndarray:

    arr = _positions(joints, "joints")
    if len(arr) < 4:
        return np.empty(0, dtype=np.float64)
    third = (arr[3:] - 3 * arr[2:-1] + 3 * arr[1:-2] - arr[:-3]) * float(fps) ** 3
    return np.linalg.norm(third, axis=-1).mean(axis=-1) / 10.0

def compute_foot_sliding_g1(
    target_sole_points: np.ndarray,
    pred_sole_points: np.ndarray,
    *,
    contact_threshold_m_per_frame: float = 1e-2,
) -> np.ndarray:

    pred, target = _paired_positions(pred_sole_points, target_sole_points)
    if len(target) < 2:
        return np.empty(0, dtype=np.float64)
    target_disp = np.linalg.norm(np.diff(target, axis=0), axis=-1)
    pred_disp = np.linalg.norm(np.diff(pred, axis=0), axis=-1)
    contact = target_disp < float(contact_threshold_m_per_frame)
    return pred_disp[contact] * 1000.0

def compute_global_metrics(
    pred_joints: np.ndarray,
    target_joints: np.ndarray,
    *,
    pred_sole_points: np.ndarray,
    target_sole_points: np.ndarray,
    fps: float = 30.0,
    mask: np.ndarray | None = None,
    chunk_length: int = 100,
    contact_threshold_m_per_frame: float = 1e-2,
    include_rte: bool = True,
) -> MetricArrays:

    pred, target = _paired_positions(pred_joints, target_joints)
    pred_feet, target_feet = _paired_positions(pred_sole_points, target_sole_points)
    if len(pred_feet) != len(pred):
        raise ValueError("Joint and sole trajectories must have the same frame count")
    pred, target, pred_feet, target_feet = _apply_mask(
        [pred, target, pred_feet, target_feet], mask
    )
    if int(chunk_length) <= 0:
        raise ValueError(f"chunk_length must be positive, got {chunk_length}")

    wa_errors: list[np.ndarray] = []
    w_errors: list[np.ndarray] = []
    for start in range(0, len(pred), int(chunk_length)):
        end = min(len(pred), start + int(chunk_length))
        pred_chunk = pred[start:end]
        target_chunk = target[start:end]

        wa_transform = align_similarity(target_chunk, pred_chunk)
        wa_pred = apply_similarity(pred_chunk, *wa_transform)
        wa_errors.append(_joint_error_mm(wa_pred, target_chunk))

        align_frames = min(2, len(pred_chunk))
        w_transform = align_similarity(target_chunk[:align_frames], pred_chunk[:align_frames])
        w_pred = apply_similarity(pred_chunk, *w_transform)
        w_errors.append(_joint_error_mm(w_pred, target_chunk))

    result = {
        "wa_mpjpe_100_mm": np.concatenate(wa_errors),
        "w_mpjpe_100_mm": np.concatenate(w_errors),

        "jitter_10m_per_s3": compute_jitter(pred, fps=fps),
        "foot_sliding_g1_mm_per_frame": compute_foot_sliding_g1(
            target_feet,
            pred_feet,
            contact_threshold_m_per_frame=contact_threshold_m_per_frame,
        ),
    }

    if include_rte:
        result["rte_percent"] = compute_rte_percent(target[:, 0], pred[:, 0])
    return result

def aggregate_metric_arrays(sequence_metrics: Sequence[Mapping[str, np.ndarray]]) -> dict[str, float]:

    if not sequence_metrics:
        raise ValueError("No sequence metrics to aggregate")
    expected = set(sequence_metrics[0])
    if any(set(metrics) != expected for metrics in sequence_metrics[1:]):
        raise ValueError("All sequences must expose identical metric keys")
    result: dict[str, float] = {}
    for key in sorted(expected):
        arrays = [np.asarray(metrics[key], dtype=np.float64).reshape(-1) for metrics in sequence_metrics]
        nonempty = [arr for arr in arrays if arr.size]
        if not nonempty:
            result[key] = float("nan")
        else:
            result[key] = float(np.concatenate(nonempty).mean())
    return result

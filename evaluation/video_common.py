from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from video_metrics import (
    aggregate_metric_arrays,
    compute_global_metrics,
    compute_local_metrics,
    compute_rte_percent,
    transform_points,
)
from video_motion import G1Kinematics, G1_MUJOCO_JOINT_NAMES, qpos_to_motion36

def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

def valid_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(valid, dtype=bool)
    padded = np.r_[False, mask, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))

def load_qpos_npz(path: str | Path, *, role: str) -> dict:
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as data:
        keys = (f"{role}_qpos_36", "qpos_36")
        key = next((name for name in keys if name in data), None)
        if key is None:
            raise KeyError(f"{path}: expected one of {keys}")
        raw_qpos = data[key]
        if raw_qpos.dtype != np.float32:
            raise TypeError(f"{path}: {key} must be float32, got {raw_qpos.dtype}")
        qpos = np.asarray(raw_qpos, dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) == 0:
            raise ValueError(f"{path}: {key} must have shape [F,36], got {qpos.shape}")
        if not np.isfinite(qpos).all():
            raise ValueError(f"{path}: qpos contains non-finite values")
        quaternion_norm = np.linalg.norm(qpos[:, 3:7], axis=1)
        if np.max(np.abs(quaternion_norm - 1.0)) > 1e-5:
            raise ValueError(f"{path}: root quaternion is not normalized")
        qpos[:, 3:7] /= quaternion_norm[:, None]

        if "fps" not in data or np.asarray(data["fps"]).size != 1:
            raise ValueError(f"{path}: fps must be a numeric scalar")
        fps = float(np.asarray(data["fps"]).reshape(()))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"{path}: invalid fps={fps}")

        if "joint_names" not in data:
            raise KeyError(f"{path}: joint_names[29] is required")
        joint_names = tuple(str(value) for value in data["joint_names"].tolist())
        if joint_names != G1_MUJOCO_JOINT_NAMES:
            raise ValueError(f"{path}: joint_names do not match G1 MuJoCo order")

        valid = None
        if "valid" in data:
            if data["valid"].dtype != np.bool_:
                raise TypeError(f"{path}: valid must be bool, got {data['valid'].dtype}")
            valid = np.asarray(data["valid"], dtype=bool)
            if valid.shape != (len(qpos),) or not valid.any():
                raise ValueError(f"{path}: valid must be non-empty bool [F]")

        camera = None
        if "T_robot_z_to_camera" in data:
            camera = np.asarray(data["T_robot_z_to_camera"], dtype=np.float64)
            if camera.shape not in ((4, 4), (len(qpos), 4, 4)):
                raise ValueError(f"{path}: invalid camera transform shape {camera.shape}")
            if not np.isfinite(camera).all():
                raise ValueError(f"{path}: camera transform contains non-finite values")

        score_start_frame = None
        if "score_start_frame" in data:
            if np.asarray(data["score_start_frame"]).size != 1:
                raise ValueError(f"{path}: score_start_frame must be a scalar")
            score_start_frame = int(np.asarray(data["score_start_frame"]).reshape(()))
            if score_start_frame < 0 or score_start_frame >= len(qpos):
                raise ValueError(f"{path}: invalid score_start_frame={score_start_frame}")

        has_sample_id = "sample_id" in data
        sample_id = str(data["sample_id"]) if has_sample_id else path.stem
        sequence_id = str(data["sequence_id"]) if "sequence_id" in data else sample_id
    return {
        "path": path,
        "qpos": qpos,
        "fps": fps,
        "valid": valid,
        "camera": camera,
        "score_start_frame": score_start_frame,
        "sample_id": sample_id,
        "has_sample_id": has_sample_id,
        "sequence_id": sequence_id,
    }

def _merge_metric_arrays(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("no valid metric segments")
    keys = set(parts[0])
    if any(set(part) != keys for part in parts[1:]):
        raise ValueError("metric segments expose different keys")
    return {
        key: np.concatenate([np.asarray(part[key]).reshape(-1) for part in parts])
        for key in keys
    }

def evaluate_qpos_pair(
    prediction_qpos: np.ndarray,
    target_qpos: np.ndarray,
    valid: np.ndarray,
    *,
    fps: float,
    mode: str,
    mask_mode: str,
    kinematics: G1Kinematics,
    camera_transform: np.ndarray | None = None,
    allow_constant_camera_invariance: bool = False,

    camera_rotation: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if mode not in ("camera", "world", "both"):
        raise ValueError(f"unsupported video mode: {mode}")
    if mask_mode not in ("legacy_compress", "contiguous_segments"):
        raise ValueError(f"unsupported mask mode: {mask_mode}")
    prediction_qpos = np.asarray(prediction_qpos, dtype=np.float64)
    target_qpos = np.asarray(target_qpos, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if prediction_qpos.shape != target_qpos.shape:
        raise ValueError(
            f"prediction/target qpos mismatch: {prediction_qpos.shape} vs {target_qpos.shape}"
        )
    if valid.shape != (len(target_qpos),) or not valid.any():
        raise ValueError(f"valid must have shape ({len(target_qpos)},) and select frames")

    prediction_fk = kinematics.forward(qpos_to_motion36(prediction_qpos))
    target_fk = kinematics.forward(qpos_to_motion36(target_qpos))

    prediction_local = target_local = None
    if mode in ("camera", "both"):

        if camera_rotation is not None and camera_transform is not None:
            raise ValueError("Provide either camera rotation or full transform, not both")
        if camera_transform is not None:
            prediction_local = transform_points(prediction_fk.joints24, camera_transform)
            target_local = transform_points(target_fk.joints24, camera_transform)
        elif camera_rotation is not None:
            rotation = np.asarray(camera_rotation, dtype=np.float64)
            if rotation.shape != (len(target_qpos), 3, 3) or not np.isfinite(rotation).all():
                raise ValueError("Camera rotations must be finite [F,3,3]")
            if not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-6, rtol=0) or not np.allclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0):
                raise ValueError("Camera rotations must be in SO(3)")
            prediction_local = prediction_fk.joints24 - prediction_fk.joints24[:, [1, 2]].mean(axis=1, keepdims=True)
            target_local = target_fk.joints24 - target_fk.joints24[:, [1, 2]].mean(axis=1, keepdims=True)
            prediction_local = np.einsum("tij,tkj->tki", rotation, prediction_local)
            target_local = np.einsum("tij,tkj->tki", rotation, target_local)
        elif allow_constant_camera_invariance:
            prediction_local = prediction_fk.joints24
            target_local = target_fk.joints24
        else:

            raise ValueError("Camera metrics require a full transform or aligned camera rotations")

    def one(index, one_valid):
        metric = {}
        if mode in ("camera", "both"):
            metric.update(
                compute_local_metrics(
                    prediction_local[index], target_local[index], fps=fps, mask=one_valid
                )
            )
        if mode in ("world", "both"):
            metric.update(
                compute_global_metrics(
                    prediction_fk.joints24[index],
                    target_fk.joints24[index],
                    pred_sole_points=prediction_fk.sole_points[index],
                    target_sole_points=target_fk.sole_points[index],
                    fps=fps,
                    mask=one_valid,
                    include_rte=mask_mode == "legacy_compress",
                )
            )
        return metric

    if mask_mode == "legacy_compress":
        return one(slice(None), valid)
    parts = [one(slice(start, end), None) for start, end in valid_runs(valid)]

    result = _merge_metric_arrays(parts)
    if mode in ("world", "both"):
        result["rte_percent"] = compute_rte_percent(
            target_fk.joints24[:, 0], prediction_fk.joints24[:, 0],
            valid=valid, allow_undefined=True,
        )
    return result

def aggregate_by_space(sample_metrics: list[dict]) -> dict:
    result = {}
    for space, keys in (
        ("camera", {"pa_mpjpe_mm", "mpjpe_mm", "accel_m_per_s2"}),
        (
            "world",
            {
                "wa_mpjpe_100_mm",
                "w_mpjpe_100_mm",
                "rte_percent",
                "jitter_10m_per_s3",
                "foot_sliding_g1_mm_per_frame",
            },
        ),
    ):
        selected = [
            {key: item["metrics"][key] for key in keys}
            for item in sample_metrics
            if keys.issubset(item["metrics"])
        ]
        result[space] = {
            "sequences": len(selected),
            "summary": aggregate_metric_arrays(selected) if selected else None,
        }
    return result

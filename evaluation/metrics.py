from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import torch
from tqdm import tqdm

from omg.benchmarks.metrics import e_acc, e_vel, g_mpjpe, mpjpe
from omg.benchmarks.runners.common import _summary_stats
from omg.generation.metrics.physical import physical_qpos_metrics
from omg.robots.g1.constants import G1_JOINT_NAMES

def read_npz(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as handle:
        return {key: np.asarray(handle[key]) for key in handle.files}

def pick_array(
    payload: dict[str, np.ndarray], keys: tuple[str, ...], *, required: bool = True
):
    for key in keys:
        if key in payload:
            return payload[key]
    if required:
        raise KeyError(f"Missing one of {keys}; available keys: {sorted(payload)}")
    return None

def _fps_array(value: Any, num_samples: int, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{name} must have numeric dtype, got {array.dtype}")
    array = array.astype(np.float32, copy=False)
    if array.ndim == 0 or array.size == 1:
        array = np.full(num_samples, float(array.reshape(-1)[0]), dtype=np.float32)
    else:
        array = array.reshape(-1)
    if array.shape != (num_samples,):
        raise ValueError(f"{name} must be scalar or [N], got {array.shape}")
    if not np.isfinite(array).all() or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    return array

def _validate_qpos(
    qpos: Any,
    *,
    name: str,
    valid: np.ndarray | None = None,
    check_quaternion: bool = True,
) -> np.ndarray:
    array = np.asarray(qpos)
    if array.dtype != np.float32:
        raise ValueError(f"{name} dtype must be float32, got {array.dtype}")
    if array.ndim != 3 or array.shape[-1] != 36:
        raise ValueError(f"{name} must have shape [N,T,36], got {array.shape}")
    if array.shape[0] < 2 or array.shape[1] < 4:
        raise ValueError(f"{name} requires N>=2 and T>=4, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")

    if not check_quaternion:
        return array
    if valid is not None:
        valid = np.asarray(valid)
        if valid.dtype != np.bool_ or valid.shape != array.shape[:2]:
            raise ValueError(f"{name} valid must be bool {array.shape[:2]}, got {valid.dtype} {valid.shape}")
        quat_norm = np.linalg.norm(array[..., 3:7][valid], axis=-1)
    else:
        quat_norm = np.linalg.norm(array[..., 3:7], axis=-1)
    if not np.allclose(quat_norm, 1.0, atol=5.0e-3, rtol=0.0):
        max_error = float(np.max(np.abs(quat_norm - 1.0)))
        raise ValueError(
            f"{name} root quaternion must be normalized wxyz; max norm error={max_error:.6g}"
        )
    return array

def _validate_identity(
    generated: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    condition: dict[str, np.ndarray],
    num_samples: int,
) -> np.ndarray:
    identities = []
    for label, payload in (
        ("generated", generated),
        ("reference", reference),
        ("condition", condition),
    ):
        value = np.asarray(pick_array(payload, ("sample_id", "sample_ids")))
        if value.dtype.kind not in {"U", "S"}:
            raise ValueError(
                f"{label} sample_id must have string dtype, got {value.dtype}"
            )
        value = value.astype(str).reshape(-1)
        if value.shape != (num_samples,):
            raise ValueError(
                f"{label} sample_id must have shape [N], got {value.shape}"
            )
        if np.any(np.char.str_len(value) == 0) or np.unique(value).size != num_samples:
            raise ValueError(
                f"{label} sample_id must contain N non-empty unique strings"
            )
        identities.append(value)
    for value in identities[1:]:
        if not np.array_equal(identities[0], value):
            raise ValueError(
                "sample_id order differs across generated/reference/condition NPZ files"
            )
    return identities[0]

def _repeat_last_valid_reference_frame(
    qpos: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, int]:

    normalized = qpos.copy()
    partial_count = 0
    for index in range(qpos.shape[0]):
        length = int(valid[index].sum())
        if length < qpos.shape[1]:
            normalized[index, length:] = normalized[index, length - 1]
            partial_count += 1
    return normalized, partial_count

def load_benchmark_inputs(
    *,
    task: str,
    generated_path: str,
    reference_path: str,
    condition_path: str,
    expected_frames: int,
    expected_fps: float,
) -> dict[str, Any]:
    generated = read_npz(generated_path)
    reference = read_npz(reference_path)
    condition = read_npz(condition_path)

    generated_qpos = _validate_qpos(
        pick_array(generated, ("qpos_36", "generated_qpos_36")),
        name="generated qpos",
        check_quaternion=False,
    )
    reference_qpos = _validate_qpos(
        pick_array(reference, ("qpos_36", "reference_qpos_36")),
        name="reference qpos",
        check_quaternion=False,
    )
    if generated_qpos.shape != reference_qpos.shape:
        raise ValueError(
            f"generated/reference qpos shapes differ: {generated_qpos.shape} != {reference_qpos.shape}"
        )
    expected_joint_names = np.asarray(G1_JOINT_NAMES, dtype=np.str_)
    for label, payload in (("generated", generated), ("reference", reference)):
        joint_names = np.asarray(pick_array(payload, ("joint_names",)))
        if joint_names.dtype.kind not in {"U", "S"}:
            raise ValueError(
                f"{label} joint_names must have string dtype, got {joint_names.dtype}"
            )
        joint_names = joint_names.astype(str).reshape(-1)
        if not np.array_equal(joint_names, expected_joint_names):
            raise ValueError(
                f"{label} joint_names must exactly match G1 MuJoCo joint order"
            )
    num_samples, num_frames, _ = generated_qpos.shape
    if int(expected_frames) > 0 and num_frames != int(expected_frames):
        raise ValueError(
            f"Movens-Intent protocol requires {expected_frames} frames, got {num_frames}"
        )

    fps_value = pick_array(generated, ("fps",))
    fps = _fps_array(fps_value, num_samples, name="fps")
    if float(expected_fps) > 0.0 and not np.allclose(
        fps, float(expected_fps), atol=1.0e-6
    ):
        raise ValueError(
            f"Movens-Intent protocol requires fps={expected_fps}, got {np.unique(fps)}"
        )
    reference_fps = pick_array(reference, ("fps",), required=False)
    if reference_fps is not None:
        reference_fps = _fps_array(reference_fps, num_samples, name="reference fps")
        if not np.allclose(reference_fps, fps, atol=1.0e-6):
            raise ValueError("generated and reference fps differ")

    generated_valid = pick_array(
        generated, ("valid", "generated_valid"), required=(task == "text")
    )
    reference_valid = pick_array(
        reference, ("valid", "reference_valid"), required=(task == "text")
    )
    if generated_valid is not None and np.asarray(generated_valid).dtype != np.bool_:
        raise ValueError(
            f"generated valid dtype must be bool, got {np.asarray(generated_valid).dtype}"
        )
    if reference_valid is not None and np.asarray(reference_valid).dtype != np.bool_:
        raise ValueError(
            f"reference valid dtype must be bool, got {np.asarray(reference_valid).dtype}"
        )
    generated_valid = (
        np.ones((num_samples, num_frames), dtype=np.bool_)
        if generated_valid is None
        else np.asarray(generated_valid)
    )
    reference_valid = (
        np.ones((num_samples, num_frames), dtype=np.bool_)
        if reference_valid is None
        else np.asarray(reference_valid)
    )
    if generated_valid.shape != (num_samples, num_frames) or reference_valid.shape != (
        num_samples,
        num_frames,
    ):
        raise ValueError("valid masks must have shape [N,T]")

    for label, valid in (("generated", generated_valid), ("reference", reference_valid)):
        for index in range(num_samples):
            length = int(valid[index].sum())
            if length <= 0 or not valid[index, :length].all() or valid[index, length:].any():
                raise ValueError(f"{label} valid must be one non-empty contiguous prefix")
    if not np.array_equal(generated_valid, reference_valid):
        raise ValueError(f"{task} generated/reference valid masks must match")

    valid_lengths = reference_valid.sum(axis=1).astype(np.int32)
    for label, payload in (("generated", generated), ("reference", reference)):
        stored_lengths = pick_array(payload, ("lengths",), required=(task == "text"))
        if stored_lengths is None:
            continue
        stored_lengths = np.asarray(stored_lengths)
        if stored_lengths.dtype.kind not in {"i", "u"} or stored_lengths.shape != (num_samples,):
            raise ValueError(f"{label} lengths must be integer [N]")
        if not np.array_equal(stored_lengths.astype(np.int32, copy=False), valid_lengths):
            raise ValueError(f"{label} lengths differs from valid prefix lengths")

    generated_qpos = _validate_qpos(
        generated_qpos, name="generated qpos", valid=generated_valid
    )
    reference_qpos = _validate_qpos(
        reference_qpos, name="reference qpos", valid=reference_valid
    )

    generated_qpos, generated_partial_count = _repeat_last_valid_reference_frame(
        generated_qpos, generated_valid
    )
    reference_qpos, reference_partial_count = _repeat_last_valid_reference_frame(
        reference_qpos, reference_valid
    )

    sample_id = _validate_identity(generated, reference, condition, num_samples)
    dataset_value = np.asarray(pick_array(condition, ("dataset", "cohort")))
    if dataset_value.dtype.kind not in {"U", "S"}:
        raise ValueError(
            f"condition dataset/cohort must have string dtype, got {dataset_value.dtype}"
        )
    dataset = dataset_value.astype(str).reshape(-1)
    if dataset.shape != (num_samples,) or np.any(np.char.str_len(dataset) == 0):
        raise ValueError("dataset/cohort must contain N non-empty strings")
    dataset_index_value = np.asarray(pick_array(condition, ("dataset_index",)))
    if dataset_index_value.dtype.kind not in {"i", "u"}:
        raise ValueError(
            f"condition dataset_index must have integer dtype, got {dataset_index_value.dtype}"
        )
    dataset_index = dataset_index_value.astype(np.int32, copy=False).reshape(-1)
    if dataset_index.shape != (num_samples,):
        raise ValueError("dataset_index must have shape [N]")
    for label, payload in (("generated", generated), ("reference", reference)):
        other_dataset = pick_array(payload, ("dataset", "cohort"), required=False)
        if other_dataset is not None:
            other_dataset = np.asarray(other_dataset)
            if other_dataset.dtype.kind not in {"U", "S"} or not np.array_equal(
                other_dataset.astype(str).reshape(-1), dataset
            ):
                raise ValueError(f"{label} dataset/cohort differs from condition order")
        other_index = pick_array(payload, ("dataset_index",), required=False)
        if other_index is not None:
            other_index = np.asarray(other_index)
            if other_index.dtype.kind not in {"i", "u"} or not np.array_equal(
                other_index.astype(np.int32, copy=False).reshape(-1), dataset_index
            ):
                raise ValueError(f"{label} dataset_index differs from condition order")

    captions_value = pick_array(
        condition, ("captions", "caption"), required=(task == "text")
    )
    if captions_value is not None and np.asarray(captions_value).dtype.kind not in {
        "U",
        "S",
    }:
        raise ValueError(
            f"captions must have string dtype, got {np.asarray(captions_value).dtype}"
        )
    captions = (
        np.asarray([f"sample {idx}" for idx in range(num_samples)], dtype=np.str_)
        if captions_value is None
        else np.asarray(captions_value).astype(str).reshape(-1)
    )
    if captions.shape != (num_samples,) or np.any(np.char.str_len(captions) == 0):
        raise ValueError("captions must contain N non-empty strings")

    if task == "text":
        generated_captions = np.asarray(pick_array(generated, ("captions", "caption")))
        if generated_captions.dtype.kind not in {"U", "S"} or not np.array_equal(
            generated_captions.astype(str).reshape(-1), captions
        ):
            raise ValueError("generated captions differ from current text condition")

    result: dict[str, Any] = {
        "generated_qpos": generated_qpos,
        "reference_qpos": reference_qpos,
        "generated_valid": generated_valid,
        "reference_valid": reference_valid,
        "fps": fps,
        "sample_id": sample_id,
        "dataset": dataset,
        "dataset_index": dataset_index,
        "captions": captions,
        "num_samples": num_samples,
        "num_frames": num_frames,
        "joint_names": expected_joint_names,
        "reference_padding_normalized_count": reference_partial_count,
        "generated_padding_normalized_count": generated_partial_count,
        "lengths": valid_lengths,
    }

    if task == "text":
        if num_samples % 32 != 0:
            raise ValueError(
                f"Movens-Intent R-Precision uses complete batch-32 groups; got N={num_samples}"
            )
    elif task == "audio":
        audio_features = pick_array(condition, ("audio_features",), required=False)
        audio_beats = pick_array(condition, ("audio_beats",), required=False)
        if audio_features is None and audio_beats is None:
            raise KeyError(
                "audio condition NPZ requires audio_features [N,T,D] or audio_beats [N,T]"
            )
        if audio_features is not None and audio_beats is not None:
            raise ValueError("Provide only one of audio_features or audio_beats")
        if audio_features is None:
            audio_beats = np.asarray(audio_beats)
            if audio_beats.dtype.kind not in {"b", "f", "i", "u"}:
                raise ValueError(
                    f"audio_beats must have bool/numeric dtype, got {audio_beats.dtype}"
                )
            audio_beats = audio_beats.astype(np.float32, copy=False)
            if audio_beats.shape != (num_samples, num_frames):
                raise ValueError(
                    f"audio_beats must have shape {(num_samples, num_frames)}"
                )
            audio_features = audio_beats[..., None]
            beat_source = "explicit_audio_beats"
        else:
            audio_features = np.asarray(audio_features)
            if audio_features.dtype != np.float32:
                raise ValueError(
                    f"audio_features dtype must be float32, got {audio_features.dtype}"
                )
            contract = np.asarray(pick_array(condition, ("audio_feature_contract",)))
            if contract.dtype.kind not in {"U", "S"} or contract.size != 1:
                raise ValueError("audio_feature_contract must be one string")
            contract = str(contract.reshape(-1)[0])
            if contract != "omg_legacy_last_channel_threshold":
                raise ValueError(
                    "audio_features requires audio_feature_contract="
                    "'omg_legacy_last_channel_threshold'; use explicit audio_beats for new data"
                )
            beat_source = "omg_legacy_last_channel_threshold"
        if audio_features.ndim != 3 or audio_features.shape[:2] != (
            num_samples,
            num_frames,
        ):
            raise ValueError(
                f"audio_features must have shape [N,{num_frames},D], got {audio_features.shape}"
            )
        if (
            beat_source == "omg_legacy_last_channel_threshold"
            and audio_features.shape[-1] != 35
        ):
            raise ValueError(
                f"OMG precomputed audio_features must have 35 channels, got {audio_features.shape[-1]}; "
                "use explicit audio_beats [N,T] when only beat annotations are available"
            )
        if not np.isfinite(audio_features).all():
            raise ValueError("audio_features contains NaN or Inf")
        audio_fps_value = pick_array(condition, ("audio_fps",), required=False)
        audio_fps = (
            fps.copy()
            if audio_fps_value is None
            else _fps_array(audio_fps_value, num_samples, name="audio_fps")
        )
        if not np.allclose(audio_fps, fps, atol=1.0e-6):
            raise ValueError(
                "Movens-Intent artifact protocol requires frame-aligned audio_fps == motion fps"
            )
        audio_type = np.asarray(pick_array(condition, ("audio_type",)))
        if audio_type.dtype.kind not in {"U", "S"}:
            raise ValueError(
                f"audio_type must have string dtype, got {audio_type.dtype}"
            )
        audio_type = np.char.lower(audio_type.astype(str).reshape(-1))
        if audio_type.shape != (num_samples,):
            raise ValueError("audio_type must have shape [N]")
        supported = np.isin(audio_type, ["music", "dance", "music_dance"])
        if not supported.all():
            bad = sorted(set(audio_type[~supported].tolist()))
            raise ValueError(
                f"BeatAlign/FID-k/FID-g/PFC is a music-dance protocol; unsupported audio_type={bad}"
            )
        result.update(
            {
                "audio_features": audio_features,
                "audio_fps": audio_fps,
                "beat_source": beat_source,
            }
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
    return result

def physical_metric_values(
    qpos: torch.Tensor,
    valid: torch.Tensor,
    fps: torch.Tensor,
    *,
    kinematics,
    device: torch.device,
    contact_height_threshold: float,
    contact_penetration_tolerance: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    representation = SimpleNamespace(kinematics=kinematics)
    values = {
        "contact_sliding_speed": [],
        "foot_ground_error": [],
        "body_jerk_mean": [],
    }
    with torch.inference_mode():
        for index in tqdm(range(qpos.shape[0]), desc="Physical metrics", unit="sample"):
            result = physical_qpos_metrics(
                qpos_36=qpos[index : index + 1].to(device),
                representation=representation,
                fps=fps[index : index + 1].to(device),
                valid=valid[index : index + 1].to(device),
                contact_height_threshold=float(contact_height_threshold),
                contact_penetration_tolerance=float(contact_penetration_tolerance),
            )
            for key in values:
                values[key].append(float(result[key].detach().cpu()))
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in values.items()}
    summary = {key: _summary_stats(value) for key, value in arrays.items()}
    summary["num_samples"] = int(qpos.shape[0])
    summary["contact_height_threshold_m"] = float(contact_height_threshold)
    summary["contact_penetration_tolerance_m"] = float(contact_penetration_tolerance)
    return summary, arrays

def _tracker_failure_flags(
    executed: np.ndarray,
    valid: np.ndarray,
    *,
    robot_xml: str,
) -> tuple[np.ndarray, np.ndarray]:
    quat = np.asarray(executed[:, :, 3:7], dtype=np.float64)
    quat /= np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1.0e-12, None)
    qx = quat[..., 1]
    qy = quat[..., 2]
    tilt = np.arccos(np.clip(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0))
    fall_frame = (
        (executed[:, :, 2] < 0.20)
        | (tilt > np.deg2rad(85.0))
        | ((executed[:, :, 2] < 0.35) & (tilt > np.deg2rad(60.0)))
    ) & valid
    run = np.zeros(executed.shape[0], dtype=np.int32)
    max_run = np.zeros(executed.shape[0], dtype=np.int32)
    for frame_index in range(executed.shape[1]):
        run = np.where(fall_frame[:, frame_index], run + 1, 0)
        max_run = np.maximum(max_run, run)
    fallen = max_run > 10

    model = mujoco.MjModel.from_xml_path(str(Path(robot_xml).expanduser().resolve()))
    if model.nq != executed.shape[-1]:
        raise ValueError(
            f"MuJoCo nq={model.nq} differs from tracker qpos dim={executed.shape[-1]}"
        )
    lower = np.full(model.nq, -np.inf, dtype=np.float64)
    upper = np.full(model.nq, np.inf, dtype=np.float64)
    for joint_id in range(model.njnt):
        address = int(model.jnt_qposadr[joint_id])
        if address >= 7 and bool(model.jnt_limited[joint_id]):
            lower[address], upper[address] = model.jnt_range[joint_id]
    violation = (
        (
            (executed < lower[None, None, :] - 1.0e-4)
            | (executed > upper[None, None, :] + 1.0e-4)
        )
        & valid[:, :, None]
    ).any(axis=(1, 2))
    return fallen, violation

def tracker_metrics_from_artifact(
    artifact_path: str | Path,
    *,
    expected_samples: int,
    expected_generated_qpos: np.ndarray,
    expected_generated_valid: np.ndarray | None = None,
    expected_generation_fps: np.ndarray,
    expected_tracker_fps: float,
    expected_holomotion_onnx: str,
    expected_dataset: np.ndarray,
    expected_dataset_index: np.ndarray,
    kinematics,
    device: torch.device,
    robot_xml: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    artifact_path = Path(artifact_path).expanduser().resolve()
    payload = read_npz(artifact_path)
    executed = np.asarray(pick_array(payload, ("executed_qpos_36",)), dtype=np.float32)
    tracker_reference = np.asarray(
        pick_array(payload, ("tracker_reference_qpos_36",)), dtype=np.float32
    )
    stored_generated = np.asarray(
        pick_array(payload, ("generated_qpos_36",)), dtype=np.float32
    )

    stored_generated_valid_value = pick_array(
        payload, ("generated_valid",), required=False
    )
    stored_generated_lengths_value = pick_array(
        payload, ("generated_lengths",), required=False
    )
    valid = np.asarray(pick_array(payload, ("tracker_valid",)), dtype=np.bool_)
    sample_index = np.asarray(
        pick_array(payload, ("sample_index",)), dtype=np.int64
    ).reshape(-1)
    generation_fps = np.asarray(
        pick_array(payload, ("generation_fps",)), dtype=np.float32
    ).reshape(-1)
    tracker_fps = np.asarray(
        pick_array(payload, ("tracker_fps",)), dtype=np.float32
    ).reshape(-1)
    holomotion_onnx = (
        np.asarray(pick_array(payload, ("holomotion_onnx",))).astype(str).reshape(-1)
    )
    artifact_dataset = (
        np.asarray(pick_array(payload, ("dataset",))).astype(str).reshape(-1)
    )
    artifact_dataset_index = np.asarray(
        pick_array(payload, ("dataset_index",)), dtype=np.int32
    ).reshape(-1)
    if (
        executed.shape != tracker_reference.shape
        or executed.ndim != 3
        or executed.shape[-1] != 36
    ):
        raise ValueError("tracker executed/reference qpos must share shape [N,T,36]")
    if valid.shape != executed.shape[:2]:
        raise ValueError("tracker_valid must have shape [N,T]")
    if executed.shape[0] != int(expected_samples):
        raise ValueError(
            f"tracker artifact has {executed.shape[0]} samples, expected {expected_samples}"
        )
    if not np.array_equal(sample_index, np.arange(expected_samples, dtype=np.int64)):
        raise ValueError("paper metrics require the complete ordered tracker cohort")
    if not np.array_equal(
        artifact_dataset, np.asarray(expected_dataset).astype(str).reshape(-1)
    ) or not np.array_equal(
        artifact_dataset_index,
        np.asarray(expected_dataset_index, dtype=np.int32).reshape(-1),
    ):
        raise ValueError(
            "tracker artifact dataset identity/order differs from current cohort"
        )
    expected_generation_fps = np.asarray(
        expected_generation_fps, dtype=np.float32
    ).reshape(-1)
    if generation_fps.shape != (expected_samples,) or not np.allclose(
        generation_fps,
        expected_generation_fps,
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError(
            "tracker artifact generation_fps differs from current generated qpos"
        )
    if tracker_fps.shape != (1,) or not np.isclose(
        tracker_fps[0], float(expected_tracker_fps), atol=1.0e-6, rtol=0.0
    ):
        raise ValueError("tracker artifact tracker_fps differs from the paper protocol")
    expected_onnx_path = Path(expected_holomotion_onnx).expanduser().resolve()
    if (
        holomotion_onnx.shape != (1,)
        or Path(holomotion_onnx[0]).expanduser().resolve() != expected_onnx_path
    ):
        raise ValueError(
            "tracker artifact HoloMotion model path differs from the current configuration"
        )
    expected_generated_qpos = np.asarray(expected_generated_qpos, dtype=np.float32)
    if stored_generated.shape != expected_generated_qpos.shape or not np.allclose(
        stored_generated,
        expected_generated_qpos,
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError(
            "tracker artifact was not produced from the current ordered generated qpos cohort"
        )
    if expected_generated_valid is None:
        expected_generated_valid = np.ones(
            expected_generated_qpos.shape[:2], dtype=np.bool_
        )
    else:
        expected_generated_valid = np.asarray(expected_generated_valid)
    if (
        expected_generated_valid.dtype != np.bool_
        or expected_generated_valid.shape != expected_generated_qpos.shape[:2]
    ):
        raise ValueError("expected_generated_valid must be bool [N,T]")
    expected_generated_lengths = expected_generated_valid.sum(axis=1).astype(np.int32)
    for index, generation_length in enumerate(expected_generated_lengths):
        generation_length = int(generation_length)
        if (
            generation_length <= 0
            or not expected_generated_valid[index, :generation_length].all()
            or expected_generated_valid[index, generation_length:].any()
        ):
            raise ValueError("expected_generated_valid must be one non-empty contiguous prefix")
    if stored_generated_valid_value is None:
        if not expected_generated_valid.all():
            raise ValueError("variable-length tracker artifact is missing generated_valid")
    else:
        stored_generated_valid = np.asarray(stored_generated_valid_value)
        if stored_generated_valid.dtype != np.bool_ or not np.array_equal(
            stored_generated_valid, expected_generated_valid
        ):
            raise ValueError("tracker artifact generated_valid differs from current cohort")
    if stored_generated_lengths_value is None:
        if not expected_generated_valid.all():
            raise ValueError("variable-length tracker artifact is missing generated_lengths")
    else:
        stored_generated_lengths = np.asarray(stored_generated_lengths_value)
        if (
            stored_generated_lengths.dtype.kind not in {"i", "u"}
            or stored_generated_lengths.shape != (expected_samples,)
            or not np.array_equal(
                stored_generated_lengths.astype(np.int32, copy=False),
                expected_generated_lengths,
            )
        ):
            raise ValueError("tracker artifact generated_lengths differs from current cohort")
    if (
        not np.isfinite(executed[valid]).all()
        or not np.isfinite(tracker_reference[valid]).all()
    ):
        raise ValueError("tracker artifact contains NaN or Inf in valid frames")
    for index in range(valid.shape[0]):
        length = int(valid[index].sum())
        generation_length = int(expected_generated_lengths[index])
        expected_frames = (
            int(
                np.floor(
                    (
                        (generation_length - 1)
                        / float(expected_generation_fps[index])
                    )
                    * float(expected_tracker_fps)
                )
            )
            + 1
        )
        if (
            length != expected_frames
            or not valid[index, :length].all()
            or valid[index, length:].any()
        ):
            raise ValueError(
                f"tracker sample {index} must contain the full {expected_frames}-frame "
                f"rollout derived from {generation_length} valid generation frames, got {length}"
            )

    values = {"g_mpjpe": [], "mpjpe": [], "e_vel": [], "e_acc": []}
    with torch.inference_mode():
        for index in tqdm(
            range(expected_samples), desc="Tracker metrics", unit="sample"
        ):
            length = int(valid[index].sum())
            executed_tensor = torch.from_numpy(executed[index : index + 1, :length]).to(
                device
            )
            reference_tensor = torch.from_numpy(
                tracker_reference[index : index + 1, :length]
            ).to(device)
            executed_body = (
                kinematics.forward_kinematics(executed_tensor)["body_pos_w"][0]
                .cpu()
                .numpy()
            )
            reference_body = (
                kinematics.forward_kinematics(reference_tensor)["body_pos_w"][0]
                .cpu()
                .numpy()
            )
            values["g_mpjpe"].append(g_mpjpe(executed_body, reference_body))
            values["mpjpe"].append(mpjpe(executed_body, reference_body, root_index=0))
            values["e_vel"].append(e_vel(executed_body, reference_body))
            values["e_acc"].append(e_acc(executed_body, reference_body))
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in values.items()}
    fallen, joint_limit = _tracker_failure_flags(executed, valid, robot_xml=robot_xml)
    summary = {
        "enabled": True,
        "num_samples": int(expected_samples),
        "artifact_path": str(artifact_path),
        "metrics": {key: _summary_stats(value) for key, value in arrays.items()},
        "failures": {
            "fall_rate_percent": 100.0 * float(fallen.mean()),
            "joint_limit_rate_percent": 100.0 * float(joint_limit.mean()),
            "fall_count": int(fallen.sum()),
            "joint_limit_count": int(joint_limit.sum()),
            "fall_definition": "root-height/tilt condition persists for at least 11 consecutive tracker frames",
            "joint_limit_epsilon_rad": 1.0e-4,
        },
    }
    arrays.update({"fall": fallen, "joint_limit": joint_limit})
    return summary, arrays

def build_paper_payload(task: str, raw_metrics: dict[str, Any]) -> dict[str, Any]:
    generated = raw_metrics["generated_target"]
    tracker = raw_metrics["tracker_executed"]
    tracker_is_eligible = tracker.get("enabled", False) and tracker.get(
        "paper_eligible", True
    )
    tracking = tracker.get("metrics", {}) if tracker_is_eligible else {}
    failures = tracker.get("failures", {}) if tracker_is_eligible else {}

    if task == "text":
        condition = generated["text"]
        columns = [
            "FID",
            "Matching",
            "R@1",
            "R@2",
            "R@3",
            "Diversity",
            "C-Slide",
            "Jerk",
            "MPJPE",
            "g-MPJPE",
            "E_vel",
            "E_acc",
            "Fall",
            "J-Limit",
        ]
        values = [
            100.0 * condition["motion_fid"],
            condition["matching_score"],
            *[100.0 * value for value in condition["r_precision"][:3]],
            condition["diversity_generated"],
            10.0 * generated["physical"]["contact_sliding_speed"]["mean"],
            generated["physical"]["body_jerk_mean"]["mean"],
        ]
        scaling = [
            "raw x100",
            "raw",
            "fraction x100",
            "fraction x100",
            "fraction x100",
            "raw",
            "m/s x10",
            "m/s^3",
        ]
    elif task == "audio":
        condition = generated["audio"]
        columns = [
            "BeatAlign",
            "FID-k",
            "FID-g",
            "PFC",
            "C-Slide",
            "Jerk",
            "MPJPE",
            "g-MPJPE",
            "E_vel",
            "E_acc",
            "Fall",
            "J-Limit",
        ]
        values = [
            condition["beat_align_generated"]["mean"],
            condition["fid_k"],
            condition["fid_g"],
            condition["pfc_generated"]["mean"],
            generated["physical"]["contact_sliding_speed"]["mean"],
            generated["physical"]["body_jerk_mean"]["mean"],
        ]
        scaling = ["raw", "raw", "raw", "raw", "m/s raw", "m/s^3"]
    else:
        raise ValueError(task)

    tracker_values = [
        tracking.get("mpjpe", {}).get("mean"),
        tracking.get("g_mpjpe", {}).get("mean"),
        tracking.get("e_vel", {}).get("mean"),
        tracking.get("e_acc", {}).get("mean"),
        failures.get("fall_rate_percent"),
        failures.get("joint_limit_rate_percent"),
    ]
    values.extend(tracker_values)
    scaling.extend(
        ["mm", "mm", "mm/frame", "mm/frame^2", "sequence percent", "sequence percent"]
    )
    complete = all(value is not None and np.isfinite(float(value)) for value in values)
    display = ["--" if value is None else f"{float(value):.2f}" for value in values]
    return {
        "task": task,
        "scope": raw_metrics.get("protocol", {}).get(
            "tracker_policy_protocol", "unspecified"
        ),
        "complete": complete,
        "columns": columns,
        "values": [None if value is None else float(value) for value in values],
        "display_values": display,
        "display_scaling": dict(zip(columns, scaling, strict=True)),
        "latex_cells": " & ".join(display),
    }

def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value

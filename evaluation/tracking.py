import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import write_json

def run_variable_length_tracker_benchmark(
    *,
    output_dir: Path,
    generated_qpos: torch.Tensor,
    reference_qpos: torch.Tensor,
    generated_valid: torch.Tensor,
    fps: torch.Tensor,
    captions: list[str],
    dataset_names: list[str],
    dataset_indices: list[int],
    tracker_args: argparse.Namespace,
) -> dict:
    from omg.tracking.holomotion.runner import HoloMotionRolloutRunner

    generated_np = generated_qpos.detach().cpu().numpy().astype(np.float32, copy=False)
    reference_np = reference_qpos.detach().cpu().numpy().astype(np.float32, copy=False)
    valid_np = generated_valid.detach().cpu().numpy()
    fps_np = fps.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    num_samples = int(generated_np.shape[0])
    if (
        generated_np.shape != reference_np.shape
        or valid_np.dtype != np.bool_
        or valid_np.shape != generated_np.shape[:2]
        or fps_np.shape != (num_samples,)
    ):
        raise ValueError("variable-length tracker inputs have inconsistent shapes")

    lengths = valid_np.sum(axis=1).astype(np.int32)
    stage_dir = output_dir / str(tracker_args.tracker_output_name)
    stage_dir.mkdir(parents=True, exist_ok=True)
    runner = HoloMotionRolloutRunner(
        holomotion_onnx=tracker_args.holomotion_onnx,
        target_fps=float(tracker_args.tracker_target_fps),
        robot_xml=tracker_args.tracker_robot_xml,
        providers=tracker_args.tracker_providers,
        control_substeps=int(tracker_args.tracker_control_substeps),
        action_clip=float(tracker_args.tracker_action_clip),
        video=False,
    )
    executed_sequences = []
    tracker_reference_sequences = []
    action_sequences = []
    sample_rows = []
    try:
        for index in tqdm(range(num_samples), desc="Tracker execute", unit="sample"):
            length = int(lengths[index])
            if (
                length <= 0
                or not valid_np[index, :length].all()
                or valid_np[index, length:].any()
            ):
                raise ValueError(f"tracker sample {index} valid is not one contiguous prefix")
            runner.clear_rollout_buffers(reset_initialized=True)
            chunk = runner.run_reference_chunk(
                generated_np[index, :length],
                reference_fps=float(fps_np[index]),
                steps=tracker_args.tracker_steps,
                plan_id=0,
                plan_start=0,
            )
            executed = np.asarray(chunk.qpos_36, dtype=np.float32)
            tracker_reference = np.asarray(runner.reference_qpos_36, dtype=np.float32)
            actions = np.asarray(runner.actions, dtype=np.float32)
            expected_tracker_frames = int(
                np.floor(
                    ((length - 1) / float(fps_np[index]))
                    * float(tracker_args.tracker_target_fps)
                )
            ) + 1
            if executed.shape != (expected_tracker_frames, 36):
                raise ValueError(
                    f"tracker sample {index}: expected {expected_tracker_frames} frames "
                    f"from {length} valid generation frames, got {executed.shape}"
                )
            if tracker_reference.shape != executed.shape or actions.shape[0] != expected_tracker_frames:
                raise ValueError(f"tracker sample {index}: output/reference/action length differs")
            executed_sequences.append(executed)
            tracker_reference_sequences.append(tracker_reference)
            action_sequences.append(actions)
            sample_rows.append(
                {
                    "sample_index": index,
                    "dataset": dataset_names[index],
                    "dataset_index": int(dataset_indices[index]),
                    "caption": captions[index],
                    "generation_frames": length,
                    "tracker_frames": expected_tracker_frames,
                    "generation_fps": float(fps_np[index]),
                    "tracker_fps": float(tracker_args.tracker_target_fps),
                }
            )
    finally:
        runner.close()

    def pad_sequences(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        max_length = max(len(sequence) for sequence in sequences)
        padded = np.empty(
            (len(sequences), max_length, *sequences[0].shape[1:]), dtype=np.float32
        )
        valid = np.zeros((len(sequences), max_length), dtype=np.bool_)
        for index, sequence in enumerate(sequences):
            length = len(sequence)
            padded[index, :length] = sequence
            padded[index, length:] = sequence[-1]
            valid[index, :length] = True
        return padded, valid

    executed_padded, tracker_valid = pad_sequences(executed_sequences)
    tracker_reference_padded, tracker_reference_valid = pad_sequences(
        tracker_reference_sequences
    )
    actions_padded, action_valid = pad_sequences(action_sequences)
    if not np.array_equal(tracker_valid, tracker_reference_valid) or not np.array_equal(
        tracker_valid, action_valid
    ):
        raise ValueError("tracker executed/reference/action valid masks differ")

    artifact_path = stage_dir / "tracker_executed_qpos.npz"
    np.savez_compressed(
        artifact_path,
        executed_qpos_36=executed_padded,
        tracker_reference_qpos_36=tracker_reference_padded,
        generated_qpos_36=generated_np,
        reference_qpos_36=reference_np,
        generated_valid=valid_np,
        generated_lengths=lengths,
        actions=actions_padded,
        tracker_valid=tracker_valid,
        tracker_fps=np.asarray([float(tracker_args.tracker_target_fps)], dtype=np.float32),
        generation_fps=fps_np,
        sample_index=np.arange(num_samples, dtype=np.int32),
        dataset=np.asarray(dataset_names, dtype=np.str_),
        dataset_index=np.asarray(dataset_indices, dtype=np.int32),
        caption=np.asarray(captions, dtype=np.str_),
        holomotion_onnx=np.asarray(
            [str(Path(tracker_args.holomotion_onnx).expanduser().resolve())], dtype=np.str_
        ),
    )
    sample_metrics_path = stage_dir / "tracker_sample_metrics.jsonl"
    sample_metrics_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sample_rows),
        encoding="utf-8",
    )
    benchmark = {
        "enabled": True,
        "num_samples": num_samples,
        "num_generated_samples": num_samples,
        "selected_indices": list(range(num_samples)),
        "holomotion_onnx": str(Path(tracker_args.holomotion_onnx).expanduser().resolve()),
        "robot_xml": str(Path(tracker_args.tracker_robot_xml).expanduser().resolve()),
        "providers": tracker_args.tracker_providers,
        "target_fps": float(tracker_args.tracker_target_fps),
        "steps": None,
        "control_substeps": int(tracker_args.tracker_control_substeps),
        "action_clip": float(tracker_args.tracker_action_clip),
        "root_index": 0,
        "artifact_path": str(artifact_path.resolve()),
        "sample_metrics_path": str(sample_metrics_path.resolve()),
        "generation_length_min": int(lengths.min()),
        "generation_length_max": int(lengths.max()),
        "tracker_length_min": int(tracker_valid.sum(axis=1).min()),
        "tracker_length_max": int(tracker_valid.sum(axis=1).max()),
        "valid_shape": list(tracker_valid.shape),
    }
    write_json(stage_dir / "benchmark_tracker.json", benchmark)
    return benchmark

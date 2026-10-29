from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

here = Path(__file__).resolve().parent
repo = here.parent

parser = argparse.ArgumentParser(
    description="Movens-Intent text, speech and music evaluator"
)
parser.add_argument("--task", choices=["text", "audio"], required=True)
parser.add_argument("--generated-path", required=True)
parser.add_argument("--reference-path", required=True)
parser.add_argument("--condition-path", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--metrics-root", type=Path, default=repo / "third_party/metrics")
parser.add_argument("--expected-frames", type=int, required=True)
parser.add_argument("--expected-fps", type=float, required=True)
parser.add_argument("--device", choices=["auto", "cpu", "cuda"], required=True)
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--validate-only", type=int, choices=[0, 1], required=True)

parser.add_argument("--kinematics-path")
parser.add_argument("--contact-height-threshold", type=float, required=True)
parser.add_argument("--contact-penetration-tolerance", type=float, required=True)

parser.add_argument("--evaluator-checkpoint", required=True)

parser.add_argument("--retrieval-evaluator-checkpoint", default=None)
parser.add_argument("--raw-text-cache", default=None, help="Frozen T5 raw embeddings for the metric captions")
parser.add_argument("--motion-key", required=True)
parser.add_argument("--t5-model", required=True)
parser.add_argument("--text-cache-dir", required=True)
parser.add_argument("--text-batch-size", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)

parser.add_argument(
    "--beat-direction", choices=["music_to_motion", "motion_to_music"], required=True
)
parser.add_argument("--beat-threshold", type=float, required=True)
parser.add_argument("--sigma-frames", type=float, required=True)
parser.add_argument("--min-motion-beat-distance-seconds", type=float, required=True)

parser.add_argument("--tracker-mode", choices=["run", "reuse", "skip"], required=True)
parser.add_argument("--tracker-artifact", required=True)
parser.add_argument("--holomotion-onnx", default=str(repo / "checkpoints/evaluation/holomotion_v1.2_motion_tracking_model/exported/model.onnx"))
parser.add_argument("--tracker-robot-xml")
parser.add_argument("--tracker-target-fps", type=float, required=True)
parser.add_argument("--tracker-providers", required=True)
parser.add_argument("--tracker-control-substeps", type=int, required=True)
parser.add_argument("--tracker-action-clip", type=float, required=True)
parser.add_argument("--tracker-steps", type=int, required=True)
args = parser.parse_args()

metrics_root = args.metrics_root.expanduser().resolve()
if args.kinematics_path is None:
    args.kinematics_path = str(metrics_root / "assets/robots/g1/g1_kinematics.json")
if args.tracker_robot_xml is None:
    args.tracker_robot_xml = str(metrics_root / "assets/holomotion/g1_29dof/scene_29dof.xml")
metrics_src = metrics_root / "src"
if not metrics_src.is_dir():
    raise FileNotFoundError(f"Metric dependency source directory does not exist: {metrics_src}")
sys.path.insert(0, str(metrics_src))
for name in ("generated_path", "reference_path", "condition_path", "output_dir", "kinematics_path",
        "evaluator_checkpoint", "retrieval_evaluator_checkpoint", "raw_text_cache", "text_cache_dir",
        "tracker_artifact", "holomotion_onnx", "tracker_robot_xml"):
    value = getattr(args, name)
    if value is not None:
        setattr(args, name, str(Path(value).expanduser().resolve()))
if Path(args.t5_model).is_dir():
    args.t5_model = str(Path(args.t5_model).resolve())
os.chdir(metrics_root)

import numpy as np
import torch
from tqdm import tqdm

from metrics import (
    build_paper_payload,
    jsonable,
    load_benchmark_inputs,
    physical_metric_values,
    tracker_metrics_from_artifact,
)
from omg.benchmarks.metrics.audio import (
    aistpp_edge_features,
    aistpp_edge_metric_summary,
    beat_align,
)
from omg.benchmarks.runners.common import (
    _embedding_distribution_metrics,
    _encode_motion_embeddings,
    _load_evaluator_motion_encoder,
    _motion_for_evaluator,
    _summary_stats,
    qpos_to_body_positions,
)

from omg.robots.g1.kinematics import G1Kinematics

def write_json(path: Path, payload: dict) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)

def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(name)

def omg_git_state(root: Path) -> dict:
    commit_result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    status_result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    )
    changes = [line for line in status_result.stdout.splitlines() if line.strip()]
    return {
        "commit": commit_result.stdout.strip(),
        "tracked_worktree_changes": changes,
        "clean": not changes,
    }

def metric_source_hashes(root: Path) -> dict[str, str]:
    relative_paths = [
        "src/omg/benchmarks/metrics/audio.py",
        "src/omg/benchmarks/metrics/distribution.py",
        "src/omg/benchmarks/metrics/tracking.py",
        "src/omg/benchmarks/runners/common.py",
        "src/omg/benchmarks/runners/text.py",
        "src/omg/benchmarks/runners/tracker_executed.py",
        "src/omg/generation/metrics/physical.py",
        "src/omg/robots/g1/kinematics.py",
    ]
    hashes = {}
    for relative_path in relative_paths:
        hashes[relative_path] = file_sha256(root / relative_path)

    hashes["movens_zero/evaluation/tracking.py"] = file_sha256(Path(__file__).with_name("tracking.py"))
    return hashes

def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def expected_tracker_provenance() -> dict:
    return {
        "schema": "movens.tracker.provenance.v1",
        "holomotion_onnx": str(Path(args.holomotion_onnx).expanduser().resolve()),
        "holomotion_onnx_sha256": file_sha256(args.holomotion_onnx),
        "robot_xml": str(Path(args.tracker_robot_xml).expanduser().resolve()),
        "robot_xml_sha256": file_sha256(args.tracker_robot_xml),
        "target_fps": float(args.tracker_target_fps),
        "providers": args.tracker_providers,
        "control_substeps": int(args.tracker_control_substeps),
        "action_clip": float(args.tracker_action_clip),
        "steps": None,
        "root_index": 0,
    }

def canonical_omg_tracker_status(actual: dict) -> dict:

    canonical_onnx = (
        metrics_root
        / "models/HoloMotion/HoloMotion_v1.2/holomotion_v1.2_motion_tracking_model/exported/model.onnx"
    )
    canonical_xml = metrics_root / "assets/holomotion/g1_29dof/scene_29dof.xml"
    canonical_onnx_hash = "96e917b7de78605b1e2ee3666ed59f3d4ed5111df3325b56e59fdfdf0ef99225"
    canonical_xml_hash = "4e6ae5f0f1d6bee36242fa02c36da1999748f94fa4d2a013f72ca1379b56bd69"
    canonical = {
        "holomotion_onnx": str(canonical_onnx.resolve()),
        "holomotion_onnx_sha256": canonical_onnx_hash,
        "robot_xml": str(canonical_xml.resolve()),
        "robot_xml_sha256": canonical_xml_hash,
        "target_fps": 50.0,
        "providers": "TensorrtExecutionProvider,CUDAExecutionProvider",
        "control_substeps": 10,
        "action_clip": 10.0,
        "steps": None,
        "root_index": 0,
    }
    compared_keys = (
        "holomotion_onnx_sha256",
        "robot_xml_sha256",
        "target_fps",
        "providers",
        "control_substeps",
        "action_clip",
        "steps",
        "root_index",
    )
    reasons = [
        f"{key}: actual={actual.get(key)!r}, canonical={canonical[key]!r}"
        for key in compared_keys
        if actual.get(key) != canonical[key]
    ]
    if args.tracker_steps != 0:
        reasons.append(
            f"tracker_steps CLI must be 0 for a full rollout, got {args.tracker_steps}"
        )
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "canonical": canonical,
    }

def validate_tracker_bundle(artifact_path: str | Path, expected: dict) -> bool:
    stage_dir = Path(artifact_path).expanduser().resolve().parent
    provenance_path = stage_dir / "movens_tracker_provenance.json"
    benchmark_path = stage_dir / "benchmark_tracker.json"
    if not provenance_path.is_file() or not benchmark_path.is_file():
        raise FileNotFoundError(
            "tracker reuse requires benchmark_tracker.json and "
            "movens_tracker_provenance.json next to the artifact"
        )
    actual = json.loads(provenance_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError(
            "tracker provenance differs from the current HoloMotion/XML/runtime configuration"
        )
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    checks = {
        "num_samples": int(data["num_samples"]),
        "num_generated_samples": int(data["num_samples"]),
        "holomotion_onnx": expected["holomotion_onnx"],
        "robot_xml": expected["robot_xml"],
        "providers": expected["providers"],
        "target_fps": expected["target_fps"],
        "steps": None,
        "control_substeps": expected["control_substeps"],
        "action_clip": expected["action_clip"],
        "root_index": 0,
    }
    for key, value in checks.items():
        if benchmark.get(key) != value:
            raise ValueError(
                f"tracker benchmark metadata mismatch for {key}: "
                f"{benchmark.get(key)!r} != {value!r}"
            )
    if benchmark.get("selected_indices") != list(range(data["num_samples"])):
        raise ValueError("tracker benchmark must contain the complete ordered cohort")
    return bool(benchmark.get("synthetic_test_only", False))

from tracking import run_variable_length_tracker_benchmark

output_dir = Path(args.output_dir).expanduser().resolve()
if output_dir.exists() and any(output_dir.iterdir()):
    raise FileExistsError(
        f"output directory must be new or empty so stale metrics cannot be mistaken for this run: {output_dir}"
    )
output_dir.mkdir(parents=True, exist_ok=True)
write_json(
    output_dir / "run_state.json",
    {"status": "running", "task": args.task, "output_dir": str(output_dir)},
)
data = load_benchmark_inputs(
    task=args.task,
    generated_path=args.generated_path,
    reference_path=args.reference_path,
    condition_path=args.condition_path,
    expected_frames=args.expected_frames,
    expected_fps=args.expected_fps,
)
protocol = {
    "name": "Movens-Intent G1 motion benchmark",
    "task": args.task,
    "metrics_root": str(metrics_root),
    "omg_git": omg_git_state(metrics_root),
    "omg_metric_source_sha256": metric_source_hashes(metrics_root),
    "num_samples": int(data["num_samples"]),
    "num_frames": int(data["num_frames"]),
    "fps": sorted(np.unique(data["fps"]).astype(float).tolist()),
    "qpos_layout": "root_xyz[3] + root_quaternion_wxyz[4] + G1_joint_angles[29]",
    "joint_names": data["joint_names"].tolist(),
    "coordinate_frame": "world Z-up, ground plane z=0",
    "tracker_policy_protocol": "HoloMotion single-policy executed tracking",
    "inputs": {
        "generated": str(Path(args.generated_path).expanduser().resolve()),
        "reference": str(Path(args.reference_path).expanduser().resolve()),
        "condition": str(Path(args.condition_path).expanduser().resolve()),
    },
    "generated_all_frames_valid": bool(data["generated_valid"].all()),
    "generated_valid_length_min": int(data["generated_valid"].sum(axis=1).min()),
    "generated_valid_length_max": int(data["generated_valid"].sum(axis=1).max()),
    "reference_valid_length_min": int(data["reference_valid"].sum(axis=1).min()),
    "reference_valid_length_max": int(data["reference_valid"].sum(axis=1).max()),
    "reference_padding_normalized_count": int(
        data["reference_padding_normalized_count"]
    ),
    "generated_padding_normalized_count": int(
        data["generated_padding_normalized_count"]
    ),
    "dataset_cohorts": sorted(set(data["dataset"].tolist())),
}
if args.task == "text":
    protocol["text_retrieval"] = "stratified fixed batch-32, seed-controlled"
else:
    protocol["audio_scope"] = "music/dance only"
    protocol["audio_beat_source"] = data["beat_source"]

if bool(args.validate_only):
    validation = {"valid": True, "protocol": protocol}
    write_json(output_dir / "validation.json", validation)
    write_json(
        output_dir / "run_state.json",
        {"status": "complete", "kind": "validation_only", "task": args.task},
    )
    print(json.dumps(validation, indent=2))
    raise SystemExit(0)

device = resolve_device(args.device)
if args.batch_size <= 0 or args.text_batch_size <= 0:
    raise ValueError("batch sizes must be positive")
print(
    f"[INFO] task={args.task} samples={data['num_samples']} frames={data['num_frames']} device={device}"
)

generated_qpos = torch.from_numpy(data["generated_qpos"])
reference_qpos = torch.from_numpy(data["reference_qpos"])
generated_valid = torch.from_numpy(data["generated_valid"])
reference_valid = torch.from_numpy(data["reference_valid"])
fps = torch.from_numpy(data["fps"])

kinematics = G1Kinematics(args.kinematics_path).to(device).eval()
model = SimpleNamespace(representation=SimpleNamespace(kinematics=kinematics))

physical_summary, sample_arrays = physical_metric_values(
    generated_qpos,
    generated_valid,
    fps,
    kinematics=kinematics,
    device=device,
    contact_height_threshold=args.contact_height_threshold,
    contact_penetration_tolerance=args.contact_penetration_tolerance,
)
physical_summary["units"] = {
    "contact_sliding_speed": "m/s",
    "foot_ground_error": "m",
    "body_jerk_mean": "m/s^3",
}

generated_target: dict = {"physical": physical_summary}

if args.task == "text":

    from omg.benchmarks.runners.text import _encode_text_embeddings, _text_retrieval_summary

    motion_encoder, evaluator_checkpoint = _load_evaluator_motion_encoder(
        args.evaluator_checkpoint,
        args.motion_key,
        device,
    )
    with torch.inference_mode():
        reference_motion = _motion_for_evaluator(
            reference_qpos.to(device),
            args.motion_key,
            kinematics=kinematics,
        )
        generated_motion = _motion_for_evaluator(
            generated_qpos.to(device),
            args.motion_key,
            kinematics=kinematics,
        )
    reference_embeddings = _encode_motion_embeddings(
        motion_encoder,
        reference_motion,
        reference_valid,
        batch_size=args.batch_size,
        device=device,
    )
    generated_embeddings = _encode_motion_embeddings(
        motion_encoder,
        generated_motion,
        generated_valid,
        batch_size=args.batch_size,
        device=device,
    )

    distribution = _embedding_distribution_metrics(
        reference_embeddings, generated_embeddings
    )
    distribution_reference_embeddings = reference_embeddings
    distribution_generated_embeddings = generated_embeddings
    retrieval_checkpoint_path = Path(args.retrieval_evaluator_checkpoint or args.evaluator_checkpoint).expanduser().resolve()
    if args.retrieval_evaluator_checkpoint is not None:
        if args.seed != 0:
            raise ValueError("Intent retrieval uses the single default seed 0")

        if retrieval_checkpoint_path != Path(args.evaluator_checkpoint).expanduser().resolve():
            del motion_encoder, evaluator_checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
            motion_encoder, evaluator_checkpoint = _load_evaluator_motion_encoder(
                str(retrieval_checkpoint_path), args.motion_key, device)
            reference_embeddings = _encode_motion_embeddings(motion_encoder, reference_motion, reference_valid,
                batch_size=args.batch_size, device=device)
            generated_embeddings = _encode_motion_embeddings(motion_encoder, generated_motion, generated_valid,
                batch_size=args.batch_size, device=device)
    raw_text_provenance = None
    text_embeddings = _encode_text_embeddings(evaluator_checkpoint, data["captions"].tolist(),
        model_name=args.t5_model, cache_dir=args.text_cache_dir, batch_size=args.text_batch_size, device=device)
    generated_retrieval = _text_retrieval_summary(
        generated_embeddings,
        text_embeddings,
        dataset_names=data["dataset"].tolist(),
        batch_size=32,
        require_full_batches=True,
        stratified=True,
        seed=args.seed,
    )
    reference_retrieval = _text_retrieval_summary(
        reference_embeddings,
        text_embeddings,
        dataset_names=data["dataset"].tolist(),
        batch_size=32,
        require_full_batches=True,
        stratified=True,
        seed=args.seed,
    )

    for retrieval in (generated_retrieval, reference_retrieval):
        retrieval.update(retrieval_seed=args.seed, retrieval_seeds=[args.seed],
                         retrieval_aggregation=f"single seed {args.seed}")
    text_metrics = {
        "motion_fid": distribution["motion_fid"],
        "motion_kid_diagnostic": distribution["motion_kid"],
        "matching_score": generated_retrieval["matching_score"],
        "r_precision": generated_retrieval["r_precision"],
        "diversity_generated": distribution["diversity_generated"],
        "generated_retrieval_diagnostics": generated_retrieval,
        "reference_retrieval_diagnostics": reference_retrieval,
        "motion_key": args.motion_key,
        "evaluator_checkpoint": str(
            Path(args.evaluator_checkpoint).expanduser().resolve()
        ),
        "text_encoder_model": str(Path(args.t5_model).expanduser().resolve()),

        "evaluator_checkpoint_sha256": file_sha256(args.evaluator_checkpoint),
        "retrieval_evaluator_checkpoint": str(retrieval_checkpoint_path),
        "retrieval_evaluator_checkpoint_sha256": file_sha256(retrieval_checkpoint_path),
        "retrieval_seed": args.seed,
        "retrieval_seeds": [args.seed],
        "retrieval_aggregation": f"single seed {args.seed}",
        "raw_text_embeddings": raw_text_provenance,
    }
    protocol["mixed_evaluator_protocol"] = {

        "unified_encoder": retrieval_checkpoint_path == Path(args.evaluator_checkpoint).expanduser().resolve(),
        "retrieval": {"checkpoint": str(retrieval_checkpoint_path), "sha256": file_sha256(retrieval_checkpoint_path),
                      "seed": args.seed, "seeds": [args.seed], "batch_size": 32, "stratified": True,
                      "aggregation": f"single seed {args.seed}"},
        "distribution": {"checkpoint": text_metrics["evaluator_checkpoint"],
                         "sha256": file_sha256(args.evaluator_checkpoint), "metrics": ["motion_fid", "diversity_generated"]},
    }
    generated_target["text"] = text_metrics
    np.savez_compressed(
        output_dir / "embeddings.npz",
        reference_embeddings=reference_embeddings.astype(np.float32),
        generated_embeddings=generated_embeddings.astype(np.float32),
        text_embeddings=text_embeddings.astype(np.float32),
        sample_id=data["sample_id"],

        distribution_reference_embeddings=distribution_reference_embeddings.astype(np.float32),
        distribution_generated_embeddings=distribution_generated_embeddings.astype(np.float32),
        retrieval_checkpoint=np.asarray(str(retrieval_checkpoint_path)),
        distribution_checkpoint=np.asarray(text_metrics["evaluator_checkpoint"]),
    )

    del motion_encoder, evaluator_checkpoint, reference_motion, generated_motion
    if device.type == "cuda":
        torch.cuda.empty_cache()
else:
    generated_body = qpos_to_body_positions(
        model,
        generated_qpos,
        batch_size=args.batch_size,
        device=device,
    ).numpy()
    reference_body = qpos_to_body_positions(
        model,
        reference_qpos,
        batch_size=args.batch_size,
        device=device,
    ).numpy()
    beat_generated = []
    beat_reference = []
    for index in tqdm(range(data["num_samples"]), desc="BeatAlign", unit="sample"):

        length = int(data["reference_valid"][index].sum())
        kwargs = {
            "audio_features": data["audio_features"][index, :length],
            "fps": float(data["fps"][index]),
            "audio_fps": float(data["audio_fps"][index]),
            "sigma_frames": args.sigma_frames,
            "direction": args.beat_direction,
            "beat_threshold": args.beat_threshold,
            "min_motion_beat_distance_seconds": args.min_motion_beat_distance_seconds,
        }
        beat_generated.append(
            float(beat_align(motion_positions=generated_body[index, :length], **kwargs))
        )
        beat_reference.append(
            float(beat_align(motion_positions=reference_body[index, :length], **kwargs))
        )
    beat_generated = np.asarray(beat_generated, dtype=np.float64)
    beat_reference = np.asarray(beat_reference, dtype=np.float64)
    parent_edges = [
        (int(parent), int(child))
        for parent, child in zip(
            kinematics.parent_body_indices.detach().cpu().tolist(),
            kinematics.child_body_indices.detach().cpu().tolist(),
            strict=True,
        )
    ]

    dance_feature_items = []
    for index in tqdm(range(data["num_samples"]), desc="Dance features", unit="sample"):
        length = int(data["reference_valid"][index].sum())
        dance_feature_items.append(
            aistpp_edge_features(
                reference_body[index : index + 1, :length],
                generated_body[index : index + 1, :length],
                fps=float(data["fps"][index]),
                root_index=0,
                up_axis="z",
                parent_edges=parent_edges,
                left_foot_indices=(5, 6),
                right_foot_indices=(11, 12),
            )
        )
    dance_features = {
        key: np.concatenate([item[key] for item in dance_feature_items], axis=0)
        for key in dance_feature_items[0]
    }
    dance_summary = aistpp_edge_metric_summary(dance_features)
    audio_metrics = {
        "beat_align_generated": _summary_stats(beat_generated),
        "beat_align_reference": _summary_stats(beat_reference),
        "beat_align_gap": _summary_stats(beat_generated - beat_reference),
        "fid_k": dance_summary["fid_k"],
        "fid_g": dance_summary["fid_g"],
        "pfc_generated": dance_summary["pfc_generated"],
        "pfc_reference": dance_summary["pfc_reference"],
        "pfc_gap": dance_summary["pfc_gap"],
        "dance_feature_diagnostics": dance_summary,
        "beat_protocol": {
            "source": data["beat_source"],
            "last_channel_threshold": args.beat_threshold,
            "direction": args.beat_direction,
            "sigma_frames": args.sigma_frames,
            "minimum_motion_beat_distance_seconds": args.min_motion_beat_distance_seconds,
        },
    }
    generated_target["audio"] = audio_metrics
    sample_arrays.update(
        {
            "beat_align_generated": beat_generated,
            "beat_align_reference": beat_reference,
            "pfc_generated": np.asarray(
                dance_features["generated_pfc"], dtype=np.float64
            ),
            "pfc_reference": np.asarray(
                dance_features["reference_pfc"], dtype=np.float64
            ),
        }
    )

tracker_provenance = None
tracker_canonical_status = None
if args.tracker_mode != "skip":
    tracker_provenance = expected_tracker_provenance()
    tracker_canonical_status = canonical_omg_tracker_status(tracker_provenance)
    protocol["tracker_configuration"] = tracker_canonical_status
    if not tracker_canonical_status["eligible"]:
        print(
            "[WARNING] tracker configuration is not canonical Movens-Intent tracking; raw metrics will "
            "be computed, but paper_metrics.complete will remain false:\n- "
            + "\n- ".join(tracker_canonical_status["reasons"])
        )

if args.tracker_mode == "run":
    if args.tracker_steps != 0:
        raise ValueError(
            "paper-compatible tracker execution requires TRACKER_STEPS=0 "
            "(full per-sample rollout derived from each valid generation length)"
        )
    tracker_args = argparse.Namespace(
        tracker_executed=True,
        holomotion_onnx=args.holomotion_onnx,
        tracker_output_name="tracker_executed",
        tracker_num_samples=None,
        tracker_sample_seed=args.seed,
        tracker_target_fps=args.tracker_target_fps,
        tracker_providers=args.tracker_providers,
        tracker_robot_xml=args.tracker_robot_xml,
        tracker_steps=None if args.tracker_steps <= 0 else args.tracker_steps,
        tracker_control_substeps=args.tracker_control_substeps,
        tracker_action_clip=args.tracker_action_clip,
        tracker_root_index=0,
    )

    tracker_run = run_variable_length_tracker_benchmark(
        output_dir=output_dir,
        generated_qpos=generated_qpos,
        reference_qpos=reference_qpos,
        generated_valid=generated_valid,
        fps=fps,
        captions=data["captions"].tolist(),
        dataset_names=data["dataset"].tolist(),
        dataset_indices=data["dataset_index"].astype(int).tolist(),
        tracker_args=tracker_args,
    )
    tracker_artifact = tracker_run["artifact_path"]
elif args.tracker_mode == "reuse":
    tracker_artifact = args.tracker_artifact
else:
    tracker_artifact = None

if tracker_artifact is None:
    tracker_summary = {
        "enabled": False,
        "reason": "tracker-mode=skip; paper row is intentionally incomplete",
    }
    tracker_arrays = {}
else:
    if args.tracker_mode == "run":
        write_json(
            Path(tracker_artifact).expanduser().resolve().parent
            / "movens_tracker_provenance.json",
            tracker_provenance,
        )
    tracker_is_synthetic = validate_tracker_bundle(tracker_artifact, tracker_provenance)
    tracker_summary, tracker_arrays = tracker_metrics_from_artifact(
        tracker_artifact,
        expected_samples=data["num_samples"],
        expected_generated_qpos=data["generated_qpos"],
        expected_generated_valid=data["generated_valid"],
        expected_generation_fps=data["fps"],
        expected_tracker_fps=args.tracker_target_fps,
        expected_holomotion_onnx=args.holomotion_onnx,
        expected_dataset=data["dataset"],
        expected_dataset_index=data["dataset_index"],
        kinematics=kinematics,
        device=device,
        robot_xml=args.tracker_robot_xml,
    )
    tracker_summary["mode"] = args.tracker_mode
    tracker_summary["paper_eligible"] = (
        not tracker_is_synthetic and tracker_canonical_status["eligible"]
    )
    tracker_summary["paper_ineligible_reasons"] = (
        ["synthetic_test_only=true"] if tracker_is_synthetic else []
    ) + tracker_canonical_status["reasons"]
    tracker_summary["tracking_reference"] = (
        "generated target qpos resampled by HoloMotion"
    )
    tracker_summary["units"] = {
        "mpjpe": "mm",
        "g_mpjpe": "mm",
        "e_vel": "mm/frame",
        "e_acc": "mm/frame^2",
        "fall_rate_percent": "sequence percent",
        "joint_limit_rate_percent": "sequence percent",
    }
    np.savez_compressed(
        output_dir / "tracker_failure_flags.npz",
        sample_id=data["sample_id"],
        **tracker_arrays,
    )

raw_metrics = {
    "protocol": protocol,
    "generated_target": generated_target,
    "tracker_executed": tracker_summary,
}
paper_metrics = build_paper_payload(args.task, raw_metrics)
write_json(output_dir / "metrics.json", raw_metrics)
write_json(output_dir / "paper_metrics.json", paper_metrics)
(output_dir / "paper_row.tex").write_text(
    paper_metrics["latex_cells"] + "\n", encoding="utf-8"
)
np.savez_compressed(
    output_dir / "sample_metrics.npz",
    sample_id=data["sample_id"],
    dataset=data["dataset"],
    dataset_index=data["dataset_index"],
    **sample_arrays,
)
write_json(
    output_dir / "run_state.json",
    {
        "status": "complete",
        "kind": "benchmark",
        "task": args.task,
        "paper_row_complete": paper_metrics["complete"],
    },
)

print(json.dumps(paper_metrics, indent=2, allow_nan=False))
print(f"[INFO] raw metrics: {output_dir / 'metrics.json'}")
print(f"[INFO] paper row:  {output_dir / 'paper_metrics.json'}")

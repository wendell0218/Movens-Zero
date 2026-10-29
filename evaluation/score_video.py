import argparse
from pathlib import Path

import numpy as np

from common import load_reference, sha256, write_json
from video_common import evaluate_qpos_pair
from video_metrics import aggregate_metric_arrays
from video_motion import G1Kinematics

parser = argparse.ArgumentParser(description="Evaluate paired G1 video trajectories in world coordinates")
parser.add_argument("--generated-path", type=Path, required=True)
parser.add_argument("--reference-path", type=Path, required=True)
parser.add_argument("--g1-xml", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--validate-only", action="store_true")
args = parser.parse_args()
prediction = load_reference(args.generated_path)
reference = load_reference(args.reference_path)
if prediction["qpos_36"].shape != reference["qpos_36"].shape or not np.array_equal(prediction["sample_id"], reference["sample_id"]):
    raise ValueError("Generated/reference identity or shapes differ")
if not np.array_equal(prediction["valid"], reference["valid"]):
    raise ValueError("Generated/reference validity differs")
if not np.array_equal(prediction["joint_names"], reference["joint_names"]):
    raise ValueError("Generated/reference joint order differs")
out = args.output_dir.resolve()
if out.exists() and any(out.iterdir()):
    raise FileExistsError("Use a new/empty output directory")
out.mkdir(parents=True, exist_ok=True)
protocol = dict(joint_space="G1-24 semantic joints; not SMPL joints or vertices", fps=30,
    coordinate_frame="world Z-up", alignment="per-frame pelvis-relative and Procrustes for local errors; world trajectory for global errors",
    ground_alignment="none", smoothing="none", samples=len(reference["sample_id"]),
    generated_sha256=sha256(args.generated_path), reference_sha256=sha256(args.reference_path), robot_sha256=sha256(args.g1_xml))
if args.validate_only:
    write_json(out / "validation.json", dict(valid=True, protocol=protocol))
    raise SystemExit(0)
kinematics = G1Kinematics(args.g1_xml)
per_sample = []
for index, sample_id in enumerate(reference["sample_id"]):
    metrics = evaluate_qpos_pair(prediction["qpos_36"][index], reference["qpos_36"][index],
        reference["valid"][index], fps=30, mode="both", mask_mode="contiguous_segments", kinematics=kinematics,
        allow_constant_camera_invariance=True)
    per_sample.append(metrics)
summary = {key: value if np.isfinite(value) else None for key, value in aggregate_metric_arrays(per_sample).items()}
write_json(out / "video_metrics.json", dict(protocol=protocol, metrics=summary,
    per_sample=[dict(sample_id=str(sample_id), **{key: float(value.mean()) if value.size else None for key, value in metrics.items()})
        for sample_id, metrics in zip(reference["sample_id"], per_sample)]))
for key, value in summary.items():
    print(f"{key}: {value}")

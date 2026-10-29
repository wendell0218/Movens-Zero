import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from common import load_reference, sha256, write_json

here = Path(__file__).resolve().parent
repo = here.parent

parser = argparse.ArgumentParser(description="Execute all generated video trajectories with HoloMotion and score tracking")
parser.add_argument("--generated-path", type=Path, required=True)
parser.add_argument("--reference-path", type=Path, required=True)
parser.add_argument("--condition-path", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--metrics-root", type=Path, default=repo / "third_party/metrics")
parser.add_argument("--tracker-onnx", type=Path, default=repo / "checkpoints/evaluation/holomotion_v1.2_motion_tracking_model/exported/model.onnx")
parser.add_argument("--tracker-xml", type=Path)
parser.add_argument("--tracker-providers", default="TensorrtExecutionProvider,CUDAExecutionProvider")
parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
args = parser.parse_args()

metrics_root = args.metrics_root.expanduser().resolve()
if args.tracker_xml is None:
    args.tracker_xml = metrics_root / "assets/holomotion/g1_29dof/scene_29dof.xml"
sys.path.insert(0, str(metrics_root / "src"))
from omg.robots.g1.kinematics import G1Kinematics
from metrics import physical_metric_values, tracker_metrics_from_artifact
from tracking import run_variable_length_tracker_benchmark

prediction, reference = load_reference(args.generated_path), load_reference(args.reference_path)
if not np.array_equal(prediction["sample_id"], reference["sample_id"]) or not np.array_equal(prediction["valid"], reference["valid"]):
    raise ValueError("Tracking input identity/validity differs")
with np.load(args.condition_path, allow_pickle=False) as packed:
    condition = {key: packed[key] for key in packed.files}
if not np.array_equal(condition["sample_id"], reference["sample_id"]):
    raise ValueError("Tracking condition identities differ")
out = args.output_dir.resolve()
if out.exists() and any(out.iterdir()):
    raise FileExistsError("Use a new/empty tracking output directory")
out.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
kinematics = G1Kinematics(str(metrics_root / "assets/robots/g1/g1_kinematics.json")).to(device).eval()
generated_qpos = torch.from_numpy(prediction["qpos_36"])
reference_qpos = torch.from_numpy(reference["qpos_36"])
valid, fps = torch.from_numpy(prediction["valid"]), torch.from_numpy(prediction["fps"])
policy_sha, robot_sha = sha256(args.tracker_onnx), sha256(args.tracker_xml)
physical, arrays = physical_metric_values(generated_qpos, valid, fps, kinematics=kinematics, device=device,
    contact_height_threshold=0.1, contact_penetration_tolerance=0.02)
settings = SimpleNamespace(tracker_output_name="tracker_executed", holomotion_onnx=str(args.tracker_onnx.resolve()),
    tracker_target_fps=50, tracker_robot_xml=str(args.tracker_xml.resolve()), tracker_providers=args.tracker_providers,
    tracker_control_substeps=10, tracker_action_clip=10, tracker_steps=None)
run = run_variable_length_tracker_benchmark(output_dir=out, generated_qpos=generated_qpos,
    reference_qpos=reference_qpos, generated_valid=valid, fps=fps, captions=condition["captions"].tolist(),
    dataset_names=condition["dataset"].tolist(), dataset_indices=condition["dataset_index"].astype(int).tolist(), tracker_args=settings)
summary, sample_arrays = tracker_metrics_from_artifact(run["artifact_path"], expected_samples=len(prediction["sample_id"]),
    expected_generated_qpos=prediction["qpos_36"], expected_generated_valid=prediction["valid"],
    expected_generation_fps=prediction["fps"], expected_tracker_fps=50, expected_holomotion_onnx=str(args.tracker_onnx.resolve()),
    expected_dataset=condition["dataset"], expected_dataset_index=condition["dataset_index"],
    kinematics=kinematics, device=device, robot_xml=str(args.tracker_xml.resolve()))
canonical = (policy_sha == "96e917b7de78605b1e2ee3666ed59f3d4ed5111df3325b56e59fdfdf0ef99225"
    and robot_sha == "4e6ae5f0f1d6bee36242fa02c36da1999748f94fa4d2a013f72ca1379b56bd69"
    and args.tracker_providers == "TensorrtExecutionProvider,CUDAExecutionProvider")
write_json(out / "tracking_metrics.json", dict(generated_physical=physical, tracker_executed=summary,
    protocol=dict(policy_sha256=policy_sha, robot_sha256=robot_sha, canonical_tracking_configuration=canonical,
        generated_sha256=sha256(args.generated_path), reference_sha256=sha256(args.reference_path),
        tracker_target_fps=50, control_substeps=10, action_clip=10, providers=args.tracker_providers,
        tracking_reference="generated trajectory resampled by HoloMotion", all_samples_executed=True)))
np.savez_compressed(out / "sample_metrics.npz", sample_id=prediction["sample_id"], **arrays, **sample_arrays)
print(summary)

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import load_reference, sha256, write_json
from download_assets import EVALUATOR_ASSET

here = Path(__file__).resolve().parent
repo = here.parent

parser = argparse.ArgumentParser(description="Score generated Movens-Zero benchmark trajectories")
parser.add_argument("--track", choices=("text", "speech", "music", "video"), required=True)
parser.add_argument("--data-dir", type=Path, required=True)
parser.add_argument("--generation-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--metrics-root", type=Path, default=repo / "third_party/metrics")
parser.add_argument("--evaluator-checkpoint", type=Path,
    help="Text/Speech evaluator; defaults to checkpoints/evaluation/movens_intent_evaluator.pt")
parser.add_argument("--t5-model", default="google-t5/t5-3b")
parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--tracking", choices=("run", "skip"), default="run")
parser.add_argument("--tracker-onnx", type=Path)
parser.add_argument("--tracker-xml", type=Path)
parser.add_argument("--video-g1-xml", type=Path, default=repo / "third_party/g1/assets/unitree_g1/g1_mocap_29dof.xml")
parser.add_argument("--tracker-providers", default="TensorrtExecutionProvider,CUDAExecutionProvider")
parser.add_argument("--validate-only", action="store_true")
args = parser.parse_args()

if args.track in ("text", "speech") and args.evaluator_checkpoint is None:
    args.evaluator_checkpoint = here.parent / "checkpoints/evaluation" / EVALUATOR_ASSET["filename"]
metrics_root = args.metrics_root.expanduser().resolve()
data = args.data_dir.resolve()
generated = args.generation_dir.resolve()
out = args.output_dir.resolve()
tracker_onnx = (args.tracker_onnx or here.parent / "checkpoints/evaluation/holomotion_v1.2_motion_tracking_model/exported/model.onnx").resolve()
tracker_xml = (args.tracker_xml or metrics_root / "assets/holomotion/g1_29dof/scene_29dof.xml").resolve()
reference = load_reference(data / "reference_qpos.npz")
generation = json.loads((generated / "generation.json").read_text())
if generation["benchmark"] != "Movens-Intent" or generation["track"] != args.track:
    raise ValueError("Generation belongs to a different benchmark or track")
if generation["reference_sha256"] != sha256(data / "reference_qpos.npz"):
    raise ValueError("Reference has changed since generation")
if generation["samples_sha256"] != sha256(data / "samples.jsonl"):
    raise ValueError("Condition manifest has changed since generation")
with np.load(generated / "generated_qpos.npz", allow_pickle=False) as packed:
    if str(packed["generation_sha256"].item()) != sha256(generated / "generation.json"):
        raise ValueError("Packed trajectories belong to another generation")
    if not np.array_equal(packed["sample_id"], reference["sample_id"]):
        raise ValueError("Generated/reference IDs or order differ")
    if packed["qpos_36"].shape != reference["qpos_36"].shape or not np.array_equal(packed["valid"], reference["valid"]):
        raise ValueError("Generated/reference shapes or valid masks differ")
if args.track in ("text", "speech") and not args.validate_only:
    if args.evaluator_checkpoint is None or not args.evaluator_checkpoint.is_file():
        raise FileNotFoundError("Movens-Intent evaluator not found; run python evaluation/download_assets.py --asset evaluator")
    if not EVALUATOR_ASSET["sha256"]:
        raise RuntimeError("The published evaluator checksum must be configured before scoring")
    if sha256(args.evaluator_checkpoint) != EVALUATOR_ASSET["sha256"]:
        raise ValueError("Intent paper protocol requires the exact Movens-Intent evaluator checkpoint")
if args.track == "video":
    command = [sys.executable, str(here / "score_video.py"), "--generated-path", str(generated / "generated_qpos.npz"),
        "--reference-path", str(data / "reference_qpos.npz"), "--output-dir", str(out),
        "--g1-xml", str(args.video_g1_xml.resolve())]
    if args.validate_only:
        command += ["--validate-only"]
else:
    if not (metrics_root / "src/omg").is_dir():
        raise FileNotFoundError("Metric dependency not found; run the evaluation setup or provide --metrics-root")
    checkpoint = str(args.evaluator_checkpoint.resolve()) if args.evaluator_checkpoint else "unused"
    task = "text" if args.track in ("text", "speech") else "audio"
    command = [sys.executable, str(here / "score_text_music.py"), "--task", task,
        "--generated-path", str(generated / "generated_qpos.npz"), "--reference-path", str(data / "reference_qpos.npz"),
        "--condition-path", str(data / "condition.npz"), "--output-dir", str(out), "--metrics-root", str(metrics_root),
        "--expected-frames", str(reference["qpos_36"].shape[1]), "--expected-fps", "30", "--device", args.device,
        "--batch-size", str(args.batch_size), "--validate-only", str(int(args.validate_only)),
        "--kinematics-path", str(metrics_root / "assets/robots/g1/g1_kinematics.json"), "--contact-height-threshold", "0.10",
        "--contact-penetration-tolerance", "0.02", "--evaluator-checkpoint", checkpoint,
        "--motion-key", "body_pos_local", "--t5-model", args.t5_model, "--text-cache-dir", str(generated / "t5_cache"),
        "--text-batch-size", "16", "--seed", "0", "--beat-direction", "music_to_motion", "--beat-threshold", "0.5",
        "--sigma-frames", "3", "--min-motion-beat-distance-seconds", "0.25", "--tracker-mode", args.tracking,
        "--tracker-artifact", "unused", "--holomotion-onnx", str(tracker_onnx),
        "--tracker-robot-xml", str(tracker_xml),
        "--tracker-target-fps", "50", "--tracker-providers", args.tracker_providers,
        "--tracker-control-substeps", "10", "--tracker-action-clip", "10", "--tracker-steps", "0"]
    if task == "text":
        command += ["--retrieval-evaluator-checkpoint", checkpoint]
subprocess.run(command, cwd=metrics_root, check=True)
if args.track == "video" and args.tracking == "run" and not args.validate_only:
    subprocess.run([sys.executable, str(here / "score_tracking.py"), "--generated-path", str(generated / "generated_qpos.npz"),
        "--reference-path", str(data / "reference_qpos.npz"), "--condition-path", str(data / "condition.npz"),
        "--output-dir", str(out / "tracking"), "--metrics-root", str(metrics_root), "--tracker-onnx", str(tracker_onnx),
        "--tracker-xml", str(tracker_xml), "--tracker-providers", args.tracker_providers, "--device", args.device], cwd=metrics_root, check=True)
if not args.validate_only:
    write_json(out / "release_protocol.json", dict(benchmark="Movens-Intent", track=args.track,
        generation=generation, generated_sha256=sha256(generated / "generated_qpos.npz"),
        evaluator_sha256=sha256(args.evaluator_checkpoint) if args.evaluator_checkpoint else None,
        retrieval_seed=0, retrieval_batch_size=32, smoothing="none", tracking=args.tracking,
        metric_implementation_sha256={file.name: sha256(file) for file in sorted(here.glob("*.py"))}))

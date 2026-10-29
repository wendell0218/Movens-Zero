import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_reference, sha256, write_json
from video_motion import motion36_to_qpos, sample_motion36
from movens_zero.config import load_config
from movens_zero.normalization import MotionNormalizer, load_initial_pose
from movens_zero.rollout_root40 import generate_root40_motion

here = Path(__file__).resolve().parent
repo = here.parent

parser = argparse.ArgumentParser(description="Generate a complete fixed benchmark cohort with a public checkpoint")
parser.add_argument("--track", choices=("text", "speech", "music", "video"), required=True)
parser.add_argument("--data-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--model-size", choices=("S", "M", "L", "XL"), default="XL")
parser.add_argument("--backbone", default="Qwen/Qwen2.5-Omni-3B")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--cfg-scale", type=float, default=1.5)
parser.add_argument("--cfg-interval", nargs=2, type=float, default=(0.2, 0.8))
parser.add_argument("--num-inference-steps", type=int, default=10)
parser.add_argument("--metrics-root", type=Path, default=repo / "third_party/metrics")
parser.add_argument("--ground-xml", type=Path, help="Override the scene used for Text/Speech/Music ground alignment")
parser.add_argument("--start-index", type=int, default=0)
parser.add_argument("--end-index", type=int, default=0)
parser.add_argument("--pack-only", action="store_true")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

metrics_root = args.metrics_root.expanduser().resolve()
if args.track != "video" and args.ground_xml is None:
    args.ground_xml = metrics_root / "assets/holomotion/g1_29dof/scene_29dof.xml"
reference = load_reference(args.data_dir / "reference_qpos.npz")
rows = [json.loads(line) for line in (args.data_dir / "samples.jsonl").read_text().splitlines() if line.strip()]
if [row["sample_id"] for row in rows] != reference["sample_id"].tolist():
    raise ValueError("Manifest and reference sample IDs/order differ")
if args.track == "text" and any(not row["caption"].strip() for row in rows):
    raise ValueError("Text captions cannot be empty")
if not 0 <= args.cfg_interval[0] < args.cfg_interval[1] <= 1 or args.cfg_scale < 0:
    raise ValueError("Invalid CFG scale or interval")
config_path = here.parent / "configs" / f"inference_{args.model_size.lower()}.json"
config = load_config(config_path)
config.cfg_scale, config.cfg_interval = args.cfg_scale, args.cfg_interval
assets = here.parent / "movens_zero/assets"
normalizer = MotionNormalizer.from_file(assets / Path(config.normalization).name)
standing = load_initial_pose(here / "assets/standard_standing.npy")
count = len(rows)
end = args.end_index or count
if not 0 <= args.start_index < end <= count:
    raise ValueError("Invalid generation shard")
initials = np.repeat(standing[None], count, axis=0)
protocol = dict(benchmark="Movens-Intent", track=args.track, samples=count,
    checkpoint_sha256=sha256(args.checkpoint), config_sha256=sha256(config_path),
    normalization_sha256=sha256(assets / Path(config.normalization).name),
    ground_xml_sha256=sha256(args.ground_xml) if args.ground_xml else None,
    generation_source_sha256=sha256(Path(__file__)),
    reference_sha256=sha256(args.data_dir / "reference_qpos.npz"),
    samples_sha256=sha256(args.data_dir / "samples.jsonl"),
    initial_state="standard_standing",
    initial_array_sha256=__import__("hashlib").sha256(initials.tobytes()).hexdigest(),
    seed=0, cfg_scale=args.cfg_scale, cfg_interval=list(args.cfg_interval),
    steps=args.num_inference_steps, native_fps=50, output_fps=30,
    smoothing="none", gt_state_each_chunk=False,
    temporal_protocol="include initial frame",
    ground_alignment="none_world_coordinates" if args.track == "video" else "sequence_constant_min_robot_collision_distance")
if args.dry_run:
    print(json.dumps(protocol, indent=2))
    raise SystemExit(0)
out = args.output_dir.resolve()
(out / "samples").mkdir(parents=True, exist_ok=True)
if (out / "generation.json").exists() and json.loads((out / "generation.json").read_text()) != protocol:
    raise ValueError("Existing output belongs to a different checkpoint/protocol; use a new directory")
write_json(out / "generation.json", protocol)
identity = sha256(out / "generation.json")
ground = None
if args.track != "video":
    import mujoco
    ground = mujoco.MjModel.from_xml_path(str(args.ground_xml.resolve()))
    floor = np.flatnonzero((ground.geom_bodyid == 0) & (ground.geom_type == mujoco.mjtGeom.mjGEOM_PLANE))
    collisions = np.flatnonzero((ground.geom_bodyid != 0) & ((ground.geom_contype != 0) | (ground.geom_conaffinity != 0)))
    if len(floor) != 1 or not len(collisions):
        raise ValueError("Ground XML must contain one floor and robot collision geometry")
    ground_data, nearest = mujoco.MjData(ground), np.zeros(6)

if not args.pack_only:
    from movens_zero.model import MovensZeroModel
    model = MovensZeroModel.from_pretrained(checkpoint_path=args.checkpoint, backbone_path=args.backbone,
        config=config, device=args.device, attn_implementation="sdpa").to(args.device).eval()
    model.requires_grad_(False)
    prompts = dict(music="Generate a dance motion that follows the rhythm and energy of this music.",
        speech="Listen for the spoken task in the audio, then carry out that task as body motion.",
        video="The clip is the reference. Imitate the person's body movement as closely as possible.")
    for index in range(args.start_index, end):
        row = rows[index]
        target = out / "samples" / (row["sample_id"] + ".npz")
        if target.exists():
            with np.load(target, allow_pickle=False) as cached:
                if str(cached["generation_sha256"].item()) != identity or str(cached["sample_id"].item()) != row["sample_id"]:
                    raise ValueError(f"Stale sample: {target}")
            continue
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        length = int(reference["lengths"][index])
        offset = 0
        native_frames = max(1, math.ceil((length - 1 + offset) * 50 / 30))
        media = args.data_dir / row["input_path"] if "input_path" in row else None
        if media is not None and sha256(media) != row["input_sha256"]:
            raise ValueError(f"Condition media changed: {media}")
        prompt = row["caption"].strip() if args.track == "text" else prompts[args.track]
        with torch.inference_mode(), torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16,
                enabled=torch.device(args.device).type == "cuda"):
            future = generate_root40_motion(model, config, normalizer, args.track, prompt, media,
                initials[index], native_frames, args.num_inference_steps, args.device)
        history = np.concatenate([initials[index:index + 1], future])
        sampled = sample_motion36(history, source_fps=50, sample_times=(np.arange(length) + offset) / 30)
        qpos = motion36_to_qpos(sampled).astype(np.float32)
        shift = 0.0
        if ground is not None:
            lowest = np.inf
            for frame in qpos:
                ground_data.qpos[:] = frame
                mujoco.mj_forward(ground, ground_data)
                lowest = min(lowest, min(mujoco.mj_geomDistance(ground, ground_data, int(floor[0]), int(g), 1000, nearest) for g in collisions))
            shift = -float(lowest)
            qpos[:, 2] += shift
        temporary = target.with_name(target.stem + f".{__import__('os').getpid()}.tmp.npz")
        np.savez_compressed(temporary, qpos_36=qpos, fps=np.float32(30), sample_id=np.asarray(row["sample_id"]),
            generation_sha256=np.asarray(identity), ground_shift_z=np.float32(shift),
            joint_names=reference["joint_names"], initial_motion36=initials[index])
        temporary.replace(target)
        print(f"[{index + 1}/{count}] {row['sample_id']}", flush=True)

if all((out / "samples" / (row["sample_id"] + ".npz")).exists() for row in rows):
    packed = np.empty_like(reference["qpos_36"])
    for index, row in enumerate(rows):
        with np.load(out / "samples" / (row["sample_id"] + ".npz"), allow_pickle=False) as sample:
            qpos = sample["qpos_36"]
            length = int(reference["lengths"][index])
            if str(sample["generation_sha256"].item()) != identity or str(sample["sample_id"].item()) != row["sample_id"] or qpos.shape != (length, 36) or not np.isfinite(qpos).all():
                raise ValueError(f"Incomplete/stale sample {row['sample_id']}")
            packed[index, :length], packed[index, length:] = qpos, qpos[-1]
    payload = {key: reference[key] for key in ("sample_id", "valid", "lengths", "joint_names", "fps")}
    payload.update(qpos_36=packed, captions=np.asarray([row["caption"] for row in rows]), generation_sha256=np.asarray(identity))
    temporary = out / f"generated_qpos.{__import__('os').getpid()}.tmp.npz"
    np.savez_compressed(temporary, **payload)
    temporary.replace(out / "generated_qpos.npz")
    print(f"Packed complete cohort: {out / 'generated_qpos.npz'}")

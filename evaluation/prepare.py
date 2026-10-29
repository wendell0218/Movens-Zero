import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common import load_reference, sha256, write_json

parser = argparse.ArgumentParser(description="Prepare the public Movens-Intent benchmark")
parser.add_argument("--track", choices=("text", "speech", "music", "video"), required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--revision", default="main")
parser.add_argument("--download-dir", type=Path, default=Path("data/Movens-Intent"))
args = parser.parse_args()

from huggingface_hub import HfApi, snapshot_download
import pyarrow.parquet as pq

revision = HfApi().dataset_info("wendell0218/Movens-Intent", revision=args.revision).sha
root = Path(snapshot_download("wendell0218/Movens-Intent", repo_type="dataset", revision=revision,
    local_dir=args.download_dir, allow_patterns=[f"data/{args.track}/*.parquet",
        f"eval_assets/v1/{args.track}/reference_qpos.npz", "eval_assets/v1/manifest.json"]))
reference_path = root / f"eval_assets/v1/{args.track}/reference_qpos.npz"
reference = load_reference(reference_path)
protocol_path = Path(__file__).parent / "protocols" / f"intent_{args.track}_v1.json"
protocol = json.loads(protocol_path.read_text())
if len(protocol["samples"]) != len(reference["sample_id"]):
    raise ValueError("Published cohort size differs from the frozen protocol")
if hashlib.sha256(reference["qpos_36"][reference["valid"]].tobytes()).hexdigest() != protocol["valid_qpos_sha256"]:
    raise ValueError("Published numerical reference differs from the frozen metric protocol")
out = args.output_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
(out / "media").mkdir(exist_ok=True)
rows = []
columns = ["sample_id", "prompt"] if args.track == "text" else ["sample_id", "transcript", "condition_audio"] if args.track == "speech" else ["sample_id", "condition_audio"] if args.track == "music" else ["sample_id", "condition_video"]
for shard in sorted((root / "data" / args.track).glob("*.parquet")):
    schema = pq.read_schema(shard)
    table = pq.read_table(shard, columns=[key for key in columns if key in schema.names])
    rows.extend(table.to_pylist())
rows_by_id = {str(row["sample_id"]): row for row in rows}
if len(rows_by_id) != len(rows) or set(rows_by_id) != set(reference["sample_id"]):
    raise ValueError("Condition and numerical reference IDs differ")
manifest = []
for index, sample_id in enumerate(reference["sample_id"]):
    source = rows_by_id[str(sample_id)]
    frozen = protocol["samples"][index]
    if frozen["sample_id"] != str(sample_id):
        raise ValueError("Public sample order differs from the frozen evaluation protocol")
    caption = str(source.get("prompt", source.get("transcript", f"{args.track} {sample_id}")))
    row = dict(sample_id=str(sample_id), caption=caption, dataset=frozen["dataset"],
        dataset_index=frozen["dataset_index"], frames=int(reference["lengths"][index]), fps=30)
    if args.track != "text":
        media = source["condition_video" if args.track == "video" else "condition_audio"]
        if not isinstance(media, dict) or media.get("bytes") is None:
            raise ValueError("Expected embedded Audio/Video bytes in the public Parquet")
        extension = Path(media.get("path") or ("clip.mp4" if args.track == "video" else "audio.wav")).suffix
        path = out / "media" / (str(sample_id) + extension)
        path.write_bytes(media["bytes"])
        row["input_path"] = str(path.relative_to(out))
        row["input_sha256"] = sha256(path)
    manifest.append(row)
condition = dict(sample_id=reference["sample_id"],
    captions=np.asarray([row["caption"] for row in manifest]),
    dataset=np.asarray([row["dataset"] for row in manifest]),
    dataset_index=np.asarray([row["dataset_index"] for row in manifest], dtype=np.int32))
if args.track == "music":
    beats = np.zeros(reference["valid"].shape, dtype=np.float32)
    for index, row in enumerate(protocol["samples"]):
        beats[index, row["audio_beat_frames_30fps"]] = 1
    condition.update(audio_beats=beats, audio_fps=reference["fps"],
        audio_type=np.repeat("music", len(manifest)))
np.savez_compressed(out / "condition.npz", **condition)
np.savez_compressed(out / "reference_qpos.npz", **reference)
(out / "samples.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest))
write_json(out / "preparation.json", dict(benchmark="Movens-Intent", track=args.track,
    dataset="wendell0218/Movens-Intent", revision=revision, samples=len(manifest),
    reference_sha256=sha256(reference_path), protocol_sha256=sha256(protocol_path),
    manifest_sha256=sha256(out / "samples.jsonl"), condition_sha256=sha256(out / "condition.npz")))
print(f"Prepared {args.track}: {len(manifest)} samples at {out}")

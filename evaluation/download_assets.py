import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from common import sha256

EVALUATOR_ASSET = dict(
    repo_id="wendell0218/Movens-Intent-Evaluator",
    revision="1d5519b485fc1107d25c0cb41d3d761949d69db1",
    filename="movens_intent_evaluator.pt",
    sha256="d9c556b4681e834f1c04a4165ef885c8fff27a01b91d371a0463471f82e7eda9",
)
TRACKER_ASSET = dict(
    repo_id="HorizonRobotics/HoloMotion_models",
    revision="b656e063caa7b564b9dce9044bb1e38afc54832a",
    filename="holomotion_v1.2_motion_tracking_model/exported/model.onnx",
    sha256="96e917b7de78605b1e2ee3666ed59f3d4ed5111df3325b56e59fdfdf0ef99225",
)


def download_asset(asset, output_dir):
    if not asset["revision"] or not asset["sha256"]:
        raise RuntimeError("The published asset revision and checksum must be configured before downloading")
    path = Path(hf_hub_download(asset["repo_id"], filename=asset["filename"],
        revision=asset["revision"], local_dir=output_dir))
    if sha256(path) != asset["sha256"]:
        raise ValueError(f"Public artifact checksum mismatch: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Download the public Movens-Intent evaluator and HoloMotion tracking policy")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/evaluation"))
    parser.add_argument("--asset", choices=("all", "evaluator", "tracker"), default="all")
    args = parser.parse_args()
    assets = dict(evaluator=EVALUATOR_ASSET, tracker=TRACKER_ASSET)
    selected = assets.values() if args.asset == "all" else [assets[args.asset]]
    for asset in selected:
        print(download_asset(asset, args.output_dir))


if __name__ == "__main__":
    main()

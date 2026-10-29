import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from movens_zero.checkpoints import MODEL_RELEASES, checkpoint_model_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=MODEL_RELEASES, default="XL")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    repository, revision = MODEL_RELEASES[args.model_size]
    destination = args.output_dir / f"Movens-Zero-{args.model_size}"
    path = hf_hub_download(repository, "model.safetensors", revision=revision, local_dir=destination)
    architecture = checkpoint_model_config(path)
    receipt = dict(repo_id=repository, revision=revision, filename="model.safetensors", model=architecture)
    (destination / "download.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Checkpoint: {Path(path).resolve()}")
    print(json.dumps(architecture, indent=2))


if __name__ == "__main__":
    main()

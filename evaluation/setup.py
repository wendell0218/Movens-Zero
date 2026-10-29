import argparse
import hashlib
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
DEPENDENCIES = (
    dict(name="metric utilities", directory="metrics", url="https://github.com/Tsinghua-MARS-Lab/OMG.git",
        revision="61e196010b332acbace223b2c449e25f454ea0a3",
        folders=("src/omg", "assets/holomotion/g1_29dof", "assets/robots/g1"),
        required=("LICENSE", "src/omg/benchmarks/runners/text.py", "src/omg/robots/g1/kinematics.py",
                  "assets/holomotion/g1_29dof/scene_29dof.xml", "assets/robots/g1/g1_kinematics.json")),
    dict(name="G1 robot assets", directory="g1", url="https://github.com/YanjieZe/GMR.git",
        revision="25f56db79ac411258b377b6d4ccba0c8ff3372ac", folders=("assets/unitree_g1",),
        required=("LICENSE", "assets/unitree_g1/LICENSE", "assets/unitree_g1/g1_mocap_29dof.xml"),
        xml_sha256="85a25128d42ad3cc8dd071c089806a2773a59ee0689edcc646aeb0201ae8f4c8"),
)


def git(path, *arguments):
    result = subprocess.run(["git", "-C", str(path), *arguments], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def prepare_dependency(specification, target):
    if target.exists():
        if not target.is_dir() or not (target / ".git").exists():
            raise RuntimeError(f"{target} already exists and is not the required Git checkout; use another dependency directory")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout",
                        specification["url"], str(target)], check=True)
        git(target, "sparse-checkout", "set", "--cone", *specification["folders"])
        git(target, "fetch", "--depth", "1", "origin", specification["revision"])
        git(target, "checkout", "--detach", specification["revision"])
    if Path(git(target, "rev-parse", "--show-toplevel")).resolve() != target.resolve():
        raise RuntimeError(f"{target} is not an independent dependency checkout")
    origin = git(target, "remote", "get-url", "origin").rstrip("/").removesuffix(".git")
    expected = specification["url"].removesuffix(".git")
    normalized = origin.replace("git@github.com:", "https://github.com/").replace("ssh://git@github.com/", "https://github.com/")
    if normalized.lower() != expected.lower():
        raise RuntimeError(f"{target} has a different origin; expected {specification['url']}")
    if git(target, "status", "--porcelain"):
        raise RuntimeError(f"{target} contains local changes; preserve them and provide a clean dependency checkout")
    current = git(target, "rev-parse", "HEAD")
    if current != specification["revision"]:
        raise RuntimeError(f"{target} is at {current}; the evaluation protocol requires {specification['revision']}. Existing checkouts are not changed")
    missing = [name for name in specification["required"] if not (target / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{target} is missing required upstream files: {', '.join(missing)}")
    if "xml_sha256" in specification:
        xml = target / "assets/unitree_g1/g1_mocap_29dof.xml"
        if hashlib.sha256(xml.read_bytes()).hexdigest() != specification["xml_sha256"]:
            raise ValueError(f"{xml} does not match the fixed G1 evaluation geometry")
        document = ET.parse(xml)
        compiler = document.find("compiler")
        mesh_root = xml.parent / (compiler.get("meshdir", "") if compiler is not None else "")
        for mesh in document.findall(".//asset/mesh"):
            path = mesh_root / mesh.attrib["file"]
            if not path.is_file():
                raise FileNotFoundError(f"Required G1 mesh is missing: {path}")
            with path.open("rb") as handle:
                if handle.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    raise RuntimeError(f"{path} is a Git LFS pointer; install Git LFS and fetch this checkout's G1 assets")
    print(f"Ready: {specification['name']} at {target} ({current})")


def main():
    parser = argparse.ArgumentParser(description="Prepare public metric utilities, G1 robot assets, evaluator and tracking policy")
    parser.add_argument("--dependency-dir", type=Path, default=ROOT / "third_party")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/evaluation")
    parser.add_argument("--skip-download", action="store_true", help="Prepare source code and robot assets without downloading model weights")
    args = parser.parse_args()
    for specification in DEPENDENCIES:
        prepare_dependency(specification, args.dependency_dir.resolve() / specification["directory"])
    if not args.skip_download:
        from download_assets import EVALUATOR_ASSET, TRACKER_ASSET, download_asset
        for asset in (EVALUATOR_ASSET, TRACKER_ASSET):
            print(download_asset(asset, args.output_dir.resolve()))


if __name__ == "__main__":
    main()

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", choices=("text", "music", "speech", "video"))
    p.add_argument("--text", default="")
    for mode in ("music", "speech", "video"):
        p.add_argument("--" + mode, type=Path)
    p.add_argument("--seconds", type=float)
    p.add_argument("--host", default=os.environ.get("MOVENS_HOST", "127.0.0.1"))
    p.add_argument(
        "--port", type=int, default=int(os.environ.get("MOVENS_PORT", "8014"))
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Validate dependencies, input and service; no DDS/WBC",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a static pose to infer two chunks; never imports or starts DDS/WBC",
    )
    args = p.parse_args()
    interactive_menu = (
        not args.modality and (not args.check_only) and (not args.dry_run)
    )
    if not args.modality:
        if not sys.stdin.isatty() or not interactive_menu:
            p.error("--modality is required for noninteractive/check-only/dry-run use")
        args.modality, args.text = ("text", "A person stands still.")
    if args.modality == "text" and (not args.text.strip()):
        p.error("--text is required for text tasks")
    if args.seconds is not None and (
        args.seconds <= 0 or args.modality not in ("text", "speech")
    ):
        p.error(
            "--seconds must be positive and applies only to text/speech; music/video follow the file"
        )
    media = getattr(args, args.modality, None) if args.modality != "text" else None
    if args.modality != "text" and (media is None or not media.is_file()):
        p.error("Selected modality requires an existing local media file")
    for mode in ("music", "speech", "video"):
        if getattr(args, mode) is not None and mode != args.modality:
            p.error(f"--{mode} requires --modality {mode}")
    import numpy, pandas, websocket, zmq
    from controller import wait_movens_server

    mode = dict(video="video_clip", speech="audio").get(args.modality, args.modality)
    if args.dry_run:
        from probe import run_probe

        return run_probe(
            args.host, args.port, args.modality, args.text, media, args.seconds or 6.0
        )
    wait_movens_server(args.host, args.port, 10.0, mode)
    if args.check_only:
        import unitree_sdk2py
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        wbc = Path(
            os.environ.get("WBC_ROOT", "~/GR00T-WholeBodyControl/gear_sonic_deploy")
        ).expanduser()
        required = [
            "target/release/g1_deploy_onnx_ref",
            "policy/release/model_decoder.onnx",
            "policy/release/model_encoder.onnx",
            "policy/release/observation_config.yaml",
            "planner/target_vel/V2/planner_sonic.onnx",
        ]
        for name in required:
            if not (wbc / name).is_file():
                raise FileNotFoundError(wbc / name)
        print("CLIENT_PREFLIGHT_OK: no DDS initialization, no WBC, no motor commands")
        return 0
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("controller.py")),
        "--visual-input-mode",
        mode,
        "--movens-host",
        args.host,
        "--movens-port",
        str(args.port),
        "--prompt",
        args.text,
        "--prewarm-video-cache",
        "1",
        "--prewarm-timeout",
        "180",
        "--wbc-zmq-host",
        "127.0.0.1",
        "--wbc-zmq-connect-host",
        "127.0.0.1",
        "--wbc-zmq-mode",
        "queue",
        "--log-dir",
        "/tmp/movens_real_logs",
    ]
    if interactive_menu:
        cmd += ["--interactive-task-menu"]
    if args.modality in ("text", "speech"):
        cmd += ["--max-seconds", str(args.seconds or 6.0)]
    if args.modality == "video":
        cmd += ["--inference-video-path", str(media.resolve()), "--inline-video-bytes"]
    elif args.modality in ("music", "speech"):
        cmd += ["--audio-path", str(media.resolve()), "--inline-audio-bytes"]
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    raise SystemExit(main())

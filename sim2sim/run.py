import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent


def wait_for(path, processes, timeout=120, patterns=()):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for name, process in processes:
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited with {process.returncode}; inspect its log")
        if path.is_file() and (not patterns or any(text in path.read_text(errors="replace") for text in patterns)):
            return
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for {path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sonic-root", type=Path, default=Path(os.environ.get("SONIC_ROOT", "~/GR00T-WholeBodyControl")).expanduser())
    parser.add_argument("--mode", choices=("text", "music", "speech", "video"), default="text")
    parser.add_argument("--prompt", default="A person walks slowly forward.")
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sim2sim"))
    parser.add_argument("--host", default=os.environ.get("MOVENS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MOVENS_PORT", "8014")))
    parser.add_argument("--domain-id", type=int, default=71)
    parser.add_argument("--zmq-port", type=int, default=5566)
    parser.add_argument("--settle-seconds", type=float, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.seconds <= 0 or args.timeout <= 0 or args.settle_seconds < 0:
        parser.error("seconds/timeout must be positive and settling nonnegative")
    if args.mode != "text" and (args.input_path is None or not args.input_path.is_file()):
        parser.error("Media tasks require an existing --input-path")
    if args.mode == "text" and not args.prompt.strip():
        parser.error("Text tasks require a nonempty prompt")
    sonic = args.sonic_root.resolve()
    wbc = sonic / "gear_sonic_deploy"
    required = [wbc / name for name in ("target/release/g1_deploy_onnx_ref", "policy/release/model_decoder.onnx",
        "policy/release/model_encoder.onnx", "policy/release/observation_config.yaml", "planner/target_vel/V2/planner_sonic.onnx")]
    required += [sonic / "gear_sonic/utils/mujoco_sim/base_sim.py", wbc / "reference/example", wbc / "scripts/setup_env.sh"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Complete the SONIC installation first:\n" + "\n".join(missing))
    for name in ("numpy", "mujoco", "pandas", "websocket", "zmq", "unitree_sdk2py"):
        if importlib.util.find_spec(name) is None:
            raise ModuleNotFoundError(f"Install {name} in the simulation environment")
    with urllib.request.urlopen(f"http://{args.host}:{args.port}/health", timeout=10) as reply:
        health = json.load(reply)
    if health.get("backend") != "checkpoint" or not health.get("checkpoint_verified") or health.get("protocol_version") != "movens-root40-v1":
        raise ValueError("Use the real checkpoint server, not a transport fixture")
    identity = dict(health=health, mode=args.mode, input_path=str(args.input_path.resolve()) if args.input_path else None,
                    prompt=args.prompt, seconds=args.seconds, domain_id=args.domain_id, state_feedback="live",
                    physics_frequency=200, observation_frequency=50, action_horizon=30, execute_horizon=20,
                    overlap_frames=10, max_inference_seconds=0.18)
    if args.check_only:
        print(json.dumps(identity, indent=2))
        return
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Choose an empty output directory; previous runs are not overwritten")
    (out / "run_config.json").write_text(json.dumps(identity, indent=2))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(sonic), str(sonic / "external_dependencies/unitree_sdk2_python"), str(ROOT), env.get("PYTHONPATH", "")])
    env["PYTHONUNBUFFERED"] = "1"
    processes, handles = [], []
    started = time.monotonic()
    status = dict(status="running", start_time=time.time())
    def launch(name, command, cwd=ROOT):
        handle = (out / (name + ".log")).open("w")
        handles.append(handle)
        process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((name, process))
        return process
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        launch("simulator", [sys.executable, "-u", str(ROOT / "sim2sim/simulator.py"), "--sonic-root", str(sonic),
            "--run-dir", str(out), "--domain-id", str(args.domain_id), "--max-seconds", str(args.timeout + 180)])
        wait_for(out / "live_state.npy", processes)
        command = [str(required[0]), "lo", str(required[1]), str(wbc / "reference/example"),
            "--obs-config", str(required[3]), "--encoder-file", str(required[2]),
            "--planner-file", str(required[4]), "--input-type", "manager", "--output-type", "all",
            "--disable-crc-check", "--domain-id", str(args.domain_id),
            "--zmq-host", "127.0.0.1", "--zmq-port", str(args.zmq_port), "--zmq-out-port", str(args.zmq_port + 1)]
        controller = launch("wbc", ["bash", "-c", "source scripts/setup_env.sh && exec " + shlex.join(command)], wbc)
        wait_for(out / "wbc.log", processes, patterns=("Initialized ZMQ endpoint interface", "[ZMQEndpointInterface] Connected"))
        controller.stdin.write(b"\n]")
        controller.stdin.flush()
        time.sleep(args.settle_seconds)
        (out / "release.flag").touch()
        time.sleep(args.settle_seconds)
        controller.stdin.write(b"#")
        controller.stdin.flush()
        mode = dict(video="video_clip", speech="audio").get(args.mode, args.mode)
        bridge_command = [sys.executable, "-u", str(ROOT / "sim2real/motion_bridge.py"),
            "--movens-host", args.host, "--movens-port", str(args.port), "--wbc-zmq-host", "127.0.0.1",
            "--wbc-zmq-port", str(args.zmq_port), "--visual-input-mode", mode,
            "--prompt", args.prompt, "--max-seconds", str(args.seconds), "--target-frames", str(round(args.seconds * 50)),
            "--live-state-npy", str(out / "live_state.npy"), "--state-feedback", "live",
            "--live-state-max-age", "0.2", "--start-index", "0",
            "--start-flag-path", str(out / "start.flag"), "--stop-flag-path", str(out / "stop.flag"),
            "--ready-flag-path", str(out / "bridge_ready.flag"), "--first-pose-flag-path", str(out / "first_pose.flag"),
            "--generated-joint-pos-csv", str(out / "generated_joint_pos.csv"),
            "--latency-log", str(out / "latency.jsonl"), "--tail-padding-frames", "50", "--tail-padding-hold-seconds", "1"]
        if args.mode == "video":
            bridge_command += ["--inference-video-path", str(args.input_path.resolve()), "--inline-video-bytes"]
        elif args.mode in ("music", "speech"):
            bridge_command += ["--audio-path", str(args.input_path.resolve()), "--inline-audio-bytes"]
        bridge = launch("bridge", bridge_command)
        wait_for(out / "bridge_ready.flag", processes)
        (out / "start.flag").touch()
        wait_for(out / "first_pose.flag", processes, timeout=args.timeout)
        (out / "record.flag").touch()
        bridge.wait(timeout=args.timeout)
        if bridge.returncode:
            raise RuntimeError(f"Bridge exited with {bridge.returncode}; inspect bridge.log")
        if "completed_stream=True" not in (out / "bridge.log").read_text():
            raise RuntimeError("Bridge did not complete the motion timeline; inspect bridge.log")
        status["status"] = "complete"
    except BaseException as exc:
        status.update(status="failed", error=str(exc) or type(exc).__name__)
        raise
    finally:
        (out / "stop.flag").touch()
        for name, process in reversed(processes):
            if process.poll() is None:
                if name == "wbc" and process.stdin is not None:
                    try:
                        process.stdin.write(b"o")
                        process.stdin.flush()
                    except BrokenPipeError:
                        pass
                try:
                    os.killpg(process.pid, signal.SIGINT)
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                except ProcessLookupError:
                    pass
        for handle in handles:
            handle.close()
        status["wall_seconds"] = time.monotonic() - started
        status["returncodes"] = {name: process.returncode for name, process in processes}
        (out / "run_status.json").write_text(json.dumps(status, indent=2))
        print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

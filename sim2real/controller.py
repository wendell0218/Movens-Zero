from __future__ import annotations
import argparse
import json
import os
import re
import select
import shutil
import signal
import shlex
import subprocess
import sys
import termios
import threading
import time
import tty
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import websocket

CTRL_HZ = 50
PROMPT = ""
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


@dataclass(frozen=True)
class VideoEntry:
    path: Path | None
    seconds: float
    target_frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument(
        "--interactive-task-menu",
        action="store_true",
        help="Choose task after standing and entering action mode",
    )
    parser.add_argument(
        "--movens-host", default=os.environ.get("MOVENS_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--movens-port", type=int, default=int(os.environ.get("MOVENS_PORT", "8014"))
    )
    parser.add_argument(
        "--visual-input-mode",
        choices=("video_clip", "remote_camera_stream", "text", "music", "audio"),
        default=os.environ.get("VISUAL_INPUT_MODE", "video_clip"),
    )
    parser.add_argument("--run-dir", default=os.environ.get("RUN_DIR", ""))
    parser.add_argument(
        "--init-state-npy", default=os.environ.get("INIT_STATE_NPY", "")
    )
    parser.add_argument(
        "--init-motion-dir", default=os.environ.get("INIT_MOTION_DIR", "")
    )
    parser.add_argument(
        "--inference-video-path", default=os.environ.get("INFERENCE_VIDEO_PATH", "")
    )
    parser.add_argument("--audio-path", default=os.environ.get("AUDIO_PATH", ""))
    parser.add_argument("--audio-sampling-rate", type=int, default=16000)
    parser.add_argument("--audio-window-seconds", type=float, default=0.6)
    parser.add_argument(
        "--inline-audio-bytes",
        action="store_true",
        default=os.environ.get("INLINE_AUDIO_BYTES", "1") == "1",
    )
    parser.add_argument(
        "--video-config-key", default=os.environ.get("VIDEO_CONFIG_KEY", "video_clip")
    )
    parser.add_argument("--video-clip-motion-fps", type=float, default=50.0)
    parser.add_argument("--video-sample-fps", type=float, default=8.0)
    parser.add_argument("--video-size-pixels", type=int, default=50176)
    parser.add_argument("--video-decoder", default="ffmpeg")
    parser.add_argument(
        "--camera-width",
        type=int,
        default=int(os.environ.get("REMOTE_CAMERA_WIDTH", "640")),
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=int(os.environ.get("REMOTE_CAMERA_HEIGHT", "480")),
    )
    parser.add_argument(
        "--remote-camera-host", default=os.environ.get("REMOTE_CAMERA_HOST", "0.0.0.0")
    )
    parser.add_argument(
        "--remote-camera-port",
        type=int,
        default=int(os.environ.get("REMOTE_CAMERA_PORT", "18080")),
    )
    parser.add_argument(
        "--remote-camera-warmup-timeout",
        type=float,
        default=float(os.environ.get("REMOTE_CAMERA_WARMUP_TIMEOUT", "30.0")),
    )
    parser.add_argument(
        "--prewarm-video-cache",
        type=int,
        default=int(os.environ.get("PREWARM_VIDEO_CACHE", "0")),
    )
    parser.add_argument(
        "--prewarm-timeout",
        type=float,
        default=float(os.environ.get("PREWARM_TIMEOUT", "180")),
    )
    parser.add_argument(
        "--inline-video-bytes",
        action="store_true",
        default=os.environ.get("INLINE_VIDEO_BYTES", "1") == "1",
    )
    parser.add_argument(
        "--require-local-video",
        action="store_true",
        default=os.environ.get("REQUIRE_LOCAL_VIDEO", "0") == "1",
    )
    parser.add_argument("--pose-chunk-frames", type=int, default=30)
    parser.add_argument("--pose-chunk-advance-frames", type=int, default=20)
    parser.add_argument("--pose-chunk-send-seconds", type=float, default=0.0)
    parser.add_argument("--start-index", type=int, default=138990)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--target-frames", type=int, default=0)
    parser.add_argument("--motion-plan-json", default="")
    parser.add_argument(
        "--live-state-npy",
        default=os.environ.get("LIVE_STATE_NPY", "/tmp/movens_wbc_live_state.npy"),
    )
    parser.add_argument(
        "--live-state-timeout",
        type=float,
        default=float(os.environ.get("LIVE_STATE_TIMEOUT", "30")),
    )
    parser.add_argument(
        "--live-state-max-age",
        type=float,
        default=float(os.environ.get("LIVE_STATE_MAX_AGE", "0.2")),
    )
    parser.add_argument("--start-flag-path", default="/tmp/movens_wbc_real_start.flag")
    parser.add_argument("--stop-flag-path", default="/tmp/movens_wbc_real_stop.flag")
    parser.add_argument(
        "--bridge-ready-flag-path", default="/tmp/movens_wbc_real_bridge_ready.flag"
    )
    parser.add_argument(
        "--first-pose-flag-path", default="/tmp/movens_wbc_real_first_pose.flag"
    )
    parser.add_argument(
        "--tail-padding-frames",
        type=int,
        default=int(os.environ.get("TAIL_PADDING_FRAMES", "80")),
    )
    parser.add_argument(
        "--tail-padding-hold-seconds",
        "--post-stream-hold-seconds",
        dest="tail_padding_hold_seconds",
        type=float,
        default=float(
            os.environ.get(
                "TAIL_PADDING_HOLD_SECONDS",
                os.environ.get("POST_STREAM_HOLD_SECONDS", "2.0"),
            )
        ),
    )
    parser.add_argument(
        "--action-timeout",
        type=float,
        default=float(os.environ.get("ACTION_TIMEOUT", "0.55")),
    )
    parser.add_argument(
        "--action-idle-timeout-seconds",
        type=float,
        default=float(os.environ.get("ACTION_IDLE_TIMEOUT_SECONDS", "10.0")),
    )
    parser.add_argument(
        "--generated-joint-pos-csv",
        default=os.environ.get("GENERATED_JOINT_POS_CSV", ""),
    )
    parser.add_argument(
        "--timing-log-interval",
        type=float,
        default=float(os.environ.get("TIMING_LOG_INTERVAL", "1.0")),
    )
    parser.add_argument(
        "--max-reference-joint-velocity",
        type=float,
        default=float(os.environ.get("MAX_REFERENCE_JOINT_VELOCITY", "0.0")),
    )
    parser.add_argument(
        "--max-first-joint-jump-rad",
        type=float,
        default=float(os.environ.get("MAX_FIRST_JOINT_JUMP_RAD", "0.35")),
    )
    parser.add_argument(
        "--min-reference-root-height",
        type=float,
        default=float(os.environ.get("MIN_REFERENCE_ROOT_HEIGHT", "0.20")),
    )
    parser.add_argument(
        "--max-reference-root-height",
        type=float,
        default=float(os.environ.get("MAX_REFERENCE_ROOT_HEIGHT", "1.50")),
    )
    parser.add_argument("--wbc-zmq-host", default=os.environ.get("WBC_ZMQ_HOST", "*"))
    parser.add_argument(
        "--wbc-zmq-port", type=int, default=int(os.environ.get("WBC_ZMQ_PORT", "5556"))
    )
    parser.add_argument(
        "--wbc-zmq-connect-host",
        default=os.environ.get("WBC_ZMQ_CONNECT_HOST", "localhost"),
        help="Host used by WBC WBC to connect to the bridge publisher.",
    )
    parser.add_argument(
        "--log-dir", default=os.environ.get("LOG_DIR", "/tmp/movens_wbc_real_logs")
    )
    parser.add_argument("--wbc-ready-timeout", type=float, default=300.0)
    parser.add_argument(
        "--wbc-zmq-mode",
        choices=("auto", "queue", "latest"),
        default=os.environ.get("WBC_ZMQ_MODE", "auto"),
    )
    parser.add_argument(
        "--first-pose-timeout",
        type=float,
        default=float(os.environ.get("FIRST_POSE_TIMEOUT", "20.0")),
    )
    parser.add_argument(
        "--planner-handoff-timeout",
        type=float,
        default=float(os.environ.get("PLANNER_HANDOFF_TIMEOUT", "15.0")),
        help="Seconds to wait for planner_motion confirmation after a bridge fault.",
    )
    return parser.parse_args()


def stream_log(name: str, process: subprocess.Popen, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        while True:
            line = process.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            print(f"[{name}] {text}", flush=True)
            f.write(text + "\n")
            f.flush()


def start_process(
    name: str,
    cmd: str,
    cwd: Path,
    log_dir: Path,
    env: dict[str, str] | None = None,
    stdin_pipe: bool = False,
) -> subprocess.Popen:
    process = subprocess.Popen(
        ["bash", "-lc", cmd],
        cwd=str(cwd),
        stdin=subprocess.PIPE if stdin_pipe else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        env=env,
    )
    thread = threading.Thread(
        target=stream_log, args=(name, process, log_dir / f"{name}.log"), daemon=True
    )
    thread.start()
    return process


def stop_process(
    name: str, process: subprocess.Popen | None, timeout: float = 8.0
) -> None:
    if process is None or process.poll() is not None:
        return
    print(f"[client] stopping {name}", flush=True)
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=5.0)
    except ProcessLookupError:
        pass


def send_wbc_key(process: subprocess.Popen | None, chars: str) -> None:
    if process is None or process.poll() is not None or process.stdin is None:
        return
    try:
        process.stdin.write(chars.encode("utf-8"))
        process.stdin.flush()
    except Exception as exc:
        print(f"[client] failed to send WBC key {chars!r}: {exc}", flush=True)


def write_flag(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def clean_paths(paths: list[Path]) -> None:
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except FileNotFoundError:
            pass


def collect_video_paths(path_text: str) -> list[Path]:
    if not path_text:
        return []
    path = Path(path_text).expanduser()
    if path.is_dir():
        return sorted(
            (
                item
                for item in path.iterdir()
                if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES
            )
        )
    return [path]


def read_motion_plan_summary(path_text: str) -> tuple[float, int]:
    if not path_text:
        return (0.0, 0)
    motion_plan = json.loads(Path(path_text).read_text(encoding="utf-8"))
    fps = int(motion_plan.get("fps", CTRL_HZ))
    total_seconds = 0.0
    total_frames = 0
    for item in motion_plan.get("segments", []):
        frames = int(item.get("num_frames", 0))
        if frames <= 0:
            frames = int(round(float(item.get("duration_seconds", 0.0)) * fps))
        if frames <= 0:
            continue
        total_frames += frames
        total_seconds += frames / float(fps)
    return (total_seconds, total_frames)


def probe_media_duration(path: Path, require_audio=False) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type:format=duration",
            "-of",
            "json",
            str(path),
        ]
        info = json.loads(subprocess.check_output(cmd, text=True, timeout=30))
        if require_audio and (
            not any((x.get("codec_type") == "audio" for x in info.get("streams", [])))
        ):
            raise ValueError("Media contains no audio stream")
        duration = float(info["format"]["duration"])
    else:
        import imageio_ffmpeg

        result = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if require_audio and "Audio:" not in result.stderr:
            raise ValueError("Media contains no audio stream")
        match = re.search("Duration: (\\d+):(\\d+):(\\d+(?:\\.\\d+)?)", result.stderr)
        if not match:
            raise ValueError(f"Media duration unavailable: {path}")
        h, m, sec = map(float, match.groups())
        duration = h * 3600 + m * 60 + sec
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Media duration must be finite and positive")
    return duration


def probe_video_duration_seconds(path: Path) -> float:
    return probe_media_duration(path)


def probe_audio_duration_seconds(path: Path, sampling_rate: int) -> float:
    return probe_media_duration(path, require_audio=True)


def resolve_motion_duration(
    args: argparse.Namespace, video_path: Path, plan_summary: tuple[float, int]
) -> tuple[float, int]:
    plan_seconds, plan_frames = plan_summary
    effective_max_seconds = float(args.max_seconds)
    target_frames = int(args.target_frames)
    if plan_seconds > 0:
        effective_max_seconds = plan_seconds
        if target_frames <= 0:
            target_frames = plan_frames
    elif target_frames > 0 and effective_max_seconds <= 0.0:
        effective_max_seconds = target_frames / float(CTRL_HZ)
    elif effective_max_seconds <= 0.0:
        effective_max_seconds = probe_video_duration_seconds(video_path)
    if target_frames <= 0 and effective_max_seconds > 0.0:
        target_frames = int(np.ceil(effective_max_seconds * CTRL_HZ))
    return (effective_max_seconds, target_frames)


def build_video_entries(args: argparse.Namespace) -> list[VideoEntry]:
    if not args.inference_video_path:
        raise ValueError("video_clip deployment requires --inference-video-path")
    input_path = Path(args.inference_video_path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"inference video path does not exist: {input_path}")
    video_paths = collect_video_paths(str(input_path))
    if not video_paths:
        suffixes = ", ".join(sorted(VIDEO_SUFFIXES))
        raise RuntimeError(f"No video files found in {input_path} ({suffixes})")
    plan_summary = read_motion_plan_summary(args.motion_plan_json)
    return [
        VideoEntry(path, *resolve_motion_duration(args, path, plan_summary))
        for path in video_paths
    ]


def build_input_entries(args: argparse.Namespace) -> list[VideoEntry]:
    if args.visual_input_mode == "video_clip":
        return build_video_entries(args)
    path = None
    if args.visual_input_mode == "text":
        if not str(args.prompt).strip():
            raise ValueError("text deployment requires a non-empty --prompt")
    elif args.visual_input_mode in {"music", "audio"}:
        path = Path(args.audio_path).expanduser() if args.audio_path else None
        if path is None or not path.is_file():
            raise FileNotFoundError(f"audio input does not exist: {path}")
    plan_seconds, plan_frames = read_motion_plan_summary(args.motion_plan_json)
    target_frames = int(args.target_frames)
    seconds = float(args.max_seconds)
    if plan_frames > 0:
        target_frames = plan_frames
        seconds = plan_seconds
    elif target_frames > 0:
        seconds = target_frames / float(CTRL_HZ)
    elif seconds > 0.0:
        target_frames = int(np.ceil(seconds * CTRL_HZ))
    elif args.visual_input_mode == "music":
        assert path is not None
        seconds = probe_audio_duration_seconds(path, int(args.audio_sampling_rate))
        target_frames = int(np.ceil(seconds * CTRL_HZ))
        print(
            f"[client] music duration from source audio: {seconds:.3f}s / {target_frames} frames",
            flush=True,
        )
    else:
        raise ValueError(
            f"{args.visual_input_mode} deployment requires --max-seconds or --target-frames"
        )
    return [VideoEntry(path, seconds, target_frames)]


def print_input_selection(
    args: argparse.Namespace, entries: list[VideoEntry], index: int
) -> None:
    if args.visual_input_mode == "text":
        entry = entries[index]
        print(
            f"[client] text input: {args.prompt!r} ({entry.seconds:.3f}s / {entry.target_frames} frames)",
            flush=True,
        )
        return
    if args.visual_input_mode == "remote_camera_stream":
        entry = entries[index]
        print(
            f"[client] remote camera input: listen={args.remote_camera_host}:{args.remote_camera_port} source={args.camera_width}x{args.camera_height} ({entry.seconds:.3f}s / {entry.target_frames} frames)",
            flush=True,
        )
        return
    if args.visual_input_mode in {"music", "audio"}:
        entry = entries[index]
        print(
            f"[client] {args.visual_input_mode} input: {entry.path} ({entry.seconds:.3f}s / {entry.target_frames} frames)",
            flush=True,
        )
        return
    entry = entries[index]
    assert entry.path is not None
    print(f"[client] video dir: {entry.path.parent}", flush=True)
    print(
        f"[client] current video [{index + 1}/{len(entries)}]: {entry.path.name} ({entry.seconds:.3f}s / {entry.target_frames} frames)",
        flush=True,
    )


def wait_movens_server(
    host: str, port: int, timeout: float, expected_input_mode: str
) -> None:
    deadline = time.time() + timeout
    ws_url = f"ws://{host}:{port}/ws"
    health_url = f"http://{host}:{port}/health"
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=3.0) as response:
                health = json.loads(response.read().decode("utf-8"))
            if health.get("status") != "ok":
                raise RuntimeError(f"unexpected health response: {health!r}")
            if (
                health.get("protocol_version") != "movens-root40-v1"
                or health.get("model_action_dim") != 40
            ):
                raise RuntimeError(
                    "server input mode/protocol is not the Movens root40 contract"
                )
            if health.get("backend") != "checkpoint" or not health.get(
                "checkpoint_verified"
            ):
                raise RuntimeError(
                    "server input mode is not verified checkpoint inference"
                )
            server_mode = str(health.get("input_mode", "")).strip().lower()
            advertised_modes = {
                str(mode).strip().lower()
                for mode in health.get("supported_input_modes", [])
            }
            if server_mode == "auto":
                if expected_input_mode not in advertised_modes:
                    raise RuntimeError(
                        f"Movens server does not support requested input mode: client={expected_input_mode}, supported={sorted(advertised_modes)}"
                    )
            elif server_mode and server_mode != expected_input_mode:
                raise RuntimeError(
                    f"Movens server input mode mismatch: client={expected_input_mode}, server={server_mode}. Restart the 4090 server with the current automatic launcher."
                )
            elif not server_mode and expected_input_mode != "video_clip":
                raise RuntimeError(
                    "Movens server does not advertise automatic multimodal support; restart the updated serve_movens-rtc-wbc.sh"
                )
            mode_text = server_mode or "legacy-video_clip"
            print(
                f"[client] Movens server ready: {ws_url} input_mode={mode_text}",
                flush=True,
            )
            return
        except Exception as exc:
            last_error = exc
            if "input mode" in str(exc).lower() or "multimodal" in str(exc).lower():
                raise
            time.sleep(1.0)
    detail = f": {last_error}" if last_error is not None else ""
    raise RuntimeError(f"Movens server is not ready: {ws_url}{detail}")


def probe_movens_server_mode(host: str, port: int, timeout: float = 3.0) -> str:
    health_url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as response:
            health = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"cannot query Movens server health at {health_url}: {exc}"
        ) from exc
    if health.get("status") != "ok":
        raise RuntimeError(f"unexpected Movens health response: {health!r}")
    return str(health.get("input_mode", "")).strip().lower()


def wait_live_state(path: Path, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            state = np.load(path)
            if state.shape == (36,) and time.time() - path.stat().st_mtime < 1.0:
                root = state[29:32]
                print(
                    f"[client] live state ready: root=({root[0]:.3f},{root[1]:.3f},{root[2]:.3f})",
                    flush=True,
                )
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"live state is not ready: {path}")


def wait_log_pattern(
    name: str,
    process: subprocess.Popen,
    log_path: Path,
    patterns: list[str],
    timeout: float,
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{name} exited early with code {process.poll()}")
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            for pattern in patterns:
                if pattern in text:
                    print(f"[client] {name} ready: {pattern}", flush=True)
                    return
        time.sleep(0.5)
    raise RuntimeError(f"{name} was not ready within {timeout:.1f}s: {log_path}")


def wait_log_pattern_since(
    name: str,
    process: subprocess.Popen,
    log_path: Path,
    patterns: list[str],
    timeout: float,
    start_offset: int,
) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{name} exited during recovery with code {process.poll()}"
            )
        if log_path.exists():
            with log_path.open("rb") as f:
                size = f.seek(0, os.SEEK_END)
                f.seek(min(max(0, start_offset), size))
                text = f.read().decode("utf-8", errors="replace")
            for pattern in patterns:
                if pattern in text:
                    return pattern
        time.sleep(0.1)
    raise RuntimeError(
        f"{name} did not confirm planner handoff within {timeout:.1f}s: {log_path}"
    )


def handoff_wbc_to_planner(
    process: subprocess.Popen | None, log_path: Path, timeout: float
) -> None:
    if process is None or process.poll() is not None:
        raise RuntimeError("WBC is not running; cannot return control to WBC planner")
    start_offset = log_path.stat().st_size if log_path.exists() else 0
    print("[client] returning WBC to WBC planner stand", flush=True)
    send_wbc_key(process, "!\n")
    marker = wait_log_pattern_since(
        "wbc",
        process,
        log_path,
        ["Reference motion name: planner_motion"],
        timeout,
        start_offset,
    )
    print(f"[client] WBC planner takeover confirmed: {marker}", flush=True)


def wait_file(
    path: Path,
    timeout: float,
    process: subprocess.Popen | None = None,
    name: str = "process",
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"{name} exited early with code {process.poll()}")
        if path.exists():
            return
        time.sleep(0.1)
    raise RuntimeError(f"timeout waiting for {path}")


WBC_PLAYBACK_PATTERN = re.compile(
    "incoming_frame_start:\\s*(\\d+),\\s*incoming_frame_end:\\s*(\\d+),.*global_playback_frame:\\s*(\\d+)"
)


def read_wbc_playback_line(
    log_path: Path, max_incoming_end: int | None = None
) -> tuple[int, int, int] | None:
    if not log_path.exists():
        return None
    last_match = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = WBC_PLAYBACK_PATTERN.search(line)
        if match is None:
            continue
        incoming_start = int(match.group(1))
        incoming_end = int(match.group(2))
        global_playback = int(match.group(3))
        if max_incoming_end is not None and incoming_end > max_incoming_end:
            continue
        last_match = (incoming_start, incoming_end, global_playback)
    return last_match


def print_wbc_tail_check(log_path: Path, motion_end_frame: int | None) -> None:
    if motion_end_frame is None or motion_end_frame < 0:
        return
    motion_line = read_wbc_playback_line(log_path, max_incoming_end=motion_end_frame)
    if motion_line is None:
        print(
            f"[client] WBC tail check: no playback line found for motion_end={motion_end_frame}",
            flush=True,
        )
        return
    incoming_start, incoming_end, global_playback = motion_line
    lag_frames = max(0, incoming_end - global_playback)
    print(
        f"[client] WBC tail check: motion_packet={incoming_start}..{incoming_end} global_playback={global_playback} lag={lag_frames} frames / {lag_frames / CTRL_HZ:.2f}s",
        flush=True,
    )
    hold_line = read_wbc_playback_line(log_path)
    if hold_line is None or hold_line[1] <= motion_end_frame:
        return
    hold_start, hold_end, hold_playback = hold_line
    print(
        f"[client] WBC hold check: hold_packet={hold_start}..{hold_end} global_playback={hold_playback}",
        flush=True,
    )


@contextmanager
def raw_terminal():
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def read_key(timeout: float = 0.2) -> str | None:
    readable, _, _ = select.select([sys.stdin], [], [], timeout)
    if not readable:
        return None
    return sys.stdin.read(1)


def resolve_wbc_zmq_mode(visual_input_mode: str, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    if visual_input_mode == "remote_camera_stream":
        return "latest"
    return "queue"


def main() -> int:
    args = parse_args()
    if not sys.stdin.isatty():
        raise RuntimeError(
            "real G1 jog requires an interactive TTY for operator control and stop keys"
        )
    args.visual_input_mode = str(args.visual_input_mode).strip().lower()
    if int(args.pose_chunk_frames) != 30 or int(args.pose_chunk_advance_frames) != 20:
        raise ValueError(
            "v26 real deployment requires POSE_CHUNK_FRAMES=30 and POSE_CHUNK_ADVANCE_FRAMES=20"
        )
    if int(args.prewarm_video_cache) not in (0, 1):
        raise ValueError("--prewarm-video-cache must be 0 or 1")
    if (
        args.live_state_max_age <= 0.0
        or args.first_pose_timeout <= 0.0
        or args.planner_handoff_timeout <= 0.0
    ):
        raise ValueError("live-state and first-pose timeouts must be positive")
    if args.tail_padding_frames < 0 or args.tail_padding_hold_seconds < 0.0:
        raise ValueError("tail-padding frames/hold seconds must be non-negative")
    if args.visual_input_mode == "remote_camera_stream":
        if args.camera_width <= 0 or args.camera_height <= 0:
            raise ValueError("remote camera width/height must be positive")
        if not 0 < args.remote_camera_port <= 65535:
            raise ValueError("remote camera port must be in 1..65535")
        if args.remote_camera_warmup_timeout <= 0.0:
            raise ValueError("remote camera warmup timeout must be positive")
    if args.visual_input_mode == "remote_camera_stream":
        raise ValueError(
            "This release supports text/music/speech/video file inputs only"
        )
    print("[Movens] Deployment initializes now; press s to start standing.", flush=True)
    video_entries = [] if args.interactive_task_menu else build_input_entries(args)
    video_index = 0
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    required_paths = {}
    if args.init_state_npy:
        required_paths["init_state_npy"] = Path(args.init_state_npy).expanduser()
    if args.init_motion_dir:
        required_paths["init_motion_dir"] = Path(args.init_motion_dir).expanduser()
    if args.motion_plan_json:
        required_paths["motion_plan_json"] = Path(args.motion_plan_json).expanduser()
    missing_paths = [
        f"{name}: {path}" for name, path in required_paths.items() if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Missing required deploy paths:\n" + "\n".join(missing_paths)
        )
    live_state = Path(args.live_state_npy)
    start_flag = Path(args.start_flag_path)
    stop_flag = Path(args.stop_flag_path)
    bridge_ready_flag = Path(args.bridge_ready_flag_path)
    first_pose_flag = Path(args.first_pose_flag_path)
    clean_paths([live_state, start_flag, stop_flag, bridge_ready_flag, first_pose_flag])
    state_process = None
    wbc_process = None
    bridge_process = None
    try:
        wait_movens_server(
            args.movens_host,
            args.movens_port,
            timeout=10.0,
            expected_input_mode=args.visual_input_mode,
        )
        process_env = os.environ.copy()
        process_env["MOVENS_DEPLOY_PYTHON"] = sys.executable
        process_env["LIVE_STATE_NPY"] = str(live_state)
        state_cmd = f"source {shlex.quote(str(script_dir / 'state_reader.sh'))}"
        state_process = start_process(
            "g1_state", state_cmd, repo_root, log_dir, env=process_env
        )
        wait_live_state(live_state, timeout=args.live_state_timeout)
        wbc_env = os.environ.copy()
        wbc_env["MOVENS_REAL_CONFIRMED"] = "1"
        wbc_env["WBC_ZMQ_HOST"] = args.wbc_zmq_connect_host
        wbc_env["WBC_ZMQ_PORT"] = str(args.wbc_zmq_port)
        resolved_wbc_zmq_mode = resolve_wbc_zmq_mode(
            args.visual_input_mode, args.wbc_zmq_mode
        )
        wbc_env["WBC_ZMQ_MODE"] = resolved_wbc_zmq_mode
        wbc_env["WBC_ZMQ_CONFLATE"] = "1" if resolved_wbc_zmq_mode == "latest" else "0"
        print(
            f"[client] WBC ZMQ mode: {resolved_wbc_zmq_mode} (visual_input_mode={args.visual_input_mode})",
            flush=True,
        )
        wbc_cmd = f"source {shlex.quote(str(script_dir / 'start_wbc.sh'))}"
        wbc_process = start_process(
            "wbc", wbc_cmd, repo_root, log_dir, env=wbc_env, stdin_pipe=True
        )
        wait_log_pattern(
            "wbc",
            wbc_process,
            log_dir / "wbc.log",
            ["[ZMQEndpointInterface] Connected", "Initialized ZMQ endpoint interface"],
            args.wbc_ready_timeout,
        )

        def start_bridge_for_video(video_entry: VideoEntry) -> subprocess.Popen:
            clean_paths([start_flag, stop_flag, bridge_ready_flag, first_pose_flag])
            bridge_args = [
                sys.executable,
                "-u",
                str(script_dir / "motion_bridge.py"),
                "--movens-host",
                args.movens_host,
                "--movens-port",
                str(args.movens_port),
                "--wbc-zmq-host",
                args.wbc_zmq_host,
                "--wbc-zmq-port",
                str(args.wbc_zmq_port),
                "--prompt",
                args.prompt,
                "--visual-input-mode",
                args.visual_input_mode,
                "--video-config-key",
                args.video_config_key,
                "--video-clip-motion-fps",
                str(args.video_clip_motion_fps),
                "--video-sample-fps",
                str(args.video_sample_fps),
                "--video-size-pixels",
                str(args.video_size_pixels),
                "--video-decoder",
                args.video_decoder,
                "--camera-width",
                str(args.camera_width),
                "--camera-height",
                str(args.camera_height),
                "--remote-camera-host",
                args.remote_camera_host,
                "--remote-camera-port",
                str(args.remote_camera_port),
                "--remote-camera-warmup-timeout",
                str(args.remote_camera_warmup_timeout),
                "--start-index",
                str(args.start_index),
                "--live-state-npy",
                str(live_state),
                "--live-state-timeout",
                str(args.live_state_timeout),
                "--live-state-max-age",
                str(args.live_state_max_age),
                "--start-flag-path",
                str(start_flag),
                "--stop-flag-path",
                str(stop_flag),
                "--ready-flag-path",
                str(bridge_ready_flag),
                "--first-pose-flag-path",
                str(first_pose_flag),
                "--target-frames",
                str(video_entry.target_frames),
                "--max-seconds",
                str(video_entry.seconds),
                "--pose-chunk-frames",
                str(args.pose_chunk_frames),
                "--pose-chunk-advance-frames",
                str(args.pose_chunk_advance_frames),
                "--pose-chunk-send-seconds",
                str(args.pose_chunk_send_seconds),
                "--tail-padding-frames",
                str(args.tail_padding_frames),
                "--tail-padding-hold-seconds",
                str(args.tail_padding_hold_seconds),
                "--action-timeout",
                str(args.action_timeout),
                "--action-idle-timeout-seconds",
                str(args.action_idle_timeout_seconds),
                "--timing-log-interval",
                str(args.timing_log_interval),
                "--max-reference-joint-velocity",
                str(args.max_reference_joint_velocity),
                "--max-first-joint-jump-rad",
                str(args.max_first_joint_jump_rad),
                "--min-reference-root-height",
                str(args.min_reference_root_height),
                "--max-reference-root-height",
                str(args.max_reference_root_height),
                "--orchestrator-managed-handoff",
            ]
            if args.visual_input_mode == "video_clip":
                assert video_entry.path is not None
                bridge_args.extend(["--inference-video-path", str(video_entry.path)])
            elif args.visual_input_mode in {"music", "audio"}:
                assert video_entry.path is not None
                bridge_args.extend(
                    [
                        "--audio-path",
                        str(video_entry.path),
                        "--audio-sampling-rate",
                        str(args.audio_sampling_rate),
                        "--audio-window-seconds",
                        str(args.audio_window_seconds),
                    ]
                )
            if args.run_dir:
                bridge_args.extend(["--run-dir", args.run_dir])
            if args.init_state_npy:
                bridge_args.extend(["--init-state-npy", args.init_state_npy])
            if args.init_motion_dir:
                bridge_args.extend(["--init-motion-dir", args.init_motion_dir])
            if args.visual_input_mode == "video_clip" and args.require_local_video:
                bridge_args.append("--require-local-video")
            if args.visual_input_mode == "video_clip" and args.inline_video_bytes:
                bridge_args.append("--inline-video-bytes")
            if args.visual_input_mode in {"music", "audio"} and args.inline_audio_bytes:
                bridge_args.append("--inline-audio-bytes")
            if int(args.prewarm_video_cache) > 0:
                bridge_args.extend(
                    [
                        "--prewarm-video-cache",
                        "--prewarm-timeout",
                        str(args.prewarm_timeout),
                    ]
                )
            if args.generated_joint_pos_csv:
                bridge_args.extend(
                    ["--generated-joint-pos-csv", args.generated_joint_pos_csv]
                )
            if args.motion_plan_json:
                bridge_args.extend(["--motion-plan-json", args.motion_plan_json])
            bridge_cmd = " ".join((shlex.quote(part) for part in bridge_args))
            process = start_process("bridge", bridge_cmd, repo_root, log_dir)
            ready_timeout = max(
                30.0,
                float(args.prewarm_timeout) + 30.0,
                float(args.remote_camera_warmup_timeout) + 30.0,
            )
            try:
                wait_file(
                    bridge_ready_flag,
                    timeout=ready_timeout,
                    process=process,
                    name="bridge",
                )
            except Exception:
                write_flag(stop_flag, str(time.time()))
                stop_process("bridge", process)
                raise
            return process

        prepared_video_index: int | None = None

        def stop_prepared_bridge() -> None:
            nonlocal bridge_process, prepared_video_index
            if bridge_process is not None and bridge_process.poll() is None:
                write_flag(stop_flag, str(time.time()))
                stop_process("bridge", bridge_process)
            bridge_process = None
            prepared_video_index = None
            clean_paths([start_flag, stop_flag, bridge_ready_flag, first_pose_flag])

        def prepare_bridge_for_current_video() -> None:
            nonlocal bridge_process, prepared_video_index
            if (
                bridge_process is not None
                and bridge_process.poll() is None
                and (prepared_video_index == video_index)
            ):
                return
            stop_prepared_bridge()
            print(
                f"[client] preparing Movens bridge/{args.visual_input_mode} cache",
                flush=True,
            )
            bridge_process = start_bridge_for_video(video_entries[video_index])
            prepared_video_index = video_index
            print("[client] bridge prepared; press ] to run Movens motion", flush=True)

        print("")
        print("[client] ready")
        key_help = "n/N=next video, " if args.visual_input_mode == "video_clip" else ""
        print(
            f"[client] keys: s=start WBC planner, ENTER=action/ZMQ mode, {key_help}]=run Movens motion, !=planner, o=stop"
        )
        from terminal_ui import TaskMenu, TaskInput, KeyReader
        import copy

        task_selected = not args.interactive_task_menu
        menu = TaskMenu(seconds=args.max_seconds if args.max_seconds > 0 else 6.0)
        if task_selected:
            print_input_selection(args, video_entries, video_index)
            mode = dict(video_clip="video", audio="speech").get(
                args.visual_input_mode, args.visual_input_mode
            )
            content = (
                args.prompt if mode == "text" else str(video_entries[video_index].path)
            )
            menu.remember(TaskInput(mode, content, video_entries[video_index].seconds))
        else:
            print(
                "[Controls] s stand -> Enter task mode -> 1/2/3/4 modality -> input -> ] run",
                flush=True,
            )
        control_started = False
        action_mode_armed = False
        action_started = False
        planner_recovery_pending = False

        def recover_wbc_after_bridge_fault() -> bool:
            nonlocal action_started, action_mode_armed, planner_recovery_pending
            try:
                handoff_wbc_to_planner(
                    wbc_process, log_dir / "wbc.log", args.planner_handoff_timeout
                )
            except Exception as exc:
                action_started = False
                action_mode_armed = True
                planner_recovery_pending = True
                print(
                    f"[client] CRITICAL: planner takeover is not confirmed; WBC is intentionally kept alive ({exc})",
                    flush=True,
                )
                print(
                    "[client] press ! to retry planner takeover, or o for an explicit global stop",
                    flush=True,
                )
                return False
            action_started = False
            action_mode_armed = False
            planner_recovery_pending = False
            return True

        def prepare_or_recover() -> bool:
            nonlocal bridge_process, prepared_video_index
            try:
                prepare_bridge_for_current_video()
                return True
            except Exception as exc:
                print(f"[client] Movens preparation failed: {exc}", flush=True)
                bridge_process = None
                prepared_video_index = None
                clean_paths([start_flag, stop_flag, bridge_ready_flag, first_pose_flag])
                recovered = recover_wbc_after_bridge_fault()
                if recovered:
                    print(
                        "[client] WBC remains active in planner stand; fix the input/server mode, then press ENTER to retry",
                        flush=True,
                    )
                return False

        with raw_terminal():
            key_reader = KeyReader(sys.stdin.fileno())
            while True:
                for name, proc in [("g1_state", state_process), ("wbc", wbc_process)]:
                    if proc is not None and proc.poll() is not None:
                        raise RuntimeError(f"{name} exited with code {proc.poll()}")
                if bridge_process is not None and bridge_process.poll() is not None:
                    bridge_code = bridge_process.poll()
                    if action_started and bridge_code == 0:
                        completed_video_index = (
                            prepared_video_index
                            if prepared_video_index is not None
                            else video_index
                        )
                        completed_entry = video_entries[completed_video_index]
                        motion_end_frame = (
                            completed_entry.target_frames - 1
                            if completed_entry.target_frames > 0
                            else None
                        )
                        print(
                            "[client] Movens motion completed; checking WBC tail before planner return",
                            flush=True,
                        )
                        print_wbc_tail_check(log_dir / "wbc.log", motion_end_frame)
                        bridge_process = None
                        prepared_video_index = None
                        if recover_wbc_after_bridge_fault():
                            print("[client] ready for next Movens motion", flush=True)
                    else:
                        print(
                            f"[client] Movens bridge fault (exit={bridge_code}); recovering to WBC planner",
                            flush=True,
                        )
                        bridge_process = None
                        prepared_video_index = None
                        recovered = recover_wbc_after_bridge_fault()
                        clean_paths(
                            [start_flag, stop_flag, bridge_ready_flag, first_pose_flag]
                        )
                        if recovered:
                            print(
                                "[client] Movens stopped, but WBC remains active in planner stand; press ENTER to prepare another attempt or o to stop",
                                flush=True,
                            )
                        continue
                key = key_reader.read(timeout=0.2)
                if key is None:
                    continue
                key_lower = key.lower()
                if menu.editing and key != "\x03":
                    chosen = menu.feed(key)
                    if chosen is not None:
                        candidate = copy.copy(args)
                        candidate.visual_input_mode = dict(
                            video="video_clip", speech="audio"
                        ).get(chosen.mode, chosen.mode)
                        candidate.prompt = (
                            chosen.content if chosen.mode == "text" else ""
                        )
                        candidate.inference_video_path = (
                            chosen.content if chosen.mode == "video" else ""
                        )
                        candidate.audio_path = (
                            chosen.content if chosen.mode in ("music", "speech") else ""
                        )
                        candidate.max_seconds = (
                            chosen.seconds if chosen.mode in ("text", "speech") else 0.0
                        )
                        candidate.target_frames = 0
                        candidate.motion_plan_json = ""
                        try:
                            entries = build_input_entries(candidate)
                        except Exception as exc:
                            print(
                                f"\x07[Input error] {exc}; the previous task is unchanged.",
                                flush=True,
                            )
                            menu.show()
                            continue
                        stop_prepared_bridge()
                        args = candidate
                        video_entries, video_index = (entries, 0)
                        task_selected = True
                        menu.remember(chosen)
                        print_input_selection(args, video_entries, video_index)
                        print(
                            "[Selected] Press ] to prepare and run; 1-4 changes input; h opens history.",
                            flush=True,
                        )
                    continue
                if key.startswith("\x1b"):
                    continue
                if key_lower == "o" or key == "\x03":
                    print("[client] stop requested", flush=True)
                    write_flag(stop_flag, str(time.time()))
                    send_wbc_key(wbc_process, "o")
                    return 0
                if planner_recovery_pending:
                    if key == "!":
                        if recover_wbc_after_bridge_fault():
                            print(
                                "[client] planner recovery confirmed; ready for another attempt",
                                flush=True,
                            )
                    else:
                        print(
                            "[client] planner recovery is pending; press ! to retry or o to stop",
                            flush=True,
                        )
                    continue
                if key_lower == "s" and (not control_started):
                    print("[client] starting WBC planner control", flush=True)
                    send_wbc_key(wbc_process, "\n]")
                    control_started = True
                    print(
                        "[client] WBC planner is active; W/A/S/D and planner keys are forwarded to WBC",
                        flush=True,
                    )
                    continue
                if not control_started:
                    print(
                        "[client] press s first to start WBC planner control",
                        flush=True,
                    )
                    continue
                if action_mode_armed and (
                    key in ("1", "2", "3", "4") or key_lower == "h"
                ):
                    if action_started:
                        print(
                            "[Task] Input changes are disabled during motion; wait or press ! to return.",
                            flush=True,
                        )
                    else:
                        menu.begin(key_lower)
                    continue
                if key_lower == "n":
                    if not task_selected or args.visual_input_mode != "video_clip":
                        send_wbc_key(wbc_process, key)
                        continue
                    if action_started:
                        print(
                            "[client] Movens motion is running; video switch ignored",
                            flush=True,
                        )
                        continue
                    if bridge_process is not None:
                        stop_prepared_bridge()
                    video_index = (video_index + 1) % len(video_entries)
                    print_input_selection(args, video_entries, video_index)
                    if action_mode_armed:
                        prepare_or_recover()
                    continue
                if key == "\n":
                    if action_started:
                        print(
                            "[client] Movens motion is running; ENTER ignored until it finishes",
                            flush=True,
                        )
                        continue
                    if action_mode_armed:
                        action_mode_armed = False
                        send_wbc_key(wbc_process, "!\n")
                        stop_prepared_bridge()
                        print(
                            "[client] switched WBC back to WBC planner stand",
                            flush=True,
                        )
                    else:
                        action_mode_armed = True
                        send_wbc_key(wbc_process, "#")
                        print(
                            "[client] switched WBC to action/ZMQ interface; press ] to run Movens motion",
                            flush=True,
                        )
                        menu.show()
                        if task_selected:
                            print_input_selection(args, video_entries, video_index)
                            prepare_or_recover()
                    continue
                if key == "#":
                    action_mode_armed = True
                    send_wbc_key(wbc_process, key)
                    print(
                        "[client] switched WBC to action/ZMQ interface; press ] to run Movens motion",
                        flush=True,
                    )
                    menu.show()
                    if task_selected:
                        print_input_selection(args, video_entries, video_index)
                        prepare_or_recover()
                    continue
                if key == "!":
                    action_mode_armed = False
                    send_wbc_key(wbc_process, "!\n")
                    stop_prepared_bridge()
                    print("[client] switched WBC back to WBC planner stand", flush=True)
                    continue
                if key == "]":
                    if action_started:
                        print("[client] Movens motion is already running", flush=True)
                        continue
                    if not action_mode_armed:
                        action_mode_armed = True
                        send_wbc_key(wbc_process, "#")
                        print(
                            "[client] switched WBC to action/ZMQ interface", flush=True
                        )
                        time.sleep(0.3)
                    if not task_selected:
                        menu.show()
                        print(
                            "[Task] Select a modality and enter its input first.",
                            flush=True,
                        )
                        continue
                    if not prepare_or_recover():
                        continue
                    print_input_selection(args, video_entries, video_index)
                    print(
                        "[client] Movens motion requested; enabling WBC ZMQ stream",
                        flush=True,
                    )
                    send_wbc_key(wbc_process, "\n")
                    write_flag(start_flag, str(time.time()))
                    print("[client] waiting for first safe pose chunk", flush=True)
                    try:
                        wait_file(
                            first_pose_flag,
                            timeout=args.first_pose_timeout,
                            process=bridge_process,
                            name="bridge",
                        )
                    except Exception as exc:
                        print(
                            f"[client] Movens failed before the first safe pose: {exc}",
                            flush=True,
                        )
                        write_flag(stop_flag, str(time.time()))
                        stop_process("bridge", bridge_process)
                        bridge_process = None
                        prepared_video_index = None
                        recovered = recover_wbc_after_bridge_fault()
                        clean_paths(
                            [start_flag, stop_flag, bridge_ready_flag, first_pose_flag]
                        )
                        if recovered:
                            print(
                                "[client] WBC remains active in planner stand; press ENTER to prepare another attempt or o to stop",
                                flush=True,
                            )
                        continue
                    action_started = True
                    print(
                        "[client] WBC ZMQ stream enabled; Movens motion is running",
                        flush=True,
                    )
                    continue
                send_wbc_key(wbc_process, key)
    except KeyboardInterrupt:
        print("[client] interrupted", flush=True)
        return 130
    except Exception as exc:
        print(f"[client] stopped by error: {exc}", flush=True)
        return 1
    finally:
        write_flag(stop_flag, str(time.time()))
        send_wbc_key(wbc_process, "o")
        stop_process("bridge", bridge_process)
        stop_process("wbc", wbc_process)
        stop_process("g1_state", state_process)
        print(f"[client] logs: {log_dir}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

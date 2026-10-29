from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
import zlib
from base64 import b64encode
from collections import deque
from io import BytesIO
from pathlib import Path
import numpy as np
import websocket
from motion_transport import (
    CTRL_DT,
    CTRL_HZ,
    WBCZMQPublisher,
    StateProvider,
    actions_to_abs_states,
    build_request_payload,
    compose_policy_state,
    decode_response_message,
    load_initial_state,
    load_state_npy,
    load_motion_prompt_plan,
)

WIRE_ACTION_DIM = 36
DEFAULT_GENERATED_JOINT_POS_CSV = (
    Path(__file__).resolve().parent / "logs/generated_joint_pos_real.csv"
)
G1_JOINT_NAMES = (
    "left_hip_pitch",
    "right_hip_pitch",
    "waist_yaw",
    "left_hip_roll",
    "right_hip_roll",
    "waist_roll",
    "left_hip_yaw",
    "right_hip_yaw",
    "waist_pitch",
    "left_knee",
    "right_knee",
    "left_shoulder_pitch",
    "right_shoulder_pitch",
    "left_ankle_pitch",
    "right_ankle_pitch",
    "left_shoulder_roll",
    "right_shoulder_roll",
    "left_ankle_roll",
    "right_ankle_roll",
    "left_shoulder_yaw",
    "right_shoulder_yaw",
    "left_elbow",
    "right_elbow",
    "left_wrist_roll",
    "right_wrist_roll",
    "left_wrist_pitch",
    "right_wrist_pitch",
    "left_wrist_yaw",
    "right_wrist_yaw",
)
G1_JOINT_LOWER = np.asarray(
    [
        -2.5307,
        -2.5307,
        -2.618,
        -0.5236,
        -2.9671,
        -0.52,
        -2.7576,
        -2.7576,
        -0.52,
        -0.087267,
        -0.087267,
        -3.0892,
        -3.0892,
        -0.87267,
        -0.87267,
        -1.5882,
        -2.2515,
        -0.2618,
        -0.2618,
        -2.618,
        -2.618,
        -1.0472,
        -1.0472,
        -1.972222054,
        -1.972222054,
        -1.614429558,
        -1.614429558,
        -1.614429558,
        -1.614429558,
    ],
    dtype=np.float32,
)
G1_JOINT_UPPER = np.asarray(
    [
        2.8798,
        2.8798,
        2.618,
        2.9671,
        0.5236,
        0.52,
        2.7576,
        2.7576,
        0.52,
        2.8798,
        2.8798,
        2.6704,
        2.6704,
        0.5236,
        0.5236,
        2.2515,
        1.5882,
        0.2618,
        0.2618,
        2.618,
        2.618,
        2.0944,
        2.0944,
        1.972222054,
        1.972222054,
        1.614429558,
        1.614429558,
        1.614429558,
        1.614429558,
    ],
    dtype=np.float32,
)
G1_REFERENCE_VELOCITY_SCALE = np.ones(len(G1_JOINT_NAMES), dtype=np.float32)
G1_MECHANICAL_VELOCITY_LIMIT = np.asarray(
    [
        32.0,
        32.0,
        32.0,
        32.0,
        32.0,
        37.0,
        32.0,
        32.0,
        37.0,
        20.0,
        20.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        22.0,
        22.0,
        22.0,
        22.0,
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--movens-host", default="localhost")
    parser.add_argument("--movens-port", type=int, default=8014)
    parser.add_argument("--wbc-zmq-host", default="*")
    parser.add_argument("--wbc-zmq-port", type=int, default=5556)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--motion-plan-json", default="")
    parser.add_argument("--run-dir", default=os.environ.get("RUN_DIR", ""))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--init-state-npy", default="")
    parser.add_argument("--init-motion-dir", default="")
    parser.add_argument("--latency-log", default="")
    parser.add_argument("--state-feedback", choices=("reference_heading", "live"), default="reference_heading")
    parser.add_argument("--live-state-npy", default="/tmp/movens_wbc_live_state.npy")
    parser.add_argument("--live-state-timeout", type=float, default=30.0)
    parser.add_argument("--live-state-max-age", type=float, default=0.5)
    parser.add_argument("--start-flag-path", default="/tmp/movens_wbc_real_start.flag")
    parser.add_argument("--stop-flag-path", default="/tmp/movens_wbc_real_stop.flag")
    parser.add_argument(
        "--ready-flag-path", default="/tmp/movens_wbc_real_bridge_ready.flag"
    )
    parser.add_argument(
        "--first-pose-flag-path", default="/tmp/movens_wbc_real_first_pose.flag"
    )
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--target-frames", type=int, default=0)
    parser.add_argument("--pose-chunk-frames", type=int, default=30)
    parser.add_argument("--pose-chunk-advance-frames", type=int, default=20)
    parser.add_argument("--pose-chunk-send-seconds", type=float, default=0.0)
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
        "--visual-input-mode",
        default=os.environ.get("VISUAL_INPUT_MODE", "video_clip"),
        choices=("video_clip", "remote_camera_stream", "text", "music", "audio"),
    )
    parser.add_argument(
        "--inference-video-path", default=os.environ.get("INFERENCE_VIDEO_PATH", "")
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
        "--inline-video-bytes",
        action="store_true",
        default=os.environ.get("INLINE_VIDEO_BYTES", "0") == "1",
    )
    parser.add_argument(
        "--require-local-video",
        action="store_true",
        default=os.environ.get("REQUIRE_LOCAL_VIDEO", "0") == "1",
    )
    parser.add_argument(
        "--prewarm-video-cache",
        "--prewarm-input",
        dest="prewarm_video_cache",
        action="store_true",
        default=os.environ.get("PREWARM_VIDEO_CACHE", "0") == "1",
    )
    parser.add_argument("--prewarm-timeout", type=float, default=90.0)
    parser.add_argument("--audio-path", default=os.environ.get("AUDIO_PATH", ""))
    parser.add_argument(
        "--audio-sampling-rate",
        type=int,
        default=int(os.environ.get("AUDIO_SAMPLING_RATE", "16000")),
    )
    parser.add_argument(
        "--audio-window-seconds",
        type=float,
        default=float(os.environ.get("AUDIO_WINDOW_SECONDS", "0.6")),
    )
    parser.add_argument(
        "--inline-audio-bytes",
        action="store_true",
        default=os.environ.get("INLINE_AUDIO_BYTES", "0") == "1",
    )
    parser.add_argument("--dummy-image-height", type=int, default=1)
    parser.add_argument("--dummy-image-width", type=int, default=1)
    parser.add_argument(
        "--action-timeout",
        type=float,
        default=float(os.environ.get("MOVENS_ACTION_TIMEOUT", "4.0")),
    )
    parser.add_argument(
        "--action-idle-timeout-seconds",
        type=float,
        default=float(os.environ.get("MOVENS_ACTION_IDLE_TIMEOUT", "10.0")),
    )
    parser.add_argument(
        "--joint-limit-margin-rad",
        type=float,
        default=float(os.environ.get("MOVENS_JOINT_LIMIT_MARGIN_RAD", "0.001")),
    )
    parser.add_argument(
        "--max-reference-joint-velocity",
        type=float,
        default=float(os.environ.get("MOVENS_MAX_REFERENCE_JOINT_VELOCITY", "0.0")),
        help="Optional reference-velocity rejection threshold in rad/s; 0 disables it like v18.",
    )
    parser.add_argument(
        "--max-first-joint-jump-rad",
        type=float,
        default=float(os.environ.get("MOVENS_MAX_FIRST_JOINT_JUMP_RAD", "0.35")),
    )
    parser.add_argument(
        "--min-reference-root-height",
        type=float,
        default=float(os.environ.get("MOVENS_MIN_REFERENCE_ROOT_HEIGHT", "0.20")),
    )
    parser.add_argument(
        "--max-reference-root-height",
        type=float,
        default=float(os.environ.get("MOVENS_MAX_REFERENCE_ROOT_HEIGHT", "1.50")),
    )
    parser.add_argument(
        "--generated-joint-pos-csv", default=str(DEFAULT_GENERATED_JOINT_POS_CSV)
    )
    parser.add_argument("--timing-log-interval", type=float, default=1.0)
    parser.add_argument(
        "--orchestrator-managed-handoff",
        action="store_true",
        help="Let the jog parent atomically switch WBC from ZMQ back to planner after a clean stream.",
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args()


def flag_exists(path: str) -> bool:
    return bool(path) and Path(path).exists()


def write_flag(path: str, text: str) -> None:
    if not path:
        return
    flag = Path(path)
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text(text, encoding="utf-8")


def wait_start_flag(start_flag: str, stop_flag: str) -> None:
    print(f"[bridge-real] waiting start flag: {start_flag}", flush=True)
    while True:
        if flag_exists(stop_flag):
            raise SystemExit("[bridge-real] stop flag received before start")
        if flag_exists(start_flag):
            print("[bridge-real] start flag received", flush=True)
            return
        time.sleep(0.1)


def init_generated_joint_csv(path: str) -> Path | None:
    if not path:
        return None
    csv_path = Path(path).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join((f"joint_{i}" for i in range(29)))
    csv_path.write_text(header + "\n", encoding="utf-8")
    print(f"[record] generated joint_pos csv: {csv_path}", flush=True)
    return csv_path


def append_generated_joint_csv(path: Path | None, joint_pos: np.ndarray) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as f:
        np.savetxt(
            f, np.asarray(joint_pos, dtype=np.float32), delimiter=",", fmt="%.6f"
        )


def append_stationary_tail(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    body_quat: np.ndarray,
    frame_indices: np.ndarray,
    tail_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = int(tail_frames)
    if count < 0:
        raise ValueError("tail padding frame count must be non-negative")
    if count == 0:
        return (joint_pos, joint_vel, body_quat, frame_indices)
    if len(frame_indices) <= 0:
        raise ValueError("cannot append stationary padding to an empty pose packet")
    padding_joint_pos = np.repeat(joint_pos[-1:], count, axis=0)
    padding_joint_vel = np.zeros_like(padding_joint_pos)
    padding_body_quat = np.repeat(body_quat[-1:], count, axis=0)
    padding_frame_indices = np.arange(
        int(frame_indices[-1]) + 1, int(frame_indices[-1]) + 1 + count, dtype=np.int64
    )
    return (
        np.concatenate([joint_pos, padding_joint_pos], axis=0),
        np.concatenate([joint_vel, padding_joint_vel], axis=0),
        np.concatenate([body_quat, padding_body_quat], axis=0),
        np.concatenate([frame_indices, padding_frame_indices], axis=0),
    )


def validate_action_chunk(
    actions: np.ndarray, args: argparse.Namespace, meta: dict
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    expected_shape = (int(args.pose_chunk_frames), WIRE_ACTION_DIM)
    if actions.shape != expected_shape:
        raise ValueError(
            f"v26 real deployment requires a full 36D absolute-pose RTC chunk: expected={expected_shape}, got={actions.shape}. Check the server checkpoint/run_config."
        )
    if not np.isfinite(actions).all():
        raise FloatingPointError("Movens server returned non-finite action values")
    expected_protocol = {
        "protocol_version": "movens-root40-v1",
        "action_horizon": int(args.pose_chunk_frames),
        "action_exec_horizon": int(args.pose_chunk_advance_frames),
        "rtc_overlap": int(args.pose_chunk_frames)
        - int(args.pose_chunk_advance_frames),
        "action_dim": WIRE_ACTION_DIM,
    }
    for key, expected in expected_protocol.items():
        if key not in meta:
            raise ValueError(
                f"Movens server response is missing required RTC metadata {key!r}"
            )
        actual = meta[key] if isinstance(expected, str) else int(meta[key])
        if actual != expected:
            raise ValueError(
                f"Movens RTC protocol mismatch for {key}: expected={expected!r}, got={actual!r}"
            )
    if "overlap_exact" not in meta or not isinstance(meta["overlap_exact"], bool):
        raise ValueError(
            "Movens server response is missing required boolean RTC metadata 'overlap_exact'"
        )
    if "chunk_start_frame" not in meta:
        raise ValueError(
            "Movens server response is missing required RTC metadata 'chunk_start_frame'"
        )
    chunk_start_frame = int(meta["chunk_start_frame"])
    if (
        chunk_start_frame < 0
        or chunk_start_frame % int(args.pose_chunk_advance_frames) != 0
    ):
        raise ValueError(
            f"Movens server returned an invalid global chunk start: start={chunk_start_frame}, advance={args.pose_chunk_advance_frames}"
        )
    controller_t = int(meta.get("controller_t", -1))
    if not 0 <= controller_t < expected_protocol["rtc_overlap"]:
        raise ValueError(
            f"Movens server published a chunk after exhausting its overlap: controller_t={controller_t}, overlap={expected_protocol['rtc_overlap']}"
        )
    return actions


def next_reference_deadline(previous_deadline, actual_send, advance_frames):
    if advance_frames <= 0:
        raise ValueError("Reference advance must be positive")
    anchor = actual_send if previous_deadline is None else previous_deadline
    return anchor + advance_frames / float(CTRL_HZ)


def validate_and_record_chunk(
    state_chunk,
    live_state,
    previous_joint,
    first_chunk,
    args,
    metadata=None,
    snapshot_dir=Path("/tmp/movens_real_logs"),
):
    comparison = np.asarray(live_state[:29] if first_chunk else previous_joint).copy()
    try:
        return validate_absolute_state_chunk(state_chunk, comparison, first_chunk, args)
    except (ValueError, FloatingPointError) as exc:
        try:
            snapshot_dir = Path(snapshot_dir)
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            path = snapshot_dir / f"rejected_chunk_{time.time_ns()}.npz"
            temporary = path.with_suffix(".tmp")
            import json

            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    state_chunk=np.asarray(state_chunk),
                    live_state=np.asarray(live_state),
                    previous_joint=np.asarray(previous_joint),
                    comparison_joint=comparison,
                    first_chunk=np.asarray(first_chunk),
                    error=np.asarray(str(exc)),
                    metadata_json=np.asarray(json.dumps(metadata or {}, default=str)),
                    limits_json=np.asarray(
                        json.dumps(
                            {
                                k: getattr(args, k)
                                for k in (
                                    "joint_limit_margin_rad",
                                    "max_first_joint_jump_rad",
                                    "max_reference_joint_velocity",
                                    "min_reference_root_height",
                                    "max_reference_root_height",
                                )
                            }
                        )
                    ),
                )
            os.replace(temporary, path)
            print(f"[bridge-real] rejected input snapshot: {path}", flush=True)
        except Exception as record_error:
            print(f"[bridge-real] snapshot write failed: {record_error}", flush=True)
        raise


def validate_absolute_state_chunk(
    state_chunk: np.ndarray,
    prev_joint: np.ndarray,
    first_chunk: bool,
    args: argparse.Namespace,
) -> np.ndarray:
    states = np.asarray(state_chunk, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 36 or states.shape[0] <= 0:
        raise ValueError(
            f"recovered WBC state chunk must have shape (N, 36), got {states.shape}"
        )
    if not np.isfinite(states).all():
        raise FloatingPointError("recovered WBC state contains non-finite values")
    margin = float(args.joint_limit_margin_rad)
    if margin < 0.0:
        raise ValueError("--joint-limit-margin-rad must be non-negative")
    joint_pos = states[:, :29]
    low_violation = joint_pos < G1_JOINT_LOWER[None, :] - margin
    high_violation = joint_pos > G1_JOINT_UPPER[None, :] + margin
    if np.any(low_violation | high_violation):
        frame, joint = np.argwhere(low_violation | high_violation)[0]
        raise ValueError(
            f"predicted joint reference violates the G1 URDF limit: frame={int(frame)} joint={int(joint)}({G1_JOINT_NAMES[int(joint)]}) value={joint_pos[frame, joint]:.4f} allowed_range=[{G1_JOINT_LOWER[joint] - margin:.4f},{G1_JOINT_UPPER[joint] + margin:.4f}]"
        )
    prev_joint = np.asarray(prev_joint, dtype=np.float32).reshape(-1)
    if prev_joint.shape != (29,) or not np.isfinite(prev_joint).all():
        raise ValueError(
            f"previous joint state must be finite 29D, got {prev_joint.shape}"
        )
    jump_joint = int(np.argmax(np.abs(joint_pos[0] - prev_joint)))
    first_jump = float(np.abs(joint_pos[0, jump_joint] - prev_joint[jump_joint]))
    if first_chunk and first_jump > float(args.max_first_joint_jump_rad):
        raise ValueError(
            f"first predicted pose is discontinuous from the live G1 pose: max_joint_jump={first_jump:.4f}rad > {args.max_first_joint_jump_rad:.4f}rad joint={jump_joint}({G1_JOINT_NAMES[jump_joint]}) live={prev_joint[jump_joint]:.4f} predicted={joint_pos[0, jump_joint]:.4f}"
        )
    joint_velocity = np.empty_like(joint_pos)
    if first_chunk:
        joint_velocity[0] = 0.0
    else:
        joint_velocity[0] = (joint_pos[0] - prev_joint) / CTRL_DT
    if joint_pos.shape[0] > 1:
        joint_velocity[1:] = np.diff(joint_pos, axis=0) / CTRL_DT
    velocity_limits = G1_MECHANICAL_VELOCITY_LIMIT.copy()
    if float(args.max_reference_joint_velocity) > 0.0:
        velocity_limits = np.minimum(
            float(args.max_reference_joint_velocity) * G1_REFERENCE_VELOCITY_SCALE,
            velocity_limits,
        )
    velocity_ratio = np.abs(joint_velocity) / velocity_limits[None, :]
    if float(np.max(velocity_ratio)) > 1.0:
        frame, joint = np.unravel_index(np.argmax(velocity_ratio), velocity_ratio.shape)
        raise ValueError(
            f"predicted joint reference velocity exceeds the real-deploy envelope: frame={int(frame)} joint={G1_JOINT_NAMES[int(joint)]} velocity={joint_velocity[frame, joint]:.3f}rad/s limit={velocity_limits[joint]:.3f}rad/s"
        )
    root_height = states[:, 31]
    if np.any(root_height < args.min_reference_root_height) or np.any(
        root_height > args.max_reference_root_height
    ):
        frame = int(
            np.argwhere(
                (root_height < args.min_reference_root_height)
                | (root_height > args.max_reference_root_height)
            )[0, 0]
        )
        raise ValueError(
            f"predicted root height is outside the real-deploy envelope: frame={frame} z={root_height[frame]:.4f}m range=[{args.min_reference_root_height:.3f},{args.max_reference_root_height:.3f}]m"
        )
    quat_norm = np.linalg.norm(states[:, 32:36], axis=1)
    if np.max(np.abs(quat_norm - 1.0)) > 0.001:
        raise ValueError("recovered body quaternion is not unit length")
    return joint_velocity.astype(np.float32)


def load_inline_video_payload(video_path: Path | None, enabled: bool) -> dict | None:
    if not enabled:
        return None
    if video_path is None:
        raise ValueError("--inline-video-bytes requires --inference-video-path")
    raw = video_path.read_bytes()
    max_bytes = int(
        os.environ.get("MOVENS_INLINE_VIDEO_MAX_BYTES", str(384 * 1024 * 1024))
    )
    if len(raw) > max_bytes:
        raise ValueError(
            f"inference video is too large for inline transfer: {len(raw)} bytes > {max_bytes}"
        )
    digest = hashlib.sha256(raw).hexdigest()
    payload = {
        "video_inline_id": digest,
        "video_sha256": digest,
        "video_filename": video_path.name,
        "video_bytes_b64": b64encode(raw).decode("ascii"),
    }
    print(
        f"[video-real] inline video enabled: bytes={len(raw)} sha256={digest[:12]}",
        flush=True,
    )
    return payload


def ffmpeg_bin() -> str:
    return os.environ.get("MOVENS_FFMPEG_BIN", "ffmpeg")


def scaled_video_size(
    width: int, height: int, video_size_pixels: int
) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid remote camera source size: {width}x{height}")
    if video_size_pixels <= 0:
        return (width, height)
    scale = math.sqrt(float(video_size_pixels) / float(max(1, width * height)))
    new_w = max(2, int(round(width * scale / 2.0)) * 2)
    new_h = max(2, int(round(height * scale / 2.0)) * 2)
    return (new_w, new_h)


def read_exact(stream, size: int) -> bytes | None:
    chunks = []
    remaining = int(size)
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class RemoteCameraStreamClipProvider:
    def __init__(self, args: argparse.Namespace) -> None:
        self.remote_host = str(args.remote_camera_host)
        self.remote_port = int(args.remote_camera_port)
        self.source_label = (
            f"remote_camera_stream:{self.remote_host}:{self.remote_port}"
        )
        self.sample_fps = float(args.video_sample_fps)
        self.duration_sec = float(args.pose_chunk_frames) / float(
            args.video_clip_motion_fps
        )
        self.clip_frames = max(1, int(math.ceil(self.duration_sec * self.sample_fps)))
        self.out_w, self.out_h = scaled_video_size(
            int(args.camera_width), int(args.camera_height), int(args.video_size_pixels)
        )
        self.frame_size = self.out_h * self.out_w * 3
        buffer_frames = max(
            self.clip_frames + int(math.ceil(self.sample_fps * 5.0)),
            self.clip_frames + 8,
        )
        self.frames: deque[tuple[int, float, np.ndarray]] = deque(maxlen=buffer_frames)
        self.lock = threading.Condition()
        self.stop_event = threading.Event()
        self.process: subprocess.Popen | None = None
        self.listener: socket.socket | None = None
        self.reader_thread: threading.Thread | None = None
        self.stderr_thread: threading.Thread | None = None
        self.remote_forward_thread: threading.Thread | None = None
        self.seq = 0
        self.error: str | None = None
        self.cached_cursor: int | None = None
        self.cached_clip: tuple[str, np.ndarray, int, int, float] | None = None
        self.cmd = [
            ffmpeg_bin(),
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            "pipe:0",
            "-an",
            "-vf",
            f"fps={self.sample_fps},scale={self.out_w}:{self.out_h}",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ]

    def start(self) -> None:
        print(
            f"[camera-real] starting source={self.source_label} sample_fps={self.sample_fps} clip={self.duration_sec:.3f}s/{self.clip_frames}frames out={self.out_w}x{self.out_h}",
            flush=True,
        )
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((self.remote_host, self.remote_port))
        self.listener.listen(1)
        self.listener.settimeout(0.2)
        print(
            f"[camera-real] waiting for remote stream on {self.remote_host}:{self.remote_port}",
            flush=True,
        )
        try:
            self.process = subprocess.Popen(
                self.cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except Exception:
            self.listener.close()
            self.listener = None
            raise
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self.remote_forward_thread = threading.Thread(
            target=self._remote_forward_loop, daemon=True
        )
        self.reader_thread.start()
        self.stderr_thread.start()
        self.remote_forward_thread.start()

    def _set_error(self, message: str) -> None:
        self.error = message
        with self.lock:
            self.lock.notify_all()

    def _reader_loop(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        while not self.stop_event.is_set():
            raw = read_exact(self.process.stdout, self.frame_size)
            if raw is None:
                if not self.stop_event.is_set():
                    code = self.process.poll()
                    self._set_error(
                        "camera ffmpeg stopped"
                        if code is None
                        else f"camera ffmpeg exited code={code}"
                    )
                return
            frame = (
                np.frombuffer(raw, dtype=np.uint8)
                .reshape(self.out_h, self.out_w, 3)
                .copy()
            )
            with self.lock:
                self.seq += 1
                self.frames.append((self.seq, time.perf_counter(), frame))
                self.lock.notify_all()

    def _stderr_loop(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for raw_line in self.process.stderr:
            if self.stop_event.is_set():
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line:
                print(f"[camera-real] ffmpeg: {line}", flush=True)

    def _remote_forward_loop(self) -> None:
        assert self.process is not None
        assert self.process.stdin is not None
        assert self.listener is not None
        conn = None
        try:
            while not self.stop_event.is_set():
                try:
                    conn, addr = self.listener.accept()
                    print(
                        f"[camera-real] remote stream connected: {addr[0]}:{addr[1]}",
                        flush=True,
                    )
                    break
                except socket.timeout:
                    continue
            if conn is None:
                return
            with conn:
                while not self.stop_event.is_set():
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    self.process.stdin.write(chunk)
            if not self.stop_event.is_set():
                print("[camera-real] remote stream ended", flush=True)
        except Exception as exc:
            if not self.stop_event.is_set():
                self._set_error(f"remote camera stream failed: {exc}")
        finally:
            try:
                self.process.stdin.close()
            except Exception:
                pass

    def wait_ready(self, timeout: float) -> None:
        deadline = time.time() + float(timeout)
        with self.lock:
            while len(self.frames) < self.clip_frames:
                if self.error:
                    raise RuntimeError(self.error)
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"remote camera warmup timed out: got {len(self.frames)}/{self.clip_frames} frames"
                    )
                self.lock.wait(timeout=min(0.1, remaining))
        print(f"[camera-real] ready frames={len(self.frames)}", flush=True)

    def get_clip(self, frame_cursor: int) -> tuple[str, np.ndarray, int, int, float]:
        cursor = int(frame_cursor)
        with self.lock:
            if self.cached_cursor == cursor and self.cached_clip is not None:
                return self.cached_clip
            if self.error and len(self.frames) < self.clip_frames:
                raise RuntimeError(self.error)
            if len(self.frames) < self.clip_frames:
                raise RuntimeError(
                    f"remote camera has only {len(self.frames)}/{self.clip_frames} frames"
                )
            selected = list(self.frames)[-self.clip_frames :]
            first_seq = selected[0][0]
            last_seq = selected[-1][0]
            last_perf = selected[-1][1]
            video = np.stack([item[2] for item in selected], axis=0)
        key_text = f"{self.source_label}|{cursor}|{first_seq}|{last_seq}|{last_perf:.6f}|{video.shape}"
        clip_id = hashlib.sha256(key_text.encode("utf-8")).hexdigest()
        self.cached_cursor = cursor
        self.cached_clip = (clip_id, video, first_seq, last_seq, last_perf)
        return self.cached_clip

    def close(self) -> None:
        self.stop_event.set()
        try:
            if self.listener is not None:
                self.listener.close()
        except Exception:
            pass
        proc = self.process
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def encode_camera_clip_frames(video: np.ndarray) -> str:
    buffer = BytesIO()
    np.save(buffer, np.asarray(video, dtype=np.uint8), allow_pickle=False)
    compressed = zlib.compress(buffer.getvalue(), level=1)
    return b64encode(compressed).decode("ascii")


def build_remote_camera_condition(
    args: argparse.Namespace,
    camera_provider: RemoteCameraStreamClipProvider | None,
    frame_cursor: int,
    sent_clip_ids: set[str] | None,
) -> dict:
    if camera_provider is None:
        raise RuntimeError("remote camera provider is not initialized")
    clip_id, video, first_seq, last_seq, last_perf = camera_provider.get_clip(
        frame_cursor
    )
    condition = {
        "input_mode": "remote_camera_stream",
        "camera_clip_id": clip_id,
        "camera_clip_source": camera_provider.source_label,
        "camera_clip_capture_first_seq": int(first_seq),
        "camera_clip_capture_last_seq": int(last_seq),
        "camera_clip_capture_age_ms": max(
            0.0, (time.perf_counter() - last_perf) * 1000.0
        ),
        "video_config_key": str(args.video_config_key),
        "video_clip_start_sec": float(frame_cursor) / float(args.video_clip_motion_fps),
        "video_clip_duration_sec": float(args.pose_chunk_frames)
        / float(args.video_clip_motion_fps),
        "video_sample_fps": float(args.video_sample_fps),
        "video_size_pixels": int(args.video_size_pixels),
        "video_decoder": "remote_camera_stream",
        "video_clip_motion_fps": float(args.video_clip_motion_fps),
    }
    if sent_clip_ids is None or clip_id not in sent_clip_ids:
        condition["camera_clip_codec"] = "npy.zlib"
        condition["camera_clip_shape"] = list(video.shape)
        condition["camera_clip_frames_b64"] = encode_camera_clip_frames(video)
        if sent_clip_ids is not None:
            sent_clip_ids.add(clip_id)
    return condition


def load_inline_audio_payload(audio_path: Path | None, enabled: bool) -> dict | None:
    if not enabled:
        return None
    if audio_path is None:
        raise ValueError("--inline-audio-bytes requires --audio-path")
    raw = audio_path.read_bytes()
    max_bytes = int(
        os.environ.get("MOVENS_INLINE_AUDIO_MAX_BYTES", str(64 * 1024 * 1024))
    )
    if len(raw) > max_bytes:
        raise ValueError(
            f"input audio is too large for inline transfer: {len(raw)} bytes > {max_bytes}"
        )
    digest = hashlib.sha256(raw).hexdigest()
    print(
        f"[audio-real] inline audio enabled: bytes={len(raw)} sha256={digest[:12]}",
        flush=True,
    )
    return {
        "audio_inline_id": digest,
        "audio_sha256": digest,
        "audio_filename": audio_path.name,
        "audio_bytes_b64": b64encode(raw).decode("ascii"),
    }


def build_input_condition(
    args: argparse.Namespace,
    video_path: Path | None,
    audio_path: Path | None,
    frame_cursor: int,
    inline_video_payload: dict | None,
    inline_audio_payload: dict | None,
    include_inline_bytes: bool,
    sent_camera_clip_ids: set[str] | None = None,
    camera_provider: RemoteCameraStreamClipProvider | None = None,
) -> dict:
    mode = str(args.visual_input_mode)
    if mode == "remote_camera_stream":
        return build_remote_camera_condition(
            args, camera_provider, frame_cursor, sent_camera_clip_ids
        )
    if mode == "video_clip":
        return build_video_condition(
            args, video_path, frame_cursor, inline_video_payload, include_inline_bytes
        ) or {"input_mode": mode}
    if mode == "text":
        return {"input_mode": mode}
    if audio_path is None:
        raise ValueError("audio deployment requires --audio-path")
    full_audio = mode == "audio"
    condition = {
        "input_mode": mode,
        "audio_path": str(audio_path),
        "audio_sampling_rate": int(args.audio_sampling_rate),
        "audio_start_sec": 0.0 if full_audio else frame_cursor / float(CTRL_HZ),
        "audio_duration_sec": 0.0 if full_audio else float(args.audio_window_seconds),
    }
    if inline_audio_payload is not None:
        for key in ("audio_inline_id", "audio_sha256", "audio_filename"):
            condition[key] = inline_audio_payload[key]
        if include_inline_bytes:
            condition["audio_bytes_b64"] = inline_audio_payload["audio_bytes_b64"]
    return condition


def build_video_condition(
    args: argparse.Namespace,
    video_path: Path | None,
    video_frame_cursor: int,
    inline_video_payload: dict | None,
    include_inline_bytes: bool,
) -> dict | None:
    if video_path is None:
        return None
    condition = {
        "input_mode": str(args.visual_input_mode),
        "video_path": str(video_path),
        "video_config_key": str(args.video_config_key),
        "video_clip_start_sec": float(video_frame_cursor)
        / float(args.video_clip_motion_fps),
        "video_clip_duration_sec": float(args.pose_chunk_frames)
        / float(args.video_clip_motion_fps),
        "video_sample_fps": float(args.video_sample_fps),
        "video_size_pixels": int(args.video_size_pixels),
        "video_decoder": str(args.video_decoder),
        "video_clip_motion_fps": float(args.video_clip_motion_fps),
    }
    if inline_video_payload is not None:
        condition["video_inline_id"] = inline_video_payload["video_inline_id"]
        condition["video_sha256"] = inline_video_payload["video_sha256"]
        condition["video_filename"] = inline_video_payload["video_filename"]
        if include_inline_bytes:
            condition["video_bytes_b64"] = inline_video_payload["video_bytes_b64"]
    return condition


def prewarm_video_cache(
    args: argparse.Namespace,
    ws_url: str,
    prompt_state: dict,
    state: np.ndarray,
    dummy_image: np.ndarray,
    video_path: Path | None,
    inline_video_payload: dict | None,
    audio_path: Path | None = None,
    inline_audio_payload: dict | None = None,
    camera_provider: RemoteCameraStreamClipProvider | None = None,
) -> bool:
    if not args.prewarm_video_cache:
        return False
    start = time.perf_counter()
    prewarm_ws = None
    include_inline_bytes = (
        inline_video_payload is not None or inline_audio_payload is not None
    )
    try:
        prewarm_ws = websocket.create_connection(ws_url, timeout=30)
        prewarm_ws.settimeout(max(30.0, float(args.prewarm_timeout)))
        input_condition = build_input_condition(
            args,
            video_path,
            audio_path,
            int(prompt_state["video_frame_cursor"]),
            inline_video_payload,
            inline_audio_payload,
            include_inline_bytes,
            set(),
            camera_provider,
        )
        prewarm_ws.send(
            build_request_payload(
                state,
                str(prompt_state["prompt"]),
                dummy_image,
                int(prompt_state["segment_target_frames"]),
                int(prompt_state["segment_index"]),
                int(prompt_state["segment_elapsed_frames"]),
                input_condition,
            )
        )
        prewarm_ws.settimeout(1.0)
        deadline = time.time() + float(args.prewarm_timeout)
        while time.time() < deadline:
            try:
                message = prewarm_ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            actions, meta = decode_response_message(message)
            if actions is None:
                continue
            actions = validate_action_chunk(actions, args, meta)
            print(
                f"[{args.visual_input_mode}-real] prewarm complete: latency={time.perf_counter() - start:.3f}s action_shape={actions.shape} server_version={meta.get('version')} chunk_version={meta.get('chunk_version')}",
                flush=True,
            )
            return True
        raise TimeoutError(f"input prewarm timed out after {args.prewarm_timeout:.1f}s")
    except Exception as exc:
        print(f"[{args.visual_input_mode}-real] prewarm failed: {exc}", flush=True)
        return False
    finally:
        try:
            if prewarm_ws is not None:
                prewarm_ws.close()
        except Exception:
            pass


def load_initial_state_real(args: argparse.Namespace) -> np.ndarray:
    if args.init_state_npy or args.init_motion_dir:
        return load_initial_state(args)
    live_state_path = (
        Path(args.live_state_npy).expanduser() if args.live_state_npy else None
    )
    if live_state_path is not None and live_state_path.exists():
        print(f"[state-real] init from live state: {live_state_path}", flush=True)
        return load_state_npy(live_state_path)
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser()
        if (
            run_dir.exists()
            and (run_dir / "argv.txt").exists()
            and (run_dir / "run_config.json").exists()
        ):
            return load_initial_state(args)
    state = np.zeros(36, dtype=np.float32)
    state[31] = float(os.environ.get("G1_ROOT_Z_FALLBACK", "0.78"))
    state[32] = 1.0
    print(
        "[state-real] no init motion/run metadata; using neutral 36D seed", flush=True
    )
    return state


def main() -> int:
    args = parse_args()
    args.visual_input_mode = str(args.visual_input_mode).strip().lower()
    if int(args.pose_chunk_frames) != 30 or int(args.pose_chunk_advance_frames) != 20:
        raise ValueError(
            f"this v26 WBC bridge requires pose chunk/advance 30/20 to match the target RTC contract, got {args.pose_chunk_frames}/{args.pose_chunk_advance_frames}"
        )
    if args.pose_chunk_send_seconds < 0.0:
        raise ValueError("--pose-chunk-send-seconds must be non-negative")
    if args.tail_padding_frames < 0 or args.tail_padding_hold_seconds < 0.0:
        raise ValueError("tail-padding frames/hold seconds must be non-negative")
    if args.live_state_max_age <= 0.0 or args.action_timeout <= 0.0:
        raise ValueError("live-state and action timeouts must be positive")
    chunk_period = float(args.pose_chunk_advance_frames) / CTRL_HZ
    chunk_coverage = float(args.pose_chunk_frames) / CTRL_HZ
    if args.action_timeout <= chunk_period:
        raise ValueError(
            f"--action-timeout must be greater than the 0.4s chunk period: period={chunk_period:.3f}s timeout={args.action_timeout:.3f}s coverage={chunk_coverage:.3f}s"
        )
    if args.action_idle_timeout_seconds <= 0.0:
        raise ValueError("--action-idle-timeout-seconds must be positive")
    if args.max_reference_joint_velocity < 0.0:
        raise ValueError(
            "--max-reference-joint-velocity must be non-negative (0 disables it)"
        )
    if args.max_first_joint_jump_rad <= 0.0:
        raise ValueError("--max-first-joint-jump-rad must be positive")
    if not 0.0 < args.min_reference_root_height < args.max_reference_root_height:
        raise ValueError("root-height safety envelope must be positive and increasing")
    ws_url = f"ws://{args.movens_host}:{args.movens_port}/ws"
    prompt = args.prompt.strip()
    if args.visual_input_mode == "text" and (not prompt):
        raise ValueError("text deployment requires a non-empty --prompt")
    if args.audio_sampling_rate <= 0 or args.audio_window_seconds <= 0.0:
        raise ValueError("audio sampling rate/window must be positive")
    if args.visual_input_mode == "remote_camera_stream":
        if args.camera_width <= 0 or args.camera_height <= 0:
            raise ValueError("remote camera width/height must be positive")
        if not 0 < args.remote_camera_port <= 65535:
            raise ValueError("remote camera port must be in 1..65535")
        if args.remote_camera_warmup_timeout <= 0.0:
            raise ValueError("remote camera warmup timeout must be positive")
    prompt_plan = (
        load_motion_prompt_plan(args.motion_plan_json, CTRL_HZ)
        if args.motion_plan_json
        else []
    )
    if prompt_plan:
        for item in prompt_plan[:-1]:
            segment_frames = int(item["frames"])
            if segment_frames % int(args.pose_chunk_advance_frames) != 0:
                raise ValueError(
                    f"every non-final motion-plan segment must align to the RTC execute horizon: segment={item['index']} frames={segment_frames} advance={args.pose_chunk_advance_frames}"
                )
        total_plan_frames = sum((int(item["frames"]) for item in prompt_plan))
        if args.target_frames <= 0:
            args.target_frames = total_plan_frames
        print(
            f"[planner-real] loaded {len(prompt_plan)} segments / {total_plan_frames} frames",
            flush=True,
        )
        for item in prompt_plan:
            print(
                f"[planner-real] segment {item['index']}: {item['frames']} frames - {item['low_level_instruction']}",
                flush=True,
            )
    else:
        total_plan_frames = 0
    target_frames = int(args.target_frames)
    if target_frames <= 0 and args.max_seconds > 0:
        target_frames = int(round(args.max_seconds * CTRL_HZ))
    prompt_lock = threading.Lock()
    prompt_state = {
        "prompt": prompt_plan[0]["prompt"] if prompt_plan else prompt,
        "segment_index": int(prompt_plan[0]["index"]) if prompt_plan else 0,
        "segment_target_frames": int(prompt_plan[0]["frames"])
        if prompt_plan
        else int(target_frames),
        "segment_elapsed_frames": 0,
        "video_frame_cursor": 0,
    }

    def update_prompt_state(sent_frames: int) -> None:
        if not prompt_plan:
            return
        cursor = 0
        selected = prompt_plan[-1]
        selected_start = max(0, total_plan_frames - int(selected["frames"]))
        for item in prompt_plan:
            next_cursor = cursor + int(item["frames"])
            if sent_frames < next_cursor:
                selected = item
                selected_start = cursor
                break
            cursor = next_cursor
        segment_elapsed = min(
            max(0, sent_frames - selected_start), int(selected["frames"])
        )
        with prompt_lock:
            old_index = int(prompt_state["segment_index"])
            prompt_state["prompt"] = selected["prompt"]
            prompt_state["segment_index"] = int(selected["index"])
            prompt_state["segment_target_frames"] = int(selected["frames"])
            prompt_state["segment_elapsed_frames"] = int(segment_elapsed)
        if old_index != int(selected["index"]):
            print(
                f"[planner-real] switch to segment {selected['index']}: {selected['low_level_instruction']}",
                flush=True,
            )

    video_path = (
        Path(args.inference_video_path).expanduser()
        if args.visual_input_mode == "video_clip" and args.inference_video_path
        else None
    )
    audio_path = (
        Path(args.audio_path).expanduser()
        if args.visual_input_mode in {"music", "audio"} and args.audio_path
        else None
    )
    if args.visual_input_mode == "video_clip" and (
        args.require_local_video or args.inline_video_bytes
    ):
        if video_path is None:
            raise ValueError("video_clip deployment requires --inference-video-path")
        if not video_path.exists():
            raise FileNotFoundError(f"inference video does not exist: {video_path}")
    if video_path is not None:
        print(
            f"[video-real] video_clip input: path={video_path} sample_fps={args.video_sample_fps} size_pixels={args.video_size_pixels} motion_fps={args.video_clip_motion_fps}",
            flush=True,
        )
    if args.visual_input_mode in {"music", "audio"}:
        if audio_path is None:
            raise ValueError(
                f"{args.visual_input_mode} deployment requires --audio-path"
            )
        if not audio_path.is_file():
            raise FileNotFoundError(f"input audio does not exist: {audio_path}")
        print(
            f"[{args.visual_input_mode}-real] input: path={audio_path} sampling_rate={args.audio_sampling_rate}Hz "
            + (
                f"window={args.audio_window_seconds:.3f}s"
                if args.visual_input_mode == "music"
                else "full waveform"
            ),
            flush=True,
        )
    camera_provider = None
    if args.visual_input_mode == "remote_camera_stream":
        camera_provider = RemoteCameraStreamClipProvider(args)
        try:
            camera_provider.start()
            camera_provider.wait_ready(args.remote_camera_warmup_timeout)
        except Exception:
            camera_provider.close()
            raise
    inline_video_payload = load_inline_video_payload(
        video_path, args.visual_input_mode == "video_clip" and args.inline_video_bytes
    )
    inline_audio_payload = load_inline_audio_payload(
        audio_path,
        args.visual_input_mode in {"music", "audio"} and args.inline_audio_bytes,
    )
    if (
        args.prewarm_video_cache
        and args.visual_input_mode == "video_clip"
        and (video_path is not None)
        and (inline_video_payload is None)
    ):
        raise ValueError(
            "--prewarm-video-cache requires --inline-video-bytes for split-machine deployment; otherwise the prewarm connection cannot establish a reusable server-side video mapping"
        )
    if (
        args.prewarm_video_cache
        and args.visual_input_mode in {"music", "audio"}
        and (inline_audio_payload is None)
    ):
        raise ValueError(
            "--prewarm-input requires --inline-audio-bytes for split-machine audio deployment"
        )
    inline_input_sent = False
    sent_camera_clip_ids: set[str] = set()
    print(
        f"[stream-real] chunk_frames={args.pose_chunk_frames} advance_frames={args.pose_chunk_advance_frames} coverage={args.pose_chunk_advance_frames / CTRL_HZ:.3f}s target_frames={target_frames}",
        flush=True,
    )
    dummy_image = np.zeros(
        (args.dummy_image_height, args.dummy_image_width, 3), dtype=np.uint8
    )
    generated_joint_csv = init_generated_joint_csv(args.generated_joint_pos_csv)
    try:
        current_state = load_initial_state_real(args)
        if prewarm_video_cache(
            args,
            ws_url,
            prompt_state,
            current_state,
            dummy_image,
            video_path,
            inline_video_payload,
            audio_path,
            inline_audio_payload,
            camera_provider,
        ):
            inline_input_sent = True
        elif args.prewarm_video_cache:
            raise RuntimeError(
                "input prewarm failed; refusing to arm the real bridge without a cached first chunk"
            )
        state_provider = StateProvider(
            args.live_state_npy, require_live=True, max_age=args.live_state_max_age
        )
        current_state = state_provider.wait_first_live(
            current_state, args.live_state_timeout
        )
        prev_joint = current_state[:29].copy()
        wbc = WBCZMQPublisher(args.wbc_zmq_host, args.wbc_zmq_port)
    except Exception:
        if camera_provider is not None:
            camera_provider.close()
        raise
    write_flag(args.ready_flag_path, str(time.time()))
    print("[bridge-real] ready", flush=True)
    ws = None
    stop_event = threading.Event()
    state_lock = threading.Lock()
    action_lock = threading.Lock()
    action_event = threading.Event()
    latest_actions = None
    latest_action_meta = {}
    latest_action_version = 0
    last_action_time = time.time()
    completed_stream = False
    last_sent_end_frame = -1
    previous_actions = None
    try:
        wait_start_flag(args.start_flag_path, args.stop_flag_path)
        ws = websocket.create_connection(ws_url, timeout=30)
        ws.settimeout(30)
        print(f"[movens-real] connected {ws_url}", flush=True)
        for _ in range(5):
            wbc.send_command(start=True, stop=False, planner=False)
            time.sleep(0.05)
        print("[wbc-real] start command sent", flush=True)
        last_action_time = time.time()

        def send_obs_loop() -> None:
            nonlocal inline_input_sent
            next_tick = time.perf_counter()
            obs_last_log = next_tick
            obs_count_since_log = 0
            obs_max_send_ms = 0.0
            obs_max_loop_ms = 0.0
            obs_max_late_ms = 0.0
            while not stop_event.is_set():
                try:
                    loop_start_perf = time.perf_counter()
                    if flag_exists(args.stop_flag_path):
                        stop_event.set()
                        break
                    with state_lock:
                        live_state = state_provider.get(current_state)
                        state = live_state.copy() if args.state_feedback == "live" else compose_policy_state(live_state, current_state)
                    with prompt_lock:
                        active_prompt = str(prompt_state["prompt"])
                        active_target_frames = int(
                            prompt_state["segment_target_frames"]
                        )
                        active_segment_index = int(prompt_state["segment_index"])
                        active_segment_elapsed_frames = int(
                            prompt_state["segment_elapsed_frames"]
                        )
                        active_video_frame_cursor = int(
                            prompt_state["video_frame_cursor"]
                        )
                    include_inline_bytes = not inline_input_sent and (
                        inline_video_payload is not None
                        or inline_audio_payload is not None
                    )
                    input_condition = build_input_condition(
                        args,
                        video_path,
                        audio_path,
                        active_video_frame_cursor,
                        inline_video_payload,
                        inline_audio_payload,
                        include_inline_bytes,
                        sent_camera_clip_ids,
                        camera_provider,
                    )
                    input_condition["client_frame"] = active_video_frame_cursor
                    camera_diag = ""
                    if input_condition.get("camera_clip_capture_last_seq") is not None:
                        camera_diag = f" camera_seq={input_condition.get('camera_clip_capture_first_seq')}..{input_condition.get('camera_clip_capture_last_seq')} camera_age={float(input_condition.get('camera_clip_capture_age_ms', 0.0)):.1f}ms"
                    payload = build_request_payload(
                        state,
                        active_prompt,
                        dummy_image,
                        active_target_frames,
                        active_segment_index,
                        active_segment_elapsed_frames,
                        input_condition,
                    )
                    ws_send_start = time.perf_counter()
                    ws.send(payload)
                    send_ms = (time.perf_counter() - ws_send_start) * 1000.0
                    obs_max_send_ms = max(obs_max_send_ms, send_ms)
                    if include_inline_bytes:
                        inline_input_sent = True
                except Exception as exc:
                    print(f"[movens-real] send stopped: {exc}", flush=True)
                    stop_event.set()
                    break
                next_tick += CTRL_DT
                sleep_time = next_tick - time.perf_counter()
                loop_ms = (time.perf_counter() - loop_start_perf) * 1000.0
                late_ms = max(0.0, -sleep_time * 1000.0)
                obs_count_since_log += 1
                obs_max_loop_ms = max(obs_max_loop_ms, loop_ms)
                obs_max_late_ms = max(obs_max_late_ms, late_ms)
                diag_now = time.perf_counter()
                diag_interval = float(args.timing_log_interval)
                if diag_interval > 0.0 and diag_now - obs_last_log >= diag_interval:
                    elapsed = max(1e-06, diag_now - obs_last_log)
                    live_age = (
                        time.time() - state_provider.last_mtime
                        if state_provider.last_mtime > 0.0
                        else -1.0
                    )
                    print(
                        f"[diag-real] obs hz={obs_count_since_log / elapsed:.1f} input_cursor={active_video_frame_cursor} segment_elapsed={active_segment_elapsed_frames} {camera_diag} live_age={live_age:.3f}s ws_send_max={obs_max_send_ms:.2f}ms loop_max={obs_max_loop_ms:.2f}ms late_max={obs_max_late_ms:.2f}ms",
                        flush=True,
                    )
                    obs_last_log = diag_now
                    obs_count_since_log = 0
                    obs_max_send_ms = 0.0
                    obs_max_loop_ms = 0.0
                    obs_max_late_ms = 0.0
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                else:
                    next_tick = time.perf_counter()

        def recv_action_loop() -> None:
            nonlocal \
                latest_actions, \
                latest_action_meta, \
                latest_action_version, \
                last_action_time
            recv_last_arrival = None
            recv_last_log = time.perf_counter()
            while not stop_event.is_set():
                try:
                    recv_wait_start = time.perf_counter()
                    message = ws.recv()
                    recv_arrival = time.perf_counter()
                except Exception as exc:
                    print(f"[movens-real] recv stopped: {exc}", flush=True)
                    stop_event.set()
                    break
                try:
                    decode_start = time.perf_counter()
                    actions, meta = decode_response_message(message)
                    decode_ms = (time.perf_counter() - decode_start) * 1000.0
                except Exception as exc:
                    print(f"[movens-real] decode stopped: {exc}", flush=True)
                    stop_event.set()
                    break
                if actions is None:
                    continue
                try:
                    actions = validate_action_chunk(actions, args, meta)
                except Exception as exc:
                    print(f"[movens-real] invalid action stopped: {exc}", flush=True)
                    stop_event.set()
                    break
                with action_lock:
                    latest_actions = actions
                    latest_action_meta = dict(meta)
                    latest_action_version += 1
                    last_action_time = time.time()
                if args.latency_log:
                    record = dict(received_wall_time=time.time(), chunk_start_frame=meta.get("chunk_start_frame"),
                                  inference_seconds=meta.get("inference_seconds"), checkpoint_sha256=meta.get("checkpoint_sha256"))
                    with Path(args.latency_log).open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                action_event.set()
                diag_interval = float(args.timing_log_interval)
                action_interval = (
                    -1.0
                    if recv_last_arrival is None
                    else recv_arrival - recv_last_arrival
                )
                recv_last_arrival = recv_arrival
                if diag_interval > 0.0 and (
                    action_interval < 0.0
                    or recv_arrival - recv_last_log >= diag_interval
                ):
                    print(
                        f"[diag-real] action_recv version={meta.get('version')} chunk={meta.get('chunk_version')} interval={action_interval:.3f}s ws_wait={(recv_arrival - recv_wait_start) * 1000.0:.1f}ms decode={decode_ms:.2f}ms shape={actions.shape}",
                        flush=True,
                    )
                    recv_last_log = recv_arrival

        sender = threading.Thread(target=send_obs_loop, daemon=True)
        receiver = threading.Thread(target=recv_action_loop, daemon=True)
        sender.start()
        receiver.start()
        frame_idx = 0
        start_time = None
        start_perf = None
        next_pose_send_perf = None
        last_sent_version = 0
        first_pose_sent = False
        stream_target_frames = (
            total_plan_frames if total_plan_frames > 0 else max(0, target_frames)
        )
        tail_padding_sent = False
        while not stop_event.is_set():
            if flag_exists(args.stop_flag_path):
                print("[bridge-real] stop flag received", flush=True)
                break
            if (
                stream_target_frames <= 0
                and start_time is not None
                and (args.max_seconds > 0)
                and (time.time() - start_time > args.max_seconds)
            ):
                print("[bridge-real] max seconds reached", flush=True)
                completed_stream = True
                break
            if stream_target_frames > 0 and frame_idx >= stream_target_frames:
                print("[bridge-real] target frames reached", flush=True)
                completed_stream = True
                break
            if next_pose_send_perf is not None:
                while not stop_event.is_set():
                    if flag_exists(args.stop_flag_path):
                        print("[bridge-real] stop flag received", flush=True)
                        stop_event.set()
                        break
                    sleep_time = next_pose_send_perf - time.perf_counter()
                    if sleep_time <= 0.0:
                        break
                    time.sleep(min(0.02, sleep_time))
                if stop_event.is_set():
                    break
            if not action_event.wait(timeout=0.2):
                if (
                    frame_idx <= 0
                    and time.time() - last_action_time
                    > args.action_idle_timeout_seconds
                ):
                    raise RuntimeError(
                        f"No initial action received from Movens server for {time.time() - last_action_time:.2f}s"
                    )
                if (
                    frame_idx > 0
                    and time.time() - last_action_time > args.action_timeout
                ):
                    raise RuntimeError(
                        f"Movens action timeout: {time.time() - last_action_time:.3f}s"
                    )
                continue
            with action_lock:
                actions = None if latest_actions is None else latest_actions.copy()
                action_meta = dict(latest_action_meta)
                action_version = latest_action_version
                action_event.clear()
            if actions is None or action_version == last_sent_version:
                continue
            last_sent_version = action_version
            chunk_start_frame = int(action_meta["chunk_start_frame"])
            if chunk_start_frame != frame_idx:
                raise RuntimeError(
                    f"Movens action sequence is discontinuous; refusing to relabel or skip a chunk: expected_start={frame_idx}, server_start={chunk_start_frame}, chunk_version={action_meta.get('chunk_version')}"
                )
            overlap_frames = int(args.pose_chunk_frames) - int(
                args.pose_chunk_advance_frames
            )
            if previous_actions is not None and bool(action_meta["overlap_exact"]):
                if not np.allclose(
                    actions[:overlap_frames],
                    previous_actions[int(args.pose_chunk_advance_frames) :],
                    rtol=1e-05,
                    atol=1e-05,
                ):
                    max_error = float(
                        np.max(
                            np.abs(
                                actions[:overlap_frames]
                                - previous_actions[
                                    int(args.pose_chunk_advance_frames) :
                                ]
                            )
                        )
                    )
                    raise RuntimeError(
                        f"Movens RTC clean-prefix mismatch; refusing a discontinuous action window: max_abs_error={max_error:.6g}"
                    )
            update_prompt_state(frame_idx)
            with state_lock:
                live_state = state_provider.get(current_state)
                decode_state = live_state.copy() if args.state_feedback == "live" else compose_policy_state(live_state, current_state)
                max_chunk_frames = int(args.pose_chunk_frames)
                if stream_target_frames > 0:
                    max_chunk_frames = min(
                        max_chunk_frames, max(0, stream_target_frames - frame_idx)
                    )
                if max_chunk_frames <= 0:
                    break
                state_chunk = actions_to_abs_states(
                    actions, decode_state, max_chunk_frames
                )
                advance_frames = min(
                    int(args.pose_chunk_advance_frames),
                    int(state_chunk.shape[0]),
                    max(1, max_chunk_frames),
                )
                next_current_state = state_chunk[advance_frames - 1].copy()
            joint_pos = state_chunk[:, :29]
            joint_vel = validate_and_record_chunk(
                state_chunk,
                live_state,
                prev_joint,
                first_chunk=start_time is None,
                args=args,
                metadata=dict(
                    action_meta,
                    live_state_mtime=state_provider.last_mtime,
                    validation_time=time.time(),
                ),
            )
            body_quat = state_chunk[:, 32:36]
            n = int(state_chunk.shape[0])
            is_final_real_chunk = (
                stream_target_frames > 0 and frame_idx + n >= stream_target_frames
            )
            if is_final_real_chunk and advance_frames < n:
                advance_frames = n
                next_current_state = state_chunk[advance_frames - 1].copy()
            frame_indices = np.arange(frame_idx, frame_idx + n, dtype=np.int64)
            send_joint_pos = joint_pos
            send_joint_vel = joint_vel
            send_body_quat = body_quat
            send_frame_indices = frame_indices
            tail_padding_count = 0
            if is_final_real_chunk and int(args.tail_padding_frames) > 0:
                tail_padding_count = int(args.tail_padding_frames)
                send_joint_pos, send_joint_vel, send_body_quat, send_frame_indices = (
                    append_stationary_tail(
                        joint_pos,
                        joint_vel,
                        body_quat,
                        frame_indices,
                        tail_padding_count,
                    )
                )
                tail_padding_sent = True
            if start_time is None:
                start_time = time.time()
                start_perf = time.perf_counter()
            pose_send_start = time.perf_counter()
            tick_late_ms = (
                max(0.0, (pose_send_start - next_pose_send_perf) * 1000.0)
                if next_pose_send_perf is not None
                else 0.0
            )
            if tick_late_ms > 180.0:
                raise RuntimeError(
                    f"Reference publication missed its timeline: {tick_late_ms:.1f}ms > 180ms"
                )
            wbc.send_pose(
                send_joint_pos,
                send_joint_vel,
                send_body_quat,
                send_frame_indices,
                catch_up=False,
            )
            pose_send_ms = (time.perf_counter() - pose_send_start) * 1000.0
            csv_start = time.perf_counter()
            append_generated_joint_csv(generated_joint_csv, joint_pos)
            csv_ms = (time.perf_counter() - csv_start) * 1000.0
            last_sent_end_frame = int(frame_indices[-1])
            if not first_pose_sent:
                write_flag(args.first_pose_flag_path, str(time.time()))
                first_pose_sent = True
                print(
                    f"[bridge-real] first pose sent: {args.first_pose_flag_path}",
                    flush=True,
                )
            with state_lock:
                current_state = next_current_state
            prev_joint = joint_pos[advance_frames - 1].copy()
            root = state_chunk[advance_frames - 1, 29:32]
            tail_root = state_chunk[-1, 29:32]
            if args.verbose or frame_idx % 50 == 0 or tail_padding_count > 0:
                wall_frame = (
                    int(round((time.perf_counter() - start_perf) * CTRL_HZ))
                    if start_perf is not None
                    else 0
                )
                send_ahead_frames = int(send_frame_indices[-1]) - wall_frame
                action_age_ms = max(0.0, (time.time() - last_action_time) * 1000.0)
                live_age = (
                    time.time() - state_provider.last_mtime
                    if state_provider.last_mtime > 0.0
                    else -1.0
                )
                print(
                    f"[bridge-real] frames={frame_idx}..{frame_idx + n - 1} wbc_frames={int(send_frame_indices[0])}..{int(send_frame_indices[-1])} advance={advance_frames} tail_padding={tail_padding_count} wall_frame={wall_frame} send_ahead={send_ahead_frames} root=({root[0]:.3f},{root[1]:.3f},{root[2]:.3f}) tail=({tail_root[0]:.3f},{tail_root[1]:.3f},{tail_root[2]:.3f}) action_shape={actions.shape} server_version={action_meta.get('version')} chunk_version={action_meta.get('chunk_version')} ctrl_frames={action_meta.get('controller_generated_frames')} action_age={action_age_ms:.1f}ms tick_late={tick_late_ms:.1f}ms pose_send={pose_send_ms:.2f}ms csv={csv_ms:.2f}ms live_age={live_age:.3f}s",
                    flush=True,
                )
            frame_idx += advance_frames
            next_pose_send_perf = next_reference_deadline(
                next_pose_send_perf, pose_send_start, advance_frames
            )
            previous_actions = actions.copy()
            update_prompt_state(frame_idx)
            with prompt_lock:
                prompt_state["video_frame_cursor"] = int(frame_idx)
            if args.pose_chunk_send_seconds > 0:
                time.sleep(args.pose_chunk_send_seconds)
        if completed_stream and tail_padding_sent:
            hold_seconds = max(0.0, float(args.tail_padding_hold_seconds))
            print(
                f"[bridge-real] tail padding: motion_end_frame={last_sent_end_frame} merged_into_final_chunk count={int(args.tail_padding_frames)} hold_seconds={hold_seconds:.2f}",
                flush=True,
            )
            if hold_seconds > 0.0:
                time.sleep(hold_seconds)
    except KeyboardInterrupt:
        print("[bridge-real] interrupted", flush=True)
        return 130
    except SystemExit as exc:
        print(str(exc), flush=True)
        return 0
    except Exception as exc:
        print(f"[bridge-real] stopped by error: {exc}", flush=True)
        return 1
    finally:
        stop_event.set()
        try:
            if completed_stream and args.orchestrator_managed_handoff:
                print(
                    "[bridge-real] clean stream; handoff delegated to jog parent",
                    flush=True,
                )
            else:
                wbc.send_command(start=False, stop=True, planner=False)
        except Exception:
            pass
        try:
            if ws is not None:
                ws.close()
        except Exception:
            pass
        if camera_provider is not None:
            camera_provider.close()
        wbc.close()
        print(f"[bridge-real] completed_stream={completed_stream}", flush=True)
        print("[bridge-real] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

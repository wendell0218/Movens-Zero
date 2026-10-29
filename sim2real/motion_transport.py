from __future__ import annotations
import argparse
import json
import struct
import time
from base64 import b64decode, b64encode
from pathlib import Path
import numpy as np
import pandas as pd
import zmq
from numpy.lib.format import descr_to_dtype, dtype_to_descr

HEADER_SIZE = 1280
CTRL_HZ = 50
CTRL_DT = 1.0 / CTRL_HZ


def normalize_quat_wxyz(q):
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-08:
        raise ValueError("Invalid root quaternion")
    return q / norm


def numpy_serialize(o):
    if isinstance(o, (np.ndarray, np.generic)):
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {
            "__numpy__": b64encode(data).decode(),
            "dtype": dtype_to_descr(o.dtype),
            "shape": o.shape,
        }
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def numpy_deserialize(dct):
    if "__numpy__" in dct:
        array = np.frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        return array.reshape(dct["shape"]) if dct["shape"] else array[0]
    return dct


def convert_numpy_in_dict(data, func):
    if isinstance(data, dict):
        if "__numpy__" in data:
            return func(data)
        return {key: convert_numpy_in_dict(value, func) for key, value in data.items()}
    if isinstance(data, list):
        return [convert_numpy_in_dict(item, func) for item in data]
    if isinstance(data, (np.ndarray, np.generic)):
        return func(data)
    return data


def format_motion_prompt(description: str) -> str:
    description = str(description).strip()
    return (
        "Generate a motion sequence that matches the following description.\nDescription: "
        + description
    )


def load_motion_prompt_plan(path_text: str, fps: int = CTRL_HZ) -> list[dict]:
    if not path_text:
        return []
    motion_plan = json.loads(Path(path_text).read_text(encoding="utf-8"))
    prompt_plan = []
    for index, item in enumerate(motion_plan.get("segments", []), start=1):
        low_level = str(item.get("low_level_instruction", "")).strip()
        if not low_level:
            raise ValueError(
                f"motion plan segment {index} has empty low_level_instruction"
            )
        frames = int(item.get("num_frames", 0))
        if frames <= 0:
            frames = int(round(float(item.get("duration_seconds", 0.0)) * fps))
        if frames <= 0:
            raise ValueError(f"motion plan segment {index} has invalid duration/frames")
        prompt_plan.append(
            {
                "index": index,
                "low_level_instruction": low_level,
                "prompt": format_motion_prompt(low_level),
                "frames": frames,
            }
        )
    if not prompt_plan:
        raise ValueError(f"No valid segments found in motion plan: {path_text}")
    return prompt_plan


def pack_header(fields: list[dict], version: int = 1, count: int = 1) -> bytes:
    header = {"v": version, "endian": "le", "count": count, "fields": fields}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"ZMQ header too large: {len(header_json)} > {HEADER_SIZE}")
    return header_json.ljust(HEADER_SIZE, b"\x00")


class WBCZMQPublisher:
    def __init__(self, host: str, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.endpoint = f"tcp://{host}:{port}"
        self.socket.bind(self.endpoint)
        time.sleep(0.5)
        print(f"[wbc] bound {self.endpoint}")

    def send_command(self, start: bool, stop: bool, planner: bool) -> None:
        fields = [
            {"name": "start", "dtype": "u8", "shape": [1]},
            {"name": "stop", "dtype": "u8", "shape": [1]},
            {"name": "planner", "dtype": "u8", "shape": [1]},
        ]
        payload = b"".join(
            [
                struct.pack("B", 1 if start else 0),
                struct.pack("B", 1 if stop else 0),
                struct.pack("B", 1 if planner else 0),
            ]
        )
        self.socket.send(b"command" + pack_header(fields) + payload)

    def send_pose(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        body_quat: np.ndarray,
        frame_index: np.ndarray,
        catch_up: bool = False,
    ) -> None:
        joint_pos = np.asarray(joint_pos, dtype=np.float32)
        joint_vel = np.asarray(joint_vel, dtype=np.float32)
        body_quat = np.asarray(body_quat, dtype=np.float32)
        frame_index = np.asarray(frame_index, dtype=np.int64)
        n = int(joint_pos.shape[0])
        fields = [
            {"name": "joint_pos", "dtype": "f32", "shape": [n, 29]},
            {"name": "joint_vel", "dtype": "f32", "shape": [n, 29]},
            {"name": "body_quat_w", "dtype": "f32", "shape": [n, 4]},
            {"name": "frame_index", "dtype": "i64", "shape": [n]},
            {"name": "catch_up", "dtype": "u8", "shape": [1]},
        ]
        payload = b"".join(
            [
                np.ascontiguousarray(joint_pos).tobytes(),
                np.ascontiguousarray(joint_vel).tobytes(),
                np.ascontiguousarray(body_quat).tobytes(),
                np.ascontiguousarray(frame_index).tobytes(),
                struct.pack("B", 1 if catch_up else 0),
            ]
        )
        self.socket.send(b"pose" + pack_header(fields, count=n) + payload)

    def close(self) -> None:
        self.socket.close()
        self.context.term()


def load_state_npy(path: Path) -> np.ndarray:
    state = np.load(path).astype(np.float32)
    if state.ndim == 2:
        state = state[-1]
    if state.shape != (36,):
        raise ValueError(f"Expected 36D state from {path}, got {state.shape}")
    state[32:36] = normalize_quat_wxyz(state[32:36])
    return state.astype(np.float32)


def load_state_from_motion_dir(motion_dir: Path) -> np.ndarray:
    joint = pd.read_csv(motion_dir / "joint_pos.csv").to_numpy(dtype=np.float32)
    body_pos = pd.read_csv(motion_dir / "body_pos.csv").to_numpy(dtype=np.float32)
    body_quat = pd.read_csv(motion_dir / "body_quat.csv").to_numpy(dtype=np.float32)
    state = np.concatenate([joint[0], body_pos[0], body_quat[0]], axis=0).astype(
        np.float32
    )
    state[32:36] = normalize_quat_wxyz(state[32:36])
    return state


class StateProvider:
    def __init__(
        self, live_state_npy: str, require_live: bool = False, max_age: float = 0.5
    ):
        self.live_state_npy = (
            Path(live_state_npy).expanduser() if live_state_npy else None
        )
        self.require_live = require_live
        self.max_age = max_age
        self.last_mtime = 0.0
        self.live_state = None
        self.last_good_load_time = 0.0
        self.last_fallback_warn_time = 0.0

    def get(self, fallback: np.ndarray) -> np.ndarray:
        if self.live_state_npy is None:
            return fallback.copy()
        try:
            stat = self.live_state_npy.stat()
            if stat.st_mtime != self.last_mtime:
                self.live_state = load_state_npy(self.live_state_npy)
                self.last_mtime = stat.st_mtime
                self.last_good_load_time = time.time()
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"[state] live state load failed: {exc}")
        if self.live_state is None:
            if self.require_live:
                raise RuntimeError(f"live state not available: {self.live_state_npy}")
            return fallback.copy()
        if self.require_live and time.time() - self.last_mtime > self.max_age:
            raise RuntimeError(
                f"live state is stale: age={time.time() - self.last_mtime:.3f}s path={self.live_state_npy}"
            )
        return self.live_state.copy()

    def get_or_fallback(self, fallback: np.ndarray) -> np.ndarray:
        try:
            return self.get(fallback)
        except RuntimeError as exc:
            now = time.time()
            if now - self.last_fallback_warn_time > 1.0:
                print(f"[state] live state fallback to reference: {exc}", flush=True)
                self.last_fallback_warn_time = now
            return fallback.copy()

    def has_live(self) -> bool:
        return self.live_state is not None

    def wait_first_live(self, fallback: np.ndarray, timeout: float) -> np.ndarray:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                state = self.get(fallback)
            except RuntimeError:
                state = fallback
            if self.has_live() and time.time() - self.last_mtime <= self.max_age:
                print(f"[state] live state ready: {self.live_state_npy}")
                return state
            time.sleep(0.05)
        raise TimeoutError(
            f"live state not ready within {timeout:.1f}s: {self.live_state_npy}"
        )


def build_request_payload(
    state: np.ndarray,
    prompt: str,
    image: np.ndarray,
    target_frames: int = 0,
    segment_index: int = 0,
    segment_elapsed_frames: int = 0,
    video_condition: dict | None = None,
) -> str:
    condition = {}
    if int(target_frames) > 0:
        condition["progress_target_frames"] = int(target_frames)
    if int(segment_index) > 0:
        condition["segment_index"] = int(segment_index)
        condition["segment_elapsed_frames"] = int(segment_elapsed_frames)
    if video_condition:
        condition.update(video_condition)
    payload = {
        "image": {},
        "state": {"states": state.astype(np.float32)},
        "gt_action": None,
        "dataset_name": None,
        "instruction": prompt,
        "history": None,
        "condition": condition,
        "timestamp": None,
    }
    return json.dumps(convert_numpy_in_dict(payload, numpy_serialize))


def decode_response_message(message: str) -> tuple[np.ndarray | None, dict]:
    data = convert_numpy_in_dict(json.loads(message), numpy_deserialize)
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    if "action" not in data or data["action"] is None:
        return (None, data)
    action = np.asarray(data["action"], dtype=np.float32)
    if action.ndim == 1:
        action = action[None, :]
    elif action.ndim == 3:
        action = action[0]
    elif action.ndim != 2:
        raise ValueError(f"Unsupported action response shape: {action.shape}")
    return (action.astype(np.float32), data)


def compose_policy_state(
    live_state: np.ndarray, reference_state: np.ndarray
) -> np.ndarray:
    state = np.asarray(live_state, dtype=np.float32).copy()
    state[29:31] = reference_state[29:31]
    state[32:36] = reference_state[32:36]
    return state


def load_initial_state(args):
    if args.init_state_npy:
        return load_state_npy(Path(args.init_state_npy).expanduser())
    if args.init_motion_dir:
        return load_state_from_motion_dir(Path(args.init_motion_dir).expanduser())
    raise ValueError(
        "Use fresh live state or an explicit initial pose, not training metadata"
    )


def actions_to_abs_states(actions, current_state, max_frames):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 36 or (not np.isfinite(actions).all()):
        raise ValueError("Expected finite absolute motion36 from the root40 server")
    result = actions[:max_frames].copy() if max_frames > 0 else actions.copy()
    for row in result:
        row[32:36] = normalize_quat_wxyz(row[32:36])
    return result

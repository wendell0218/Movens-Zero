from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path
from threading import Lock

import numpy as np


LOWSTATE_TOPIC = "rt/lowstate"
ODOM_TOPIC = "rt/odommodestate"
NUM_G1_JOINTS = 29
# Unitree lowstate indexes use the hardware/MuJoCo order.  SEED csv_sonic,
# Psi0 and Sonic MotionSequence use IsaacLab order.  This is the same mapping
# used by run_sonic_sim_record_online.py and Sonic's policy_parameters.hpp.
MUJOCO_TO_ISAACLAB = np.asarray(
    [
        0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
        16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
    ],
    dtype=np.int64,
)

RUNNING = True


class LatestBuffer:
    def __init__(self):
        self._data = None
        self._lock = Lock()

    def set(self, data) -> None:
        with self._lock:
            self._data = data

    def get(self):
        with self._lock:
            return self._data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-state-npy", default="/tmp/psi0_sonic_live_state.npy")
    parser.add_argument("--period", type=float, default=0.02)
    parser.add_argument("--root-z-fallback", type=float, default=0.78)
    parser.add_argument("--quat-order", choices=["wxyz", "xyzw"], default="wxyz")
    parser.add_argument("--require-odom", action="store_true", default=False)
    parser.add_argument(
        "--status-period",
        type=float,
        default=2.0,
        help="Seconds between periodic state summaries; 0 disables repeats after the first status.",
    )
    return parser.parse_args()


def stop_handler(signum, frame) -> None:
    global RUNNING
    RUNNING = False


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1e-6:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


def convert_quat(quat, quat_order: str) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(-1)
    if quat.shape[0] != 4:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if quat_order == "xyzw":
        quat = quat[[3, 0, 1, 2]]
    return normalize_quat_wxyz(quat)


def get_lowstate_joint_q(lowstate) -> np.ndarray | None:
    if lowstate is None or not hasattr(lowstate, "motor_state"):
        return None
    try:
        q = np.asarray([lowstate.motor_state[i].q for i in range(NUM_G1_JOINTS)], dtype=np.float32)
    except Exception:
        return None
    if q.shape != (NUM_G1_JOINTS,) or not np.all(np.isfinite(q)):
        return None
    return q


def get_lowstate_quat(lowstate, quat_order: str) -> np.ndarray | None:
    imu = getattr(lowstate, "imu_state", None)
    if imu is None or not hasattr(imu, "quaternion"):
        return None
    quat = convert_quat(imu.quaternion, quat_order)
    if np.all(np.isfinite(quat)):
        return quat
    return None


def get_root_from_odom(odom, quat_order: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    if odom is None:
        return None, None

    root_pos = None
    if hasattr(odom, "position"):
        pos = np.asarray(odom.position, dtype=np.float32).reshape(-1)
        if pos.shape[0] >= 3 and np.all(np.isfinite(pos[:3])):
            root_pos = pos[:3].copy()

    root_quat = None
    imu = getattr(odom, "imu_state", None)
    if imu is not None and hasattr(imu, "quaternion"):
        root_quat = convert_quat(imu.quaternion, quat_order)

    return root_pos, root_quat


def write_state_atomic(path: Path, state: np.ndarray) -> None:
    tmp_path = Path(str(path) + ".tmp")
    with tmp_path.open("wb") as f:
        np.save(f, state.astype(np.float32))
    os.replace(tmp_path, path)


def main() -> int:
    args = parse_args()
    if args.status_period < 0.0:
        raise SystemExit("[g1-state] --status-period must be non-negative")
    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_._SportModeState_ import SportModeState_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    except Exception as exc:
        raise SystemExit(f"[g1-state] failed to import Unitree SDK2 python package: {exc}") from exc

    live_state_path = Path(args.live_state_npy)
    live_state_path.parent.mkdir(parents=True, exist_ok=True)

    odom_buffer = LatestBuffer()

    print("[g1-state] initializing Unitree DDS")
    ChannelFactoryInitialize(0, os.environ.get("G1_NETWORK_INTERFACE", "enP8p1s0"))

    lowstate_sub = ChannelSubscriber(LOWSTATE_TOPIC, LowState_)
    lowstate_sub.Init()

    odom_sub = ChannelSubscriber(ODOM_TOPIC, SportModeState_)
    odom_sub.Init(lambda msg: odom_buffer.set(msg), 1)

    print(f"[g1-state] lowstate topic: {LOWSTATE_TOPIC}")
    print(f"[g1-state] odom topic: {ODOM_TOPIC}")
    print(f"[g1-state] live_state: {live_state_path}")
    print(f"[g1-state] period: {args.period:.3f}s")

    next_tick = time.perf_counter()
    next_status = 0.0
    wrote_once = False
    warned_odom = False
    warned_root_z = False

    while RUNNING:
        lowstate = lowstate_sub.Read()
        q = get_lowstate_joint_q(lowstate)
        if q is None:
            if time.time() >= next_status:
                print("[g1-state] waiting for valid lowstate motor q")
                next_status = (
                    time.time() + args.status_period
                    if args.status_period > 0.0
                    else float("inf")
                )
            time.sleep(0.01)
            continue

        odom = odom_buffer.get()
        odom_pos, odom_quat = get_root_from_odom(odom, args.quat_order)
        lowstate_quat = get_lowstate_quat(lowstate, args.quat_order)

        if odom is None and args.require_odom:
            if time.time() >= next_status:
                print("[g1-state] waiting for odom because --require-odom is set")
                next_status = (
                    time.time() + args.status_period
                    if args.status_period > 0.0
                    else float("inf")
                )
            time.sleep(0.01)
            continue

        if odom is None and not warned_odom:
            print("[g1-state] odom not ready; root xy stays zero and root quat uses lowstate IMU if available")
            warned_odom = True

        root_pos = np.zeros(3, dtype=np.float32) if odom_pos is None else odom_pos.astype(np.float32)
        root_quat = odom_quat if odom_quat is not None else lowstate_quat
        if root_quat is None:
            root_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Unitree odom z can be unavailable or reported as zero on some stacks.
        # Psi0/Sonic 36D state still needs a plausible pelvis height; 0.78m is
        # the standing-height scale observed in the existing Seed/Sonic data.
        if not np.isfinite(root_pos[2]) or root_pos[2] < 0.2 or root_pos[2] > 1.5:
            root_pos[2] = np.float32(args.root_z_fallback)
            if not warned_root_z:
                print(f"[g1-state] using root z fallback: {args.root_z_fallback:.3f} m")
                warned_root_z = True

        state = np.zeros(36, dtype=np.float32)
        state[:29] = q[MUJOCO_TO_ISAACLAB]
        state[29:32] = root_pos
        state[32:36] = normalize_quat_wxyz(root_quat)
        write_state_atomic(live_state_path, state)

        if not wrote_once or time.time() >= next_status:
            root = state[29:32]
            print(
                "[g1-state] wrote live_state "
                f"q0={state[0]:.3f} root=({root[0]:.3f},{root[1]:.3f},{root[2]:.3f})"
            )
            wrote_once = True
            next_status = (
                time.time() + args.status_period
                if args.status_period > 0.0
                else float("inf")
            )

        next_tick += args.period
        sleep_time = next_tick - time.perf_counter()
        if sleep_time > 0.0:
            time.sleep(sleep_time)
        else:
            next_tick = time.perf_counter()

    print("[g1-state] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

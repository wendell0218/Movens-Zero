import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sonic-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--domain-id", type=int, default=71)
    parser.add_argument("--sim-frequency", type=int, default=200)
    parser.add_argument("--max-seconds", type=float, default=600)
    args = parser.parse_args()
    if args.sim_frequency < 50 or args.sim_frequency % 50:
        parser.error("sim-frequency must be a positive multiple of 50")
    sys.path.insert(0, str(args.sonic_root))
    from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    config = SimLoopConfig(enable_onscreen=False, enable_offscreen=False,
        interface="lo", sim_frequency=args.sim_frequency, verbose=False)
    settings = config.load_wbc_yaml()
    settings.update(ENV_NAME=config.env_name, INTERFACE="lo", DOMAIN_ID=args.domain_id,
        ENABLE_ONSCREEN=False, ENABLE_OFFSCREEN=False, ENABLE_ELASTIC_BAND=True,
        SIMULATE_DT=1.0 / args.sim_frequency)
    simulator = BaseSimulator(config=settings, env_name=config.env_name, onscreen=False, offscreen=False)
    env = simulator.sim_env
    args.run_dir.mkdir(parents=True, exist_ok=True)
    running = True
    def stop(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    order = np.asarray([0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28])
    states, qposes, times = [], [], []
    start = time.monotonic()
    deadline = start
    sim_step = 0
    max_lag = 0.0
    record_start = None
    status = dict(status="running", domain_id=args.domain_id, interface="lo", sim_frequency=args.sim_frequency)
    (args.run_dir / "simulator_ready.json").write_text(json.dumps(status))
    try:
        while running and time.monotonic() - start < args.max_seconds:
            if (args.run_dir / "stop.flag").exists():
                break
            if (args.run_dir / "release.flag").exists() and env.elastic_band is not None:
                env.elastic_band.enable = False
            env.sim_step()
            now = time.monotonic()
            if sim_step % (args.sim_frequency // 50) == 0:
                joint = np.asarray(env.mj_data.qpos[env.body_joint_index + 7 - 1], dtype=np.float32)
                state = np.concatenate([joint[order], env.mj_data.qpos[:7]]).astype(np.float32)
                temporary = args.run_dir / "live_state.tmp"
                with temporary.open("wb") as handle:
                    np.save(handle, state)
                os.replace(temporary, args.run_dir / "live_state.npy")
                if (args.run_dir / "record.flag").exists():
                    if record_start is None:
                        record_start = now
                    states.append(state.copy())
                    qposes.append(env.mj_data.qpos.copy())
                    times.append(now - record_start)
            sim_step += 1
            deadline += 1.0 / args.sim_frequency
            lag = time.monotonic() - deadline
            max_lag = max(max_lag, lag)
            if lag > 0.2:
                raise RuntimeError(f"Physics fell {lag:.3f}s behind wall time; reduce CPU contention")
            if lag < 0:
                time.sleep(-lag)
        status["status"] = "complete"
    except BaseException as exc:
        status.update(status="failed", error=str(exc))
        raise
    finally:
        status.update(physics_steps=sim_step, physics_seconds=sim_step / args.sim_frequency,
                      wall_seconds=time.monotonic() - start, max_lag_seconds=max_lag,
                      recorded_frames=len(states), scene=str(args.sonic_root / settings["ROBOT_SCENE"]))
        if states:
            np.savez_compressed(args.run_dir / "simulated_states.npz", motion36=np.stack(states),
                qpos=np.stack(qposes), wall_time=np.asarray(times), scene=np.asarray(status["scene"]))
        (args.run_dir / "simulation_status.json").write_text(json.dumps(status, indent=2))
        simulator.close()


if __name__ == "__main__":
    main()

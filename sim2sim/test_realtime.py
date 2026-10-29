import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


class ClockTests(unittest.TestCase):
    def run_simulator(self, directory, delay=0):
        spec = importlib.util.spec_from_file_location("movens_simulator", Path(__file__).with_name("simulator.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fake_base = ModuleType("gear_sonic.utils.mujoco_sim.base_sim")
        fake_config = ModuleType("gear_sonic.utils.mujoco_sim.configs")
        qpos = np.zeros(36)
        qpos[2:4] = [0.78, 1]
        def step():
            if delay:
                time.sleep(delay)
            qpos[0] += 0.005
        class Config:
            def __init__(self, **kwargs):
                self.env_name = "test"
            def load_wbc_yaml(self):
                return {"ROBOT_SCENE": "scene.xml"}
        class Simulator:
            def __init__(self, **kwargs):
                self.sim_env = SimpleNamespace(mj_data=SimpleNamespace(qpos=qpos),
                    body_joint_index=np.arange(1, 30), sim_step=step, elastic_band=None)
            def close(self):
                pass
        fake_base.BaseSimulator = Simulator
        fake_config.SimLoopConfig = Config
        arguments = ["simulator.py", "--sonic-root", directory, "--run-dir", directory,
                     "--max-seconds", "0.18"]
        with patch.dict(sys.modules, {fake_base.__name__: fake_base, fake_config.__name__: fake_config}), patch.object(sys, "argv", arguments), patch.object(module.signal, "signal"):
            module.main()

    def test_physics_does_not_wait_for_a_blocked_inference_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "record.flag").touch()
            observed = []
            def slow_inference():
                deadline = time.monotonic() + 1
                while not (root / "live_state.npy").exists():
                    if time.monotonic() > deadline:
                        return
                    time.sleep(0.001)
                first = np.load(root / "live_state.npy")[29]
                time.sleep(0.1)
                observed.append(float(np.load(root / "live_state.npy")[29] - first))
            worker = threading.Thread(target=slow_inference)
            worker.start()
            self.run_simulator(directory)
            worker.join()
            self.assertTrue(observed)
            self.assertGreater(observed[0], 0.05)
            status = json.loads((root / "simulation_status.json").read_text())
            self.assertEqual(status["status"], "complete")
            self.assertLess(abs(status["physics_seconds"] - status["wall_seconds"]), 0.03)
            with np.load(root / "simulated_states.npz") as record:
                self.assertGreater(len(record["motion36"]), 5)
                self.assertTrue(np.all(np.diff(record["wall_time"]) > 0))

    def test_slow_physics_fails_instead_of_stopping_the_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "behind wall time"):
                self.run_simulator(directory, delay=0.22)
            status = json.loads((Path(directory) / "simulation_status.json").read_text())
            self.assertEqual(status["status"], "failed")


if __name__ == "__main__":
    unittest.main()

import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parent


class ReleaseTests(unittest.TestCase):
    def test_root40_config_not_repository_39d(self):
        config = json.loads((ROOT / "config/run_config.json").read_text())
        self.assertEqual(config["model"]["action_dim"], 40)
        self.assertEqual(config["model"]["odim"], 40)
        self.assertEqual(config["model"]["motion_representation"], "root40_v1")
        self.assertEqual(
            config["data"]["transform"]["repack"]["root40_contact_protocol"],
            "isaac_leg_delta_v2",
        )
        self.assertNotIn("train", config)
        self.assertNotIn("wandb", config)

    def test_deployment_entrypoints_resolve(self):
        import importlib.util

        for name in (
            "client",
            "server",
            "controller",
            "motion_bridge",
            "motion_transport",
            "state_reader",
            "root40_runtime.model",
        ):
            spec = importlib.util.find_spec(name)
            self.assertIsNotNone(spec, name)
            self.assertTrue(Path(spec.origin).resolve().is_relative_to(ROOT), name)

    def test_packaged_pose(self):
        q = np.asarray(
            json.loads((ROOT / "config/standard_standing.json").read_text())["qpos_36"]
        )
        self.assertEqual(q.shape, (36,))
        self.assertTrue(np.isfinite(q).all())
        self.assertAlmostEqual(float(np.linalg.norm(q[3:7])), 1.0, places=5)

    def test_preflight_rejects_wrong_weight(self):
        from backend import checkpoint_preflight
        with tempfile.NamedTemporaryFile() as handle:
            handle.write(b"not a checkpoint")
            handle.flush()
            with self.assertRaises(Exception):
                checkpoint_preflight(weight=handle.name)

    def test_preflight_explicit_hash_pin(self):
        from backend import checkpoint_preflight
        architecture = dict(action_dim=40, state_dim=40, action_horizon=30)
        with tempfile.NamedTemporaryFile() as handle, patch("movens_zero.checkpoints.checkpoint_model_config", return_value=architecture):
            handle.write(b"test identity")
            handle.flush()
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                checkpoint_preflight(weight=handle.name, expected_sha256="0" * 64)
            verified = checkpoint_preflight(weight=handle.name)
            self.assertFalse(verified["hash_pinned"])
            verified = checkpoint_preflight(weight=handle.name, expected_sha256=verified["sha256"])
            self.assertTrue(verified["hash_pinned"])

    def test_preflight_rejects_incompatible_representation(self):
        from backend import checkpoint_preflight
        with tempfile.NamedTemporaryFile() as handle, patch("movens_zero.checkpoints.checkpoint_model_config", return_value=dict(action_dim=39, state_dim=39, action_horizon=30)):
            with self.assertRaisesRegex(ValueError, "root40"):
                checkpoint_preflight(weight=handle.name)

    def test_controller_environment_and_launch_files(self):
        for name in ("server.sh", "client.sh", "start_wbc.sh", "state_reader.sh"):
            self.assertTrue((ROOT / name).is_file(), name)
        script = (ROOT / "start_wbc.sh").read_text()
        controller = (ROOT / "controller.py").read_text()
        self.assertIn("WBC_ZMQ_MODE", script)
        self.assertIn("WBC_ZMQ_MODE", controller)
        from protocol import PROTOCOL

        self.assertEqual(PROTOCOL, "movens-root40-v1")


if __name__ == "__main__":
    unittest.main(verbosity=2)

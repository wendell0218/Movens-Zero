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

        with tempfile.NamedTemporaryFile() as f:
            f.write(b"not a checkpoint")
            f.flush()
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                checkpoint_preflight(weight=f.name)

    def test_packaged_normalization_contract(self):
        from backend import checkpoint_preflight

        expected = json.loads((ROOT / "config/release_manifest.json").read_text())[
            "checkpoint_sha256"
        ]
        with (
            tempfile.NamedTemporaryFile() as f,
            patch("backend.hash_file", return_value=expected),
        ):
            result = checkpoint_preflight(
                project="/nonexistent/training/repository", weight=f.name
            )
            self.assertEqual(Path(result["release"]), ROOT / "config")
            self.assertEqual(result["config"]["model"]["action_chunk_size"], 30)

    def test_python_sources_have_no_comments_or_docstrings(self):
        import io
        import tokenize

        for path in ROOT.rglob("*.py"):
            source = path.read_text()
            self.assertFalse(
                any(
                    t.type == tokenize.COMMENT
                    for t in tokenize.generate_tokens(io.StringIO(source).readline)
                ),
                str(path),
            )
            for node in ast.walk(ast.parse(source)):
                if isinstance(
                    node,
                    (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    self.assertIsNone(ast.get_docstring(node), str(path))

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

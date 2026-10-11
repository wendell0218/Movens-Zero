"""Release completeness checks; no robot or checkpoint loading."""
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
        config = json.loads((ROOT/'config/run_config.json').read_text())
        self.assertEqual(config['model']['action_dim'], 40)
        self.assertEqual(config['model']['odim'], 40)
        self.assertEqual(config['model']['motion_representation'], 'root40_v1')
        self.assertEqual(config['data']['transform']['repack']['root40_contact_protocol'], 'isaac_leg_delta_v2')
        self.assertNotIn('train', config)
        self.assertNotIn('wandb', config)

    def test_no_internal_psi_imports(self):
        for path in ROOT.rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith('psi.'), str(path))
                elif isinstance(node, ast.Import):
                    self.assertFalse(any(n.name == 'psi' or n.name.startswith('psi.') for n in node.names), str(path))

    def test_packaged_pose(self):
        q = np.asarray(json.loads((ROOT/'config/standard_standing.json').read_text())['qpos_36'])
        self.assertEqual(q.shape, (36,))
        self.assertTrue(np.isfinite(q).all())
        self.assertAlmostEqual(float(np.linalg.norm(q[3:7])), 1.0, places=5)

    def test_preflight_rejects_wrong_weight(self):
        from backend import checkpoint_preflight
        with tempfile.NamedTemporaryFile() as f:
            f.write(b'not a checkpoint'); f.flush()
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                checkpoint_preflight(weight=f.name)

    def test_packaged_normalization_contract(self):
        from backend import checkpoint_preflight
        expected=json.loads((ROOT/'config/release_manifest.json').read_text())['checkpoint_sha256']
        with tempfile.NamedTemporaryFile() as f, patch('backend.hash_file', return_value=expected):
            result=checkpoint_preflight(project='/nonexistent/training/repository', weight=f.name)
            self.assertEqual(Path(result['release']), ROOT/'config')
            self.assertEqual(result['config']['model']['action_chunk_size'], 30)

if __name__ == '__main__':
    unittest.main(verbosity=2)

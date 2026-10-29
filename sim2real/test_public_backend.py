from pathlib import Path
import unittest

import numpy as np
import torch

from backend import Root40Policy, PROJECT
from movens_zero.config import load_config
from movens_zero.normalization import MotionNormalizer
from movens_zero.root40 import build_root40_action, choose_heading_anchor


class BackendTests(unittest.TestCase):
    def test_large_live_state_correction_does_not_clip_the_clean_prefix(self):
        initial = np.zeros((1, 36), dtype=np.float32)
        initial[0, 31:33] = (0.78, 1)
        future = np.repeat(initial, 30, axis=0)
        future[:, 29] = np.arange(30) * 0.002
        normalizer = MotionNormalizer.from_file(PROJECT / "assets/normalization.json")
        raw = build_root40_action(future, initial, choose_heading_anchor(initial), contact_protocol="isaac_leg_delta_v2")
        received_prefix = []
        class Model:
            def predict_action(self, **kwargs):
                return torch.from_numpy(normalizer.normalize_action(raw)[None])
            def predict_action_rtc(self, previous_actions, **kwargs):
                received_prefix.append(previous_actions.numpy().copy())
                return previous_actions
        policy = Root40Policy.__new__(Root40Policy)
        policy.torch, policy.device = torch, torch.device("cpu")
        policy.config = load_config(PROJECT / "configs/inference.json")
        policy.contact = "isaac_leg_delta_v2"
        policy.normalizer, policy.model, policy.video_cache = normalizer, Model(), {}
        first, previous = policy.infer(initial[0], "text", "Walk forward.", None, 0, 300)
        observed = future[19].copy()
        observed[29] += 10
        second, _ = policy.infer(observed, "text", "Walk forward.", None, 20, 300, previous)
        self.assertTrue(np.any(np.abs(received_prefix[0][0, :10]) > 1))
        np.testing.assert_allclose(second[:10], first[20:], atol=1e-5)


if __name__ == "__main__":
    unittest.main()

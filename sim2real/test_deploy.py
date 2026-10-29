import argparse
import ast
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from protocol import encode, decode, observation, response
from motion_transport import (
    actions_to_abs_states,
    compose_policy_state,
    pack_header,
    StateProvider,
)
from motion_bridge import (
    validate_action_chunk,
    validate_absolute_state_chunk,
    parse_args as bridge_args,
)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.state = np.zeros(36, dtype=np.float32)
        self.state[31] = 0.78
        self.state[32] = 1
        self.args = argparse.Namespace(
            pose_chunk_frames=30, pose_chunk_advance_frames=20
        )

    def test_wire_roundtrip(self):
        np.testing.assert_array_equal(
            decode(encode({"state": self.state}))["state"], self.state
        )

    def test_object_dtype_rejected(self):
        with self.assertRaises(ValueError):
            decode('{"__numpy__":"", "dtype":"O", "shape":[0]}')

    def test_observation(self):
        for mode in ["text", "music", "audio", "speech", "video", "video_clip"]:
            result = observation(
                dict(
                    state={"states": self.state},
                    instruction="Stand",
                    condition=dict(input_mode=mode, progress_target_frames=300),
                )
            )
            self.assertIn(result[1], ["text", "music", "speech", "video"])

    def test_empty_quaternion_rejected(self):
        with self.assertRaises(ValueError):
            observation(dict(state={"states": np.zeros(36)}, condition={}))

    def test_action_contract(self):
        arr = np.repeat(self.state[None], 30, axis=0)
        meta = response(arr, 0, 0.1, "hash", "checkpoint")
        validate_action_chunk(arr, self.args, meta)
        np.testing.assert_array_equal(
            actions_to_abs_states(arr, self.state, 20), arr[:20]
        )

    def test_legacy_rejected(self):
        arr = np.zeros((30, 39), dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_action_chunk(arr, self.args, response(arr, 0, 0, "", "checkpoint"))

    def test_wrong_protocol_rejected(self):
        arr = np.repeat(self.state[None], 30, axis=0)
        meta = response(arr, 0, 0, "", "checkpoint")
        meta["protocol_version"] = "movens-wbc-rtc-v1"
        with self.assertRaises(ValueError):
            validate_action_chunk(arr, self.args, meta)

    def test_state_freshness(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "live.npy"
            np.save(path, self.state)
            provider = StateProvider(str(path), require_live=True, max_age=1)
            np.testing.assert_array_equal(provider.get(self.state), self.state)
            import os, time

            os.utime(path, (time.time() - 10, time.time() - 10))
            with self.assertRaises(RuntimeError):
                provider.get(self.state)

    def test_joint_order(self):
        from state_reader import MUJOCO_TO_ISAACLAB

        self.assertEqual(sorted(MUJOCO_TO_ISAACLAB.tolist()), list(range(29)))
        self.assertEqual(
            MUJOCO_TO_ISAACLAB.tolist(),
            [
                0,
                6,
                12,
                1,
                7,
                13,
                2,
                8,
                14,
                3,
                9,
                15,
                22,
                4,
                10,
                16,
                23,
                5,
                11,
                17,
                24,
                18,
                25,
                19,
                26,
                20,
                27,
                21,
                28,
            ],
        )

    def test_wbc_header(self):
        header = pack_header([{"name": "joint_pos", "dtype": "f32", "shape": [30, 29]}])
        self.assertEqual(len(header), 1280)
        self.assertEqual(json.loads(header.rstrip(b"\x00"))["endian"], "le")

    def test_reference_guards(self):
        args = argparse.Namespace(
            joint_limit_margin_rad=0.0,
            max_first_joint_jump_rad=0.35,
            max_reference_joint_velocity=5.0,
            min_reference_root_height=0.2,
            max_reference_root_height=1.5,
        )
        arr = np.repeat(self.state[None], 30, axis=0)
        validate_absolute_state_chunk(arr, self.state[:29], True, args)
        for field, value, exc in [
            (0, 100, ValueError),
            (31, 0.1, ValueError),
            (32, 2, ValueError),
            (0, float("nan"), FloatingPointError),
        ]:
            invalid = arr.copy()
            invalid[0, field] = value
            with self.assertRaises(exc):
                validate_absolute_state_chunk(invalid, self.state[:29], True, args)
        jump = arr.copy()
        jump[0, 0] = 0.4
        with self.assertRaises(ValueError):
            validate_absolute_state_chunk(jump, self.state[:29], True, args)
        fast = arr.copy()
        fast[1, 0] = 0.2
        with self.assertRaises(ValueError):
            validate_absolute_state_chunk(fast, self.state[:29], True, args)

    def test_mechanical_velocity_bound_is_always_active(self):
        args = argparse.Namespace(
            joint_limit_margin_rad=0.0,
            max_first_joint_jump_rad=0.35,
            max_reference_joint_velocity=0.0,
            min_reference_root_height=0.2,
            max_reference_root_height=1.5,
        )
        arr = np.repeat(self.state[None], 30, axis=0)
        arr[1, 0] = 1.0
        with self.assertRaises(ValueError):
            validate_absolute_state_chunk(arr, self.state[:29], True, args)

    def test_latest_pose_handoff_and_rejection_snapshot(self):
        from motion_bridge import validate_and_record_chunk
        import tempfile

        args = argparse.Namespace(
            joint_limit_margin_rad=0.0,
            max_first_joint_jump_rad=0.35,
            max_reference_joint_velocity=5.0,
            min_reference_root_height=0.2,
            max_reference_root_height=1.5,
        )
        live = self.state.copy()
        live[0] = 0.5
        arr = np.repeat(live[None], 30, axis=0)
        with tempfile.TemporaryDirectory() as directory:
            validate_and_record_chunk(
                arr, live, self.state[:29], True, args, snapshot_dir=directory
            )
            self.assertEqual(list(Path(directory).iterdir()), [])
            invalid = arr.copy()
            invalid[:, 0] = 0.0
            with self.assertRaisesRegex(ValueError, "left_hip_pitch"):
                validate_and_record_chunk(
                    invalid,
                    live,
                    self.state[:29],
                    True,
                    args,
                    metadata={"chunk_start_frame": 0},
                    snapshot_dir=directory,
                )
            paths = list(Path(directory).glob("*.npz"))
            self.assertEqual(len(paths), 1)
            with np.load(paths[0], allow_pickle=False) as snapshot:
                np.testing.assert_array_equal(snapshot["comparison_joint"], live[:29])
                np.testing.assert_array_equal(snapshot["state_chunk"], invalid)
                self.assertEqual(
                    json.loads(str(snapshot["metadata_json"]))["chunk_start_frame"], 0
                )
                with self.assertRaisesRegex(ValueError, "left_hip_pitch"):
                    validate_absolute_state_chunk(
                        snapshot["state_chunk"],
                        snapshot["comparison_joint"],
                        bool(snapshot["first_chunk"]),
                        args,
                    )
            validate_and_record_chunk(
                arr, self.state, live[:29], False, args, snapshot_dir=directory
            )

    def test_preserved_precision_helpers(self):
        root = Path(__file__).resolve().parent

        def canonical(n):
            if isinstance(n, ast.AST):
                return [
                    type(n).__name__,
                    {
                        k: canonical(v)
                        for k, v in ast.iter_fields(n)
                        if v is not None and v != [] and (k != "type_params")
                    },
                ]
            if isinstance(n, list):
                return [canonical(x) for x in n]
            return n

        def nodes(path):
            return {
                n.name: json.dumps(canonical(n), sort_keys=True)
                for n in ast.parse(path.read_text()).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            }

        import hashlib

        original = json.loads((root / "config/precision_ast.json").read_text())
        distilled = nodes(root / "precision.py")
        for name in distilled:
            self.assertEqual(
                hashlib.sha256(distilled[name].encode()).hexdigest(),
                original[name],
                name,
            )

    def test_code_compiles(self):
        for path in Path(__file__).parent.glob("*.py"):
            compile(path.read_text(), str(path), "exec")


if __name__ == "__main__":
    unittest.main(verbosity=2)

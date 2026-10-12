from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
from root40_runtime.motion_repr40 import (
    choose_heading_anchor,
    build_root40_action,
    recover_motion_abs_from_root40_action,
    reencode_root40_prefix,
)

state = np.zeros((1, 36), np.float32)
state[0, 31] = 0.78
state[0, 32] = 1
future = np.repeat(state, 30, axis=0)
future[:, 29] = np.arange(1, 31) * 0.001
anchor = choose_heading_anchor(state)
raw = build_root40_action(future, state, anchor, contact_protocol="isaac_leg_delta_v2")
np.testing.assert_allclose(
    recover_motion_abs_from_root40_action(raw, state, anchor), future, atol=1e-06
)
current = future[19:20].copy()
current[:, 29] += 0.003
current[:, 31] -= 0.01
new_anchor = choose_heading_anchor(current)
prefix = reencode_root40_prefix(
    raw,
    state,
    anchor,
    current,
    new_anchor,
    start_index=20,
    length=10,
    contact_protocol="isaac_leg_delta_v2",
)
np.testing.assert_allclose(
    recover_motion_abs_from_root40_action(prefix, current, new_anchor),
    future[20:],
    atol=1e-06,
)
print(
    "ROOT40_CODEC_OK roundtrip=30x36 rebased_overlap=10x36 live_state_perturbation=preserved"
)

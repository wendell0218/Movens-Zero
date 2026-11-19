from __future__ import annotations

import numpy as np

from .jerk_core import conic, root_restore, smoothing
from .jerk_core.kinematics import FK_DOUBLE, LIMITS, original_mask, physical, torch

CONFIGURATION = {
    'pose_and_root': smoothing.CONFIGURATION,
    'height': root_restore.HEIGHT_CONFIGURATION,
    'height_shape': root_restore.SHAPE_CONFIGURATION,
    'root_residual': root_restore.RESIDUAL_CONFIGURATION,
    'conic_root': conic.CONFIGURATION,
}
def postprocess_qpos(qpos, valid=None, fps=30.):

    raw = np.asarray(qpos)
    if raw.ndim != 2 or raw.shape[1] != 36 or raw.dtype not in (np.float32, np.float64):
        raise ValueError('Expected float32/64 [T,36] G1 qpos')
    if not np.isfinite(fps) or fps != 30.:
        raise ValueError('Native 30 Hz required; no resampling is performed')
    mask = original_mask(valid, len(raw))
    change = np.diff(np.r_[False, mask, False].astype(int))
    starts, ends = np.flatnonzero(change == 1), np.flatnonzero(change == -1)
    before = raw.tobytes()


    with torch.inference_mode():
        filtered, _ = smoothing.smooth_pose_root(
            raw, valid=mask, joint_limits=LIMITS,
            configuration=smoothing.CONFIGURATION, kinematics=FK_DOUBLE)
        seed, _ = root_restore.restore_height(raw, filtered, valid=mask)
        seed, _ = root_restore.restore_height_shape(raw, seed, valid=mask)
        seed, _ = root_restore.restore_residual(raw, seed, valid=mask)


    output, records = seed.copy(), []
    for start, end in zip(starts, ends):
        if end-start < 4:
            output[start:end] = raw[start:end]
            record = {'short_segment_unchanged': True, 'jerk': None}
        else:
            output[start:end], details = conic.optimize_segment(raw[start:end], seed[start:end])
            record = {'raw_jerk': physical(raw[start:end])[0],
                      'seed_jerk': details['seed_Jerk'], 'jerk': details['Jerk'],
                      'exact_seed_events': details['exact_seed_events'],
                      'iterations': len(details['history'])}
        records.append(dict(record, start=int(start), end=int(end)))

    a, b = raw[mask].astype(float), output[mask].astype(float)
    lower, upper = LIMITS.T
    old_excess = np.maximum(np.maximum(lower-a[:, 7:], a[:, 7:]-upper), 0)
    new_excess = np.maximum(np.maximum(lower-b[:, 7:], b[:, 7:]-upper), 0)
    checks = {
        'input_unchanged': raw.tobytes() == before,
        'shape_dtype': output.shape == raw.shape and output.dtype == raw.dtype,
        'first_full_states': output[starts].tobytes() == raw[starts].tobytes(),
        'invalid_bytes': output[~mask].tobytes() == raw[~mask].tobytes(),
        'pose_from_seed': output[:, 3:].tobytes() == seed[:, 3:].tobytes(),
        'root_bound_3cm': float(np.linalg.norm(b[:, :3]-a[:, :3], axis=-1).max(initial=0)) <= .03,
        'unit_quaternion': float(np.abs(np.linalg.norm(b[:, 3:7], axis=-1)-1).max(initial=0)) <= 2e-5,
        'finite': bool(np.isfinite(b).all()),
        'joint_limits_no_new_worse': bool(np.all(new_excess <= old_excess)),
        'seed_events': all(row.get('exact_seed_events', True) for row in records),
        'short_segments': all(output[s:e].tobytes() == raw[s:e].tobytes()
                              for s, e in zip(starts, ends) if e-s < 4),
    }
    if not all(checks.values()):
        raise ValueError('Output safety checks failed: '+repr(checks))
    return output, {'fps': 30., 'checks': checks, 'segments': records}

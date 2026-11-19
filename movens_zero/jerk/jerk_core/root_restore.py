

import numpy as np

from .kinematics import predicted_valleys
from .smoothing import _system

MEAN_CONFIGURATION = {'moment_mode': 1, 'maximum_root_change_m': 0.03}
AMPLITUDE_CONFIGURATION = {'moment_mode': 1, 'maximum_root_change_m': 0.03, 'height_variance_fraction': 1.0}
HEIGHT_CONFIGURATION = {
    'moment_mode': 1,
    'maximum_root_change_m': 0.03,
    'height_variance_fraction': 1.0,
    'speed_valley_guard': 1,
    'valley_distance_frames': 8,
    'backtracking_steps': 12,
}
SHAPE_CONFIGURATION = {
    'moment_mode': 1,
    'maximum_root_change_m': 0.03,
    'height_residual_d3': 64.0,
    'height_recovery_fraction': 0.5,
    'valley_distance_frames': 8,
    'backtracking_steps': 12,
}
RESIDUAL_CONFIGURATION = {
    'maximum_root_change_m': 0.03,
    'root_residual_d3': 0.05,
    'root_residual_d6': 64.0,
    'root_recovery_fraction': 1.0,
    'horizontal_recovery_fraction': 0.25,
    'valley_distance_frames': 8,
    'backtracking_steps': 12,
}


def restore_mean(raw_qpos, filtered_qpos, valid=None, configuration=None):

    raw, filtered = np.asarray(raw_qpos), np.asarray(filtered_qpos)
    if raw.ndim != 2 or raw.shape[1] != 36 or filtered.shape != raw.shape:
        raise ValueError("Expected two same-shaped[T,36] G1 predictions")
    if raw.dtype not in (np.float32, np.float64) or filtered.dtype != raw.dtype:
        raise ValueError("Both predictions must have the same float32/float64 dtype")
    if valid is None:
        mask = np.ones(len(raw), bool)
    elif np.ndim(valid) == 0:
        if isinstance(valid, (bool, np.bool_)) or not np.isfinite(valid) or int(valid) != valid or not 0 <= valid <= len(raw):
            raise ValueError("Valid prefix length must be an integer in[0,T]")
        mask = np.arange(len(raw)) < int(valid)
    else:
        mask = np.asarray(valid)
        if mask.dtype != bool or mask.shape != (len(raw),):
            raise ValueError("Expected a boolean[T] original validity mask")
    cfg = dict(MEAN_CONFIGURATION if configuration is None else configuration)
    if set(cfg) != set(MEAN_CONFIGURATION):
        raise ValueError("Declare exactly the two global stage parameters")
    mode, bound = cfg["moment_mode"], cfg["maximum_root_change_m"]
    if isinstance(mode, (bool, np.bool_)) or not np.isfinite(mode) or mode not in (0, 1, 2):
        raise ValueError("moment_mode must be0/1/2")
    if not np.isfinite(bound) or not 0 <= bound <= .03:
        raise ValueError("maximum_root_change_m must be finite in[0,.03]")
    for values in (raw, filtered):
        if not np.isfinite(values[mask]).all():
            raise ValueError("Valid prediction frames must be finite")
        if np.any(np.abs(np.linalg.norm(values[mask, 3:7].astype(np.float64), axis=-1)-1) > 2e-5):
            raise ValueError("Valid wxyz quaternion norms must satisfy the original2e-5 contract")
    if raw[~mask].tobytes() != filtered[~mask].tobytes():
        raise ValueError("Existing filtered input must preserve original invalid frames bitwise")
    changes = np.diff(np.r_[False, mask, False].astype(int))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    if raw[starts].tobytes() != filtered[starts].tobytes():
        raise ValueError("Existing filtered input must preserve every original segment first state bitwise")
    result, records = filtered.copy(), []
    for start, end in zip(starts, ends):
        original_root = raw[start:end, :3].astype(np.float64)
        previous_root = filtered[start:end, :3].astype(np.float64)
        delta = previous_root-original_root
        length = end-start
        correction = np.zeros_like(delta)
        coefficients = np.zeros((2, 3), np.float64)
        if mode and length > 1 and np.any(delta):
            t = np.linspace(0., 1., length)
            basis = np.stack([t, t*t], axis=1)
            means = basis.mean(axis=0)
            if mode == 1:

                gram = basis.T@basis
                direction = np.linalg.lstsq(gram, means, rcond=None)[0]
                denominator = float(means@direction)
                if not denominator > 0:
                    raise ValueError("Degenerate mean restoration polynomial")
                coefficients = direction[:, None]*delta.mean(axis=0)[None]/denominator
            else:

                constraints = np.stack([means, np.ones(2)], axis=0)
                targets = np.stack([delta.mean(axis=0), delta[-1]], axis=0)
                coefficients = np.linalg.lstsq(constraints, targets, rcond=None)[0]
            correction = basis@coefficients
        adjusted = delta-correction
        adjusted[0] = 0.0
        if mode == 2:

            adjusted[-1] = 0.0
        unbounded_max = float(np.linalg.norm(adjusted, axis=-1).max())
        previous_max = float(np.linalg.norm(delta, axis=-1).max())
        ulp_guard = 4*np.sqrt(3)*float(np.max(np.abs(np.spacing(raw[start:end, :3]))))
        blend = min(1., float(bound)/unbounded_max) if unbounded_max else 1.
        if mode == 0 and previous_max <= bound:
            stored = filtered[start:end, :3].copy()
            blend = 1.
        else:
            for attempt in range(6):
                stored = (original_root+blend*adjusted).astype(raw.dtype)
                stored[0] = raw[start, :3]
                if mode == 2:
                    stored[-1] = raw[end-1, :3]
                actual_max = float(np.linalg.norm(stored.astype(np.float64)-original_root, axis=-1).max())
                if actual_max <= bound:
                    break

                safe_bound = max(0., float(bound)-ulp_guard)
                blend *= safe_bound/actual_max
            else:
                raise ValueError("Strict stored root bound failed; refusing to widen it")
        result[start:end, :3] = stored
        stored_delta = stored.astype(np.float64)-original_root
        unbounded_d3_error = np.diff(adjusted, n=3, axis=0)-np.diff(delta, n=3, axis=0)
        stored_d3_change = np.diff(stored.astype(np.float64), n=3, axis=0)-np.diff(previous_root, n=3, axis=0)
        records.append({"start": int(start), "end": int(end), "length": int(end-start)})
    if result[:, 3:].tobytes() != filtered[:, 3:].tobytes():
        raise ValueError("Stage changed pose/quaternions/joints")
    if result[starts].tobytes() != raw[starts].tobytes() or result[~mask].tobytes() != raw[~mask].tobytes():
        raise ValueError("Stage changed segment first states or invalid frames")
    root_max = float(np.linalg.norm(result[mask, :3].astype(np.float64)-raw[mask, :3].astype(np.float64), axis=-1).max()) if mask.any() else 0.
    if root_max > bound:
        raise ValueError("Stored output exceeds strict root bound")
    metadata = {"configuration": cfg, "segments": records}
    return result, metadata


def restore_amplitude(raw_qpos, filtered_qpos, valid=None, configuration=None):

    cfg = dict(AMPLITUDE_CONFIGURATION if configuration is None else configuration)
    if set(cfg) != set(AMPLITUDE_CONFIGURATION) or cfg["moment_mode"] != 1:
        raise ValueError("Declare the same mean-preserving global stage, without task dispatch")
    fraction = cfg["height_variance_fraction"]
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("height_variance_fraction must be finite in[0,1]")
    old_cfg = {key: cfg[key] for key in MEAN_CONFIGURATION}
    original, previous = np.asarray(raw_qpos), np.asarray(filtered_qpos)
    result, mean_information = restore_mean(original, previous, valid=valid, configuration=old_cfg)
    records = []
    for segment in mean_information["segments"]:
        start, end = segment["start"], segment["end"]
        root = original[start:end, :3].astype(np.float64)
        current = result[start:end, :3].astype(np.float64)
        length = end-start
        raw_relative = root[:, 2]-root[0, 2]
        mean_raw = float(raw_relative.mean())
        raw_variance = float(np.mean((raw_relative-mean_raw)**2))
        initial_relative = current[:, 2]-root[0, 2]
        initial_variance = float(np.mean((initial_relative-initial_relative.mean())**2))
        gain, feasible, reason = 1.0, True, "control_or_single_frame"
        derivative_error = 0.0
        if fraction and length > 1:
            t = np.linspace(0., 1., length)

            y = initial_relative-2*t*(float(initial_relative.mean())-mean_raw)
            bridge = 2*t*mean_raw
            residual = y-bridge
            centered_bridge = bridge-bridge.mean()
            centered_residual = residual-residual.mean()
            variance_y = float(np.mean((y-y.mean())**2))
            target = (1-fraction)*variance_y+fraction*raw_variance
            a = float(np.mean(centered_residual**2))
            h = float(np.mean(centered_bridge*centered_residual))
            c = float(np.mean(centered_bridge**2))-target
            discriminant = h*h-a*c
            roots = []
            if a == 0:
                if h != 0:
                    roots = [-c/(2*h)]
                elif c == 0:
                    roots = [1.0]
            elif discriminant >= 0:

                q = -h-np.copysign(np.sqrt(discriminant), h)
                roots = [q/a, c/q] if q else [0.0]
            roots = [float(value) for value in roots if np.isfinite(value) and value >= 0]
            feasible = bool(roots)
            reason = "nonnegative_variance_gain_found" if feasible else "no_feasible_nonnegative_variance_gain_noop"
            gain = min(roots, key=lambda value: abs(value-1)) if feasible else 1.0
            adjusted_relative = bridge+gain*residual
            current[:, 2] = root[0, 2]+adjusted_relative
            if length >= 4:
                derivative_error = float(np.abs(np.diff(adjusted_relative, n=3)-gain*np.diff(y, n=3)).max()*27000)
        else:
            target = initial_variance
        current[0] = root[0]
        if not np.isfinite(current).all():
            raise ValueError("Nonfinite amplitude candidate")
        delta = current-root
        maximum = float(np.linalg.norm(delta, axis=-1).max())
        bound = float(cfg["maximum_root_change_m"])
        ulp_guard = 4*np.sqrt(3)*float(np.max(np.abs(np.spacing(original[start:end, :3]))))
        safe_bound = max(0., bound-ulp_guard)
        blend = min(1., safe_bound/maximum) if maximum > bound else 1.
        for attempt in range(6):
            stored = (root+blend*delta).astype(original.dtype)
            stored[0] = original[start, :3]
            actual_maximum = float(np.linalg.norm(stored.astype(np.float64)-root, axis=-1).max())
            if actual_maximum <= bound:
                break
            blend *= safe_bound/actual_maximum
        else:
            raise ValueError("Strict stored3cm bound failed; refusing wider tolerance")
        result[start:end, :3] = stored
        stored_relative = stored[:, 2].astype(np.float64)-root[0, 2]
        stored_variance = float(np.mean((stored_relative-stored_relative.mean())**2))
        unbounded_relative = current[:, 2]-root[0, 2]
        records.append({"start": int(start), "end": int(end), "length": int(end-start)})
    if result[:, 3:].tobytes() != previous[:, 3:].tobytes():
        raise ValueError("Amplitude stage changed pose")
    for segment in mean_information["segments"]:
        first = segment["start"]
        if result[first].tobytes() != original[first].tobytes():
            raise ValueError("Amplitude stage changed an original first state")
    metadata = {"configuration": cfg, "segments": records}
    return result, metadata


def restore_height(raw_qpos, filtered_qpos, valid=None, configuration=None):

    cfg = dict(HEIGHT_CONFIGURATION if configuration is None else configuration)
    if set(cfg) != set(HEIGHT_CONFIGURATION) or cfg["speed_valley_guard"] not in (0, 1):
        raise ValueError("Declare exactly the shared global guard configuration")
    for key, minimum in (("valley_distance_frames", 1), ("backtracking_steps", 0)):
        if not np.isfinite(cfg[key]) or int(cfg[key]) != cfg[key] or cfg[key] < minimum:
            raise ValueError("Invalid global integer parameter: "+key)
    if cfg["backtracking_steps"] > 32:
        raise ValueError("backtracking_steps exceeds the declared numerical safety range")
    original = np.asarray(raw_qpos)
    amp_cfg = {key: cfg[key] for key in AMPLITUDE_CONFIGURATION}
    proposed, amp_info = restore_amplitude(original, filtered_qpos, valid=valid, configuration=amp_cfg)
    anchor_cfg = {key: cfg[key] for key in MEAN_CONFIGURATION}
    anchor, mean_info = restore_mean(original, filtered_qpos, valid=valid, configuration=anchor_cfg)
    result, records = proposed.copy(), []
    if cfg["speed_valley_guard"]:
        result = anchor.copy()
        for segment in mean_info["segments"]:
            start, end = segment["start"], segment["end"]
            before = anchor[start:end]
            expected = predicted_valleys(before, int(cfg["valley_distance_frames"]))
            delta = proposed[start:end, :3].astype(np.float64)-before[:, :3].astype(np.float64)
            original_root = original[start:end, :3].astype(np.float64)
            selected, weight, accepted_at = before.copy(), 0., None
            for iteration in range(int(cfg["backtracking_steps"])+1):
                current_weight = 2.**(-iteration)
                candidate = before.copy()
                candidate[:, :3] = (before[:, :3].astype(np.float64)+current_weight*delta).astype(original.dtype)
                candidate[0] = original[start]
                root_change = float(np.linalg.norm(candidate[:, :3].astype(np.float64)-original_root, axis=-1).max())

                if root_change > cfg["maximum_root_change_m"]:
                    continue
                actual = predicted_valleys(candidate, int(cfg["valley_distance_frames"]))
                if np.array_equal(expected, actual):
                    selected, weight, accepted_at = candidate, current_weight, iteration
                    break
            final_valleys = predicted_valleys(selected, int(cfg["valley_distance_frames"]))
            if not np.array_equal(final_valleys, expected):
                raise ValueError("Actual stored prediction velocity valleys changed")
            result[start:end] = selected
            records.append({"start": int(start), "end": int(end), "length": int(end-start)})
    metadata = {"configuration": cfg, "segments": records}
    return result, metadata


def restore_height_shape(raw_qpos, filtered_qpos, valid=None, configuration=None):

    cfg = dict(SHAPE_CONFIGURATION if configuration is None else configuration)
    if set(cfg) != set(SHAPE_CONFIGURATION) or cfg['moment_mode'] != 1:
        raise ValueError('Declare exactly one shared mean/shape recovery configuration')
    if any(not np.isfinite(v) or v < 0 for v in cfg.values()):
        raise ValueError('All parameters must be finite and nonnegative')
    if not 0 <= cfg['height_recovery_fraction'] <= 1 or cfg['maximum_root_change_m'] > .03:
        raise ValueError('Invalid fraction or root bound')
    for key, minimum in (('valley_distance_frames', 1), ('backtracking_steps', 0)):
        if int(cfg[key]) != cfg[key] or cfg[key] < minimum:
            raise ValueError('Invalid integer parameter: '+key)
    if cfg['backtracking_steps'] > 32:
        raise ValueError('Backtracking exceeds the declared numerical range')
    raw, parent = np.asarray(raw_qpos), np.asarray(filtered_qpos)
    mean_cfg = {key: cfg[key] for key in MEAN_CONFIGURATION}
    anchor, mean_info = restore_mean(raw, parent, valid=valid, configuration=mean_cfg)
    result, records = parent.copy(), []
    for segment in mean_info['segments']:
        start, end, length = segment['start'], segment['end'], segment['length']
        original_root = raw[start:end, :3].astype(np.float64)
        before = parent[start:end]
        parent_change = float(np.linalg.norm(before[:, :3].astype(np.float64)-original_root, axis=-1).max())
        if parent_change > cfg['maximum_root_change_m']:
            raise ValueError('Input parent already exceeds the strict original-relative root bound')

        residual = original_root[:, 2]-anchor[start:end, 2].astype(np.float64)
        correction = np.zeros(length, dtype=np.float64)
        if length > 1 and cfg['height_recovery_fraction']:
            _, solver = _system(length, cfg['height_residual_d3'], 0.)
            solution = solver.solve(residual[1:])
            direction = solver.solve(np.ones(length-1, dtype=np.float64))
            correction[1:] = solution-direction*(solution.sum()/direction.sum())
        proposed = anchor[start:end, :3].astype(np.float64)
        proposed[:, 2] += cfg['height_recovery_fraction']*correction
        delta = proposed-before[:, :3].astype(np.float64)
        expected = predicted_valleys(before, int(cfg['valley_distance_frames']))
        selected, weight, accepted_at = before.copy(), 0., None
        if cfg['height_recovery_fraction']:
            for iteration in range(int(cfg['backtracking_steps'])+1):
                alpha = 2.**(-iteration)
                candidate = before.copy()
                candidate[:, :3] = (before[:, :3].astype(np.float64)+alpha*delta).astype(raw.dtype)
                candidate[0] = raw[start]
                bound = float(np.linalg.norm(candidate[:, :3].astype(np.float64)-original_root, axis=-1).max())
                if bound <= cfg['maximum_root_change_m'] and np.array_equal(
                        predicted_valleys(candidate, int(cfg['valley_distance_frames'])), expected):
                    selected, weight, accepted_at = candidate, alpha, iteration
                    break
        actual = predicted_valleys(selected, int(cfg['valley_distance_frames']))
        if not np.array_equal(actual, expected):
            raise ValueError('Stored output changed an actual original-prediction velocity valley')
        result[start:end] = selected
        records.append({"start": int(start), "end": int(end), "length": int(end-start)})
    if result[:, 3:].tobytes() != parent[:, 3:].tobytes():
        raise ValueError('Shape recovery changed frozen pose')
    return result, {"configuration": cfg, "segments": records}


def restore_residual(raw_qpos, filtered_qpos, valid=None, configuration=None):

    cfg = dict(RESIDUAL_CONFIGURATION if configuration is None else configuration)
    if set(cfg) != set(RESIDUAL_CONFIGURATION):
        raise ValueError('Declare exactly one shared root-vector configuration')
    if any(isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 0 for value in cfg.values()):
        raise ValueError('All numeric parameters must be finite and nonnegative, not boolean')

    if cfg['maximum_root_change_m'] > .03 or cfg['root_recovery_fraction'] > 1 or cfg['horizontal_recovery_fraction'] > 1:
        raise ValueError('Original-relative root bound/fraction exceeded')
    for key, minimum, maximum in (('valley_distance_frames', 1, None), ('backtracking_steps', 0, 32)):
        if int(cfg[key]) != cfg[key] or cfg[key] < minimum or (maximum is not None and cfg[key] > maximum):
            raise ValueError('Invalid integer parameter: '+key)
    raw, parent = np.asarray(raw_qpos), np.asarray(filtered_qpos)

    _, validation = restore_mean(raw, parent, valid=valid,
        configuration={'moment_mode': 0, 'maximum_root_change_m': cfg['maximum_root_change_m']})
    result, records = parent.copy(), []
    for segment in validation['segments']:
        start, end, length = segment['start'], segment['end'], segment['length']
        original_root = raw[start:end, :3].astype(np.float64)
        before = parent[start:end]
        previous_root = before[:, :3].astype(np.float64)
        if float(np.linalg.norm(previous_root-original_root, axis=-1).max()) > cfg['maximum_root_change_m']:
            raise ValueError('Input parent already exceeds strict original-relative root bound')
        residual = original_root-previous_root

        residual[:, :2] *= cfg['horizontal_recovery_fraction']
        correction = np.zeros_like(residual)
        if length > 1 and cfg['root_recovery_fraction']:

            _, factor = _system(length, cfg['root_residual_d3'], cfg['root_residual_d6'])
            solution = factor.solve(residual[1:])
            direction = factor.solve(np.ones(length-1, dtype=np.float64))
            correction[1:] = solution-direction[:, None]*(solution.sum(axis=0)/direction.sum())[None]
        expected = predicted_valleys(before, int(cfg['valley_distance_frames']))
        selected, weight, accepted_at = before.copy(), 0., None
        if cfg['root_recovery_fraction']:
            for iteration in range(int(cfg['backtracking_steps'])+1):
                alpha = 2.**(-iteration)
                candidate = before.copy()
                candidate[:, :3] = (previous_root+alpha*cfg['root_recovery_fraction']*correction).astype(raw.dtype)
                candidate[0] = raw[start]
                bound = float(np.linalg.norm(candidate[:, :3].astype(np.float64)-original_root, axis=-1).max())
                if bound <= cfg['maximum_root_change_m'] and np.array_equal(
                        predicted_valleys(candidate, int(cfg['valley_distance_frames'])), expected):
                    selected, weight, accepted_at = candidate, alpha, iteration
                    break
        actual = predicted_valleys(selected, int(cfg['valley_distance_frames']))
        if not np.array_equal(actual, expected):
            raise ValueError('Stored output changed a fresh G1 prediction speed valley')
        result[start:end] = selected
        records.append({"start": int(start), "end": int(end), "length": int(end-start)})
    if result[:, 3:].tobytes() != parent[:, 3:].tobytes():
        raise ValueError('Root-only stage changed pose')
    return result, {"configuration": cfg, "segments": records}

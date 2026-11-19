

from functools import lru_cache
from math import comb

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import torch

CONFIGURATION = {
    'joint_d3': 0.05,
    'joint_d6': 2.0,
    'quaternion_d3': 0.03,
    'quaternion_d6': 0.2,
    'root_centroid_d3': 3,
    'maximum_root_change_m': 0.03,
    'iterations': 512,
    'learning_rate': 0.00012,
    'root_fidelity_weight': 0.95,
    'event_max_weight': 1000000,
    'confidence_power': 20,
    'event_margin_fraction': 0.25,
    'contact_velocity_weight': 0,
    'root_velocity_max_weight': 0,
    'root_acceleration_max_weight': 100,
    'waveform_confidence_power': 20,
    'acceleration_scale_free': 1,
}

POSE_CONFIGURATION = {key: CONFIGURATION[key] for key in (
    'joint_d3', 'joint_d6', 'quaternion_d3', 'quaternion_d6',
    'root_centroid_d3', 'maximum_root_change_m',
)}


@lru_cache(maxsize=128)
def _system(length, strength3, strength6):

    penalty = sparse.csc_matrix((length, length))
    for order, strength in ((3, strength3), (6, strength6)):
        if strength == 0 or length <= order:
            continue
        difference = sparse.diags(
            [(-1)**(order-k) * comb(order, k) * np.ones(length-order)
             for k in range(order+1)],
            np.arange(order+1), shape=(length-order, length), format="csc")
        penalty = penalty + strength * (difference.T @ difference)
    matrix = sparse.eye(length, format="csc") + penalty
    return penalty.tocsc(), splu(matrix[1:, 1:].tocsc())


def _smooth(values, strength3, strength6=0.0):
    values = np.asarray(values, np.float64)
    if len(values) < 4 or strength3 == strength6 == 0:
        return values.copy()
    penalty, solver = _system(len(values), strength3, strength6)
    result = values.copy()

    result[1:] = solver.solve(values[1:] - penalty[1:, :1] @ values[:1])
    return result


def _safe_joints(raw, proposed, limits, dtype):

    if limits is None:
        return proposed, np.ones(29)
    lower, upper = limits[:, 0], limits[:, 1]

    inside_lower = lower.astype(dtype)
    inside_upper = upper.astype(dtype)
    inside_lower = np.where(inside_lower.astype(float) < lower,
                           np.nextafter(inside_lower, np.array(np.inf, dtype=dtype)), inside_lower)
    inside_upper = np.where(inside_upper.astype(float) > upper,
                           np.nextafter(inside_upper, np.array(-np.inf, dtype=dtype)), inside_upper)
    delta = proposed - raw
    alpha = np.ones(29)
    for j in range(29):
        if np.any((raw[:, j] < lower[j]) | (raw[:, j] > upper[j])):
            alpha[j] = 0.0
            continue
        positive, negative = delta[:, j] > 0, delta[:, j] < 0
        ratios = np.r_[(float(inside_upper[j])-raw[positive, j])/delta[positive, j],
                       (float(inside_lower[j])-raw[negative, j])/delta[negative, j]]
        if ratios.size:
            alpha[j] = min(1.0, max(0.0, float(ratios.min())))
            if alpha[j] < 1.0:
                alpha[j] = np.nextafter(alpha[j], 0.0)

    return raw + delta * alpha, alpha


def filter_pose(qpos, valid=None, joint_limits=None, configuration=None, kinematics=None):
    original = np.asarray(qpos)
    if original.ndim != 2 or original.shape[1] != 36 or original.dtype not in (np.float32, np.float64):
        raise ValueError("Expected float32/64 [T,36] G1 qpos_36")
    if valid is None:
        mask = np.ones(len(original), bool)
    elif np.ndim(valid) == 0:
        if not np.isfinite(valid) or int(valid) != valid or not 0 <= valid <= len(original):
            raise ValueError("Valid prefix must be an integer in [0,T]")
        mask = np.arange(len(original)) < int(valid)
    else:
        mask = np.asarray(valid)
        if mask.dtype != bool or mask.shape != (len(original),):
            raise ValueError("Expected boolean [T] valid mask")
    if not np.isfinite(original[mask]).all():
        raise ValueError("Valid frames must be finite")
    if np.any(np.abs(np.linalg.norm(original[mask, 3:7], axis=-1)-1) > 2e-5):
        raise ValueError("Valid root rotations must be unit wxyz")
    limits = None if joint_limits is None else np.asarray(joint_limits, float)
    if limits is not None and (limits.shape != (29, 2) or np.isnan(limits).any()
                              or np.any(limits[:, 0] > limits[:, 1])):
        raise ValueError("Expected ordered [29,2] joint limits")
    cfg = dict(POSE_CONFIGURATION if configuration is None else configuration)
    if set(cfg) != set(POSE_CONFIGURATION):
        raise ValueError("Configuration must declare exactly the global numerical parameters")
    if any(not np.isfinite(v) or v < 0 for v in cfg.values()):
        raise ValueError("All parameters must be finite and nonnegative")
    if cfg["root_centroid_d3"] > 0 and kinematics is None:
        raise ValueError("Actual G1 FK is required for root centroid optimization")
    changes = np.diff(np.r_[False, mask, False].astype(int))
    segments = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
    result, summaries = original.copy(), []
    for start, end in segments:
        raw = original[start:end].astype(float)
        if len(raw) < 4:
            summaries.append({"start": int(start), "end": int(end), "short_unmodified": True})
            continue
        filtered = raw.copy()
        proposed = _smooth(raw[:, 7:], cfg["joint_d3"], cfg["joint_d6"])
        filtered[:, 7:], alpha = _safe_joints(raw[:, 7:], proposed, limits, original.dtype)
        if cfg["quaternion_d3"] or cfg["quaternion_d6"]:
            unit = raw[:, 3:7] / np.linalg.norm(raw[:, 3:7], axis=-1, keepdims=True)
            signs = np.ones(len(raw))
            signs[1:] = np.cumprod(np.where((unit[1:]*unit[:-1]).sum(-1) < 0, -1.0, 1.0))
            rotation = _smooth(unit * signs[:, None], cfg["quaternion_d3"], cfg["quaternion_d6"])
            norm = np.linalg.norm(rotation, axis=-1, keepdims=True)
            if not np.isfinite(norm).all() or np.any(norm < 0.25):
                raise ValueError("Unsafe quaternion norm; refusing trajectory")
            filtered[:, 3:7] = rotation / norm * signs[:, None]
        pose = filtered.astype(original.dtype)
        pose[0] = original[start]
        root_scale, root_change = 1.0, 0.0
        if cfg["root_centroid_d3"]:
            with torch.inference_mode():
                body = kinematics.forward_body_positions(torch.from_numpy(pose.astype(np.float64))).numpy()
            centroid = body.mean(axis=1)
            penalty, solver = _system(len(raw), cfg["root_centroid_d3"], 0.0)
            delta = np.zeros((len(raw), 3), np.float64)
            delta[1:] = solver.solve(-(penalty @ centroid)[1:])
            root_change = float(np.linalg.norm(delta, axis=-1).max())
            if root_change > cfg["maximum_root_change_m"] and root_change > 0:

                root_scale = cfg["maximum_root_change_m"] / root_change
            pose[:, :3] = (raw[:, :3] + root_scale * delta).astype(original.dtype)
            pose[0] = original[start]
        result[start:end] = pose
        summaries.append({"start": int(start), "end": int(end),
                          "joint_whole_trajectory_blends": alpha.tolist(),
                          "root_delta_unbounded_max_m": root_change,
                          "root_whole_trajectory_scale": root_scale})
    if limits is not None:
        raw_bad = (original[mask, 7:] < limits[:, 0]) | (original[mask, 7:] > limits[:, 1])
        new_bad = (result[mask, 7:] < limits[:, 0]) | (result[mask, 7:] > limits[:, 1])
        if np.any(new_bad & ~raw_bad):
            raise ValueError("Additional joint-limit violation after storage; refusing candidate")
    metadata = {"configuration": cfg, "valid_frames": int(mask.sum()), "segments": summaries}
    return result, metadata


@torch.inference_mode(mode=False)
def smooth_pose_root(qpos, valid=None, joint_limits=None, configuration=None, kinematics=None):

    cfg = dict(CONFIGURATION if configuration is None else configuration)
    if set(cfg) != set(CONFIGURATION):
        raise ValueError("Declare exactly the shared numeric configuration, without task dispatch")
    if any(not np.isfinite(value) or value < 0 for value in cfg.values()):
        raise ValueError("Configuration must be finite and nonnegative")
    if int(cfg["iterations"]) != cfg["iterations"] or cfg["iterations"] < 1:
        raise ValueError("iterations must be a positive integer")
    if kinematics is None:
        raise ValueError("Fixed actual G1 kinematics are required")
    original = np.asarray(qpos)
    pose_cfg = {key: cfg[key] for key in POSE_CONFIGURATION}
    result, information = filter_pose(original, valid=valid, joint_limits=joint_limits,
                                               configuration=pose_cfg, kinematics=kinematics)
    summaries = []

    physical_indices = np.r_[0:3, 7:36]

    for segment in information["segments"]:
        start, end = segment["start"], segment["end"]
        if end-start < 4:
            continue
        raw, initial = original[start:end], result[start:end].copy()
        static_position = np.all(raw[:, physical_indices] == raw[:1, physical_indices])
        static_rotation = np.all(np.all(raw[:, 3:7] == raw[:1, 3:7], axis=1)
                                 | np.all(raw[:, 3:7] == -raw[:1, 3:7], axis=1))
        if static_position and static_rotation:
            result[start:end] = raw
            summaries.append({"start": start, "end": end, "static_bitwise_unmodified": True})
            continue
        with torch.no_grad():
            raw_fk = kinematics.forward_kinematics(torch.from_numpy(raw.astype(np.float64)))
            pose_fk = kinematics.forward_kinematics(torch.from_numpy(initial.astype(np.float64)))
            raw_body = raw_fk["body_pos_w"].detach()
            relative = (pose_fk["body_pos_w"] - torch.from_numpy(initial[:, :3].astype(float))[:, None]).detach()
            sole, radii = kinematics.get_sole_proxy_points(raw_fk["body_pos_w"], raw_fk["body_quat_w"])
            fixed_sole, _ = kinematics.get_sole_proxy_points(pose_fk["body_pos_w"], pose_fk["body_quat_w"])
            relative_sole = (fixed_sole-torch.from_numpy(initial[:, :3].astype(float))[:, None]).detach()
        raw_root = torch.from_numpy(raw[:, :3].astype(np.float64))
        raw_velocity = np.diff(raw_body.numpy(), axis=0)
        demeaned = raw_velocity-raw_velocity.mean(axis=0, keepdims=True)
        low = gaussian_filter1d(demeaned, sigma=1.0, axis=0, mode="nearest", truncate=4.0)
        low -= low.mean(axis=0, keepdims=True)
        confidence = float(np.clip(np.square(low).sum()/(np.square(demeaned).sum()+1e-8), 0., 1.))
        effective_event_weight = cfg["event_max_weight"] * confidence**cfg["confidence_power"]
        effective_velocity_weight = cfg["root_velocity_max_weight"] * confidence**cfg["waveform_confidence_power"]
        effective_acceleration_weight = cfg["root_acceleration_max_weight"] * confidence**cfg["waveform_confidence_power"]
        raw_root_velocity = (raw_root[1:]-raw_root[:-1])*30.0
        raw_root_acceleration = (raw_root[2:]-2*raw_root[1:-1]+raw_root[:-2])*900.0
        velocity_scale2 = raw_root_velocity.square().mean().clamp_min(0.01**2)
        acceleration_scale2 = raw_root_acceleration.square().mean().clamp_min(0.1**2)

        raw_speed = np.linalg.norm(np.diff(raw_body.numpy().astype(np.float32), axis=0), axis=-1).mean(axis=-1)
        events = find_peaks(-raw_speed, distance=8)[0]
        scale = max(float(raw_speed.mean()), 1e-5)
        contrasts = np.minimum(raw_speed[events-1]-raw_speed[events], raw_speed[events+1]-raw_speed[events])
        margins = torch.from_numpy((cfg["event_margin_fraction"]*np.maximum(contrasts, 0)).astype(float))
        event_ids = torch.as_tensor(events, dtype=torch.long)
        bottom = sole[..., 2]-radii
        contact = ((bottom >= -0.02) & (bottom <= 0.12)).detach()
        intervals = contact[1:] & contact[:-1]
        def stored_candidate(root):
            proposed = initial.copy()
            delta = root.detach().numpy()-raw_root.numpy()
            maximum = float(np.linalg.norm(delta, axis=-1).max())

            stored_origin = raw[:, :3].astype(original.dtype)
            ulp_guard = 4.0*np.sqrt(3.0)*float(np.max(np.abs(np.spacing(stored_origin))))
            safe_bound = max(0.0, cfg["maximum_root_change_m"]-ulp_guard)
            blend = min(1.0, safe_bound/maximum) if maximum else 1.0

            for _ in range(4):
                proposed[:, :3] = (raw_root.numpy()+blend*delta).astype(original.dtype)
                proposed[0] = raw[0]
                stored_maximum = float(np.linalg.norm(proposed[:, :3].astype(float)-raw_root.numpy(), axis=-1).max())
                if stored_maximum <= cfg["maximum_root_change_m"]:
                    return proposed
                blend *= safe_bound/stored_maximum*(1-8*np.finfo(original.dtype).eps)
            raise ValueError("Stored root bound exceeded; refusing candidate")
        initial = stored_candidate(torch.from_numpy(initial[:, :3].astype(float)))
        free_delta = torch.tensor((initial[1:, :3].astype(float)-raw[1:, :3]), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([free_delta], lr=cfg["learning_rate"])

        def objective(root):
            body = relative+root[:, None]
            jerk_vectors = body[3:]-3.0*body[2:-1]+3.0*body[1:-2]-body[:-3]
            jerk = jerk_vectors.norm(dim=-1).mean()*27000.0

            fidelity = ((root-raw_root)/0.005).square().sum(dim=-1).mean()
            velocity_delta = (root[1:]-root[:-1])*30.0-raw_root_velocity
            acceleration = (root[2:]-2*root[1:-1]+root[:-2])*900.0
            if cfg["acceleration_scale_free"]:

                attenuation = (acceleration*raw_root_acceleration).sum()/raw_root_acceleration.square().sum().clamp_min(1e-8)
                attenuation = attenuation.clamp(0.0, 1.0)
            else:
                attenuation = torch.ones((), dtype=torch.float64)
            acceleration_delta = acceleration-attenuation*raw_root_acceleration
            velocity_loss = velocity_delta.square().mean()/velocity_scale2
            acceleration_loss = acceleration_delta.square().mean()/acceleration_scale2
            speed = (body[1:]-body[:-1]).norm(dim=-1).mean(dim=-1)
            if len(events):
                left = torch.relu(speed[event_ids]+margins-speed[event_ids-1])
                right = torch.relu(speed[event_ids]+margins-speed[event_ids+1])
                ordinal = (left.square()+right.square()).mean()/scale**2
            else:
                ordinal = speed.sum()*0.0
            sole_pos = relative_sole+root[:, None]
            horizontal_velocity = ((sole_pos[1:]-sole_pos[:-1])[..., :2]*30.0).square().sum(-1)
            contact_loss = (horizontal_velocity*intervals).sum()/intervals.sum().clamp_min(1)/0.08**2
            total = jerk/100.0 + cfg["root_fidelity_weight"]*fidelity + effective_event_weight*ordinal
            total = total+cfg["contact_velocity_weight"]*contact_loss

            total = total+effective_velocity_weight*velocity_loss+effective_acceleration_weight*acceleration_loss
            return total, jerk, ordinal, contact_loss, velocity_loss, acceleration_loss

        best_objective = float("inf")
        best, best_iteration, best_values = initial.copy(), 0, None
        with torch.enable_grad():
            for iteration in range(int(cfg["iterations"])+1):
                delta = torch.cat([torch.zeros((1, 3), dtype=torch.float64), free_delta], dim=0)
                root = raw_root+delta

                with torch.no_grad():
                    proposed = stored_candidate(root)
                    stored_root = torch.from_numpy(proposed[:, :3].astype(float))
                    values = objective(stored_root)
                    actual_objective = float(values[0])
                    if actual_objective < best_objective:
                        best, best_objective, best_iteration = proposed, actual_objective, iteration
                        best_values = [float(value) for value in values]
                if iteration == cfg["iterations"]:
                    break
                optimizer.zero_grad(set_to_none=True)
                loss = objective(root)[0]
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite root objective")
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    maximum = free_delta.norm(dim=-1).max()
                    if maximum > cfg["maximum_root_change_m"]:
                        free_delta.mul_(cfg["maximum_root_change_m"]/maximum)
        result[start:end] = best
        summaries.append({"start": start, "end": end, "raw_motion_event_indices": events.tolist(),
                          "raw_low_frequency_velocity_confidence": confidence,
                          "effective_event_weight": effective_event_weight,
                          "effective_velocity_weight": effective_velocity_weight,
                          "effective_acceleration_weight": effective_acceleration_weight,
                          "best_iteration": best_iteration,
                          "best_objective": best_values[0], "actual_stored_world_jerk": best_values[1],
                          "ordinal_penalty": best_values[2], "contact_velocity_penalty": best_values[3],
                          "root_velocity_waveform_penalty": best_values[4],
                          "root_acceleration_waveform_penalty": best_values[5]})
    metadata = {"configuration": cfg, "base": information, "root_optimization": summaries}
    return result, metadata

import numpy as np


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-08, 1.0, norm)
    out = quat / norm
    if out.shape[-1] == 4:
        out = np.where(out[..., :1] < 0.0, -out, out)
    return out.astype(np.float32)


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    flat = matrix.reshape(-1, 3, 3)
    quats = []
    for rot in flat:
        trace = float(np.trace(rot))
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            quat = np.array(
                [
                    0.25 * s,
                    (rot[2, 1] - rot[1, 2]) / s,
                    (rot[0, 2] - rot[2, 0]) / s,
                    (rot[1, 0] - rot[0, 1]) / s,
                ],
                dtype=np.float32,
            )
        else:
            idx = int(np.argmax(np.diag(rot)))
            if idx == 0:
                s = np.sqrt(max(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2], 1e-08)) * 2.0
                quat = np.array(
                    [
                        (rot[2, 1] - rot[1, 2]) / s,
                        0.25 * s,
                        (rot[0, 1] + rot[1, 0]) / s,
                        (rot[0, 2] + rot[2, 0]) / s,
                    ],
                    dtype=np.float32,
                )
            elif idx == 1:
                s = np.sqrt(max(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2], 1e-08)) * 2.0
                quat = np.array(
                    [
                        (rot[0, 2] - rot[2, 0]) / s,
                        (rot[0, 1] + rot[1, 0]) / s,
                        0.25 * s,
                        (rot[1, 2] + rot[2, 1]) / s,
                    ],
                    dtype=np.float32,
                )
            else:
                s = np.sqrt(max(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1], 1e-08)) * 2.0
                quat = np.array(
                    [
                        (rot[1, 0] - rot[0, 1]) / s,
                        (rot[0, 2] + rot[2, 0]) / s,
                        (rot[1, 2] + rot[2, 1]) / s,
                        0.25 * s,
                    ],
                    dtype=np.float32,
                )
        quats.append(quat)
    out = np.stack(quats, axis=0).reshape(matrix.shape[:-2] + (4,))
    return normalize_quat_wxyz(out)


def estimate_two_foot_contacts(
    joint_delta: np.ndarray, root_delta_local: np.ndarray
) -> np.ndarray:
    left_leg_motion = np.linalg.norm(joint_delta[..., 0:6], axis=-1)
    right_leg_motion = np.linalg.norm(joint_delta[..., 6:12], axis=-1)
    root_z_motion = np.abs(root_delta_local[..., 2])
    left = np.logical_and(left_leg_motion < 0.035, root_z_motion < 0.025)
    right = np.logical_and(right_leg_motion < 0.035, root_z_motion < 0.025)
    return np.stack([left, right], axis=-1).astype(np.float32)

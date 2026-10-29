import hashlib
import json
import os
from pathlib import Path

import numpy as np

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

def load_reference(path):
    with np.load(path, allow_pickle=False) as packed:
        result = {key: packed[key] for key in packed.files}
    qpos = result["qpos_36"]
    count, frames, channels = qpos.shape
    if channels != 36 or qpos.dtype != np.float32:
        raise ValueError("Reference must contain float32 qpos_36 [N,T,36]")
    valid = result.get("valid", np.ones((count, frames), dtype=bool))
    lengths = result.get("lengths", valid.sum(axis=1)).astype(np.int32)
    if valid.dtype != np.bool_ or not np.array_equal(valid, np.arange(frames)[None] < lengths[:, None]):
        raise ValueError("Reference valid mask must contain one nonempty prefix per sample")
    if np.any(lengths < 1) or not np.isfinite(qpos[valid]).all():
        raise ValueError("Invalid reference values or sequence lengths")
    if not np.allclose(np.linalg.norm(qpos[..., 3:7][valid], axis=-1), 1, atol=1e-5):
        raise ValueError("Reference quaternions must use normalized wxyz order")
    ids = result.get("sample_id", np.asarray([f"{i:06d}" for i in range(count)]))
    if len(set(ids.astype(str))) != count:
        raise ValueError("Reference sample IDs must be unique")
    fps = np.asarray(result.get("fps", 30), dtype=np.float32).reshape(-1)
    if len(fps) == 1:
        fps = np.repeat(fps, count)
    if fps.shape != (count,) or not np.allclose(fps, 30):
        raise ValueError("Published benchmark protocol uses 30 FPS")
    for index, length in enumerate(lengths):
        qpos[index, length:] = qpos[index, length - 1]
    result.update(qpos_36=qpos, valid=valid, lengths=lengths, sample_id=ids.astype(str), fps=fps)
    return result

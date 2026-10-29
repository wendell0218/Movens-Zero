import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle
import tarfile

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


MUJOCO_TO_ISAAC = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
                   16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
BODY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


class ReferenceUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def resample_motion(motion, times, fps=50):
    motion, times = np.asarray(motion, dtype=np.float64), np.asarray(times, dtype=np.float64)
    if len(times) < 2 or len(times) != len(motion) or np.any(np.diff(times) <= 0):
        raise ValueError("Motion timestamps must be strictly increasing")
    if motion.shape != (len(times), 36) or not np.isfinite(motion).all():
        raise ValueError("Expected finite motion36")
    if np.any(np.linalg.norm(motion[:, 32:36], axis=-1) < 1e-8):
        raise ValueError("Zero root quaternion")
    times = times - times[0]
    output_times = np.arange(int(np.floor(times[-1] * fps)) + 1) / fps
    values = np.stack([np.interp(output_times, times, motion[:, i]) for i in range(32)], axis=-1)
    rotations = Rotation.from_quat(motion[:, [33, 34, 35, 32]])
    quaternion = Slerp(times, rotations)(output_times).as_quat()[:, [3, 0, 1, 2]]
    return np.concatenate([values, quaternion], axis=-1).astype(np.float32)


def decode_motion(payload, target, robot=None, reference_quaternion_order="xyzw"):
    if target == "g1_reference":
        data = ReferenceUnpickler(io.BytesIO(payload)).load()
        joints = np.asarray(data["dof_pos"])
        if joints.shape[1:] != (29,):
            raise ValueError("The reference must contain exactly 29 MuJoCo body joints")
        quaternion = np.asarray(data["root_rot"])
        if reference_quaternion_order == "xyzw":
            quaternion = quaternion[:, [3, 0, 1, 2]]
        motion = np.concatenate([joints[:, MUJOCO_TO_ISAAC], data["root_pos"], quaternion], axis=-1)
        times = np.arange(len(motion)) / float(data["fps"])
    else:
        if robot is None:
            raise ValueError("Tracking trajectories require --robot-xml matching their full qpos")
        import mujoco
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            qpos, times = data["qpos"], data["times"]
        if qpos.shape != (len(times), robot.nq):
            raise ValueError(f"Tracking qpos {qpos.shape} does not match XML nq={robot.nq}")
        indices = []
        for name in BODY_JOINTS:
            index = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_JOINT, name)
            if index < 0:
                raise ValueError(f"Robot XML lacks {name}")
            indices.append(int(robot.jnt_qposadr[index]))
        motion = np.concatenate([qpos[:, indices][:, MUJOCO_TO_ISAAC], qpos[:, :7]], axis=-1)
    return resample_motion(motion, times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Downloaded wendell0218/Movens-5M snapshot")
    parser.add_argument("--output", default="data/movens_demo")
    parser.add_argument("--target", choices=["g1_reference", "g1_tracking"], default="g1_reference")
    parser.add_argument("--robot-xml", help="Required with g1_tracking; must match its full MuJoCo qpos")
    parser.add_argument("--reference-quaternion-order", choices=["xyzw", "wxyz"], default="xyzw")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    source, output = Path(args.source).resolve(strict=True), Path(args.output).resolve()
    records = [json.loads(line) for line in (source / "motion/manifest.jsonl").read_text().splitlines() if line.strip()]
    if args.limit:
        records = records[:args.limit]
    robot = None
    if args.target == "g1_tracking":
        if not args.robot_xml:
            parser.error("g1_tracking requires --robot-xml; 50D tracking qpos cannot be sliced as a 36D G1 pose")
        import mujoco
        robot = mujoco.MjModel.from_xml_path(str(Path(args.robot_xml).resolve(strict=True)))
    grouped = {}
    for record in records:
        grouped.setdefault((record["archive"], record["parquet_shard"]), {})[int(record["row_in_shard"])] = record
    (output / "motion").mkdir(parents=True, exist_ok=True)
    (output / "video").mkdir(exist_ok=True)
    rows = []
    for (archive, shard), wanted in grouped.items():
        with tarfile.open(source / archive) as tar:
            for row_number, batch in enumerate(pq.ParquetFile(source / shard).iter_batches(
                    batch_size=1, columns=["sample_id", "sample_name", "motion_caption", "original_video"])):
                if row_number > max(wanted):
                    break
                if row_number not in wanted:
                    continue
                record = wanted[row_number]
                row = batch.to_pylist()[0]
                if str(row["sample_name"]) != str(record["sample_name"]):
                    raise ValueError("Parquet row identity differs from the motion manifest")
                member = record["files"][args.target]
                payload = tar.extractfile(member).read()
                expected = record.get("sha256", {}).get(args.target)
                if expected and hashlib.sha256(payload).hexdigest() != expected:
                    raise ValueError(f"Hash mismatch: {member}")
                motion = decode_motion(payload, args.target, robot, args.reference_quaternion_order)
                name = Path(record["sample_name"]).name
                if name != record["sample_name"] or name in (".", ".."):
                    raise ValueError("Invalid sample name")
                np.save(output / "motion" / f"{name}.npy", motion)
                video = row["original_video"]
                if not isinstance(video, dict) or video.get("bytes") is None:
                    raise ValueError("Expected embedded original_video bytes")
                (output / "video" / f"{name}.mp4").write_bytes(video["bytes"])
                common = dict(id=name, motion=f"motion/{name}.npy", fps=50, source_target=args.target)
                rows.append(dict(common, mode="video", media=f"video/{name}.mp4", text=""))
                if row["motion_caption"] and str(row["motion_caption"]).strip():
                    rows.append(dict(common, mode="text", text=str(row["motion_caption"])))
    manifest = output / "train.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(json.dumps(dict(manifest=str(manifest), records=len(rows), target=args.target,
        note="Public 1000-sample subset; this is not the full 5M training corpus")))


if __name__ == "__main__":
    main()

import argparse
import os
from pathlib import Path
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    if min(args.fps, args.width, args.height) <= 0 or args.width % 2 or args.height % 2:
        parser.error("Use positive fps and even dimensions")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    with np.load(args.run_dir / "simulated_states.npz", allow_pickle=False) as record:
        qposes = record["qpos"].copy()
        times = record["wall_time"].copy()
        scene = Path(str(record["scene"]))
    if not len(times) or len(times) != len(qposes) or np.any(np.diff(times) < 0):
        raise ValueError("Invalid physics recording")
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 3.5
    camera.azimuth = 120
    camera.elevation = -20
    encoder = None
    try:
        with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
            encoder = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "pipe:0", "-an",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                str(args.run_dir / "simulation.mp4")], stdin=subprocess.PIPE)
            for t in np.arange(times[0], times[-1] + 1e-6, 1 / args.fps):
                index = min(np.searchsorted(times, t), len(times) - 1)
                if index > 0 and t - times[index - 1] < times[index] - t:
                    index -= 1
                data.qpos[:] = qposes[index]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = data.qpos[:3]
                camera.lookat[2] += 0.3
                renderer.update_scene(data, camera=camera)
                encoder.stdin.write(renderer.render().tobytes())
    finally:
        if encoder is not None:
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("FFmpeg failed")
    print(args.run_dir / "simulation.mp4")


if __name__ == "__main__":
    main()

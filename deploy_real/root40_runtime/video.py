from __future__ import annotations
import os, json, subprocess
from pathlib import Path
from functools import lru_cache
import numpy as np


def ffmpeg_bin() -> str:
    return os.environ.get("MOVENS_FFMPEG_BIN", "ffmpeg")


def ffprobe_bin() -> str:
    return os.environ.get("MOVENS_FFPROBE_BIN", "ffprobe")


def _parse_rate(value: str) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        num, den = value.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else 0.0
    return float(value)


@lru_cache(maxsize=4096)
def _cached_video_probe(video_path: str, probe_path: str, file_version: tuple) -> dict:
    cmd = [
        probe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,duration,start_pts,time_base:format=duration,format_name",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffprobe failed for {video_path}: {msg}")
    info = json.loads(proc.stdout.decode("utf-8"))
    return info


def _video_probe_info(video_path: Path) -> dict:
    video_path = Path(video_path).absolute()
    stat = video_path.stat()
    version = (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    return _cached_video_probe(str(video_path), ffprobe_bin(), version)


def _probe_video(video_path: Path) -> tuple[int, int, float, float]:
    info = _video_probe_info(video_path)
    streams = info.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found by ffprobe: {video_path}")
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    src_fps = _parse_rate(
        str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/0")
    )
    duration_sec = float(
        stream.get("duration") or info.get("format", {}).get("duration") or 0.0
    )
    if width <= 0 or height <= 0 or src_fps <= 0 or (duration_sec <= 0):
        raise RuntimeError(
            f"Bad video metadata at {video_path}: width={width}, height={height}, fps={src_fps}, duration={duration_sec}"
        )
    return (width, height, src_fps, duration_sec)


def load_video_clip_array_ffmpeg(
    video_path: Path,
    start_sec: float,
    duration_sec: float,
    video_sample_fps: float = 1.0,
    video_size_pixels: int = 50176,
) -> tuple[np.ndarray, float, float]:
    width, height, src_fps, source_duration_sec = _probe_video(video_path)
    sample_fps = float(video_sample_fps)
    if sample_fps <= 0:
        raise ValueError(f"video_sample_fps must be positive, got {video_sample_fps}")
    clip_start = max(0.0, float(start_sec))
    clip_duration = max(1.0 / sample_fps, float(duration_sec))
    if source_duration_sec > 0 and clip_start + clip_duration > source_duration_sec:
        clip_start = max(0.0, source_duration_sec - clip_duration)
    num_frames = max(1, int(np.ceil(clip_duration * sample_fps)))
    if video_size_pixels > 0:
        scale = float(np.sqrt(float(video_size_pixels) / float(max(1, width * height))))
        new_w = max(2, int(round(width * scale / 2.0)) * 2)
        new_h = max(2, int(round(height * scale / 2.0)) * 2)
    else:
        new_w, new_h = (width, height)
    seek_options = []
    video_filter = f"setpts=PTS-STARTPTS,fps={sample_fps},scale={new_w}:{new_h}"
    seek_sec = max(0.0, float(np.floor(clip_start - max(2.0, 2.0 / sample_fps))))
    if seek_sec > 0 and os.environ.get("MOVENS_VIDEO_FAST_SEEK", "1") != "0":
        info = _video_probe_info(video_path)
        stream = info["streams"][0]
        formats = set(info.get("format", {}).get("format_name", "").split(","))
        start_pts = stream.get("start_pts")
        time_base = str(stream.get("time_base") or "0/0")
        if (
            formats & {"mov", "mp4", "matroska", "webm"}
            and isinstance(start_pts, int)
            and (_parse_rate(time_base) > 0)
        ):
            seek_options = ["-copyts", "-noaccurate_seek", "-ss", f"{seek_sec:.6f}"]
            video_filter = f"setpts=PTS-({start_pts}*{time_base})/TB,fps={sample_fps},scale={new_w}:{new_h}"
    cmd = [
        ffmpeg_bin(),
        "-v",
        "error",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        *seek_options,
        "-i",
        str(video_path),
        "-ss",
        f"{clip_start:.6f}",
        "-t",
        f"{clip_duration:.6f}",
        "-vf",
        video_filter,
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-threads",
        "1",
        "-",
    ]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"ffmpeg failed to decode video clip from {video_path}: {msg}"
        )
    frame_size = new_h * new_w * 3
    n_frames = len(proc.stdout) // frame_size
    if n_frames <= 0:
        raise RuntimeError(f"No frames decoded from {video_path}")
    frames = np.frombuffer(proc.stdout[: n_frames * frame_size], dtype=np.uint8)
    frames = frames.reshape(n_frames, new_h, new_w, 3).copy()
    if frames.shape[0] < num_frames:
        pad = np.repeat(frames[-1:], num_frames - frames.shape[0], axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    return (frames[:num_frames], src_fps, clip_duration)

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


def decode_sound(path: Path, sampling_rate: int) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(sampling_rate),
        "-f",
        "f32le",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed to decode {path}: {message}")
    sound = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if sound.size == 0:
        raise RuntimeError(f"No samples decoded from {path}")
    return sound


def sound_segment(
    sound: np.ndarray,
    sampling_rate: int,
    start_seconds: float,
    duration_seconds: float,
) -> np.ndarray:
    count = max(1, int(round(duration_seconds * sampling_rate)))
    start = int(round(start_seconds * sampling_rate))
    end = start + count
    segment = np.zeros(count, dtype=np.float32)
    source_start = max(0, start)
    source_end = min(sound.shape[0], end)
    if source_end > source_start:
        target_start = source_start - start
        segment[target_start : target_start + source_end - source_start] = sound[
            source_start:source_end
        ]
    return segment


def parse_rate(value: str) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" not in value:
        return float(value)
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator) if float(denominator) else 0.0


def probe_video(path: Path) -> tuple[int, int, float, float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffprobe failed for {path}: {message}")
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = parse_rate(
        str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/0")
    )
    duration = float(
        stream.get("duration") or data.get("format", {}).get("duration") or 0.0
    )
    if width <= 0 or height <= 0 or fps <= 0 or duration <= 0:
        raise RuntimeError(
            f"Invalid video metadata: width={width}, height={height}, fps={fps}, duration={duration}"
        )
    return width, height, fps, duration


def decode_video_segment(
    path: Path,
    start_seconds: float,
    duration_seconds: float,
    sample_fps: float,
    size_pixels: int,
) -> tuple[np.ndarray, float, float]:
    width, height, source_fps, source_duration = probe_video(path)
    start = max(0.0, float(start_seconds))
    duration = max(1.0 / sample_fps, float(duration_seconds))
    if start + duration > source_duration:
        start = max(0.0, source_duration - duration)
    frame_count = max(1, int(np.ceil(duration * sample_fps)))
    scale = float(np.sqrt(float(size_pixels) / float(width * height)))
    target_width = max(2, int(round(width * scale / 2.0)) * 2)
    target_height = max(2, int(round(height * scale / 2.0)) * 2)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-threads",
        "1",
        "-i",
        str(path),
        "-ss",
        f"{start:.6f}",
        "-t",
        f"{duration:.6f}",
        "-vf",
        f"setpts=PTS-STARTPTS,fps={sample_fps},scale={target_width}:{target_height}",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed to decode {path}: {message}")
    frame_size = target_height * target_width * 3
    decoded = len(result.stdout) // frame_size
    if decoded <= 0:
        raise RuntimeError(f"No frames decoded from {path}")
    frames = np.frombuffer(
        result.stdout[: decoded * frame_size], dtype=np.uint8
    ).reshape(decoded, target_height, target_width, 3)
    if decoded < frame_count:
        padding = np.repeat(frames[-1:], frame_count - decoded, axis=0)
        frames = np.concatenate([frames, padding])
    return frames[:frame_count].copy(), source_fps, source_duration

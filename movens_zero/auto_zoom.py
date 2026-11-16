from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from movens_zero.media import decode_video_segment, probe_video

SMALL_PIXELS = 90.0
TARGET_PIXELS = 120.0
DETECTION_FPS = 5.0
DETECTION_HEIGHT = 720
SCORE_THRESHOLD = 0.5


def _decode_for_detection(path: Path, width: int, height: int) -> tuple[np.ndarray, float]:
    scale = min(1.0, DETECTION_HEIGHT / float(height))
    target_width = max(2, int(round(width * scale / 2.0)) * 2)
    target_height = max(2, int(round(height * scale / 2.0)) * 2)
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-vf",
               f"fps={DETECTION_FPS},scale={target_width}:{target_height}",
               "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {path}: {result.stderr.decode('utf-8', errors='ignore')}")
    frame_size = target_width * target_height * 3
    count = len(result.stdout) // frame_size
    frames = np.frombuffer(result.stdout[: count * frame_size], dtype=np.uint8)
    return frames.reshape(count, target_height, target_width, 3), width / float(target_width)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / max(union, 1e-9)


def detect_main_person(path: Path, device: str = "cuda:0") -> tuple[np.ndarray, np.ndarray, int, int]:
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

    width, height, _, _ = probe_video(path)
    frames, scale = _decode_for_detection(path, width, height)
    detector = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
                                          box_score_thresh=SCORE_THRESHOLD).to(device).eval()
    times, boxes, previous = [], [], None
    with torch.inference_mode():
        for start in range(0, len(frames), 8):
            batch = torch.from_numpy(frames[start:start + 8]).to(device).permute(0, 3, 1, 2).float() / 255.0
            for offset, output in enumerate(detector(list(batch))):
                found = output["boxes"][output["labels"] == 1].float().cpu().numpy() * scale
                if not len(found):
                    continue
                areas = (found[:, 2] - found[:, 0]) * (found[:, 3] - found[:, 1])
                pick = int(np.argmax(areas))
                if previous is not None:
                    overlaps = [_iou(previous, box) for box in found]
                    if max(overlaps) > 0.1:
                        pick = int(np.argmax(overlaps))
                previous = found[pick]
                times.append((start + offset) / DETECTION_FPS)
                boxes.append(previous)
    del detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.asarray(times, dtype=np.float64), np.asarray(boxes, dtype=np.float64).reshape(-1, 4), width, height


class PersonZoom:

    def __init__(self, times: np.ndarray, boxes: np.ndarray, width: int, height: int, size_pixels: int):
        self.times, self.boxes, self.width, self.height = times, boxes, width, height
        person = float(np.median(boxes[:, 3] - boxes[:, 1])) if len(boxes) else 0.0
        self.person_pixels = person * float(np.sqrt(size_pixels / float(width * height)))
        self.enabled = bool(len(boxes)) and self.person_pixels < SMALL_PIXELS

    @classmethod
    def from_video(cls, path: Path, size_pixels: int, device: str = "cuda:0") -> "PersonZoom":
        return cls(*detect_main_person(Path(path), device), size_pixels=size_pixels)

    def view(self, start_seconds: float, duration_seconds: float) -> tuple[float, float, float, float]:
        inside = (self.times >= start_seconds - 1e-6) & (self.times <= start_seconds + duration_seconds + 1e-6)
        boxes = self.boxes[inside] if inside.any() else self.boxes[[int(np.argmin(np.abs(self.times - start_seconds)))]]
        center_x = (boxes[:, 0].min() + boxes[:, 2].max()) / 2
        center_y = (boxes[:, 1].min() + boxes[:, 3].max()) / 2
        ratio = self.person_pixels / TARGET_PIXELS
        w, h = self.width * ratio, self.height * ratio
        return float(np.clip(center_x - w / 2, 0, self.width - w)), float(np.clip(center_y - h / 2, 0, self.height - h)), w, h

    def decode(self, path, start_seconds, duration_seconds, sample_fps, size_pixels):
        if not self.enabled:
            return decode_video_segment(path, start_seconds, duration_seconds, sample_fps, size_pixels)
        frames, source_fps, source_duration = decode_video_segment(
            path, start_seconds, duration_seconds, sample_fps, self.width * self.height)
        scale = float(np.sqrt(float(size_pixels) / float(self.width * self.height)))
        target_width = max(2, int(round(self.width * scale / 2.0)) * 2)
        target_height = max(2, int(round(self.height * scale / 2.0)) * 2)
        x0, y0, w, h = self.view(float(start_seconds), float(duration_seconds))
        left, top = int(np.floor(x0)), int(np.floor(y0))
        right, bottom = min(self.width, int(np.ceil(x0 + w))), min(self.height, int(np.ceil(y0 + h)))
        crop = torch.from_numpy(np.ascontiguousarray(frames[:, top:bottom, left:right])).permute(0, 3, 1, 2).float()
        resized = F.interpolate(crop, size=(target_height, target_width), mode="bilinear", align_corners=False, antialias=True)
        zoomed = resized.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        return np.ascontiguousarray(zoomed), source_fps, source_duration

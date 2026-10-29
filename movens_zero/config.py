from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: _to_namespace(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def load_config(path: str | Path) -> SimpleNamespace:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(data["modalities"]) != {"text", "music", "speech", "video"}:
        raise ValueError("Modalities must be text, music, speech, and video")
    model = data["model"]
    horizon = int(model["action_horizon"])
    execute = int(model["action_exec_horizon"])
    max_delay = int(model["rtc_max_delay"])
    if not 0 < execute < horizon:
        raise ValueError("action_exec_horizon must be between zero and action_horizon")
    if horizon - execute >= max_delay:
        raise ValueError("RTC overlap must be smaller than rtc_max_delay")
    config = _to_namespace(data)
    if not isinstance(config, SimpleNamespace):
        raise TypeError("Inference configuration must be a JSON object")
    return config

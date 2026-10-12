from __future__ import annotations
import random
from collections.abc import Mapping, Sequence
from typing import Any, List
import pandas as pd

DEFAULT_MOTION_TEXT_FIELDS: List[str] = [
    "content_short_description",
    "content_short_description_2",
    "content_technical_description",
    "content_natural_desc_1",
    "content_natural_desc_2",
    "content_natural_desc_3",
    "content_natural_desc_4",
]
SYSTEM_PROMPT_POOLS: dict[str, list[str]] = {
    "system_first": [
        "Generate a motion sequence that matches the following description.",
        "Convert the following language instruction into an executable motion sequence.",
        "From a motion-control perspective, realize the following behavior description.",
        "Read the following description and generate a matching motion.",
        "Complete the motion generation task based on the description below.",
        "Generate a trackable motion sequence centered on the following description.",
        "Given the description below, produce a continuous full-body motion clip.",
        "Use the text instruction below to synthesize a coherent motion trajectory.",
        "Please generate motion conditioned on the following user request.",
        "Generate physically plausible motion according to the description that follows.",
    ],
    "desc_first": [
        "Based on the description above, generate a matching motion sequence.",
        "Using the user request above, synthesize a smooth full-body motion trajectory.",
        "Take the instruction above and convert it into an executable motion sequence.",
        "Generate stable and semantically consistent motion from the description above.",
        "Ground the behavior described above into time-series motion output.",
    ],
    "neutral": [
        "You are a motion generation assistant. Produce a natural and coherent full-body motion from the provided text.",
        "Synthesize a full-body motion trajectory from the text description.",
        "Generate a realistic motion performance consistent with the text prompt.",
        "Output a complete motion sequence from the requirement description without semantic drift.",
        "Generate a motion trajectory that satisfies the given text instruction.",
    ],
}
PROMPT_COMPOSITION_TEMPLATES_BY_LAYOUT: dict[str, list[str]] = {
    "system_first": [
        "{system}\nDescription: {desc}",
        "{system}\nUser request: {desc}",
        "{system} {desc}",
        '{system}\nInput text: "{desc}"',
        "{system}\nBehavior description: {desc}",
        "{system} (Description: {desc})",
    ],
    "desc_first": [
        "Description: {desc}\n{system}",
        "User request: {desc}\n{system}",
        "{desc}\n{system}",
        'Input text: "{desc}"\n{system}',
        "Behavior description: {desc}\n{system}",
    ],
    "neutral": [
        "{system}\nDescription: {desc}",
        "Description: {desc}\n{system}",
        "{system} {desc}",
        "{desc}\n{system}",
        "{system}\nInput text: {desc}",
    ],
}


def _row_has_key(row: Any, key: str) -> bool:
    if isinstance(row, Mapping):
        return key in row
    index = getattr(row, "index", None)
    return key in index if index is not None else hasattr(row, key)


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    if _row_has_key(row, key):
        return row[key]
    return getattr(row, key, default)


def collect_motion_text_candidates(
    row: Any, text_fields: Sequence[str] | None = None
) -> list[str]:
    candidates: list[str] = []
    seen = set()
    for key in text_fields or DEFAULT_MOTION_TEXT_FIELDS:
        if not _row_has_key(row, key):
            continue
        value = _row_get(row, key)
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        signature = text.lower()
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(text)
    return candidates


def compose_motion_prompt(desc: str, *, rng: Any = random) -> str:
    desc = str(desc).strip()
    if not desc:
        desc = "seed motion"
    layout = rng.choice(list(SYSTEM_PROMPT_POOLS.keys()))
    system_prompt = rng.choice(SYSTEM_PROMPT_POOLS[layout])
    template = rng.choice(PROMPT_COMPOSITION_TEMPLATES_BY_LAYOUT[layout])
    return template.format(system=system_prompt, desc=desc).strip()


def compose_motion_prompt_from_candidates(
    candidates: Sequence[str], *, fallback: str = "seed motion", rng: Any = random
) -> str:
    pool = [str(c).strip() for c in candidates if str(c).strip()]
    if not pool:
        pool = [str(fallback).strip() or "seed motion"]
    return compose_motion_prompt(rng.choice(pool), rng=rng)


def compose_motion_prompt_from_row(
    row: Any,
    text_fields: Sequence[str] | None = None,
    *,
    fallback_key: str = "move_name",
    rng: Any = random,
) -> str:
    fallback = str(_row_get(row, fallback_key, "seed motion")).strip() or "seed motion"
    return compose_motion_prompt_from_candidates(
        collect_motion_text_candidates(row, text_fields), fallback=fallback, rng=rng
    )

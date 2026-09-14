from __future__ import annotations

import random


SYSTEM_PROMPTS = {
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

TEMPLATES = {
    "system_first": [
        "{system}\nDescription: {description}",
        "{system}\nUser request: {description}",
        "{system} {description}",
        '{system}\nInput text: "{description}"',
        "{system}\nBehavior description: {description}",
        "{system} (Description: {description})",
    ],
    "desc_first": [
        "Description: {description}\n{system}",
        "User request: {description}\n{system}",
        "{description}\n{system}",
        'Input text: "{description}"\n{system}',
        "Behavior description: {description}\n{system}",
    ],
    "neutral": [
        "{system}\nDescription: {description}",
        "Description: {description}\n{system}",
        "{system} {description}",
        "{description}\n{system}",
        "{system}\nInput text: {description}",
    ],
}


def compose_motion_prompt(description: str, generator: random.Random) -> str:
    description = description.strip()
    if not description:
        raise ValueError("Prompt cannot be empty")
    layout = generator.choice(list(SYSTEM_PROMPTS))
    system = generator.choice(SYSTEM_PROMPTS[layout])
    template = generator.choice(TEMPLATES[layout])
    return template.format(system=system, description=description).strip()

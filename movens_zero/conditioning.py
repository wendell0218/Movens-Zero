DEFAULT_PROMPTS = {
    "text": "Generate a motion sequence that matches the following description.",
    "music": "Generate a dance motion that follows the rhythm and energy of this music.",
    "speech": "Listen for the spoken task in the audio, then carry out that task as body motion.",
    "video": "The clip is the reference. Imitate the person's body movement as closely as possible.",
}

TEXT_PROMPT_FORMAT = "{prompt}\nDescription: {description}"


def build_prompt(mode, prompt=None, description=None):
    if mode not in DEFAULT_PROMPTS:
        raise ValueError(f"Unknown modality: {mode}")
    prompt = DEFAULT_PROMPTS[mode] if prompt is None else prompt.strip()
    if not prompt:
        raise ValueError("--prompt cannot be empty")
    if mode == "text":
        if description is None or not description.strip():
            raise ValueError("--description is required and cannot be empty for text mode")
        return TEXT_PROMPT_FORMAT.format(prompt=prompt, description=description.strip())
    if description is not None:
        raise ValueError("--description is only valid for text mode")
    return prompt

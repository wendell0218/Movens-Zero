from pathlib import Path

from safetensors import safe_open


MODEL_RELEASES = {
    "S": ("wendell0218/Movens-Zero-S", "5e229e1e398bdf33c309059e5ea3b561002a5515"),
    "M": ("wendell0218/Movens-Zero-M", "7ac8742c9bcd80900fd210965f454fa1bb6d8e91"),
    "L": ("wendell0218/Movens-Zero-L", "0755e4d06d4bb9f302ecc740a23d3865d4581590"),
    "XL": ("wendell0218/Movens-Zero-XL", "792a44d5f6be74079abdbcc199fa33e5ca5bfce5"),
}


def checkpoint_model_config(path: str | Path) -> dict[str, int]:
    with safe_open(str(path), framework="pt", device="cpu") as weights:
        keys = list(weights.keys())
        prefix = "action_header." if any(k.startswith("action_header.") for k in keys) else "action_expert."
        projection = weights.get_slice(prefix + "action_proj_in.ac_proj.2.weight").get_shape()
        output = weights.get_slice(prefix + "action_proj_out.linear.weight").get_shape()
        state = weights.get_slice(prefix + "obs_proj._obs_proc.1.weight").get_shape()
        position = weights.get_slice(prefix + "action_proj_in.dec_pos").get_shape()
        blocks = {int(k.split("transformer_blocks.", 1)[1].split(".")[0]) for k in keys if k.startswith(prefix + "transformer_blocks.")}
    width, dimension = projection
    if output != [dimension, width] or state[0] != width or blocks != set(range(len(blocks))):
        raise ValueError("Checkpoint has an inconsistent action expert architecture")
    if width % 64 or dimension not in (39, 40):
        raise ValueError(f"Unsupported action expert: width={width}, dimension={dimension}")
    return dict(hidden_dim=width, num_blocks=len(blocks), num_heads=width // 64,
                action_dim=dimension, state_dim=state[1], action_horizon=position[0])

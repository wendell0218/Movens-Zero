import torch
from root40_runtime.progress import build_progress_conditions


def configure_projection_fp32(model):
    if model.training:
        raise RuntimeError(
            "Projection-FP32 configuration is inference-only; call eval() first"
        )
    projection = model.action_header.action_proj_in
    master = {
        name: value.detach().clone() for name, value in projection.state_dict().items()
    }
    if any(
        (
            value.is_floating_point() and value.dtype != torch.float32
            for value in master.values()
        )
    ):
        raise RuntimeError(
            "Load original FP32 checkpoint projection weights before configuring inference"
        )
    model.action_header.bfloat16()
    projection.float().load_state_dict(master, strict=True)
    original_forward = projection.forward

    def projection_forward(noisy_actions, position_offset=0):
        with torch.autocast(device_type=noisy_actions.device.type, enabled=False):
            result = original_forward(
                noisy_actions.float(), position_offset=position_offset
            )
        if result.dtype != torch.float32:
            raise RuntimeError("Action input projection must produce FP32 tokens")
        return result

    projection.forward = projection_forward
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for name, parameter in model.action_header.named_parameters():
        expected = (
            torch.float32 if name.startswith("action_proj_in.") else torch.bfloat16
        )
        if parameter.dtype != expected:
            raise RuntimeError(
                f"Unexpected inference parameter precision: {name}: {parameter.dtype}"
            )


def build_case_progress(modality, frame_id, total_frames, action_horizon=30):
    if modality == "video":
        return build_progress_conditions(
            frame_id=0, total_frames=action_horizon + 1, action_horizon=action_horizon
        )
    return build_progress_conditions(
        frame_id=frame_id, total_frames=total_frames, action_horizon=action_horizon
    )


def step_flow_fp32(original_step, model_output, timestep, sample, **kwargs):
    with torch.autocast(device_type=sample.device.type, enabled=False):
        result = original_step(
            model_output=model_output.float(),
            timestep=timestep,
            sample=sample.float(),
            **kwargs,
        )
    value = result[0] if isinstance(result, tuple) else result.prev_sample
    if value.dtype != torch.float32:
        raise RuntimeError(
            "The root40 flow sampler must preserve FP32 samples at every step"
        )
    return result


class InferenceProcessor:
    def __init__(self, processor, device_type="cpu"):
        self.processor = processor
        self.device_type = device_type

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, *args, **kwargs):
        with (
            torch.autocast("cpu", enabled=False),
            torch.autocast(self.device_type, enabled=False),
        ):
            return self.processor(*args, **kwargs)

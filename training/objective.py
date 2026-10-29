import torch
import torch.nn as nn


def flow_random_batch(actions, seed, microstep, rank=0, world_size=1, max_delay=11):
    batch = len(actions)
    count = batch * world_size
    generator = torch.Generator().manual_seed(seed + microstep * 104729)
    sigma = (torch.arange(count) + torch.rand(count, generator=generator)) / count
    sigma = sigma[torch.randperm(count, generator=generator)]
    delays = torch.floor((torch.arange(count) + torch.rand((), generator=generator))
                         / count * max_delay).long()
    delays = delays[torch.randperm(count, generator=generator)]
    selection = slice(rank * batch, (rank + 1) * batch)
    sigma, delays = sigma[selection].to(actions.device), delays[selection].to(actions.device)
    noise_rng = torch.Generator(device=actions.device).manual_seed(
        seed + microstep * 1000003 + rank * 10007)
    noise = torch.randn(actions.shape, device=actions.device, dtype=torch.float32,
                        generator=noise_rng)
    return sigma, delays, noise


class FlowObjective(nn.Module):
    def __init__(self, model, cfg_dropout=0.1, group_weights=None):
        super().__init__()
        self.model, self.cfg_dropout = model, cfg_dropout
        weights = torch.ones(40) / 40
        groups = {"joint_abs": slice(0, 29), "root_delta_xy": slice(29, 31),
                  "root_rot6": slice(31, 37), "root_height": slice(37, 38),
                  "foot_contact": slice(38, 40)}
        for name, value in (group_weights or {}).items():
            if name not in groups or value < 0:
                raise ValueError(f"Invalid loss group: {name}={value}")
            weights[groups[name]] *= float(value)
        self.register_buffer("loss_weights", weights)

    def forward(self, inputs, actions, states, progress, valid, sigma, delays, noise):
        prefix = torch.arange(actions.shape[1], device=actions.device)[None] < delays[:, None]
        sigma_frames = sigma[:, None].expand_as(prefix).masked_fill(prefix, 0)
        noisy = (1 - sigma_frames[..., None]) * actions + sigma_frames[..., None] * noise
        with self.model._autocast():
            context = self.model._extract_context(inputs)
            context_mask = inputs["attention_mask"].clone()
            dropped = torch.rand(len(actions), device=actions.device) < self.cfg_dropout
            context_mask = context_mask.masked_fill(dropped[:, None], 0)
            prediction = self.model.action_expert(noisy,
                sigma_frames * self.model.train_diffusion_steps, context, states,
                progress if self.model.action_expert.use_future_progress else None,
                context_mask=context_mask)
        mask = valid & ~prefix
        if torch.any(mask.sum(1) == 0):
            raise ValueError("RTC prefix consumed every valid target frame; use longer motions")
        error = (prediction.float() - (noise - actions).float()).square()
        return ((error * mask[..., None]).sum(1).mean(0) * self.loss_weights).sum()

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)


MODES = ("text", "music", "speech", "video")
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        if name not in config:
            raise KeyError(f"Missing model config field: {name}")
        return config[name]
    if not hasattr(config, name):
        raise AttributeError(f"Missing model config field: {name}")
    return getattr(config, name)


def _model_config(config: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get("model", config)
    return getattr(config, "model", config)


def _validate_rtc(
    action_horizon: int, execution_horizon: int, delay: int, max_delay: int
) -> int:
    if not 0 < execution_horizon < action_horizon:
        raise ValueError(
            f"RTC requires 0 < execution horizon < action horizon, got "
            f"{execution_horizon} and {action_horizon}"
        )
    overlap = action_horizon - execution_horizon
    if delay != overlap:
        raise ValueError(f"RTC delay must be {overlap}, got {delay}")
    if overlap >= max_delay:
        raise ValueError(
            f"RTC overlap must be smaller than max delay, got {overlap} and {max_delay}"
        )
    return overlap


class _TimeEmbedding(nn.Module):
    def __init__(self, time_dim: int, output_dim: int) -> None:
        super().__init__()
        half_dim = time_dim // 2
        scale = np.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim) * -scale).float()
        self.w = nn.Parameter(frequencies, requires_grad=False)
        self.out_net = nn.Sequential(
            nn.Linear(time_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        phase = timestep[..., None] * self.w
        return self.out_net(torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1))


class _PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_length: int = 5000) -> None:
        super().__init__()
        encoding = torch.zeros(max_length, hidden_dim)
        position = torch.arange(max_length, dtype=torch.float).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float)
            * -(np.log(10000.0) / hidden_dim)
        )
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        self.pe = nn.Parameter(encoding.unsqueeze(1), requires_grad=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.pe[: tokens.shape[0]].repeat(1, tokens.shape[1], 1)


class _AdaptiveLayerNormContinuous(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_dim, hidden_dim * 2)
        self.norm = nn.LayerNorm(
            hidden_dim,
            elementwise_affine=False,
            eps=1e-5,
            bias=True,
        )

    def forward(self, tokens: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim == 2:
            embedding = embedding.unsqueeze(1)
        if embedding.ndim != 3:
            raise ValueError(f"Invalid conditioning rank: {embedding.ndim}")
        scale, shift = self.linear(self.silu(embedding).to(tokens.dtype)).chunk(
            2, dim=-1
        )
        return self.norm(tokens) * (1 + scale) + shift


class _AdaptiveLayerNormZero(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_dim, hidden_dim * 6)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        tokens: torch.Tensor,
        embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if embedding.ndim == 2:
            embedding = embedding.unsqueeze(1)
        if embedding.ndim != 3:
            raise ValueError(f"Invalid conditioning rank: {embedding.ndim}")
        values = self.linear(self.silu(embedding))
        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_feedforward,
            scale_feedforward,
            gate_feedforward,
        ) = values.chunk(6, dim=-1)
        normalized = self.norm(tokens) * (1 + scale_attention) + shift_attention
        return (
            normalized,
            gate_attention,
            shift_feedforward,
            scale_feedforward,
            gate_feedforward,
        )


class _JointAttentionProcessor:
    def __call__(
        self,
        attention: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]
        query = attention.to_q(hidden_states)
        key = attention.to_k(hidden_states)
        value = attention.to_v(hidden_states)
        head_dim = key.shape[-1] // attention.heads
        query = query.view(batch_size, -1, attention.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attention.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attention.heads, head_dim).transpose(1, 2)
        if attention.norm_q is not None:
            query = attention.norm_q(query)
        if attention.norm_k is not None:
            key = attention.norm_k(key)
        context_query = attention.add_q_proj(encoder_hidden_states)
        context_key = attention.add_k_proj(encoder_hidden_states)
        context_value = attention.add_v_proj(encoder_hidden_states)
        context_query = context_query.view(
            batch_size, -1, attention.heads, head_dim
        ).transpose(1, 2)
        context_key = context_key.view(
            batch_size, -1, attention.heads, head_dim
        ).transpose(1, 2)
        context_value = context_value.view(
            batch_size, -1, attention.heads, head_dim
        ).transpose(1, 2)
        if attention.norm_added_q is not None:
            context_query = attention.norm_added_q(context_query)
        if attention.norm_added_k is not None:
            context_key = attention.norm_added_k(context_key)
        query = torch.cat((query, context_query), dim=2)
        key = torch.cat((key, context_key), dim=2)
        value = torch.cat((value, context_value), dim=2)
        joint_mask = None
        if attention_mask is not None:
            prefix = torch.ones(
                hidden_states.shape[0],
                1,
                1,
                hidden_states.shape[1],
                device=attention_mask.device,
                dtype=torch.bool,
            )
            joint_mask = torch.cat(
                (prefix, (attention_mask == 1)[:, None, None, :]), dim=-1
            )
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=joint_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).reshape(
            batch_size, -1, attention.heads * head_dim
        )
        output = output.to(query.dtype)
        hidden_output = output[:, :residual_length]
        context_output = output[:, residual_length:]
        if not attention.context_pre_only:
            context_output = attention.to_add_out(context_output)
        hidden_output = attention.to_out[0](hidden_output)
        hidden_output = attention.to_out[1](hidden_output)
        return hidden_output, context_output


class _ObservationProjection(nn.Module):
    def __init__(self, state_dim: int, view_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.enc_pos = _PositionalEncoding(hidden_dim)
        self.views_proj = nn.Linear(view_dim, hidden_dim)
        self.feat_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.feat_traj_norm = nn.LayerNorm(
            hidden_dim, elementwise_affine=False, eps=1e-6
        )
        self._obs_proc = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(state_dim, hidden_dim)
        )
        self.post_proc = nn.Sequential(nn.Identity(), nn.Identity(), nn.Dropout(0.1))

    def forward(self, views: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        view_tokens = self.views_proj(views)
        batch_size, view_count, token_count, hidden_dim = view_tokens.shape
        tokens = view_tokens.reshape(batch_size, view_count * token_count, hidden_dim)
        state_tokens = self._obs_proc(states)
        tokens = self.post_proc(torch.cat((tokens, state_tokens), dim=1))
        transposed = tokens.transpose(0, 1)
        return (transposed + self.enc_pos(transposed)).transpose(0, 1)


class _ActionProjectionIn(nn.Module):
    def __init__(self, action_horizon: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.ac_proj = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_dim, hidden_dim),
        )
        self.dec_pos = nn.Parameter(torch.empty(action_horizon, hidden_dim))
        nn.init.xavier_uniform_(self.dec_pos)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        batch_size = actions.shape[0]
        tokens = self.ac_proj(actions.reshape(batch_size, -1, self.action_dim))
        if tokens.shape[1] != self.action_horizon:
            raise ValueError(
                f"Expected {self.action_horizon} action tokens, got {tokens.shape[1]}"
            )
        return tokens + self.dec_pos.unsqueeze(0)


class _ActionProjectionOut(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, action_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 2)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, tokens: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim == 2:
            embedding = embedding.unsqueeze(1)
        if embedding.ndim != 3:
            raise ValueError(f"Invalid conditioning rank: {embedding.ndim}")
        shift, scale = self.adaLN_modulation(embedding).chunk(2, dim=-1)
        return self.linear(self.norm_final(tokens) * (1 + scale) + shift)


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        final_block: bool,
        qk_norm: str | None,
    ) -> None:
        super().__init__()
        head_dim = hidden_dim // num_heads
        self.context_pre_only = final_block
        self.norm1_act = _AdaptiveLayerNormZero(hidden_dim)
        if final_block:
            self.norm1_obs = _AdaptiveLayerNormContinuous(hidden_dim)
        else:
            self.norm1_obs = _AdaptiveLayerNormZero(hidden_dim)
        self.attn = Attention(
            query_dim=hidden_dim,
            cross_attention_dim=None,
            added_kv_proj_dim=hidden_dim,
            dim_head=head_dim,
            heads=num_heads,
            out_dim=hidden_dim,
            context_pre_only=final_block,
            bias=True,
            processor=_JointAttentionProcessor(),
            qk_norm=qk_norm,
            eps=1e-6,
        )
        self.norm2_act = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.ff_act = FeedForward(
            dim=hidden_dim,
            dim_out=hidden_dim,
            activation_fn="gelu-approximate",
        )
        if final_block:
            self.norm2_obs = None
            self.ff_obs = None
        else:
            self.norm2_obs = nn.LayerNorm(
                hidden_dim, elementwise_affine=False, eps=1e-6
            )
            self.ff_obs = FeedForward(
                dim=hidden_dim,
                dim_out=hidden_dim,
                activation_fn="gelu-approximate",
            )

    def forward(
        self,
        action_tokens: torch.Tensor,
        observation_tokens: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        (
            action_norm,
            action_attention_gate,
            action_shift,
            action_scale,
            action_feedforward_gate,
        ) = self.norm1_act(action_tokens, time_embedding)
        observation_embedding = (
            time_embedding[:, -1] if time_embedding.ndim > 2 else time_embedding
        )
        if self.context_pre_only:
            observation_norm = self.norm1_obs(observation_tokens, observation_embedding)
            observation_attention_gate = None
            observation_shift = None
            observation_scale = None
            observation_feedforward_gate = None
        else:
            (
                observation_norm,
                observation_attention_gate,
                observation_shift,
                observation_scale,
                observation_feedforward_gate,
            ) = self.norm1_obs(observation_tokens, observation_embedding)
        action_attention, observation_attention = self.attn(
            hidden_states=action_norm,
            encoder_hidden_states=observation_norm,
        )
        action_tokens = action_tokens + action_attention_gate * action_attention
        action_norm = self.norm2_act(action_tokens)
        action_norm = action_norm * (1 + action_scale) + action_shift
        action_tokens = action_tokens + action_feedforward_gate * self.ff_act(
            action_norm
        )
        if self.context_pre_only:
            return action_tokens, None
        if (
            observation_attention_gate is None
            or observation_shift is None
            or observation_scale is None
            or observation_feedforward_gate is None
            or self.norm2_obs is None
            or self.ff_obs is None
        ):
            raise RuntimeError("Incomplete observation branch")
        observation_tokens = (
            observation_tokens + observation_attention_gate * observation_attention
        )
        observation_norm = self.norm2_obs(observation_tokens)
        observation_norm = (
            observation_norm * (1 + observation_scale) + observation_shift
        )
        observation_tokens = (
            observation_tokens
            + observation_feedforward_gate * self.ff_obs(observation_norm)
        )
        return action_tokens, observation_tokens


class _ActionExpert(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.action_dim = int(_config_value(config, "action_dim"))
        self.action_horizon = int(_config_value(config, "action_horizon"))
        self.state_dim = int(_config_value(config, "state_dim"))
        self.hidden_dim = int(_config_value(config, "hidden_dim"))
        self.num_blocks = int(_config_value(config, "num_blocks"))
        self.num_heads = int(_config_value(config, "num_heads"))
        self.view_dim = int(_config_value(config, "view_feature_dim"))
        self.use_future_progress = bool(_config_value(config, "use_future_progress"))
        self.future_progress_dim = int(_config_value(config, "future_progress_dim"))
        self.time_ins_embed = _TimeEmbedding(256, self.hidden_dim)
        self.obs_proj = _ObservationProjection(
            self.state_dim, self.view_dim, self.hidden_dim
        )
        self.action_proj_in = _ActionProjectionIn(
            self.action_horizon,
            self.action_dim,
            self.hidden_dim,
        )
        if self.use_future_progress:
            self.future_progress_mlp = nn.Sequential(
                nn.Linear(self.future_progress_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            nn.init.zeros_(self.future_progress_mlp[-1].weight)
            nn.init.zeros_(self.future_progress_mlp[-1].bias)
        else:
            self.future_progress_mlp = None
        qk_norm = _config_value(config, "qk_norm")
        self.transformer_blocks = nn.ModuleList(
            [
                _TransformerBlock(
                    self.hidden_dim,
                    self.num_heads,
                    index == self.num_blocks - 1,
                    qk_norm,
                )
                for index in range(self.num_blocks)
            ]
        )
        self.action_proj_out = _ActionProjectionOut(self.hidden_dim, self.action_dim)

    def forward(
        self,
        action_samples: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        states: torch.Tensor,
        future_progress: torch.Tensor | None,
    ) -> torch.Tensor:
        time_embedding = self.time_ins_embed(timestep)
        action_tokens = self.action_proj_in(action_samples)
        if self.use_future_progress:
            if future_progress is None or self.future_progress_mlp is None:
                raise ValueError("Future progress is required by this checkpoint")
            if future_progress.shape[:2] != action_tokens.shape[:2]:
                raise ValueError(
                    f"Future progress shape {future_progress.shape} does not match "
                    f"action timeline {action_tokens.shape[:2]}"
                )
            progress = future_progress.to(action_tokens.device, action_tokens.dtype)
            action_tokens = action_tokens + self.future_progress_mlp(progress)
        observation_tokens = self.obs_proj(context, states)
        for block in self.transformer_blocks:
            action_tokens, next_observation_tokens = block(
                action_tokens,
                observation_tokens,
                time_embedding,
            )
            if next_observation_tokens is not None:
                observation_tokens = next_observation_tokens
        return self.action_proj_out(action_tokens, time_embedding)


class MovensZeroModel(nn.Module):
    def __init__(self, backbone: nn.Module, processor: Any, config: Any) -> None:
        super().__init__()
        self.config = _model_config(config)
        self.backbone = backbone
        self.processor = processor
        self.action_expert = _ActionExpert(self.config)
        self.action_dim = int(_config_value(self.config, "action_dim"))
        self.action_horizon = int(_config_value(self.config, "action_horizon"))
        self.action_exec_horizon = int(
            _config_value(self.config, "action_exec_horizon")
        )
        self.rtc_max_delay = int(_config_value(self.config, "rtc_max_delay"))
        self.train_diffusion_steps = int(
            _config_value(self.config, "train_diffusion_steps")
        )
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.train_diffusion_steps
        )
        self.mode = "text"
        self.runtime_device = torch.device("cpu")
        self.backbone_dtype = next(self.backbone.parameters()).dtype
        _validate_rtc(
            self.action_horizon,
            self.action_exec_horizon,
            self.action_horizon - self.action_exec_horizon,
            self.rtc_max_delay,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        backbone_path: str | Path,
        config: Any,
        device: str | torch.device = "cuda:0",
        torch_dtype: torch.dtype | None = None,
        attn_implementation: str = "sdpa",
    ) -> MovensZeroModel:
        if attn_implementation not in ("sdpa", "flash_attention_2"):
            raise ValueError(
                f"Unsupported attention implementation: {attn_implementation}"
            )
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        backbone_reference = str(backbone_path)
        local_backbone = Path(backbone_reference).expanduser()
        if (local_backbone / "config.json").is_file():
            backbone_reference = str(local_backbone.resolve())
        backbone_config = AutoConfig.from_pretrained(backbone_reference)
        if getattr(backbone_config, "model_type", None) != "qwen2_5_omni":
            raise ValueError(
                f"Expected a Qwen2.5-Omni backbone, got {getattr(backbone_config, 'model_type', None)}"
            )
        runtime_device = torch.device(device)
        if torch_dtype is None:
            torch_dtype = (
                torch.bfloat16 if runtime_device.type == "cuda" else torch.float32
            )
        backbone = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            backbone_reference,
            attn_implementation=attn_implementation,
            dtype=torch_dtype,
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(backbone_reference)
        model = cls(backbone, processor, config)
        checkpoint = load_file(str(checkpoint_path), device="cpu")
        expert_state = {}
        invalid_keys = []
        for key, value in checkpoint.items():
            if key.startswith("action_header."):
                expert_state[key[len("action_header.") :]] = value
            elif key.startswith("action_expert."):
                expert_state[key[len("action_expert.") :]] = value
            else:
                invalid_keys.append(key)
        if invalid_keys:
            raise RuntimeError(
                f"Unsupported checkpoint tensor keys: {invalid_keys[:8]}"
            )
        model.action_expert.load_state_dict(expert_state, strict=True)
        model.runtime_device = runtime_device
        model.backbone_dtype = torch_dtype
        model.to(runtime_device)
        model.eval()
        return model

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode

    def _messages(
        self, prompts: Sequence[str], mode: str
    ) -> list[list[dict[str, Any]]]:
        messages = []
        for prompt in prompts:
            content = []
            if mode in ("music", "speech"):
                content.append({"type": "audio", "audio": "audio.wav"})
            elif mode == "video":
                content.append({"type": "video", "video": "video.mp4"})
            if prompt.strip() or mode == "text":
                content.append({"type": "text", "text": prompt.strip()})
            messages.append(
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                    },
                    {"role": "user", "content": content},
                ]
            )
        return messages

    def _prepare_inputs(
        self,
        prompts: Sequence[str],
        mode: str,
        sounds: Sequence[np.ndarray] | None,
        videos: Sequence[np.ndarray] | None,
        video_sample_fps: float,
        video_size_pixels: int,
    ) -> dict[str, torch.Tensor]:
        messages = self._messages(prompts, mode)
        texts = [
            self.processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]
        if mode == "text":
            inputs = self.processor(text=texts, padding=True, return_tensors="pt")
        elif mode in ("music", "speech"):
            if sounds is None or len(sounds) != len(prompts):
                raise ValueError(f"{mode} mode requires one sound waveform per prompt")
            inputs = self.processor(
                text=texts,
                audio=[np.asarray(sound, dtype=np.float32) for sound in sounds],
                padding=True,
                return_tensors="pt",
            )
        else:
            if videos is None or len(videos) != len(prompts):
                raise ValueError("video mode requires one RGB video array per prompt")
            inputs = self.processor(
                text=texts,
                videos=[np.asarray(video, dtype=np.uint8) for video in videos],
                padding=True,
                return_tensors="pt",
                use_audio_in_video=False,
                videos_kwargs={
                    "size": {
                        "shortest_edge": int(video_size_pixels),
                        "longest_edge": int(video_size_pixels),
                    },
                    "fps": float(video_sample_fps),
                },
            )
            pixels = inputs["pixel_values_videos"]
            inputs["pixel_values_videos"] = pixels.to(torch.float16).to(torch.float32)
        return {
            key: value.to(self.runtime_device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }

    def _extract_context(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        input_features = inputs.get("input_features")
        pixel_values = inputs.get("pixel_values")
        pixel_values_videos = inputs.get("pixel_values_videos")
        image_grid_thw = inputs.get("image_grid_thw")
        video_grid_thw = inputs.get("video_grid_thw")
        feature_attention_mask = inputs.get("feature_attention_mask")
        video_second_per_grid = inputs.get("video_second_per_grid")
        embedded = self.backbone.get_input_embeddings()(input_ids)
        if input_features is not None:
            sound_lengths = (
                torch.sum(feature_attention_mask, dim=1)
                if feature_attention_mask is not None
                else None
            )
            sound_features = self.backbone.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=sound_lengths,
            )
            sound_features = sound_features.to(embedded.device, embedded.dtype)
            _, _, sound_mask = self.backbone.get_placeholder_mask(
                input_ids,
                inputs_embeds=embedded,
            )
            embedded = embedded.masked_scatter(sound_mask, sound_features)
        if pixel_values is not None:
            image_features = self.backbone.get_image_features(
                pixel_values, image_grid_thw
            )
            image_features = image_features.to(embedded.device, embedded.dtype)
            image_mask, _, _ = self.backbone.get_placeholder_mask(
                input_ids,
                inputs_embeds=embedded,
                image_features=image_features,
            )
            embedded = embedded.masked_scatter(image_mask, image_features)
        if pixel_values_videos is not None:
            video_features = self.backbone.get_video_features(
                pixel_values_videos,
                video_grid_thw,
            )
            video_features = video_features.to(embedded.device, embedded.dtype)
            _, video_mask, _ = self.backbone.get_placeholder_mask(
                input_ids,
                inputs_embeds=embedded,
                video_features=video_features,
            )
            embedded = embedded.masked_scatter(video_mask, video_features)
        sound_lengths = (
            torch.sum(feature_attention_mask, dim=1)
            if feature_attention_mask is not None
            else None
        )
        position_ids = None
        if attention_mask is not None:
            position_ids, _ = self.backbone.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask,
                False,
                sound_lengths,
                video_second_per_grid,
            )
        outputs = self.backbone.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=embedded,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1].unsqueeze(1)

    def _normalize_batch(
        self,
        states: torch.Tensor | np.ndarray,
        prompt: str | Sequence[str],
        future_progress: torch.Tensor | np.ndarray | None,
        sound: np.ndarray | Sequence[np.ndarray] | None,
        video: np.ndarray | Sequence[np.ndarray] | None,
    ) -> tuple[
        torch.Tensor,
        list[str],
        torch.Tensor | None,
        list[np.ndarray] | None,
        list[np.ndarray] | None,
        bool,
    ]:
        states_tensor = torch.as_tensor(
            states, dtype=torch.float32, device=self.runtime_device
        )
        unbatched = states_tensor.ndim == 2
        if unbatched:
            states_tensor = states_tensor.unsqueeze(0)
        if states_tensor.ndim != 3:
            raise ValueError(f"States must have rank 2 or 3, got {states_tensor.ndim}")
        batch_size = states_tensor.shape[0]
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if len(prompts) != batch_size:
            raise ValueError(f"Expected {batch_size} prompts, got {len(prompts)}")
        progress_tensor = None
        if future_progress is not None:
            progress_tensor = torch.as_tensor(
                future_progress,
                dtype=torch.float32,
                device=self.runtime_device,
            )
            if progress_tensor.ndim == 2:
                progress_tensor = progress_tensor.unsqueeze(0)
            if progress_tensor.shape[:2] != (batch_size, self.action_horizon):
                raise ValueError(
                    f"Expected future progress shape ({batch_size}, {self.action_horizon}, D), "
                    f"got {tuple(progress_tensor.shape)}"
                )
        sounds = None
        if sound is not None:
            if isinstance(sound, np.ndarray) and sound.ndim == 1:
                sounds = [sound]
            else:
                sounds = list(sound)
        videos = None
        if video is not None:
            if isinstance(video, np.ndarray) and video.ndim == 4:
                videos = [video]
            else:
                videos = list(video)
        return states_tensor, prompts, progress_tensor, sounds, videos, unbatched

    def _autocast(self) -> Any:
        if self.runtime_device.type == "cuda" and self.backbone_dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            return torch.autocast("cuda", dtype=self.backbone_dtype)
        return contextlib.nullcontext()

    @torch.inference_mode()
    def predict_action(
        self,
        states: torch.Tensor | np.ndarray,
        prompt: str | Sequence[str],
        mode: str | None = None,
        num_inference_steps: int = 10,
        future_progress: torch.Tensor | np.ndarray | None = None,
        sound: np.ndarray | Sequence[np.ndarray] | None = None,
        video: np.ndarray | Sequence[np.ndarray] | None = None,
        video_sample_fps: float = 8.0,
        video_size_pixels: int = 50176,
    ) -> torch.Tensor:
        selected_mode = self.mode if mode is None else mode
        self.set_mode(selected_mode)
        states_tensor, prompts, progress, sounds, videos, unbatched = (
            self._normalize_batch(
                states,
                prompt,
                future_progress,
                sound,
                video,
            )
        )
        inputs = self._prepare_inputs(
            prompts,
            selected_mode,
            sounds,
            videos,
            video_sample_fps,
            video_size_pixels,
        )
        batch_size = states_tensor.shape[0]
        with self._autocast():
            context = self._extract_context(inputs).to(states_tensor.dtype)
            actions = torch.randn(
                batch_size,
                self.action_horizon,
                self.action_dim,
                device=self.runtime_device,
            )
            self.noise_scheduler.set_timesteps(num_inference_steps)
            for timestep in self.noise_scheduler.timesteps:
                batched_timestep = timestep.expand(batch_size).to(self.runtime_device)
                velocity = self.action_expert(
                    actions,
                    batched_timestep,
                    context,
                    states_tensor,
                    progress,
                )
                actions = self.noise_scheduler.step(
                    model_output=velocity,
                    timestep=timestep,
                    sample=actions,
                ).prev_sample
        actions = actions.float()
        return actions[0] if unbatched else actions

    @torch.inference_mode()
    def predict_action_rtc(
        self,
        states: torch.Tensor | np.ndarray,
        prompt: str | Sequence[str],
        previous_actions: torch.Tensor | np.ndarray,
        inference_delay: int,
        max_delay: int,
        mode: str | None = None,
        num_inference_steps: int = 10,
        future_progress: torch.Tensor | np.ndarray | None = None,
        sound: np.ndarray | Sequence[np.ndarray] | None = None,
        video: np.ndarray | Sequence[np.ndarray] | None = None,
        video_sample_fps: float = 8.0,
        video_size_pixels: int = 50176,
    ) -> torch.Tensor:
        selected_mode = self.mode if mode is None else mode
        self.set_mode(selected_mode)
        delay = _validate_rtc(
            self.action_horizon,
            self.action_exec_horizon,
            int(inference_delay),
            int(max_delay),
        )
        if int(max_delay) != self.rtc_max_delay:
            raise ValueError(
                f"RTC max delay does not match checkpoint config: {max_delay} and {self.rtc_max_delay}"
            )
        states_tensor, prompts, progress, sounds, videos, unbatched = (
            self._normalize_batch(
                states,
                prompt,
                future_progress,
                sound,
                video,
            )
        )
        batch_size = states_tensor.shape[0]
        previous = torch.as_tensor(
            previous_actions,
            dtype=torch.float32,
            device=self.runtime_device,
        )
        if previous.ndim == 2:
            previous = previous.unsqueeze(0)
        expected_shape = (batch_size, self.action_horizon, self.action_dim)
        if tuple(previous.shape) != expected_shape:
            raise ValueError(
                f"Expected previous actions shape {expected_shape}, got {tuple(previous.shape)}"
            )
        inputs = self._prepare_inputs(
            prompts,
            selected_mode,
            sounds,
            videos,
            video_sample_fps,
            video_size_pixels,
        )
        prefix_mask = (
            torch.arange(self.action_horizon, device=self.runtime_device)[None, :]
            < delay
        ).expand(batch_size, -1)
        with self._autocast():
            context = self._extract_context(inputs).to(states_tensor.dtype)
            actions = torch.randn(
                batch_size,
                self.action_horizon,
                self.action_dim,
                device=self.runtime_device,
            )
            self.noise_scheduler.set_timesteps(num_inference_steps)
            for timestep in self.noise_scheduler.timesteps:
                batched_timestep = torch.where(
                    prefix_mask,
                    torch.zeros((), device=self.runtime_device, dtype=timestep.dtype),
                    timestep,
                )
                actions = torch.where(prefix_mask[:, :, None], previous, actions)
                velocity = self.action_expert(
                    actions,
                    batched_timestep,
                    context,
                    states_tensor,
                    progress,
                )
                actions = self.noise_scheduler.step(
                    model_output=velocity,
                    timestep=timestep,
                    sample=actions,
                ).prev_sample
            actions = torch.where(prefix_mask[:, :, None], previous, actions)
        actions = actions.float()
        return actions[0] if unbatched else actions

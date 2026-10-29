import os
import json
import copy
import contextlib
from tqdm import tqdm
from typing import List, Optional, Tuple, Union, Dict, Any
from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import Self
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from dataclasses import dataclass
from diffusers.utils.outputs import BaseOutput
from diffusers.configuration_utils import ConfigMixin, FrozenDict, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings
from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0
from diffusers.models.attention import FeedForward, _chunked_feed_forward

LaunchConfig = Any
from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniProcessor,
    AutoConfig,
    AutoProcessor,
    Qwen2TokenizerFast,
    Qwen3VLProcessor,
)
from qwen_vl_utils import process_vision_info
from .utils import initialize_overwatch, count_parameters

overwatch = initialize_overwatch(__name__)
from transformers.models.siglip import SiglipModel
from transformers.models.dinov2 import Dinov2Model
from diffusers.utils import logging

logger = logging.get_logger(__name__)
QWEN3VL_VARIANT = "Qwen/Qwen3-VL-2B-Instruct"
OMNI_DEFAULT_SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."


def _validate_rtc_inference_contract(
    *,
    action_horizon: int,
    action_exec_horizon: int,
    inference_delay: int,
    max_delay: int,
) -> int:
    horizon = int(action_horizon)
    execute = int(action_exec_horizon)
    delay = int(inference_delay)
    delay_limit = int(max_delay)
    if not 0 < execute < horizon:
        raise ValueError(
            f"RTC requires 0 < action_exec_horizon < action_horizon, got execute={execute}, horizon={horizon}"
        )
    expected_overlap = horizon - execute
    if delay != expected_overlap:
        raise ValueError(
            f"RTC inference_delay must equal the overlap used by training and deployment: got inference_delay={delay}, expected={expected_overlap} (action_horizon={horizon}, action_exec_horizon={execute})"
        )
    if expected_overlap >= delay_limit:
        raise ValueError(
            f"RTC overlap must be smaller than max_delay, got overlap={expected_overlap}, max_delay={delay_limit}"
        )
    return expected_overlap


@dataclass
class HumanFoundationModelOutput(BaseOutput):
    action: "torch.Tensor"
    absolute_height_losses: Optional[Dict[str, torch.Tensor]] = None

    def to_tuple(self):
        return (None, self.action)


@dataclass
class ActionTransformerModelOutput(BaseOutput):
    action: "torch.Tensor"

    def to_tuple(self):
        return (None, self.action)


class _TimeNetwork(nn.Module):
    def __init__(self, time_dim, out_dim, learnable_w=False):
        assert time_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(time_dim // 2)
        super().__init__()
        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.w = nn.Parameter(w, requires_grad=learnable_w)
        self.out_net = nn.Sequential(
            nn.Linear(time_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        x = x[..., None] * self.w
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
        return self.out_net(x)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        if isinstance(dim, int):
            dim = (dim,)
        self.dim = torch.Size(dim)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)
        return hidden_states


if torch.__version__ >= "2.1.0":
    LayerNorm = nn.LayerNorm
else:

    class LayerNorm(nn.Module):
        def __init__(
            self,
            dim,
            eps: float = 1e-05,
            elementwise_affine: bool = True,
            bias: bool = True,
        ):
            super().__init__()
            self.eps = eps
            if isinstance(dim, int):
                dim = (dim,)
            self.dim = torch.Size(dim)
            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(dim))
                self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
            else:
                self.weight = None
                self.bias = None

        def forward(self, input):
            return F.layer_norm(input, self.dim, self.weight, self.bias, self.eps)


class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-05,
        bias=True,
        norm_type="layer_norm",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(
            conditioning_embedding_dim, embedding_dim * 2, bias=bias
        )
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(
        self, x: torch.Tensor, conditioning_embedding: torch.Tensor
    ) -> torch.Tensor:
        if len(conditioning_embedding.shape) == 2:
            conditioning_embedding = conditioning_embedding.unsqueeze(1)
        elif len(conditioning_embedding.shape) == 3:
            pass
        else:
            raise ValueError(
                f"Invalid shape of conditioning_embedding: {conditioning_embedding.shape}"
            )
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class AdaLayerNormZero(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        norm_type="layer_norm",
        bias=True,
    ):
        super().__init__()
        if num_embeddings is not None:
            self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        else:
            self.emb = None
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-06)
        elif norm_type == "fp32_layer_norm":
            self.norm = FP32LayerNorm(
                embedding_dim, elementwise_affine=False, bias=False
            )
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
            assert False, "Not implemented with emb shape == 3, (B, T, D)"
        if len(emb.shape) == 2:
            emb = emb.unsqueeze(1)
        elif len(emb.shape) == 3:
            pass
        else:
            raise ValueError(f"Invalid shape of emb: {emb.shape}")
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(
            6, dim=-1
        )
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        pe = self.pe[: x.shape[0]]
        pe = pe.repeat((1, x.shape[1], 1))
        return pe.detach().clone()


class FilmConditioning(nn.Module):
    def __init__(self, in_dim: int, num_channels: int):
        super(FilmConditioning, self).__init__()
        self._projection_add = nn.Linear(in_dim, num_channels)
        self._projection_mult = nn.Linear(in_dim, num_channels)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, conv_filters, conditioning):
        assert len(conditioning.shape) == 2
        projected_cond_add = self._projection_add(conditioning)
        projected_cond_mult = self._projection_mult(conditioning)
        if len(conv_filters.shape) == 4:
            projected_cond_add = projected_cond_add.unsqueeze(-1).unsqueeze(-1)
            projected_cond_mult = projected_cond_mult.unsqueeze(-1).unsqueeze(-1)
        elif len(conv_filters.shape) == 3:
            projected_cond_add = projected_cond_add.unsqueeze(1)
            projected_cond_mult = projected_cond_mult.unsqueeze(1)
        else:
            assert len(conv_filters.shape) == 2
        result = (1 + projected_cond_mult) * conv_filters + projected_cond_add
        return result


class JointVLAAttnProcessor:
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
        context_pre_only=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "JointAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        batch_size = hidden_states.shape[0]
        query = attn.to_q(hidden_states)
        assert attn.to_k is not None
        key = attn.to_k(hidden_states)
        assert attn.to_v is not None
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(
                    encoder_hidden_states_key_proj
                )
            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)
            if attention_mask is not None:
                assert attention_mask.dtype == torch.float32
                attn_mask = torch.cat(
                    [
                        torch.ones(
                            hidden_states.shape[0],
                            1,
                            1,
                            hidden_states.shape[1],
                            device=attention_mask.device,
                        ).to(torch.bool),
                        (attention_mask == 1)[:, None, None, :],
                    ],
                    dim=-1,
                )
            else:
                attn_mask = None
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attn_mask
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return (hidden_states, encoder_hidden_states)


class ObservationProjection(nn.Module):
    def __init__(
        self,
        action_pred_horizon: int = 50,
        action_dim: int = 7,
        output_dim: int = 512,
        hidden_dim: int = 1280,
        n_conditions: int = 1,
        timestep_in_dim: int = 320,
        timestep_flip_sin_to_cos: bool = True,
        timestep_freq_shift: int = 0,
        token_fusion: str = "concat",
        resnet_store_path: str = "cache/visual_features/resnet18/IN_1M_resnet18.pth",
        odim: int = 32,
        view_feature_dim: int = 1920,
        use_film: bool = False,
    ):
        super().__init__()
        self.action_pred_horizon = action_pred_horizon
        self.action_dim = action_dim
        self.enc_pos = _PositionalEncoding(d_model=output_dim)
        n_cams = 0
        imgs_per_cam = 1
        self.camera_indices = []
        self._n_cams = n_cams
        self._share_cam_features = False
        early_fusion = False
        text_dim = 768
        self.token_fusion = token_fusion
        if self._share_cam_features:
            ...
        else:
            if n_conditions > 0:
                from dit_policy.data4robotics.models.resnet import ResNet

                features = ResNet(
                    size=18,
                    weights="IMAGENET1K_V1",
                    restore_path=resnet_store_path,
                    avg_pool=False,
                    conv_repeat=0,
                    norm_cfg={"name": "group_norm", "num_groups": 16},
                )
            else:
                features = None
            feat_list = [copy.deepcopy(features) for _ in range(0, n_conditions)]
            self.visual_features = nn.ModuleList(feat_list)
            self.views_proj = nn.Linear(view_feature_dim, output_dim, bias=True)
        self.early_fusion = early_fusion
        self.imgs_per_cam = imgs_per_cam
        imgs_per_cam = 1 if early_fusion else imgs_per_cam
        self._n_conditions = n_conditions
        self.embed_proj = nn.Identity()
        if self.token_fusion != "perceiver" and self._n_conditions > 0:
            assert features is not None
            self._token_dim = features.embed_dim
            if output_dim != self._token_dim:
                self.embed_proj = nn.Linear(self._token_dim, output_dim, bias=False)
        else:
            self._token_dim = 0
        self._n_tokens = (
            imgs_per_cam * n_cams * self._token_dim + n_conditions * self._token_dim
        )
        self.traj_n_token = self._token_dim
        self.feat_norm = nn.LayerNorm(output_dim, elementwise_affine=False, eps=1e-06)
        self.feat_traj_norm = nn.LayerNorm(
            output_dim, elementwise_affine=False, eps=1e-06
        )
        if self.token_fusion == "cross":
            self.attn = Attention(
                query_dim=self._token_dim,
                cross_attention_dim=1920,
                dim_head=64,
                out_dim=output_dim,
                heads=output_dim // 64,
                bias=True,
                eps=1e-06,
            )
        elif self.token_fusion == "perceiver":
            num_queries = 256
            self.latents1 = nn.Parameter(
                torch.randn(1, num_queries, hidden_dim) / hidden_dim**0.5
            )
            self.cross_attn1 = Attention(
                query_dim=hidden_dim,
                cross_attention_dim=hidden_dim,
                dim_head=64,
                out_dim=output_dim,
                heads=output_dim // 64,
                bias=True,
                eps=1e-06,
            )
            self.latents2 = nn.Parameter(
                torch.randn(1, num_queries, hidden_dim) / hidden_dim**0.5
            )
            self.proj_in2 = nn.Linear(self._token_dim, hidden_dim)
            self.cross_attn2 = Attention(
                query_dim=hidden_dim,
                cross_attention_dim=hidden_dim,
                dim_head=64,
                out_dim=output_dim,
                heads=output_dim // 64,
                bias=True,
                eps=1e-06,
            )
        self.film = FilmConditioning(2048, output_dim) if use_film else None
        self._obs_strat = "add_token"
        self._n_tokens += 1
        self._obs_proc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(odim, output_dim))
        dropout = 0.1
        linear_proj = nn.Identity()
        norm = nn.Identity()
        self.post_proc = nn.Sequential(linear_proj, norm, nn.Dropout(dropout))

    def forward(
        self, views, obs, traj2ds=None, text_embeddings=None, vlm_attn_mask=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s_t, vlm_attn_mask = self.tokenize_obs(
            views,
            obs,
            traj2ds,
            text_embeddings=text_embeddings,
            vlm_attn_mask=vlm_attn_mask,
        )
        s_t = s_t.transpose(0, 1)
        pos = self.enc_pos(s_t)
        return ((s_t + pos).transpose(0, 1), vlm_attn_mask)

    def tokenize_obs(
        self,
        views,
        obs,
        traj2ds=None,
        flatten=False,
        text_embeddings=None,
        vlm_attn_mask=None,
    ):
        view_tokens = self.views_proj(views)
        B, V, S, D = view_tokens.shape
        if traj2ds is not None:
            traj_tokens = self.embed(
                {}, {f"cond{i}": traj2ds[:, i] for i in range(traj2ds.shape[1])}
            )
            if self.token_fusion == "concat":
                traj_tokens = self.embed_proj(traj_tokens)
                tokens = torch.cat([view_tokens.view(B, V * S, D), traj_tokens], dim=1)
            elif self.token_fusion == "cross":
                traj_tokens_list, view_tokens_list = ([], [])
                for v_idx in range(traj2ds.shape[1]):
                    view_tokens = self.attn(
                        hidden_states=traj_tokens[
                            :,
                            v_idx * self.traj_n_token,
                            (v_idx + 1) * self.traj_n_token,
                        ],
                        encoder_hidden_states=views[:, v_idx],
                    )
                    traj_tokens_list.append(traj_tokens)
                    view_tokens_list.append(view_tokens)
                traj_tokens = torch.cat(traj_tokens_list, dim=1)
                traj_tokens = self.embed_proj(traj_tokens)
                view_tokens = torch.cat(view_tokens_list, dim=1)
                tokens = torch.cat([view_tokens, traj_tokens], dim=1)
            elif self.token_fusion == "perceiver":
                latents1 = self.latents1.repeat(B, 1, 1)
                tokens1 = self.cross_attn1(
                    hidden_states=latents1,
                    encoder_hidden_states=view_tokens.view(B, V * S, D),
                )
                latents2 = self.latents2.repeat(B, 1, 1)
                traj_tokens = self.proj_in2(traj_tokens)
                tokens2 = self.cross_attn2(
                    hidden_states=latents2, encoder_hidden_states=traj_tokens
                )
                tokens = torch.cat([tokens1, tokens2], dim=1)
            else:
                raise ValueError
        else:
            assert self._n_conditions == 0, "inconsistent confg"
            tokens = view_tokens.view(B, V * S, D)
        if self.film is not None and text_embeddings is not None:
            tokens = self.film(tokens, text_embeddings)
        if self._obs_strat == "add_token":
            obs_token = self._obs_proc(obs)
            tokens = torch.cat((tokens, obs_token), 1)
            if vlm_attn_mask is not None:
                obs_token_count = obs_token.shape[1] if len(obs_token.shape) == 3 else 1
                obs_token_mask = torch.ones(
                    (vlm_attn_mask.shape[0], obs_token_count),
                    device=vlm_attn_mask.device,
                )
                vlm_attn_mask = torch.cat([vlm_attn_mask, obs_token_mask], 1)
        elif self._obs_strat == "pad_img_tokens":
            obs = self._obs_proc(obs)
            obs = obs[:, None].repeat((1, tokens.shape[1], 1))
            tokens = torch.cat((obs, tokens), 2)
            vlm_attn_mask = None
        else:
            assert self._obs_strat is None
        tokens = self.post_proc(tokens)
        if flatten:
            return (tokens.reshape((tokens.shape[0], -1)), vlm_attn_mask)
        return (tokens, vlm_attn_mask)

    def embed(self, imgs, conditions=None):

        def embed_helper(net, im):
            if self.early_fusion and len(im.shape) == 5:
                T = im.shape[1]
                im = torch.cat([im[:, t] for t in range(T)], 1)
                embeds = net(im)
            elif len(im.shape) == 5:
                B, T, C, H, W = im.shape
                embeds = net(im.reshape((B * T, C, H, W)))
                embeds = embeds.reshape((B, -1, net.embed_dim))
            else:
                assert len(im.shape) == 4
                embeds = net(im)
            return embeds

        if self._share_cam_features:
            embeds = [
                embed_helper(self.visual_features, imgs[f"cam{i}"])
                for i in self.camera_indices
            ]
            if conditions is not None:
                assert self._n_conditions > 0
                embeds += [
                    embed_helper(self.visual_features, conditions[f"cond{i}"])
                    for i in range(self._n_conditions)
                ]
        else:
            assert conditions is not None
            cond_encoders = self.visual_features
            embeds = [
                embed_helper(net, conditions[f"cond{0}"])
                for i, net in enumerate(cond_encoders)
            ]
        assert conditions is not None
        in_dtype = (
            next(iter(imgs.values())).dtype
            if len(imgs) > 0
            else next(iter(conditions.values())).dtype
        )
        embeds = torch.cat(embeds, dim=1).to(dtype=in_dtype)
        return embeds


class ActionProjectionIn(nn.Module):
    def __init__(self, action_pred_horizon, action_dim, output_dim):
        super().__init__()
        self.action_pred_horizon = action_pred_horizon
        self.action_dim = action_dim
        self.force_fp32 = int(action_dim) == 40
        self.ac_proj = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_dim, output_dim),
        )
        self.dec_pos = nn.Parameter(
            torch.empty(action_pred_horizon, output_dim), requires_grad=True
        )
        nn.init.xavier_uniform_(self.dec_pos.data)

    def forward(
        self, noisy_actions: torch.Tensor, position_offset: int = 0
    ) -> torch.Tensor:
        if self.force_fp32 and any(
            (parameter.dtype != torch.float32 for parameter in self.parameters())
        ):
            raise RuntimeError(
                "Root40 action input projection requires original FP32 parameters"
            )
        precision_context = (
            torch.autocast(device_type=noisy_actions.device.type, enabled=False)
            if self.force_fp32
            else contextlib.nullcontext()
        )
        with precision_context:
            B = noisy_actions.shape[0]
            projection_input = (
                noisy_actions.float() if self.force_fp32 else noisy_actions
            )
            noise_acs = projection_input.reshape(B, -1, self.action_dim)
            ac_tokens = self.ac_proj(noise_acs)
            token_count = ac_tokens.shape[1]
            position_end = int(position_offset) + token_count
            if position_offset < 0 or position_end > self.action_pred_horizon:
                raise ValueError(
                    f"Action token positions exceed the configured horizon: offset={position_offset}, tokens={token_count}, horizon={self.action_pred_horizon}"
                )
            action_hidden_states = ac_tokens + self.dec_pos[
                int(position_offset) : position_end
            ].unsqueeze(0)
        return action_hidden_states


class ActionProjectionOut(nn.Module):
    def __init__(self, hidden_size, action_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.linear = nn.Linear(hidden_size, action_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.reset_parameters()

    def reset_parameters(self):
        modulation = self.adaLN_modulation[-1]
        assert isinstance(modulation, nn.Linear)
        nn.init.zeros_(modulation.weight)
        nn.init.zeros_(modulation.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, t, cond=None):
        if len(t.shape) == 2:
            t = t.unsqueeze(1)
        elif len(t.shape) == 3:
            pass
        else:
            raise ValueError(f"Invalid shape of t: {t.shape}")
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=-1)
        x = self.norm_final(x) * (1 + scale) + shift
        x = self.linear(x)
        return x


class VLATransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = "fp32_layer_norm",
        use_dual_attention: bool = False,
        training_phase: str = "joint",
    ):
        super().__init__()
        self.context_pre_only = context_pre_only
        context_norm_type = (
            "ada_norm_continous" if context_pre_only else "ada_norm_zero"
        )
        self.norm1_act = AdaLayerNormZero(dim)
        logger.debug(f"AdaLN act parameters: {count_parameters(self.norm1_act):,}")
        if context_norm_type == "ada_norm_continous":
            self.norm1_obs = AdaLayerNormContinuous(
                dim,
                dim,
                elementwise_affine=False,
                eps=1e-06,
                bias=True,
                norm_type="layer_norm",
            )
            logger.debug(f"AdaLN obs parameters: {count_parameters(self.norm1_obs):,}")
        elif context_norm_type == "ada_norm_zero":
            self.norm1_obs = AdaLayerNormZero(dim)
            logger.debug(f"AdaLN obs parameters: {count_parameters(self.norm1_obs):,}")
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
            )
        if hasattr(F, "scaled_dot_product_attention"):
            processor = JointVLAAttnProcessor(
                query_dim=dim,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                bias=True,
                context_pre_only=context_pre_only,
            )
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=1e-06,
        )
        logger.debug(f"Joint Attention parameters: {count_parameters(self.attn):,}")
        self.norm2_act = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-06)
        self.ff_act = FeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate"
        )
        logger.debug(f"FF act parameters: {count_parameters(self.ff_act):,}")
        if not context_pre_only:
            self.norm2_obs = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-06)
            self.ff_obs = FeedForward(
                dim=dim, dim_out=dim, activation_fn="gelu-approximate"
            )
            logger.debug(f"FF obs parameters: {count_parameters(self.ff_obs):,}")
        else:
            self.norm2_obs = None
            self.ff_obs = None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self,
        action_hidden_states: torch.Tensor,
        obs_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        obs_token_mask: Optional[torch.Tensor] = None,
        joint_attention_kwargs=None,
    ):
        (
            norm_action_hidden_states,
            gate_msa_act,
            shift_mlp_act,
            scale_mlp_act,
            gate_mlp_act,
        ) = self.norm1_act(action_hidden_states, emb=temb)
        if self.context_pre_only:
            norm_obs_hidden_states = self.norm1_obs(
                obs_hidden_states, temb[:, -1] if len(temb.shape) > 2 else temb
            )
            gate_msa_obs, shift_mlp_obs, scale_mlp_obs, gate_mlp_obs = (
                None,
                None,
                None,
                None,
            )
        else:
            (
                norm_obs_hidden_states,
                gate_msa_obs,
                shift_mlp_obs,
                scale_mlp_obs,
                gate_mlp_obs,
            ) = self.norm1_obs(
                obs_hidden_states, emb=temb[:, -1] if len(temb.shape) > 2 else temb
            )
        act_attn_output, obs_attn_output = self.attn(
            hidden_states=norm_action_hidden_states,
            encoder_hidden_states=norm_obs_hidden_states,
            attention_mask=obs_token_mask,
        )
        act_attn_output = gate_msa_act * act_attn_output
        action_hidden_states = action_hidden_states + act_attn_output
        norm_action_hidden_states = self.norm2_act(action_hidden_states)
        norm_action_hidden_states = (
            norm_action_hidden_states * (1 + scale_mlp_act) + shift_mlp_act
        )
        ff_action_output = self.ff_act(norm_action_hidden_states)
        ff_action_output = gate_mlp_act * ff_action_output
        action_hidden_states = action_hidden_states + ff_action_output
        if self.context_pre_only:
            obs_hidden_states = None
        else:
            assert (
                gate_msa_obs is not None
                and shift_mlp_obs is not None
                and (scale_mlp_obs is not None)
                and (gate_mlp_obs is not None)
            )
            assert self.norm2_obs is not None
            obs_attn_output = gate_msa_obs * obs_attn_output
            obs_hidden_states = obs_hidden_states + obs_attn_output
            norm_obs_hidden_states = self.norm2_obs(obs_hidden_states)
            norm_obs_hidden_states = (
                norm_obs_hidden_states * (1 + scale_mlp_obs) + shift_mlp_obs
            )
            assert self.ff_obs is not None
            ff_obs_output = self.ff_obs(norm_obs_hidden_states)
            ff_obs_output = gate_mlp_obs * ff_obs_output
            obs_hidden_states = obs_hidden_states + ff_obs_output
        return (action_hidden_states, obs_hidden_states)


class ActionTransformerModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["ActionTransformerBlock"]

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs
    ):
        model = ActionTransformerModel()
        torch_dtype = kwargs.pop("torch_dtype", torch.float32)
        if pretrained_model_name_or_path is not None and "checkpoints" in str(
            pretrained_model_name_or_path
        ):
            if os.path.exists(pretrained_model_name_or_path):
                from safetensors.torch import load_file

                file_path = f"{pretrained_model_name_or_path}/model.safetensors"
                assert os.path.exists(file_path), (
                    f"Checkpoint {file_path} does not exist."
                )
                model_state_dict = load_file(file_path)
                missing_keys, unexpected_keys = model.load_state_dict(
                    model_state_dict, strict=False
                )
                if len(unexpected_keys) > 0:
                    logger.info(
                        f"Unexpected keys in the checkpoint: {unexpected_keys}. "
                    )
                if len(missing_keys) > 0:
                    logger.info(f"Missing keys in the checkpoint: {missing_keys}.")
                logger.info(f"Loaded model from {pretrained_model_name_or_path}")
            else:
                raise ValueError(
                    f"Checkpoint {pretrained_model_name_or_path} does not exist. "
                )
        model.to(torch_dtype)
        return model

    @register_to_config
    def __init__(
        self,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        caption_projection_dim: int = 1536,
        pooled_projection_dim: int = 2048,
        pos_embed_max_size: int = 192,
        dual_attention_layers: Tuple[int, ...] = (),
        qk_norm: Optional[str] = "fp32_layer_norm",
        training_phase: str = "joint",
        action_num_blocks: int = 6,
        action_pred_horizon: int = 6,
        action_dim: int = 7,
        action_hidden_dim: int = 1536,
        action_nheads: int = 24,
        n_conditions: int = 0,
        token_fusion: str = "concat",
        resnet_store_path: str = "cache/visual_features/resnet18/IN_1M_resnet18.pth",
        odim: int = 32,
        view_feature_dim: int = 1920,
        use_film: bool = False,
        combined_temb: bool = False,
        use_future_progress_embedding: bool = False,
        future_progress_dim: int = 5,
    ):
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim
        self.combined_temb = combined_temb
        if self.combined_temb:
            self.time_ins_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=self.inner_dim,
                pooled_projection_dim=pooled_projection_dim,
            )
        else:
            self.time_ins_embed = _TimeNetwork(time_dim=256, out_dim=action_hidden_dim)
        self.obs_proj = ObservationProjection(
            action_pred_horizon=action_pred_horizon,
            action_dim=action_dim,
            hidden_dim=self.inner_dim,
            output_dim=action_hidden_dim,
            n_conditions=n_conditions,
            token_fusion=token_fusion,
            resnet_store_path=resnet_store_path,
            odim=odim,
            view_feature_dim=view_feature_dim,
            use_film=use_film,
        )
        total_params = sum(
            (p.numel() for p in self.obs_proj.parameters() if p.requires_grad)
        )
        logger.debug(f"ObservationEncoder parameters: {total_params:,}")
        self.action_proj_in = ActionProjectionIn(
            action_pred_horizon=action_pred_horizon,
            action_dim=action_dim,
            output_dim=action_hidden_dim,
        )
        self.use_future_progress_embedding = bool(use_future_progress_embedding)
        self.future_progress_dim = int(future_progress_dim)
        if self.use_future_progress_embedding:
            self.future_progress_mlp = nn.Sequential(
                nn.Linear(self.future_progress_dim, action_hidden_dim),
                nn.SiLU(),
                nn.Linear(action_hidden_dim, action_hidden_dim),
            )
            nn.init.zeros_(self.future_progress_mlp[-1].weight)
            nn.init.zeros_(self.future_progress_mlp[-1].bias)
        else:
            self.future_progress_mlp = None
        total_params = sum(
            (p.numel() for p in self.action_proj_in.parameters() if p.requires_grad)
        )
        logger.debug(f"ActionProjectionIn parameters: {total_params:,}")
        self.transformer_blocks = nn.ModuleList(
            [
                VLATransformerBlock(
                    dim=action_hidden_dim,
                    num_attention_heads=action_nheads,
                    attention_head_dim=attention_head_dim,
                    context_pre_only=i == action_num_blocks - 1,
                    qk_norm=qk_norm,
                    training_phase=training_phase,
                )
                for i in range(action_num_blocks)
            ]
        )
        total_params = sum(
            (
                p.numel()
                for p in self.transformer_blocks[0].parameters()
                if p.requires_grad
            )
        )
        logger.debug(f"ActionTransformerBlock parameters: {total_params:,}")
        self.action_proj_out = ActionProjectionOut(
            hidden_size=action_hidden_dim, action_dim=action_dim
        )
        total_params = sum((p.numel() for p in self.action_proj_out.parameters()))
        logger.debug(f"ActionProjectionOut parameters: {total_params:,}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        block_controlnet_hidden_states: Optional[List] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        vlm_attn_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
    ) -> Union[List[torch.Tensor], ActionTransformerModelOutput]:
        assert joint_attention_kwargs is not None
        if self.config.n_conditions > 0:
            assert (
                joint_attention_kwargs["traj2ds"].shape[1] == self.config.n_conditions
            ), "model n_conditions does not match"
        if self.combined_temb:
            temb = self.time_ins_embed(timestep, pooled_projections)
        else:
            temb = self.time_ins_embed(timestep)
        noisy_action = joint_attention_kwargs.pop("action_hidden_embeds")
        known_action_prefix = joint_attention_kwargs.pop("known_action_prefix", None)
        prefix_len = 0
        if known_action_prefix is not None:
            if known_action_prefix.ndim != noisy_action.ndim:
                raise ValueError(
                    f"known_action_prefix and action_hidden_embeds must have the same rank, got {known_action_prefix.ndim} and {noisy_action.ndim}"
                )
            if (
                known_action_prefix.shape[0] != noisy_action.shape[0]
                or known_action_prefix.shape[-1] != noisy_action.shape[-1]
            ):
                raise ValueError(
                    "known_action_prefix batch/action dimensions must match action_hidden_embeds"
                )
            prefix_len = int(known_action_prefix.shape[1])
        if prefix_len + noisy_action.shape[1] != self.config.action_pred_horizon:
            raise ValueError(
                f"RTC prefix plus postfix must fill the configured action horizon: prefix={prefix_len}, postfix={noisy_action.shape[1]}, horizon={self.config.action_pred_horizon}"
            )
        action_hidden_states = self.action_proj_in(
            noisy_action, position_offset=prefix_len
        )
        prefix_hidden_states = None
        if known_action_prefix is not None:
            prefix_hidden_states = self.action_proj_in(
                known_action_prefix, position_offset=0
            )
        if self.use_future_progress_embedding:
            future_progress = joint_attention_kwargs.get("future_progress", None)
            if future_progress is None:
                raise ValueError(
                    "use_future_progress_embedding=True requires future_progress tensor"
                )
            assert self.future_progress_mlp is not None
            future_progress = future_progress.to(
                device=action_hidden_states.device, dtype=action_hidden_states.dtype
            )
            expected_progress_shape = (
                action_hidden_states.shape[0],
                prefix_len + action_hidden_states.shape[1],
            )
            if future_progress.shape[:2] != expected_progress_shape:
                raise ValueError(
                    f"future_progress shape {future_progress.shape} does not match full action timeline {expected_progress_shape}"
                )
            if prefix_hidden_states is not None:
                prefix_hidden_states = prefix_hidden_states + self.future_progress_mlp(
                    future_progress[:, :prefix_len]
                )
            action_hidden_states = action_hidden_states + self.future_progress_mlp(
                future_progress[:, prefix_len:]
            )
        obs_hidden_states, obs_token_mask = self.obs_proj(
            views=joint_attention_kwargs["views"],
            obs=joint_attention_kwargs["obs"],
            traj2ds=joint_attention_kwargs["traj2ds"],
            text_embeddings=pooled_projections,
            vlm_attn_mask=vlm_attn_mask,
        )
        if prefix_hidden_states is not None:
            obs_hidden_states = torch.cat(
                [prefix_hidden_states, obs_hidden_states], dim=1
            )
            if obs_token_mask is not None:
                prefix_token_mask = torch.ones(
                    (obs_token_mask.shape[0], prefix_len),
                    device=obs_token_mask.device,
                    dtype=obs_token_mask.dtype,
                )
                obs_token_mask = torch.cat([prefix_token_mask, obs_token_mask], dim=1)
        for index_block, block in enumerate(self.transformer_blocks):
            is_skip = (
                True
                if skip_layers is not None and index_block in skip_layers
                else False
            )
            if is_skip:
                continue
            action_hidden_states, obs_hidden_states = block(
                action_hidden_states=action_hidden_states,
                obs_hidden_states=obs_hidden_states,
                temb=temb,
                obs_token_mask=obs_token_mask,
            )
        action_output = self.action_proj_out(x=action_hidden_states, t=temb)
        return ActionTransformerModelOutput(action=action_output)


class DiTActionTransformerModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        caption_projection_dim: int = 1536,
        pooled_projection_dim: int = 2048,
        pos_embed_max_size: int = 192,
        dual_attention_layers: Tuple[int, ...] = (),
        qk_norm: Optional[str] = None,
        training_phase: str = "joint",
        action_num_blocks: int = 6,
        action_pred_horizon: int = 6,
        action_dim: int = 7,
        action_hidden_dim: int = 1536,
        action_nheads: int = 24,
        n_conditions: int = 0,
        token_fusion: str = "concat",
        resnet_store_path: str = "cache/visual_features/resnet18/IN_1M_resnet18.pth",
        odim: int = 32,
        view_feature_dim: int = 1920,
        use_film: bool = False,
        combined_temb: bool = False,
        use_future_progress_embedding: bool = False,
        future_progress_dim: int = 5,
    ):
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim
        if use_future_progress_embedding:
            raise ValueError(
                "DiTActionTransformerModel does not support future progress embedding"
            )
        self.obs_proj = ObservationProjection(
            action_pred_horizon=action_pred_horizon,
            action_dim=action_dim,
            hidden_dim=self.inner_dim,
            output_dim=action_hidden_dim,
            n_conditions=n_conditions,
            token_fusion="concat",
            resnet_store_path=resnet_store_path,
            odim=odim,
            view_feature_dim=view_feature_dim,
            use_film=use_film,
        )
        total_params = sum(
            (p.numel() for p in self.obs_proj.parameters() if p.requires_grad)
        )
        logger.debug(f"ObservationEncoder parameters: {total_params:,}")
        num_query_tokens = 256
        num_heads_ = 16
        self.query_tokens = nn.Parameter(
            torch.randn(num_query_tokens, action_hidden_dim)
        )
        self.cross_attention = CrossAttentionBlock(action_hidden_dim, num_heads_)
        self.DiT = DiT(
            in_channels=action_dim,
            depth=24,
            num_heads=16,
            token_size=action_hidden_dim,
            learn_sigma=False,
            class_dropout_prob=0,
            future_action_window_size=action_pred_horizon - 1,
            past_action_window_size=0,
            n_conditon_token=num_query_tokens,
        )
        total_params = sum(
            (p.numel() for p in self.DiT.parameters() if p.requires_grad)
        )
        logger.debug(f"DiT parameters: {total_params:,}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        block_controlnet_hidden_states: Optional[List] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        vlm_attn_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
    ) -> Union[List[torch.Tensor], ActionTransformerModelOutput]:
        assert joint_attention_kwargs is not None
        if joint_attention_kwargs.get("known_action_prefix", None) is not None:
            raise ValueError(
                "Context-only RTC is implemented for ActionTransformerModel; DiTActionTransformerModel cannot consume known_action_prefix."
            )
        if self.config.n_conditions > 0:
            assert (
                joint_attention_kwargs["traj2ds"].shape[1] == self.config.n_conditions
            ), "model n_conditions does not match"
        noisy_action = joint_attention_kwargs.pop("action_hidden_embeds")
        condition_tokens, vlm_attn_mask = self.obs_proj(
            views=joint_attention_kwargs["views"],
            obs=joint_attention_kwargs["obs"],
            traj2ds=joint_attention_kwargs["traj2ds"],
            text_embeddings=pooled_projections,
            vlm_attn_mask=vlm_attn_mask,
        )
        query = self.query_tokens.unsqueeze(0).expand(condition_tokens.shape[0], -1, -1)
        queried = self.cross_attention(query, condition_tokens)
        action_output = self.DiT(noisy_action, timestep, queried)
        return ActionTransformerModelOutput(action=action_output)


class MovensModel(nn.Module):
    def __init__(self, model_cfg, vlm_model: Optional[nn.Module]):
        super().__init__()
        self.action_horizon = int(model_cfg.action_chunk_size)
        self.action_dim = int(model_cfg.action_dim)
        self.action_exec_horizon = int(model_cfg.action_exec_horizon)
        self.rtc_max_delay = int(model_cfg.max_delay)
        self.rtc_expected_overlap = self.action_horizon - self.action_exec_horizon
        if bool(getattr(model_cfg, "rtc", False)):
            _validate_rtc_inference_contract(
                action_horizon=self.action_horizon,
                action_exec_horizon=self.action_exec_horizon,
                inference_delay=self.rtc_expected_overlap,
                max_delay=self.rtc_max_delay,
            )
        if model_cfg.use_dit:
            self.action_header = DiTActionTransformerModel(
                resnet_store_path=model_cfg.resnet_store_path,
                odim=model_cfg.odim,
                action_dim=model_cfg.action_dim,
                action_pred_horizon=model_cfg.action_chunk_size,
                view_feature_dim=model_cfg.view_feature_dim,
                use_film=model_cfg.use_film,
                combined_temb=model_cfg.combined_temb,
                action_hidden_dim=model_cfg.hidden_dim,
                qk_norm=getattr(model_cfg, "qk_norm", "fp32_layer_norm"),
                use_future_progress_embedding=getattr(
                    model_cfg, "use_future_progress_embedding", False
                ),
                future_progress_dim=getattr(model_cfg, "future_progress_dim", 5),
            )
        else:
            action_nheads = int(model_cfg.nhead)
            if model_cfg.hidden_dim % action_nheads != 0:
                raise ValueError(
                    f"Action hidden dim {model_cfg.hidden_dim} must be divisible by nhead {action_nheads}"
                )
            action_head_dim = model_cfg.hidden_dim // action_nheads
            self.action_header = ActionTransformerModel(
                attention_head_dim=action_head_dim,
                num_attention_heads=action_nheads,
                action_num_blocks=model_cfg.num_blocks,
                action_nheads=action_nheads,
                resnet_store_path=model_cfg.resnet_store_path,
                odim=model_cfg.odim,
                action_dim=model_cfg.action_dim,
                action_pred_horizon=model_cfg.action_chunk_size,
                view_feature_dim=model_cfg.view_feature_dim,
                use_film=model_cfg.use_film,
                combined_temb=model_cfg.combined_temb,
                action_hidden_dim=model_cfg.hidden_dim,
                qk_norm=getattr(model_cfg, "qk_norm", "fp32_layer_norm"),
                use_future_progress_embedding=getattr(
                    model_cfg, "use_future_progress_embedding", False
                ),
                future_progress_dim=getattr(model_cfg, "future_progress_dim", 5),
            )
        total_params = sum((p.numel() for p in self.action_header.parameters()))
        overwatch.info(f"Total ActionTransformerModel parameters: {total_params:,}")
        self.vlm_model = vlm_model
        if self.vlm_model is None:
            overwatch.info(
                "Total VLM Backbone parameters: 0 (cached hidden states only)"
            )
        else:
            total_params = sum((p.numel() for p in self.vlm_model.parameters()))
            overwatch.info(f"Total VLM Backbone parameters: {total_params:,}")

    @classmethod
    def from_pretrained(
        cls,
        run_dir,
        ckpt_step,
        launch_config: LaunchConfig,
        device,
        *,
        allow_root40: bool = False,
    ):
        if (
            getattr(launch_config.model, "motion_representation", "legacy")
            == "root40_v1"
            or launch_config.model.action_dim == 40
        ):
            if not allow_root40:
                raise ValueError(
                    "Legacy inference cannot decode root40_v1. An anchor-aware 40D loader must pass allow_root40=True and use the root40_v1 codec/RTC rebasing API."
                )
            from .validation import validate_root40_config

            validate_root40_config(launch_config)
        if not os.path.exists(run_dir):
            raise ValueError(f"Pretrained model path {run_dir} does not exist.")
        from safetensors.torch import load_file

        if str(ckpt_step) in {"best", "best_ckpt"}:
            ckpt_dir = f"{run_dir}/checkpoints/best_ckpt"
        else:
            ckpt_dir = f"{run_dir}/checkpoints/ckpt_{ckpt_step}"
        if not os.path.exists(ckpt_dir):
            root_ckpt_dir = f"{run_dir}/ckpt_{ckpt_step}"
            if os.path.exists(root_ckpt_dir):
                ckpt_dir = root_ckpt_dir
        file_path = f"{ckpt_dir}/model.safetensors"
        index_path = f"{ckpt_dir}/model.safetensors.index.json"
        if os.path.exists(file_path):
            state_dict = load_file(file_path, device="cpu")
        elif os.path.exists(index_path):
            with open(index_path, "r") as f:
                index_obj = json.load(f)
            weight_map = index_obj.get("weight_map", {})
            shard_files = sorted(set(weight_map.values()))
            state_dict = {}
            for shard_file in shard_files:
                shard_path = os.path.join(ckpt_dir, shard_file)
                if not os.path.exists(shard_path):
                    raise ValueError(f"Checkpoint shard {shard_path} does not exist.")
                state_dict.update(load_file(shard_path, device="cpu"))
        else:
            raise ValueError(f"Checkpoint file {file_path} does not exist.")
        configured_model_name_or_path = str(launch_config.model.model_name_or_path)
        model_path_override = os.getenv("MOVENS_PRETRAIN_BACKBONE_PATH", "").strip()
        model_name_or_path = os.path.expanduser(
            model_path_override or configured_model_name_or_path
        )
        if model_path_override:
            overwatch.info(
                f"Using MOVENS_PRETRAIN_BACKBONE_PATH override for inference: {model_name_or_path} (run_config={configured_model_name_or_path})"
            )
        if os.path.isabs(model_name_or_path) and (
            not os.path.isfile(os.path.join(model_name_or_path, "config.json"))
        ):
            source = (
                "MOVENS_PRETRAIN_BACKBONE_PATH"
                if model_path_override
                else "run_config.json"
            )
            raise FileNotFoundError(
                f"VLM backbone from {source} is not a valid local model directory: {model_name_or_path} (missing config.json). Set MOVENS_PRETRAIN_BACKBONE_PATH to a complete local Qwen checkpoint."
            )
        is_omni = "omni" in model_name_or_path.lower()
        force_float32_infer = os.getenv("MOVENS_INFER_FLOAT32", "0") == "1"
        if force_float32_infer:
            attn_impl = "sdpa"
            vlm_dtype = torch.float32
        else:
            attn_impl = (
                "flash_attention_2" if str(device).startswith("cuda") else "sdpa"
            )
            vlm_dtype = torch.bfloat16
        if is_omni:
            vlm_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                model_name_or_path, attn_implementation=attn_impl, dtype=vlm_dtype
            )
        else:
            vlm_config = AutoConfig.from_pretrained(model_name_or_path)
            vlm_config._attn_implementation = attn_impl
            vlm_model = Qwen3VLForConditionalGeneration(vlm_config)
        vlm_model = vlm_model.to(dtype=vlm_dtype)
        vlm_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("action_header."):
                continue
            elif k.startswith("vlm_model."):
                vlm_state_dict[k.replace("vlm_model.", "")] = v
            elif k.startswith("motion_encoder."):
                continue
            else:
                assert False, "check here"
        if vlm_state_dict:
            if "lm_head.weight" not in vlm_state_dict:
                if "model.language_model.embed_tokens.weight" in vlm_state_dict:
                    vlm_state_dict["lm_head.weight"] = vlm_state_dict[
                        "model.language_model.embed_tokens.weight"
                    ]
                elif "model.embed_tokens.weight" in vlm_state_dict:
                    vlm_state_dict["lm_head.weight"] = vlm_state_dict[
                        "model.embed_tokens.weight"
                    ]
            if "lm_head.weight" not in vlm_state_dict:
                raise KeyError(
                    "Checkpoint contains VLM weights but neither lm_head.weight nor a supported embed_tokens.weight key was found."
                )
            if (
                vlm_state_dict["lm_head.weight"].shape[0]
                != vlm_model.lm_head.weight.shape[0]
            ):
                vlm_model.resize_token_embeddings(
                    vlm_state_dict["lm_head.weight"].shape[0],
                    pad_to_multiple_of=192,
                    mean_resizing=True,
                )
                overwatch.info(
                    f"Resized model token embeddings to {vlm_model.lm_head.weight.shape[0]}"
                )
            vlm_model.load_state_dict(vlm_state_dict, strict=True)
            overwatch.info("loaded vlm_backbone checkpoint successfully.")
        else:
            if not is_omni:
                raise ValueError(
                    "Checkpoint contains no VLM weights, but this non-Omni inference path only initialized the VLM from config. A pretrained base VLM is required."
                )
            overwatch.info(
                f"Checkpoint {ckpt_dir} contains no VLM weights; retaining pretrained VLM backbone from {model_name_or_path}."
            )
        model = cls(model_cfg=launch_config.model, vlm_model=vlm_model)
        action_head_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("vlm_model."):
                continue
            elif k.startswith("action_header."):
                action_head_state_dict[k.replace("action_header.", "")] = v
            elif k.startswith("motion_encoder."):
                continue
            else:
                assert False, "check here"
        model.action_header.load_state_dict(action_head_state_dict, strict=True)
        overwatch.info("loaded action head checkpoint successfully.")
        if hasattr(model, "motion_encoder"):
            motion_encoder_state_dict = {
                k.replace("motion_encoder.", ""): v
                for k, v in state_dict.items()
                if k.startswith("motion_encoder.")
            }
            if motion_encoder_state_dict:
                model.motion_encoder.load_state_dict(
                    motion_encoder_state_dict, strict=True
                )
                overwatch.info("loaded motion encoder checkpoint successfully.")
        if is_omni:
            model.vlm_processor = Qwen2_5OmniProcessor.from_pretrained(
                model_name_or_path
            )
        else:
            model.vlm_processor = AutoProcessor.from_pretrained(model_name_or_path)
        if launch_config.model.noise_scheduler == "ddpm":
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler

            scheduler = DDIMScheduler(
                num_train_timesteps=launch_config.model.train_diffusion_steps,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type="epsilon",
            )
        elif launch_config.model.noise_scheduler == "flow":
            from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
                FlowMatchEulerDiscreteScheduler,
            )

            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=launch_config.model.train_diffusion_steps
            )
        else:
            raise ValueError(
                f"Unsupported noise scheduler: {launch_config.model.noise_scheduler}"
            )
        model.noise_scheduler = scheduler
        model.action_horizon = launch_config.model.action_chunk_size
        model.action_dim = launch_config.model.action_dim
        model.text_only = bool(
            getattr(launch_config.data.transform.model, "text_only", False)
        )
        model.audio_only = bool(
            getattr(launch_config.data.transform.model, "audio_only", False)
        )
        model.video_only = bool(
            getattr(launch_config.data.transform.model, "video_only", False)
        )
        model.video_size_pixels = int(
            getattr(launch_config.data.transform.model, "video_size_pixels", 50176)
        )
        model.use_delta_actions = bool(
            getattr(launch_config.data.transform.repack, "use_delta_actions", False)
        )
        model.root_delta_actions = bool(
            getattr(launch_config.data.transform.repack, "root_delta_actions", False)
        )
        model.legacy39_actions = bool(
            getattr(launch_config.data.transform.repack, "legacy39_actions", False)
        )
        model.root40_actions = bool(
            getattr(launch_config.data.transform.repack, "root40_actions", False)
        )
        model.motion_representation = getattr(
            launch_config.model, "motion_representation", "legacy"
        )
        model.infer_autocast_dtype = vlm_dtype
        model.device = device
        return model

    def _action_condition_hidden_states(
        self,
        vlm_hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del input_ids, attention_mask
        return vlm_hidden_states.unsqueeze(1)

    def _omni_hidden_states_without_lm_head(
        self, vlm_kwargs: dict[str, Any]
    ) -> torch.Tensor:
        input_ids = vlm_kwargs.get("input_ids", None)
        attention_mask = vlm_kwargs.get("attention_mask", None)
        input_features = vlm_kwargs.get("input_features", None)
        pixel_values = vlm_kwargs.get("pixel_values", None)
        pixel_values_videos = vlm_kwargs.get("pixel_values_videos", None)
        image_grid_thw = vlm_kwargs.get("image_grid_thw", None)
        video_grid_thw = vlm_kwargs.get("video_grid_thw", None)
        feature_attention_mask = vlm_kwargs.get("feature_attention_mask", None)
        audio_feature_lengths = vlm_kwargs.get("audio_feature_lengths", None)
        video_second_per_grid = vlm_kwargs.get("video_second_per_grid", None)
        inputs_embeds = vlm_kwargs.get("inputs_embeds", None)
        position_ids = vlm_kwargs.get("position_ids", None)
        cache_position = vlm_kwargs.get("cache_position", None)
        use_audio_in_video = vlm_kwargs.get("use_audio_in_video", None)
        if inputs_embeds is None:
            inputs_embeds = self.vlm_model.get_input_embeddings()(input_ids)
        if input_features is not None:
            audio_features = self.vlm_model.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_features = audio_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, _, audio_mask = self.vlm_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
        if pixel_values is not None:
            image_embeds = self.vlm_model.get_image_features(
                pixel_values, image_grid_thw
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _, _ = self.vlm_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        if pixel_values_videos is not None:
            video_embeds = self.vlm_model.get_video_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask, _ = self.vlm_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None
        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.vlm_model.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.vlm_model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                self.vlm_model.rope_deltas = rope_deltas - delta0
            else:
                batch_size, seq_length = input_ids.shape
                delta = (
                    cache_position[0] + self.vlm_model.rope_deltas
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        outputs = self.vlm_model.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=vlm_kwargs.get("past_key_values", None),
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=vlm_kwargs.get("output_attentions", None),
            output_hidden_states=True,
            return_dict=True,
            cache_position=cache_position,
        )
        return outputs.hidden_states[-1]

    def _extract_vlm_hidden_states(self, vlm_kwargs: dict[str, Any]) -> torch.Tensor:
        if (
            self.vlm_model.__class__.__name__
            == "Qwen2_5OmniThinkerForConditionalGeneration"
        ):
            return self._omni_hidden_states_without_lm_head(vlm_kwargs)
        return self.vlm_model(**vlm_kwargs).hidden_states[-1]

    def _build_video_only_inputs(
        self,
        texts,
        video=None,
        video_embed=None,
        video_size_pixels: int = 50176,
        video_sample_fps: float = 1.0,
        video_pixel_cache_dtype: str = "none",
    ):
        if video_embed is not None:
            pixel_values_videos = torch.as_tensor(
                video_embed["pixel_values_videos"],
                dtype=torch.float32,
                device=self.device,
            )
            video_grid_thw = torch.as_tensor(
                video_embed["video_grid_thw"], dtype=torch.long, device=self.device
            )
            if video_grid_thw.dim() == 1:
                video_grid_thw = video_grid_thw.unsqueeze(0)
            if "video_second_per_grid" in video_embed:
                video_second_per_grid = torch.as_tensor(
                    video_embed["video_second_per_grid"],
                    dtype=torch.float32,
                    device=self.device,
                ).reshape(-1)
            else:
                video_second_per_grid = torch.tensor(
                    [
                        float(self.vlm_processor.video_processor.temporal_patch_size)
                        / float(video_sample_fps)
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
            texts = self.vlm_processor.replace_multimodal_special_tokens(
                texts,
                iter([]),
                iter([]),
                iter([video_grid_thw[0].detach().cpu()]),
                iter([float(video_second_per_grid.reshape(-1)[0].item())]),
                False,
                25,
                2.0,
            )
            text_inputs = self.vlm_processor.tokenizer(
                texts, padding=True, return_tensors="pt"
            )
            inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
            inputs["pixel_values_videos"] = pixel_values_videos
            inputs["video_grid_thw"] = video_grid_thw
            inputs["video_second_per_grid"] = video_second_per_grid
            return inputs
        inputs = self.vlm_processor(
            text=texts,
            videos=[np.asarray(video, dtype=np.uint8)],
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
        ).to(self.device)
        cache_dtype = str(video_pixel_cache_dtype).strip().lower()
        if cache_dtype not in {"", "none"}:
            quant_dtype = {
                "fp16": torch.float16,
                "float16": torch.float16,
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
            }.get(cache_dtype)
            if quant_dtype is None:
                raise ValueError(
                    f"Unsupported video_pixel_cache_dtype={video_pixel_cache_dtype!r}"
                )
            inputs["pixel_values_videos"] = (
                inputs["pixel_values_videos"]
                .to(dtype=quant_dtype)
                .to(dtype=torch.float32)
            )
        return inputs

    @torch.inference_mode()
    def predict_action(
        self,
        observations: Optional[List[List[Image.Image]]],
        states: torch.Tensor,
        instructions: List[str],
        num_inference_steps: int,
        traj2ds,
        **kwargs: str,
    ) -> torch.Tensor:
        bsz = states.shape[0]
        cfg_scale = float(kwargs.get("cfg_scale", 1.0))
        if not np.isfinite(cfg_scale) or cfg_scale < 0:
            raise ValueError("cfg_scale must be finite and non-negative")
        cfg_interval = kwargs.get("cfg_interval")
        if cfg_interval is not None:
            cfg_low, cfg_high = (float(value) for value in cfg_interval)
            if not (
                np.isfinite(cfg_low)
                and np.isfinite(cfg_high)
                and (0 <= cfg_low < cfg_high <= 1)
            ):
                raise ValueError("cfg_interval must satisfy 0 <= low < high <= 1")
        progress = kwargs.get("progress", None)
        future_progress = kwargs.get("future_progress", None)
        text_only = bool(getattr(self, "text_only", False))
        audio_only = bool(getattr(self, "audio_only", False))
        video_only = bool(getattr(self, "video_only", False))
        batch_input_ids = []
        batch_attention_mask = []
        batch_pixel_values = (
            [] if not text_only and (not audio_only) and (not video_only) else None
        )
        batch_image_grid_thw = (
            [] if not text_only and (not audio_only) and (not video_only) else None
        )
        batch_input_features = [] if audio_only else None
        batch_feature_attention_mask = [] if audio_only else None
        batch_pixel_values_videos = [] if video_only else None
        batch_video_grid_thw = [] if video_only else None
        batch_video_second_per_grid = [] if video_only else None
        if audio_only:
            audios = kwargs.get("audios", None)
            assert audios is not None, (
                "audios must be provided for audio-only inference"
            )
            audio_items = list(audios)
            audio_instructions = list(instructions or [""] * len(audio_items))
            assert len(audio_instructions) == len(audio_items), (
                "instructions must match audios for audio-only inference"
            )
            items = list(zip(audio_items, audio_instructions))
        elif video_only:
            video_embeds = kwargs.get("video_embeds", None)
            if video_embeds is not None:
                video_paths = kwargs.get(
                    "video_paths", ["video.mp4"] * len(video_embeds)
                )
                items = list(zip(video_embeds, video_paths, instructions))
            else:
                videos = kwargs.get("videos", None)
                assert videos is not None, (
                    "videos must be provided for video-only inference"
                )
                video_paths = kwargs.get("video_paths", ["video.mp4"] * len(videos))
                items = list(zip(videos, video_paths, instructions))
        elif text_only:
            items = [(None, instruction) for instruction in instructions]
        else:
            assert observations is not None, (
                "observations must be provided for non-text-only inference"
            )
            items = list(zip(observations, instructions))
        for item in items:
            if audio_only:
                audio, instruction = item
                content = [{"type": "audio", "audio": "audio.wav"}]
                if str(instruction).strip():
                    content.append({"type": "text", "text": str(instruction).strip()})
                messages = [
                    [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": OMNI_DEFAULT_SYSTEM_PROMPT}
                            ],
                        },
                        {"role": "user", "content": content},
                    ]
                ]
            elif video_only:
                video_item, video_path, instruction = item
                content = [{"type": "video", "video": str(video_path or "video.mp4")}]
                if str(instruction).strip():
                    content.append({"type": "text", "text": str(instruction).strip()})
                messages = [
                    [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": OMNI_DEFAULT_SYSTEM_PROMPT}
                            ],
                        },
                        {"role": "user", "content": content},
                    ]
                ]
            elif text_only:
                _, instruction = item
                messages = [
                    [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": OMNI_DEFAULT_SYSTEM_PROMPT}
                            ],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": instruction}],
                        },
                    ]
                ]
            else:
                observation, instruction = item
                assert observation is not None
                messages = []
                content = [{"type": "image", "image": img} for img in observation]
                content.append({"type": "text", "text": instruction})
                user_msg = {"role": "user", "content": content}
                messages.append([user_msg])
            texts = [
                self.vlm_processor.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True
                )
                for m in messages
            ]
            if audio_only:
                inputs = self.vlm_processor(
                    text=texts,
                    audio=[np.asarray(audio, dtype=np.float32)],
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
            elif video_only:
                video_size_pixels = int(
                    kwargs.get(
                        "video_size_pixels", getattr(self, "video_size_pixels", 50176)
                    )
                )
                video_sample_fps = float(kwargs.get("video_sample_fps", 1.0))
                inputs = self._build_video_only_inputs(
                    texts,
                    video=None if isinstance(video_item, dict) else video_item,
                    video_embed=video_item if isinstance(video_item, dict) else None,
                    video_size_pixels=video_size_pixels,
                    video_sample_fps=video_sample_fps,
                    video_pixel_cache_dtype=str(
                        kwargs.get("video_pixel_cache_dtype", "none")
                    ),
                )
            elif text_only:
                inputs = self.vlm_processor(
                    text=texts, padding=True, return_tensors="pt"
                ).to(self.device)
            else:
                image_inputs, video_inputs = process_vision_info(
                    messages, image_patch_size=16
                )
                inputs = self.vlm_processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
            batch_input_ids.append(inputs["input_ids"].squeeze(0))
            batch_attention_mask.append(inputs["attention_mask"].squeeze(0))
            if audio_only:
                assert (
                    batch_input_features is not None
                    and batch_feature_attention_mask is not None
                )
                batch_input_features.append(inputs["input_features"].squeeze(0))
                batch_feature_attention_mask.append(
                    inputs["feature_attention_mask"].squeeze(0)
                )
            if video_only:
                assert (
                    batch_pixel_values_videos is not None
                    and batch_video_grid_thw is not None
                )
                assert batch_video_second_per_grid is not None
                batch_pixel_values_videos.append(
                    inputs["pixel_values_videos"].squeeze(0)
                )
                batch_video_grid_thw.append(inputs["video_grid_thw"].squeeze(0))
                batch_video_second_per_grid.append(
                    inputs["video_second_per_grid"].reshape(-1)
                )
            if not text_only and (not audio_only) and (not video_only):
                assert (
                    batch_pixel_values is not None and batch_image_grid_thw is not None
                )
                batch_pixel_values.append(inputs["pixel_values"])
                batch_image_grid_thw.append(inputs["image_grid_thw"].squeeze(0))
        pad_token_id = int(getattr(self.vlm_processor.tokenizer, "pad_token_id", 0))
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=pad_token_id
        )
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
            batch_attention_mask, batch_first=True, padding_value=0
        )
        if audio_only:
            assert (
                batch_input_features is not None
                and batch_feature_attention_mask is not None
            )
            max_audio_len = max(
                (int(feature.shape[-1]) for feature in batch_input_features)
            )
            padded_input_features = []
            padded_feature_attention_mask = []
            for feature, feature_mask in zip(
                batch_input_features, batch_feature_attention_mask
            ):
                pad_len = max_audio_len - int(feature.shape[-1])
                if pad_len > 0:
                    feature_pad_shape = list(feature.shape)
                    feature_pad_shape[-1] = pad_len
                    feature = torch.cat(
                        [
                            feature,
                            torch.zeros(
                                feature_pad_shape,
                                dtype=feature.dtype,
                                device=feature.device,
                            ),
                        ],
                        dim=-1,
                    )
                    mask_pad_shape = list(feature_mask.shape)
                    mask_pad_shape[-1] = pad_len
                    feature_mask = torch.cat(
                        [
                            feature_mask,
                            torch.zeros(
                                mask_pad_shape,
                                dtype=feature_mask.dtype,
                                device=feature_mask.device,
                            ),
                        ],
                        dim=-1,
                    )
                padded_input_features.append(feature)
                padded_feature_attention_mask.append(feature_mask)
            batch_input_features = torch.stack(padded_input_features)
            batch_feature_attention_mask = torch.stack(padded_feature_attention_mask)
        if not text_only and (not audio_only) and (not video_only):
            assert batch_pixel_values is not None and batch_image_grid_thw is not None
            batch_pixel_values = torch.stack(batch_pixel_values)
            batch_image_grid_thw = torch.stack(batch_image_grid_thw)
        if video_only:
            assert (
                batch_pixel_values_videos is not None
                and batch_video_grid_thw is not None
            )
            assert batch_video_second_per_grid is not None
            batch_pixel_values_videos = torch.cat(batch_pixel_values_videos, dim=0)
            batch_video_grid_thw = torch.stack(batch_video_grid_thw)
            batch_video_second_per_grid = torch.cat(batch_video_second_per_grid, dim=0)
        infer_dtype = getattr(self, "infer_autocast_dtype", torch.bfloat16)
        if str(self.device).startswith("cuda") and infer_dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            autocast_ctx = torch.autocast("cuda", dtype=infer_dtype)
        else:
            autocast_ctx = contextlib.nullcontext()
        with autocast_ctx:
            if audio_only:
                output = self.vlm_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    input_features=batch_input_features,
                    feature_attention_mask=batch_feature_attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
            elif video_only:
                output = self.vlm_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    pixel_values_videos=batch_pixel_values_videos,
                    video_grid_thw=batch_video_grid_thw,
                    video_second_per_grid=batch_video_second_per_grid,
                    output_hidden_states=True,
                    return_dict=True,
                )
            elif text_only:
                output = self.vlm_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
            else:
                output = self.vlm_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    pixel_values=batch_pixel_values,
                    image_grid_thw=batch_image_grid_thw,
                    output_hidden_states=True,
                    return_dict=True,
                )
            vlm_hidden_states_ = output.hidden_states
            vlm_hidden_states = vlm_hidden_states_[-1]
            vlm_hidden_states = self._action_condition_hidden_states(
                vlm_hidden_states.to(dtype=states.dtype),
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
            )
            action_samples = torch.randn(
                bsz, self.action_horizon, self.action_dim, device=self.device
            )
            self.noise_scheduler.set_timesteps(num_inference_steps)
            for step_index, timestep in enumerate(self.noise_scheduler.timesteps):
                effective_scale = cfg_scale
                if cfg_interval is not None and (
                    not cfg_low
                    <= float(self.noise_scheduler.sigmas[step_index])
                    <= cfg_high
                ):
                    effective_scale = 1.0
                batched_timestep = timestep.expand(bsz).to(self.device)
                model_pred = self.action_header(
                    hidden_states=None,
                    timestep=batched_timestep,
                    joint_attention_kwargs=dict(
                        action_hidden_embeds=action_samples,
                        views=vlm_hidden_states,
                        obs=states,
                        traj2ds=traj2ds,
                        progress=progress,
                        future_progress=future_progress,
                    ),
                    vlm_attn_mask=batch_attention_mask
                    if effective_scale != 1.0
                    else None,
                    return_dict=True,
                ).action
                if effective_scale != 1.0:
                    uncond_pred = self.action_header(
                        hidden_states=None,
                        timestep=batched_timestep,
                        joint_attention_kwargs=dict(
                            action_hidden_embeds=action_samples,
                            views=vlm_hidden_states,
                            obs=states,
                            traj2ds=traj2ds,
                            progress=progress,
                            future_progress=future_progress,
                        ),
                        vlm_attn_mask=torch.zeros_like(batch_attention_mask),
                        return_dict=True,
                    ).action
                    model_pred = uncond_pred + effective_scale * (
                        model_pred - uncond_pred
                    )
                action_samples = self.noise_scheduler.step(
                    model_output=model_pred, timestep=timestep, sample=action_samples
                ).prev_sample
        return action_samples.float()

    @torch.inference_mode()
    def predict_action_with_training_rtc_flow(
        self,
        observations: Optional[List[List[Image.Image]]],
        states: torch.Tensor,
        instructions: List[str],
        num_inference_steps: int,
        traj2ds,
        prev_actions: torch.Tensor = None,
        inference_delay: int = 0,
        max_delay: int = 0,
        **kwargs: str,
    ) -> torch.Tensor:
        H = self.action_horizon
        bsz = states.shape[0]
        cfg_scale = float(kwargs.get("cfg_scale", 1.0))
        if not np.isfinite(cfg_scale) or cfg_scale < 0:
            raise ValueError("cfg_scale must be finite and non-negative")
        cfg_interval = kwargs.get("cfg_interval")
        if cfg_interval is not None:
            cfg_low, cfg_high = (float(value) for value in cfg_interval)
            if not (
                np.isfinite(cfg_low)
                and np.isfinite(cfg_high)
                and (0 <= cfg_low < cfg_high <= 1)
            ):
                raise ValueError("cfg_interval must satisfy 0 <= low < high <= 1")
        if prev_actions is None or inference_delay <= 0 or max_delay <= 0:
            raise ValueError(
                "prev_actions, a positive inference_delay, and max_delay must be provided for RTC inference"
            )
        if prev_actions.ndim != 3 or prev_actions.shape != (bsz, H, self.action_dim):
            raise ValueError(
                f"prev_actions must have shape ({bsz}, {H}, {self.action_dim}), got {tuple(prev_actions.shape)}"
            )
        prev_actions = prev_actions.to(device=self.device, dtype=torch.float32)
        if not hasattr(self, "action_exec_horizon") or not hasattr(
            self, "rtc_max_delay"
        ):
            raise RuntimeError(
                "RTC model is missing action_exec_horizon/max_delay deployment metadata"
            )
        if int(max_delay) != int(self.rtc_max_delay):
            raise ValueError(
                f"RTC caller max_delay does not match the loaded model config: caller={max_delay}, model={self.rtc_max_delay}"
            )
        d = _validate_rtc_inference_contract(
            action_horizon=H,
            action_exec_horizon=int(self.action_exec_horizon),
            inference_delay=int(inference_delay),
            max_delay=int(self.rtc_max_delay),
        )
        progress = kwargs.get("progress", None)
        future_progress = kwargs.get("future_progress", None)
        text_only = bool(getattr(self, "text_only", False))
        audio_only = bool(getattr(self, "audio_only", False))
        video_only = bool(getattr(self, "video_only", False))
        batch_input_ids = []
        batch_attention_mask = []
        batch_pixel_values = (
            [] if not text_only and (not audio_only) and (not video_only) else None
        )
        batch_image_grid_thw = (
            [] if not text_only and (not audio_only) and (not video_only) else None
        )
        batch_input_features = [] if audio_only else None
        batch_feature_attention_mask = [] if audio_only else None
        batch_pixel_values_videos = [] if video_only else None
        batch_video_grid_thw = [] if video_only else None
        batch_video_second_per_grid = [] if video_only else None
        if audio_only:
            audios = kwargs.get("audios", None)
            assert audios is not None, (
                "audios must be provided for audio-only inference"
            )
            audio_items = list(audios)
            audio_instructions = list(instructions or [""] * len(audio_items))
            assert len(audio_instructions) == len(audio_items), (
                "instructions must match audios for audio-only inference"
            )
            items = list(zip(audio_items, audio_instructions))
        elif video_only:
            video_embeds = kwargs.get("video_embeds", None)
            if video_embeds is not None:
                video_paths = kwargs.get(
                    "video_paths", ["video.mp4"] * len(video_embeds)
                )
                items = list(zip(video_embeds, video_paths, instructions))
            else:
                videos = kwargs.get("videos", None)
                assert videos is not None, (
                    "videos must be provided for video-only inference"
                )
                video_paths = kwargs.get("video_paths", ["video.mp4"] * len(videos))
                items = list(zip(videos, video_paths, instructions))
        elif text_only:
            items = [(None, instruction) for instruction in instructions]
        else:
            assert observations is not None, (
                "observations must be provided for non-text-only inference"
            )
            items = list(zip(observations, instructions))
        for item in items:
            if audio_only:
                audio, instruction = item
                content = [{"type": "audio", "audio": "audio.wav"}]
                if str(instruction).strip():
                    content.append({"type": "text", "text": str(instruction).strip()})
                messages = [
                    [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": OMNI_DEFAULT_SYSTEM_PROMPT}
                            ],
                        },
                        {"role": "user", "content": content},
                    ]
                ]
            elif video_only:
                video_item, video_path, instruction = item
                content = [{"type": "video", "video": str(video_path or "video.mp4")}]
                if str(instruction).strip():
                    content.append({"type": "text", "text": str(instruction).strip()})
                messages = [
                    [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": OMNI_DEFAULT_SYSTEM_PROMPT}
                            ],
                        },
                        {"role": "user", "content": content},
                    ]
                ]
            elif text_only:
                _, instruction = item
                messages = [
                    [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": OMNI_DEFAULT_SYSTEM_PROMPT}
                            ],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": instruction}],
                        },
                    ]
                ]
            else:
                observation, instruction = item
                assert observation is not None
                messages = []
                content = [{"type": "image", "image": img} for img in observation]
                content.append({"type": "text", "text": instruction})
                user_msg = {"role": "user", "content": content}
                messages.append([user_msg])
            texts = [
                self.vlm_processor.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True
                )
                for m in messages
            ]
            if audio_only:
                inputs = self.vlm_processor(
                    text=texts,
                    audio=[np.asarray(audio, dtype=np.float32)],
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
            elif video_only:
                video_size_pixels = int(
                    kwargs.get(
                        "video_size_pixels", getattr(self, "video_size_pixels", 50176)
                    )
                )
                video_sample_fps = float(kwargs.get("video_sample_fps", 1.0))
                inputs = self._build_video_only_inputs(
                    texts,
                    video=None if isinstance(video_item, dict) else video_item,
                    video_embed=video_item if isinstance(video_item, dict) else None,
                    video_size_pixels=video_size_pixels,
                    video_sample_fps=video_sample_fps,
                    video_pixel_cache_dtype=str(
                        kwargs.get("video_pixel_cache_dtype", "none")
                    ),
                )
            elif text_only:
                inputs = self.vlm_processor(
                    text=texts, padding=True, return_tensors="pt"
                ).to(self.device)
            else:
                image_inputs, video_inputs = process_vision_info(
                    messages, image_patch_size=16
                )
                inputs = self.vlm_processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
            batch_input_ids.append(inputs["input_ids"].squeeze(0))
            batch_attention_mask.append(inputs["attention_mask"].squeeze(0))
            if audio_only:
                assert (
                    batch_input_features is not None
                    and batch_feature_attention_mask is not None
                )
                batch_input_features.append(inputs["input_features"].squeeze(0))
                batch_feature_attention_mask.append(
                    inputs["feature_attention_mask"].squeeze(0)
                )
            if video_only:
                assert (
                    batch_pixel_values_videos is not None
                    and batch_video_grid_thw is not None
                )
                assert batch_video_second_per_grid is not None
                batch_pixel_values_videos.append(
                    inputs["pixel_values_videos"].squeeze(0)
                )
                batch_video_grid_thw.append(inputs["video_grid_thw"].squeeze(0))
                batch_video_second_per_grid.append(
                    inputs["video_second_per_grid"].reshape(-1)
                )
            if not text_only and (not audio_only) and (not video_only):
                assert (
                    batch_pixel_values is not None and batch_image_grid_thw is not None
                )
                batch_pixel_values.append(inputs["pixel_values"])
                batch_image_grid_thw.append(inputs["image_grid_thw"].squeeze(0))
        pad_token_id = int(getattr(self.vlm_processor.tokenizer, "pad_token_id", 0))
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=pad_token_id
        )
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
            batch_attention_mask, batch_first=True, padding_value=0
        )
        if audio_only:
            assert (
                batch_input_features is not None
                and batch_feature_attention_mask is not None
            )
            max_audio_len = max(
                (int(feature.shape[-1]) for feature in batch_input_features)
            )
            padded_input_features = []
            padded_feature_attention_mask = []
            for feature, feature_mask in zip(
                batch_input_features, batch_feature_attention_mask
            ):
                pad_len = max_audio_len - int(feature.shape[-1])
                if pad_len > 0:
                    feature_pad_shape = list(feature.shape)
                    feature_pad_shape[-1] = pad_len
                    feature = torch.cat(
                        [
                            feature,
                            torch.zeros(
                                feature_pad_shape,
                                dtype=feature.dtype,
                                device=feature.device,
                            ),
                        ],
                        dim=-1,
                    )
                    mask_pad_shape = list(feature_mask.shape)
                    mask_pad_shape[-1] = pad_len
                    feature_mask = torch.cat(
                        [
                            feature_mask,
                            torch.zeros(
                                mask_pad_shape,
                                dtype=feature_mask.dtype,
                                device=feature_mask.device,
                            ),
                        ],
                        dim=-1,
                    )
                padded_input_features.append(feature)
                padded_feature_attention_mask.append(feature_mask)
            batch_input_features = torch.stack(padded_input_features)
            batch_feature_attention_mask = torch.stack(padded_feature_attention_mask)
        if not text_only and (not audio_only) and (not video_only):
            assert batch_pixel_values is not None and batch_image_grid_thw is not None
            batch_pixel_values = torch.stack(batch_pixel_values)
            batch_image_grid_thw = torch.stack(batch_image_grid_thw)
        if video_only:
            assert (
                batch_pixel_values_videos is not None
                and batch_video_grid_thw is not None
            )
            assert batch_video_second_per_grid is not None
            batch_pixel_values_videos = torch.cat(batch_pixel_values_videos, dim=0)
            batch_video_grid_thw = torch.stack(batch_video_grid_thw)
            batch_video_second_per_grid = torch.cat(batch_video_second_per_grid, dim=0)
        infer_dtype = getattr(self, "infer_autocast_dtype", torch.bfloat16)
        if str(self.device).startswith("cuda") and infer_dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            autocast_ctx = torch.autocast("cuda", dtype=infer_dtype)
        else:
            autocast_ctx = contextlib.nullcontext()
        with autocast_ctx:
            if audio_only:
                output = self.vlm_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    input_features=batch_input_features,
                    feature_attention_mask=batch_feature_attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
            elif video_only:
                output = self.vlm_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    pixel_values_videos=batch_pixel_values_videos,
                    video_grid_thw=batch_video_grid_thw,
                    video_second_per_grid=batch_video_second_per_grid,
                    output_hidden_states=True,
                    return_dict=True,
                )
            elif text_only:
                output = self.vlm_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
            else:
                output = self.vlm_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    pixel_values=batch_pixel_values,
                    image_grid_thw=batch_image_grid_thw,
                    output_hidden_states=True,
                    return_dict=True,
                )
            vlm_hidden_states_ = output.hidden_states
            vlm_hidden_states = vlm_hidden_states_[-1]
            vlm_hidden_states = self._action_condition_hidden_states(
                vlm_hidden_states.to(dtype=states.dtype),
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
            )
            action_samples = torch.randn(
                bsz, self.action_horizon, self.action_dim, device=self.device
            )
            prefix_mask = (torch.arange(H, device=self.device)[None, :] < d).expand(
                bsz, -1
            )
            self.noise_scheduler.set_timesteps(num_inference_steps)
            for step_index, timestep in enumerate(self.noise_scheduler.timesteps):
                effective_scale = cfg_scale
                if cfg_interval is not None and (
                    not cfg_low
                    <= float(self.noise_scheduler.sigmas[step_index])
                    <= cfg_high
                ):
                    effective_scale = 1.0
                batched_timestep = torch.where(
                    prefix_mask,
                    torch.zeros((), device=self.device, dtype=timestep.dtype),
                    timestep,
                )
                action_samples = torch.where(
                    prefix_mask[:, :, None], prev_actions, action_samples
                )
                model_pred = self.action_header(
                    hidden_states=None,
                    timestep=batched_timestep,
                    joint_attention_kwargs=dict(
                        action_hidden_embeds=action_samples,
                        known_action_prefix=None,
                        views=vlm_hidden_states,
                        obs=states,
                        traj2ds=traj2ds,
                        progress=progress,
                        future_progress=future_progress,
                    ),
                    vlm_attn_mask=batch_attention_mask
                    if effective_scale != 1.0
                    else None,
                    return_dict=True,
                ).action
                if effective_scale != 1.0:
                    uncond_pred = self.action_header(
                        hidden_states=None,
                        timestep=batched_timestep,
                        joint_attention_kwargs=dict(
                            action_hidden_embeds=action_samples,
                            known_action_prefix=None,
                            views=vlm_hidden_states,
                            obs=states,
                            traj2ds=traj2ds,
                            progress=progress,
                            future_progress=future_progress,
                        ),
                        vlm_attn_mask=torch.zeros_like(batch_attention_mask),
                        return_dict=True,
                    ).action
                    model_pred = uncond_pred + effective_scale * (
                        model_pred - uncond_pred
                    )
                action_samples = self.noise_scheduler.step(
                    model_output=model_pred, timestep=timestep, sample=action_samples
                ).prev_sample
            action_samples = torch.where(
                prefix_mask[:, :, None], prev_actions, action_samples
            )
        return action_samples.float()

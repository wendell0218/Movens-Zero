import json
from pathlib import Path
import numpy as np

def validate_root40_config(cfg):
    repack = cfg.data.transform.repack
    model = cfg.model
    enabled = bool(getattr(repack, 'root40_actions', False))
    version = getattr(model, 'motion_representation', 'legacy')
    auxiliary_weight = float(getattr(getattr(cfg, 'train', None), 'absolute_height_loss_weight', 0.0))
    if auxiliary_weight > 0:
        train = cfg.train
        field = cfg.data.transform.field
        if not enabled or version != 'root40_v1' or field.action_norm_type != 'bounds':
            raise ValueError('Sampled absolute height requires normalized root40_v1')
        if not model.rtc or model.noise_scheduler != 'flow' or int(model.train_diffusion_steps) != 1000:
            raise ValueError('Sampled absolute height requires RTC flow1000')
        if (int(model.action_chunk_size), int(model.action_exec_horizon), int(model.max_delay)) != (30, 20, 11):
            raise ValueError('Sampled absolute height requires RTC 30/20 horizons and max_delay=11')
        if getattr(cfg.data, 'sampler', None) != 'task_mixture' or int(train.gradient_accumulation_steps) != 1:
            raise ValueError('Sampled absolute height requires task_mixture and gradient_accumulation_steps=1')
        if float(train.jerk_loss_source_fps) != 50 or int(train.absolute_height_denoising_steps) != 10:
            raise ValueError('Sampled absolute height requires native 50 Hz and ten Euler steps')
        if int(train.absolute_height_chunks) not in (2, 3):
            raise ValueError('Sampled absolute height supports two or three chunks')
        if not np.allclose([field.action_min[37], field.action_max[37], field.state_min[37], field.state_max[37]], [0, 8, 0, 8]):
            raise ValueError('Sampled absolute height preserves action/state height bounds [0,8] m')
    if not enabled:
        if version == 'root40_v1' or int(model.action_dim) == 40:
            raise ValueError('root40_v1 requires repack.root40_actions=True')
        return
    if version != 'root40_v1' or int(model.action_dim) != 40 or int(model.odim) != 40:
        raise ValueError("root40_actions requires model.motion_representation='root40_v1', action_dim=40, odim=40")
    if getattr(cfg, 'initial_action_checkpoint', None):
        raise ValueError('root40_v1 does not accept initial_action_checkpoint; use a compatible full resume')
    if any((bool(getattr(repack, key, False)) for key in ('textop_actions', 'use_delta_actions', 'root_delta_actions'))):
        raise ValueError('root40_v1 cannot use a legacy action encoder')
    field = cfg.data.transform.field
    for transform in (repack, field):
        if getattr(transform, 'pad_action_dim', None) not in (None, 40) or getattr(transform, 'pad_state_dim', None) not in (None, 40):
            raise ValueError('root40_v1 cannot pad or truncate to legacy dimensions')
    if field.action_norm_type == 'none':
        return
    if field.action_norm_type != 'bounds':
        raise ValueError('root40_v1 requires explicitly approved 40D bounds or raw diagnostic mode')

    path = Path(field.stat_path).expanduser()
    if not path.is_file():
        raise ValueError('root40_v1 requires a versioned 40D statistics file')
    stats = json.loads(path.read_text())
    metadata = stats.get('meta', {})
    if metadata.get('motion_representation') != 'root40_v1' or metadata.get('motion_fps') != 50:
        raise ValueError('root40_v1 statistics must declare the matching representation and 50 Hz sampling')
    for key in (field.stat_action_key, field.stat_state_key):
        low = np.asarray(stats.get(key, {}).get('min', []), dtype=np.float64)
        high = np.asarray(stats.get(key, {}).get('max', []), dtype=np.float64)
        if low.shape != (40,) or high.shape != (40,) or (not np.isfinite([low, high]).all()) or np.any(low >= high):
            raise ValueError('root40_v1 statistics require 40 finite, positive-width bounds; legacy padding is forbidden')
    for name in ('min', 'max'):
        if stats[field.stat_action_key][name] != stats[field.stat_state_key][name]:
            raise ValueError('root40_v1 action and state bounds must use the same protocol')
    for name in ('action_min', 'action_max', 'state_min', 'state_max'):
        if getattr(field, name, None) is None or len(getattr(field, name)) != 40:
            raise ValueError('root40_v1 requires populated 40D action and state normalization')

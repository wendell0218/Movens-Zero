"""Wire contract shared by inference service and non-actuating diagnostics."""
import base64
import json
import numpy as np
PROTOCOL = 'movens-root40-sonic-v1'
MODES = {'text': 'text', 'music': 'music', 'audio': 'speech', 'speech': 'speech', 'video_clip': 'video', 'video': 'video'}

def encode(value):
    def default(obj):
        if isinstance(obj, np.ndarray):
            return {'__numpy__': base64.b64encode(obj.tobytes()).decode(), 'dtype': obj.dtype.str, 'shape': list(obj.shape)}
        raise TypeError(type(obj).__name__)
    return json.dumps(value, default=default, allow_nan=False)

def decode(text):
    def hook(obj):
        if '__numpy__' not in obj:
            return obj
        dtype = np.dtype(obj['dtype'])
        if dtype.hasobject or dtype.kind not in 'biuf':
            raise ValueError('Unsupported wire dtype')
        shape = tuple(int(x) for x in obj['shape'])
        if any(x < 0 for x in shape) or len(shape) > 5:
            raise ValueError('Invalid wire shape')
        raw = base64.b64decode(obj['__numpy__'], validate=True)
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    return json.loads(text, object_hook=hook)

def observation(packet):
    state = np.asarray(packet['state']['states'], dtype=np.float32)
    if state.shape != (36,) or not np.isfinite(state).all() or np.linalg.norm(state[32:36]) < 1e-8:
        raise ValueError('Expected a finite motion36 observation with a nonzero quaternion')
    condition = packet.get('condition') or {}
    mode = MODES.get(condition.get('input_mode'))
    if mode is None:
        raise ValueError('Choose text/music/speech/video')
    target = int(condition.get('progress_target_frames', 0))
    if not 1 <= target <= 180000:
        raise ValueError('A bounded positive target frame count is required')
    prompt = str(packet.get('instruction') or '').strip()
    if mode == 'text' and not prompt:
        raise ValueError('Text task needs a nonempty instruction')
    frame = int(condition.get('client_frame', 0))
    if frame < 0:
        raise ValueError('Negative client frame')
    return state, mode, prompt, condition, target, frame

def response(absolute, frame, elapsed, checkpoint, backend):
    return dict(action=absolute, protocol_version=PROTOCOL, action_horizon=30, action_exec_horizon=20,
                rtc_overlap=10, action_dim=36, model_action_dim=40, representation='motion36_isaaclab_wxyz',
                chunk_start_frame=frame, chunk_version=frame // 20 + 1, version=frame // 20 + 1,
                controller_generated_frames=frame, controller_t=0, overlap_exact=frame > 0,
                inference_seconds=elapsed, checkpoint_sha256=checkpoint, backend=backend)

"""Non-actuating endpoint test. Never imports DDS, Sonic or the robot orchestrator."""
import argparse
import base64
import hashlib
import json
import select
from pathlib import Path
import time
import urllib.request
import numpy as np
import websocket
from protocol import PROTOCOL, encode, decode

def run_probe(host,port,mode='text',prompt='A person stands still.',media=None,seconds=6.0,allow_fixture=False):
    with urllib.request.urlopen(f'http://{host}:{port}/health',timeout=5) as r:
        health = json.load(r)
    assert health['protocol_version'] == PROTOCOL, health
    if not allow_fixture:
        assert health['backend'] == 'checkpoint' and health['checkpoint_verified'], health
    pose = json.loads((Path(__file__).resolve().parent/'config/standard_standing.json').read_text())['qpos_36']
    # Canonical MuJoCo qpos -> IsaacLab motion36 (the same map as the live state writer).
    mapping = [0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28]
    pose = np.asarray(pose,dtype=np.float32)
    state = np.concatenate([pose[7:][mapping],pose[:3],pose[3:7]])
    condition = dict(input_mode=dict(speech='audio',video='video_clip').get(mode,mode),progress_target_frames=max(61,round(seconds*50)),client_frame=0)
    if media:
        raw = Path(media).read_bytes(); key='video' if mode=='video' else 'audio'
        digest=hashlib.sha256(raw).hexdigest()
        condition.update({key+'_sha256':digest,key+'_inline_id':digest,key+'_filename':Path(media).name,key+'_bytes_b64':base64.b64encode(raw).decode()})
    ws=websocket.create_connection(f'ws://{host}:{port}/ws',timeout=180,http_proxy_host=None)
    previous=None
    start=time.monotonic()
    try:
        for frame in (0,20):
            condition['client_frame']=frame
            packet=dict(state={'states':state},condition=condition,instruction=prompt)
            ws.send(encode(packet))
            # Read readiness polling must not shrink the socket's write timeout.
            # Download contention may delay a heartbeat without invalidating its frame.
            ws.settimeout(5.0)
            deadline=time.monotonic()+180
            while True:
                ready,_,_=select.select([ws.sock],[],[],0.02)
                if ready:
                    result=decode(ws.recv()); break
                if time.monotonic()>deadline: raise TimeoutError('Inference response timed out')
                for k in ('video_bytes_b64','audio_bytes_b64'): condition.pop(k,None)
                ws.send(encode(packet))
            if result.get('error'): raise RuntimeError(result['error'])
            action=np.asarray(result['action'])
            assert result['chunk_start_frame']==frame and action.shape==(30,36) and np.isfinite(action).all(),result
            assert result['model_action_dim']==40 and result['protocol_version']==PROTOCOL,result
            if previous is not None: np.testing.assert_allclose(action[:10],previous[20:],rtol=1e-5,atol=1e-5)
            previous=action; state=action[19].copy()
            print(json.dumps(dict(frame=frame,shape=list(action.shape),backend=result['backend'],checkpoint_sha256=result['checkpoint_sha256'],inference_seconds=result['inference_seconds'])),flush=True)
        print(f'PROBE_OK mode={mode} chunks=2 elapsed={time.monotonic()-start:.3f}s DDS=NOT_STARTED WBC=NOT_STARTED')
    finally:
        ws.close()
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--host',required=True); p.add_argument('--port',type=int,default=8014)
    p.add_argument('--allow-transport-fixture',action='store_true'); p.add_argument('--modality',choices=['text','music','speech','video'],default='text'); p.add_argument('--media',type=Path)
    a=p.parse_args(); raise SystemExit(run_probe(a.host,a.port,a.modality,media=a.media,allow_fixture=a.allow_transport_fixture))

import asyncio
import base64
import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
from starlette.websockets import WebSocketDisconnect
from server import make_app, MediaStore, TransportFixture
from protocol import encode, decode


class FakeWebSocket:
    def __init__(self, packets):
        self.packets = asyncio.Queue()
        for packet in packets:
            self.packets.put_nowait(packet)
        self.messages = []
        self.closed = False

    async def accept(self):
        pass

    async def receive_text(self):
        item = await self.packets.get()
        if item is None:
            raise WebSocketDisconnect()
        return encode(item)

    async def send_text(self, text):
        self.messages.append(decode(text))
        if len(self.messages) >= 1:
            self.packets.put_nowait(None)

    async def close(self, code=1000):
        self.closed = True


class ServerTests(unittest.IsolatedAsyncioTestCase):
    def packet(self):
        state = np.zeros(36, np.float32)
        state[31] = 0.78
        state[32] = 1
        return dict(
            state={"states": state},
            instruction="Stand still",
            condition=dict(
                input_mode="text", progress_target_frames=300, client_frame=0
            ),
        )

    async def test_fixture_health_is_explicit(self):
        app = make_app(TransportFixture(), fixture=True)
        self.addCleanup(app.state.media_store.temp.cleanup)
        health = next((r.endpoint for r in app.routes if r.path == "/health"))
        data = await health()
        self.assertFalse(data["checkpoint_verified"])
        self.assertEqual(data["backend"], "transport_fixture")

    async def test_wrong_initial_ack_rejected(self):
        app = make_app(TransportFixture(), fixture=True)
        self.addCleanup(app.state.media_store.temp.cleanup)
        handler = next((r.endpoint for r in app.routes if r.path == "/ws"))
        packet = self.packet()
        packet["condition"]["client_frame"] = 20
        ws = FakeWebSocket([packet])
        await handler(ws)
        self.assertIn("start at frame zero", ws.messages[0]["error"])
        self.assertTrue(ws.closed)

    async def test_first_upload_not_overwritten_by_heartbeat(self):
        app = make_app(TransportFixture(), fixture=True)
        self.addCleanup(app.state.media_store.temp.cleanup)
        handler = next((r.endpoint for r in app.routes if r.path == "/ws"))
        raw = b"fixture-for-upload-retention"
        digest = hashlib.sha256(raw).hexdigest()
        packet = self.packet()
        packet["condition"].update(
            input_mode="video_clip",
            video_sha256=digest,
            video_filename="test.mp4",
            video_bytes_b64=base64.b64encode(raw).decode(),
        )
        heartbeat = self.packet()
        heartbeat["condition"].update(input_mode="video_clip", video_sha256=digest)
        ws = FakeWebSocket([packet, heartbeat])
        await handler(ws)
        self.assertIn("action", ws.messages[0])
        self.assertEqual(ws.messages[0]["action"].shape, (30, 36))

    async def test_media_hash_mismatch(self):
        store = MediaStore()
        self.addCleanup(store.temp.cleanup)
        with self.assertRaises(ValueError):
            store.prepare(
                "video",
                dict(
                    video_sha256="a" * 64,
                    video_filename="test.mp4",
                    video_bytes_b64=base64.b64encode(b"bad").decode(),
                ),
            )

    async def test_no_server_local_path_fallback(self):
        store = MediaStore()
        self.addCleanup(store.temp.cleanup)
        with self.assertRaises(ValueError):
            store.prepare("speech", dict(audio_path="/etc/passwd"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

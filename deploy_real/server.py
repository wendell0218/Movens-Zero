from __future__ import annotations
import argparse
from contextlib import asynccontextmanager
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from protocol import PROTOCOL, encode, decode, observation, response

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))


class MediaStore:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="movens-real-media-")
        self.cache = {}

    def prepare(self, mode, condition):
        if mode == "text":
            return None
        key = "video" if mode == "video" else "audio"
        digest = str(condition.get(key + "_sha256", ""))
        if len(digest) != 64 or any((c not in "0123456789abcdef" for c in digest)):
            raise ValueError("Media requires a SHA256 identity")
        cache_key = (key, digest)
        if cache_key in self.cache:
            return self.cache[cache_key]
        if len(self.cache) >= 8:
            raise ValueError(
                "Media cache full; restart the service before loading additional media"
            )
        raw = base64.b64decode(condition.get(key + "_bytes_b64", ""), validate=True)
        limit = (384 if key == "video" else 64) * 1024 * 1024
        if not raw or len(raw) > limit or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(
                "Missing/oversized media or SHA256 mismatch; upload from the client"
            )
        suffix = Path(str(condition.get(key + "_filename", ""))).suffix.lower()
        if suffix not in (
            ".mp4",
            ".mov",
            ".mkv",
            ".avi",
            ".webm",
            ".m4v",
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".m4a",
        ):
            raise ValueError("Unsupported media format")
        path = Path(self.temp.name) / (digest + suffix)
        path.write_bytes(raw)
        media = dict(path=path)
        if key == "audio":
            decoded = subprocess.check_output(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-threads",
                    "1",
                    "-i",
                    str(path),
                    "-t",
                    "3600",
                    "-f",
                    "f32le",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "pipe:1",
                ],
                timeout=120,
            )
            media["audio"] = np.frombuffer(decoded, dtype=np.float32).copy()
            if not len(media["audio"]) or not np.isfinite(media["audio"]).all():
                raise ValueError("Invalid decoded audio")
        self.cache[cache_key] = media
        return media


class TransportFixture:
    def infer(self, state, mode, prompt, media, frame, target, previous=None):
        return (np.repeat(np.asarray(state)[None], 30, axis=0), {})


def make_app(policy, checkpoint="", fixture=False):
    store = MediaStore()

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            store.temp.cleanup()

    app = FastAPI(lifespan=lifespan)
    app.state.media_store = store
    busy = False
    backend = "transport_fixture" if fixture else "checkpoint"

    @app.get("/health")
    async def health():
        return dict(
            status="ok",
            input_mode="auto",
            supported_input_modes=["text", "music", "audio", "video_clip"],
            protocol_version=PROTOCOL,
            model_action_dim=40,
            action_dim=36,
            checkpoint_sha256=checkpoint,
            checkpoint_verified=bool(checkpoint) and (not fixture),
            backend=backend,
            busy=busy,
        )

    @app.websocket("/ws")
    async def websocket(ws: WebSocket):
        nonlocal busy
        await ws.accept()
        if busy:
            await ws.send_text(
                encode({"error": "Another client owns the model; disconnect it first"})
            )
            await ws.close(code=1013)
            return
        busy = True
        receiver = None
        latest = None
        last_rx = time.monotonic()
        updated = asyncio.Event()
        disconnected = False

        async def receive():
            nonlocal latest, last_rx, disconnected
            try:
                while True:
                    latest = decode(await ws.receive_text())
                    last_rx = time.monotonic()
                    updated.set()
            finally:
                disconnected = True
                updated.set()

        try:
            initial = decode(await asyncio.wait_for(ws.receive_text(), timeout=15))
            latest = initial
            last_rx = time.monotonic()
            state, mode, prompt, cond, target, client_frame = observation(initial)
            receiver = asyncio.create_task(receive())
            if client_frame != 0:
                raise ValueError("New sessions must start at frame zero")
            media = await asyncio.to_thread(store.prepare, mode, cond)
            identity = (
                mode,
                prompt,
                target,
                cond.get("video_sha256"),
                cond.get("audio_sha256"),
            )
            first_tick = time.monotonic()
            first, first_previous = await asyncio.to_thread(
                policy.infer, state, mode, prompt, media, 0, target
            )
            first_latency = time.monotonic() - first_tick
            if not fixture:
                for _ in range(2):
                    tick = time.monotonic()
                    await asyncio.to_thread(
                        policy.infer,
                        first[19],
                        mode,
                        prompt,
                        media,
                        20,
                        target,
                        first_previous,
                    )
                    latency = time.monotonic() - tick
                if latency > 0.18:
                    raise RuntimeError(
                        f"RTC preflight deadline exceeded: {latency:.6f}s > 0.180000s"
                    )
            if disconnected:
                return
            previous, last_absolute = (first_previous, first)
            await ws.send_text(
                encode(response(first, 0, first_latency, checkpoint, backend))
            )
            next_emit = time.monotonic() + 0.4
            frame = 20
            while frame < target and (not disconnected):
                deadline = next_emit + 0.18
                while not disconnected:
                    state, new_mode, new_prompt, cond, new_target, ack = observation(
                        latest
                    )
                    if (
                        new_mode,
                        new_prompt,
                        new_target,
                        cond.get("video_sha256"),
                        cond.get("audio_sha256"),
                    ) != identity:
                        raise ValueError("Task changes require a new websocket session")
                    if ack == frame:
                        break
                    if ack > frame:
                        if ack >= target:
                            return
                        raise ValueError(
                            f"Out-of-order client acknowledgement {ack}, expected {frame}"
                        )
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            "Client acknowledgement missed the RTC overlap deadline"
                        )
                    updated.clear()
                    try:
                        await asyncio.wait_for(updated.wait(), timeout=0.02)
                    except asyncio.TimeoutError:
                        pass
                if disconnected:
                    break
                await asyncio.sleep(max(0, next_emit - 0.18 - time.monotonic()))
                if time.monotonic() - last_rx > 0.2:
                    raise TimeoutError("Stale websocket observation")
                state, _, _, _, _, ack = observation(latest)
                tick = time.monotonic()
                absolute, next_previous = await asyncio.to_thread(
                    policy.infer, state, mode, prompt, media, frame, target, previous
                )
                elapsed = time.monotonic() - tick
                if not fixture and (
                    elapsed > 0.18 or time.monotonic() > next_emit + 0.18
                ):
                    raise TimeoutError(
                        f"RTC deadline exceeded: inference={elapsed:.6f}s"
                    )
                if not np.allclose(
                    absolute[:10], last_absolute[20:], rtol=1e-05, atol=1e-05
                ):
                    raise ValueError(
                        "Rebased RTC absolute-pose overlap is discontinuous"
                    )
                await asyncio.sleep(max(0, next_emit - time.monotonic()))
                if disconnected:
                    break
                await ws.send_text(
                    encode(response(absolute, frame, elapsed, checkpoint, backend))
                )
                previous, last_absolute = (next_previous, absolute)
                frame += 20
                next_emit += 0.4
        except (WebSocketDisconnect, ConnectionError):
            pass
        except Exception as exc:
            print(f"[server] session stopped: {exc}", flush=True)
            try:
                await ws.send_text(encode({"error": str(exc)}))
            except Exception:
                pass
        finally:
            if receiver:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
            try:
                await ws.close()
            except Exception:
                pass
            busy = False

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("SERVER_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("MOVENS_PORT", "8014"))
    )
    parser.add_argument("--checkpoint", default=os.environ.get("MOVENS_CHECKPOINT"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--transport-test",
        action="store_true",
        help="Synthetic transport diagnostic; rejected by the robot launcher",
    )
    args = parser.parse_args()
    if args.transport_test:
        if args.port == 8014:
            parser.error(
                "Transport diagnostics must use a separate port, not the robot service port 8014"
            )
        app = make_app(TransportFixture(), fixture=True)
    else:
        from backend import checkpoint_preflight, Root40Policy

        verified = checkpoint_preflight(weight=args.checkpoint)
        print(
            json.dumps({k: v for k, v in verified.items() if k != "config"}), flush=True
        )
        if args.check_only:
            return
        app = make_app(Root40Policy(verified), verified["sha256"])
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ws_max_size=576 * 1024 * 1024,
        log_level="info",
    )


if __name__ == "__main__":
    main()

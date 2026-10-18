"""Single-worker live service. Run: uvicorn pipeline.server:app --host 0.0.0.0"""
import asyncio
import json
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from .collect import WEB, collect


@asynccontextmanager
async def lifespan(app):
    async def poll():
        while True:
            try:
                await asyncio.to_thread(collect, int(os.getenv('CAMERA_LIMIT', '48')))
            except Exception:
                import logging
                logging.exception('Collection failed; retaining last successful snapshot')
            await asyncio.sleep(max(120, int(os.getenv('POLL_SECONDS', '120'))))
    task = asyncio.create_task(poll())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)


@app.get('/api/events')
async def events():
    async def stream():
        previous = None
        while True:
            path = WEB / 'data/latest.json'
            current = path.read_text() if path.exists() else '{}'
            if current != previous:
                yield 'data: ' + current + '\n\n'
                previous = current
            else:
                yield ': heartbeat\n\n'
            await asyncio.sleep(5)
    return StreamingResponse(stream(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache'})


@app.get('/api/cameras/{camera_id}/stream')
async def camera_stream(camera_id: int):
    async def stream():
        while True:
            path = WEB / 'frames' / f'{camera_id}.jpg'
            if path.exists():
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + path.read_bytes() + b'\r\n'
            await asyncio.sleep(10)
    return StreamingResponse(stream(), media_type='multipart/x-mixed-replace; boundary=frame')


app.mount('/', StaticFiles(directory=WEB, html=True), name='dashboard')

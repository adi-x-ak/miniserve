"""Streaming byte events over HTTP; each app owns one scheduler."""

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from miniserve.model import Transformer, load_model
from miniserve.scheduler import Scheduler


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=32, ge=1)


def create_app(model: Transformer | None = None, slots: int = 8, max_pending: int = 128) -> FastAPI:
    torch.set_num_threads(1)
    scheduler = Scheduler(model or load_model(), slots=slots, max_pending=max_pending)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def pump() -> None:
            while True:
                scheduler.step()
                await asyncio.sleep(0.001)

        task = asyncio.create_task(pump())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(title="MiniServe", lifespan=lifespan)
    app.state.scheduler = scheduler

    @app.get("/health")
    def health() -> dict[str, int | str]:
        return {
            "status": "ok",
            "active": len(scheduler.active),
            "pending": len(scheduler.pending),
            "cache_bytes": scheduler.cache.bytes,
            "completed": scheduler.completed,
            "cancelled": scheduler.cancelled,
        }

    @app.post("/generate")
    async def generate(request: GenerateRequest) -> StreamingResponse:
        try:
            job = scheduler.submit(request.prompt, request.max_tokens)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except OverflowError as error:
            raise HTTPException(status_code=429, detail=str(error)) from error

        async def events():
            try:
                while True:
                    while job.buffer:
                        # Byte values remain lossless when UTF-8 characters span tokens.
                        yield f"data: {json.dumps({'byte': job.buffer.popleft()})}\n\n"
                    if job.finished:
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        break
                    await asyncio.sleep(0.001)
            finally:
                job.cancelled = True

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a small byte Transformer on CPU.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(
        create_app(load_model(args.checkpoint), slots=args.slots), host="127.0.0.1", port=args.port
    )


if __name__ == "__main__":
    main()

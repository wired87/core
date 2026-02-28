"""
Grid streamer: low-latency binary streaming of time_db frames to WebSocket.
Uses bounded queue, full-frame updates, and backpressure-aware frame dropping.
"""

from __future__ import annotations

import asyncio
import os
import struct
from typing import Any, Awaitable, Callable

import numpy as np


def build_grid_frame(step: int, data: np.ndarray) -> bytes:
    """
    Build binary frame: 8-byte header (step, n) + raw float32 data.
    Little-endian.
    """
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.ravel()
    n = int(arr.size)
    header = struct.pack("<II", step, n)
    return header + arr.tobytes()


class GridStreamer:
    """
    Streams grid frames to WebSocket with bounded queue and backpressure handling.
    Sim never blocks; frames are dropped if queue is full or send times out.
    """

    def __init__(
        self,
        send_fn: Callable[[bytes], Awaitable[None]],
        *,
        queue_size: int = 2,
        send_timeout: float = 0.05,
    ):
        """
        Args:
            send_fn: Async callable(bytes) that sends to WebSocket, e.g.
                     lambda b: consumer.send(bytes_data=b)
            queue_size: Max frames to buffer; excess dropped.
            send_timeout: Seconds to wait for send before dropping frame.
        """
        self._send_fn = send_fn
        self._queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(maxsize=queue_size)
        self._send_timeout = send_timeout
        self._task: asyncio.Task[None] | None = None
        self._running = False

    def put_frame(self, step: int, data: np.ndarray) -> None:
        """
        Non-blocking: enqueue frame. Drops oldest if full.
        Call from sim loop after each step.
        """
        if not os.getenv("GRID_STREAM_ENABLED", "false").lower() in ("true", "1"):
            return
        payload = build_grid_frame(step, data)
        try:
            self._queue.put_nowait((step, payload))
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait((step, payload))
            except asyncio.QueueFull:
                pass

    async def _sender_loop(self) -> None:
        while self._running:
            try:
                step, payload = await self._queue.get()
                try:
                    await asyncio.wait_for(
                        self._send_fn(payload),
                        timeout=self._send_timeout,
                    )
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    pass
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    def start(self) -> None:
        """Start sender task. Call when WebSocket is connected."""
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._sender_loop())

    def stop(self) -> None:
        """Stop sender task. Call on disconnect."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            self._task = None

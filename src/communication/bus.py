from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from .acl import ACLEnvelope, ACLSemanticRegistry, validate_acl_semantics


Handler = Callable[[ACLEnvelope], Awaitable[None]]
PublishHook = Callable[[ACLEnvelope], Awaitable[None]]


@dataclass
class BusConfig:
    queue_maxsize: int = 0  # 0 = unbounded


class AsyncBus:
    """
    Simple asyncio message bus with per-receiver queues.

    This bus is intentionally minimal:
    - Route by ``envelope.receiver``.
    - Each receiver has a single consumer handler.
    """

    def __init__(
        self,
        config: Optional[BusConfig] = None,
        on_publish: Optional[PublishHook] = None,
        *,
        strict_semantics: bool = True,
    ):
        self.config = config or BusConfig()
        self._on_publish = on_publish
        self._strict_semantics = strict_semantics
        self._acl_registry = ACLSemanticRegistry()
        self._queues: dict[str, asyncio.Queue[ACLEnvelope]] = {}
        self._handlers: dict[str, Handler] = {}
        self._tasks: list[asyncio.Task] = []
        self._stopped = asyncio.Event()

    def register(self, receiver: str, handler: Handler) -> None:
        self._handlers[receiver] = handler
        if receiver not in self._queues:
            self._queues[receiver] = asyncio.Queue(maxsize=self.config.queue_maxsize)

    async def publish(self, env: ACLEnvelope) -> None:
        if self._strict_semantics:
            validate_acl_semantics(env, self._acl_registry)
            self._acl_registry.apply_publish_effects(env)
        if self._on_publish is not None:
            await self._on_publish(env)
        if env.receiver not in self._queues:
            self._queues[env.receiver] = asyncio.Queue(maxsize=self.config.queue_maxsize)
        await self._queues[env.receiver].put(env)

    async def start(self) -> None:
        self._stopped.clear()
        for receiver, q in self._queues.items():
            handler = self._handlers.get(receiver)
            if handler is None:
                continue
            self._tasks.append(asyncio.create_task(self._run_consumer(receiver, q, handler)))

    async def stop(self) -> None:
        self._stopped.set()
        for t in list(self._tasks):
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _run_consumer(
        self,
        receiver: str,
        q: asyncio.Queue[ACLEnvelope],
        handler: Handler,
    ) -> None:
        _ = receiver
        while not self._stopped.is_set():
            env = await q.get()
            try:
                await handler(env)
            finally:
                q.task_done()


"""
redis_bus.py – Redis-Backed EventBus with AsyncIO Fallback

Drop-in replacement for the in-process EventBus in langgraph_engine.py.
When Redis is available: uses Redis Pub/Sub so multiple gateway replicas
share the same event stream (horizontal scaling).
When Redis is unavailable: silently falls back to the in-memory asyncio.Queue bus.

Usage:
    from redis_bus import get_event_bus
    event_bus = get_event_bus()   # returns Redis or in-memory implementation
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, List, Optional

import structlog

from config import settings

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Interface (matches the existing EventBus API)
# ─────────────────────────────────────────────────────────────────────────────

class AbstractEventBus:
    async def publish(self, thread_id: str, event: dict) -> None: ...
    def subscribe(self, thread_id: str) -> asyncio.Queue: ...
    def unsubscribe(self, thread_id: str, q: asyncio.Queue) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# In-Memory Bus (already in langgraph_engine.py – replicated here for clarity)
# ─────────────────────────────────────────────────────────────────────────────

class InMemoryEventBus(AbstractEventBus):
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}

    def subscribe(self, thread_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=512)
        self._subscribers.setdefault(thread_id, []).append(q)
        return q

    def unsubscribe(self, thread_id: str, q: asyncio.Queue) -> None:
        subs = self._subscribers.get(thread_id, [])
        if q in subs:
            subs.remove(q)

    async def publish(self, thread_id: str, event: dict) -> None:
        event.setdefault("ts", time.time())
        for q in list(self._subscribers.get(thread_id, [])):
            try: q.put_nowait(event)
            except asyncio.QueueFull: pass
        for q in list(self._subscribers.get("*", [])):
            try: q.put_nowait({"thread_id": thread_id, **event})
            except asyncio.QueueFull: pass


# ─────────────────────────────────────────────────────────────────────────────
# Redis Pub/Sub Bus
# ─────────────────────────────────────────────────────────────────────────────

class RedisEventBus(AbstractEventBus):
    """
    Uses Redis Pub/Sub channels:
      - "events:{thread_id}" for per-thread events
      - "events:*"           for the global wildcard feed (PSUBSCRIBE)

    Each subscriber spawns a background task that reads from Redis and
    forwards messages into a local asyncio.Queue for the caller.
    """

    CHANNEL_PREFIX = "orchestrator:events"

    def __init__(self, redis_url: str) -> None:
        import redis.asyncio as aioredis  # type: ignore
        self._redis = aioredis.from_url(redis_url, decode_responses=True)
        self._local_queues: Dict[str, List[asyncio.Queue]] = {}
        self._reader_tasks: Dict[str, asyncio.Task] = {}
        log.info("redis_event_bus_init", url=redis_url)

    def _channel(self, thread_id: str) -> str:
        return f"{self.CHANNEL_PREFIX}:{thread_id}"

    async def publish(self, thread_id: str, event: dict) -> None:
        event.setdefault("ts", time.time())
        payload = json.dumps({"thread_id": thread_id, **event})
        await self._redis.publish(self._channel(thread_id), payload)
        # Also publish to wildcard channel for global dashboard feed
        await self._redis.publish(self._channel("*"), payload)

    def subscribe(self, thread_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=512)
        self._local_queues.setdefault(thread_id, []).append(q)

        # Start reader task if not already running for this channel
        if thread_id not in self._reader_tasks:
            task = asyncio.create_task(self._reader_loop(thread_id))
            self._reader_tasks[thread_id] = task

        return q

    def unsubscribe(self, thread_id: str, q: asyncio.Queue) -> None:
        subs = self._local_queues.get(thread_id, [])
        if q in subs:
            subs.remove(q)

    async def _reader_loop(self, thread_id: str) -> None:
        """Background task: subscribes to Redis channel and fans out to local queues."""
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(self._channel(thread_id))
        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                try:
                    event = json.loads(message["data"])
                except Exception:
                    continue
                for q in list(self._local_queues.get(thread_id, [])):
                    try: q.put_nowait(event)
                    except asyncio.QueueFull: pass
        except Exception as exc:
            log.error("redis_reader_error", thread_id=thread_id, error=str(exc))
        finally:
            try: await pubsub.unsubscribe(self._channel(thread_id))
            except Exception: pass
            self._reader_tasks.pop(thread_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

_cached_bus: Optional[AbstractEventBus] = None

def get_event_bus() -> AbstractEventBus:
    global _cached_bus
    if _cached_bus is not None:
        return _cached_bus

    # Fast in-memory bus default unless Redis is explicitly configured
    redis_url = getattr(settings, "redis_url", None)
    if not redis_url:
        _cached_bus = InMemoryEventBus()
        log.info("event_bus", backend="in_memory")
        return _cached_bus

    try:
        import redis as sync_redis
        r = sync_redis.from_url(redis_url, socket_connect_timeout=0.1, socket_timeout=0.1)
        r.ping()
        r.close()
        _cached_bus = RedisEventBus(redis_url)
        log.info("event_bus", backend="redis", url=redis_url)
    except Exception as exc:
        log.info("event_bus", backend="in_memory", reason=str(exc))
        _cached_bus = InMemoryEventBus()

    return _cached_bus

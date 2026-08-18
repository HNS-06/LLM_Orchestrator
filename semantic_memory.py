"""
semantic_memory.py – Qdrant vector store integration for long-term semantic memory.

Responsibilities:
  - Store episodic memories (user messages + AI responses) as dense vectors.
  - Retrieve top-k semantically similar memories on each new turn.
  - Graceful mock fallback when Qdrant is not reachable (USE_QDRANT_MEMORY=False).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import structlog

from config import settings

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Memory Record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryRecord:
    content: str
    thread_id: str
    role: str = "user"                # "user" | "assistant"
    score: float = 1.0                # cosine similarity score (0–1)
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "thread_id": self.thread_id,
            "role": self.role,
            "content": self.content,
            "score": round(self.score, 4),
            "created_at": self.created_at,
        }


# ─────────────────────────────────────────────────────────────────────────────
# In-Memory Mock Store (fallback when Qdrant is offline)
# ─────────────────────────────────────────────────────────────────────────────

class MockMemoryStore:
    """Simple keyword-overlap mock for offline testing."""

    def __init__(self) -> None:
        self._records: List[MemoryRecord] = []

    async def upsert(self, record: MemoryRecord) -> None:
        self._records.append(record)
        log.debug("mock_memory_upsert", record_id=record.record_id)

    async def search(self, query: str, thread_id: Optional[str] = None, top_k: int = 5) -> List[MemoryRecord]:
        query_tokens = set(query.lower().split())
        scored: List[tuple[float, MemoryRecord]] = []
        for rec in self._records:
            if thread_id and rec.thread_id != thread_id:
                continue
            overlap = len(query_tokens & set(rec.content.lower().split()))
            if overlap:
                scored.append((overlap / max(len(query_tokens), 1), rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, rec in scored[:top_k]:
            rec.score = score
            results.append(rec)
        return results

    async def stats(self) -> dict:
        return {"backend": "mock", "total_records": len(self._records)}


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant Production Store
# ─────────────────────────────────────────────────────────────────────────────

class QdrantMemoryStore:
    """Production Qdrant-backed semantic memory using text-embedding-3-small."""

    def __init__(self) -> None:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        self._collection = settings.qdrant_collection
        self._dim = settings.embedding_dim
        self._Distance = Distance
        self._VectorParams = VectorParams
        self._initialized = False

    async def _ensure_collection(self) -> None:
        if self._initialized:
            return
        from qdrant_client.models import VectorParams, Distance
        existing = await self._client.get_collections()
        names = [c.name for c in existing.collections]
        if self._collection not in names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )
            log.info("qdrant_collection_created", collection=self._collection)
        self._initialized = True

    async def _embed(self, text: str) -> List[float]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        resp = await client.embeddings.create(model="text-embedding-3-small", input=text)
        return resp.data[0].embedding

    async def upsert(self, record: MemoryRecord) -> None:
        await self._ensure_collection()
        vector = await self._embed(record.content)
        from qdrant_client.models import PointStruct
        point = PointStruct(
            id=record.record_id,
            vector=vector,
            payload={
                "thread_id": record.thread_id,
                "role": record.role,
                "content": record.content,
                "created_at": record.created_at,
            },
        )
        await self._client.upsert(collection_name=self._collection, points=[point])
        log.info("qdrant_upsert", record_id=record.record_id, thread_id=record.thread_id)

    async def search(self, query: str, thread_id: Optional[str] = None, top_k: int = 5) -> List[MemoryRecord]:
        await self._ensure_collection()
        query_vector = await self._embed(query)

        search_filter = None
        if thread_id:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            search_filter = Filter(
                must=[FieldCondition(key="thread_id", match=MatchValue(value=thread_id))]
            )

        results = await self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
        )
        records = []
        for hit in results:
            p = hit.payload or {}
            records.append(MemoryRecord(
                record_id=str(hit.id),
                thread_id=p.get("thread_id", ""),
                role=p.get("role", "user"),
                content=p.get("content", ""),
                score=float(hit.score),
                created_at=p.get("created_at", 0.0),
            ))
        return records

    async def stats(self) -> dict:
        await self._ensure_collection()
        info = await self._client.get_collection(self._collection)
        return {
            "backend": "qdrant",
            "collection": self._collection,
            "total_vectors": info.vectors_count,
            "indexed_vectors": info.indexed_vectors_count,
            "status": info.status.value if info.status else "unknown",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_memory_store() -> QdrantMemoryStore | MockMemoryStore:
    if settings.use_qdrant_memory:
        try:
            store = QdrantMemoryStore()
            log.info("memory_store", backend="qdrant")
            return store
        except Exception as exc:
            log.warning("qdrant_fallback", error=str(exc))
    log.info("memory_store", backend="mock")
    return MockMemoryStore()


memory_store = get_memory_store()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────

async def store_interaction(thread_id: str, user_msg: str, ai_response: str) -> None:
    """Persist a user–assistant exchange to long-term memory."""
    await memory_store.upsert(MemoryRecord(content=user_msg, thread_id=thread_id, role="user"))
    await memory_store.upsert(MemoryRecord(content=ai_response, thread_id=thread_id, role="assistant"))


async def recall_context(query: str, thread_id: Optional[str] = None, top_k: int = 5) -> List[dict]:
    """Retrieve semantically similar past interactions."""
    records = await memory_store.search(query=query, thread_id=thread_id, top_k=top_k)
    return [r.to_dict() for r in records]


async def memory_stats() -> dict:
    """Return current memory store statistics."""
    return await memory_store.stats()

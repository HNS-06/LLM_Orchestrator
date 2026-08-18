---
id: AGENT-0201
title: Short-Term State & Checkpointing (Postgres)
category: Memory Systems
tags: [postgres, checkpointing, psycopg3, thread-state, persistence, short-term-memory]
updated: 2026-08-16
status: verified
---

# Short-Term State & Checkpointing (Postgres)

Short-term memory in LangGraph is the **in-flight conversation state** for a single `thread_id`. It is ephemeral within a single process but persisted externally via `PostgresSaver` to survive restarts and enable cross-process continuity.

## Lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant G as FastAPI Gateway
    participant L as LangGraph Engine
    participant P as Postgres

    C->>G: POST /orchestrate {message, thread_id}
    G->>L: invoke(state, config={thread_id})
    L->>P: READ last checkpoint for thread_id
    P-->>L: AgentState snapshot
    L->>L: Execute nodes (supervisor → worker → ...)
    L->>P: WRITE new checkpoint after each node
    L-->>G: Final state
    G-->>C: Response + trace
```

## Checkpoint Table Schema

```sql
-- Auto-created by checkpointer.setup()
CREATE TABLE checkpoints (
    thread_id       TEXT NOT NULL,
    checkpoint_id   TEXT NOT NULL,
    parent_id       TEXT,
    type            TEXT,
    checkpoint      BYTEA NOT NULL,
    metadata        BYTEA,
    PRIMARY KEY (thread_id, checkpoint_id)
);

CREATE INDEX ON checkpoints (thread_id, checkpoint_id DESC);
```

## psycopg3 Connection Pool

```python
from psycopg_pool import AsyncConnectionPool

pool = AsyncConnectionPool(
    conninfo="postgresql://postgres:postgres@localhost:5432/agent_ops",
    min_size=5,
    max_size=20,
    timeout=30.0,
)
```

> **Target**: 5–15 ms p95 write latency with pool size = 20 connections.

## State Snapshot Retrieval (FastAPI)

```python
config = {"configurable": {"thread_id": thread_id}}
state_snapshot = compiled_graph.get_state(config)

# Access channel values
messages = state_snapshot.values.get("messages", [])
retry_count = state_snapshot.values.get("retry_count", 0)
next_step = state_snapshot.values.get("next_step", "")
```

## Thread History (All Checkpoints)

```python
# Walk backwards through checkpoint history
history = list(compiled_graph.get_state_history(config))
for snapshot in history:
    print(snapshot.config["configurable"]["checkpoint_id"])
    print(snapshot.values)
```

## Key Properties

| Property | Value |
|---|---|
| Scope | Single `thread_id` session |
| Persistence | External Postgres table |
| Reducer | `operator.add` (append) for message lists |
| Access pattern | Key-value by `(thread_id, checkpoint_id)` |
| Eviction | Manual or TTL-based cleanup |

## Related

- [[LangGraph State Machine & Checkpointing]]
- [[Long-Term Semantic Vector Memory (Qdrant)]]
- [[00 - MOC (Agentic Systems Map)]]

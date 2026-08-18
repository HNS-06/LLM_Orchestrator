---
id: AGENT-0104
title: LangGraph State Machine & Checkpointing
category: Core Architectures
tags: [langgraph, persistence, postgres, state-management, checkpointing, reducers]
updated: 2026-08-16
status: verified
---

# LangGraph State Machine & Checkpointing

State machines govern deterministic multi-turn loops. Production architectures mandate externalizing runtime state from memory to handle process restarts, horizontal scaling, and human-in-the-loop interrupts.

## State Channel Mechanics

### TypedDict with Reducers

```python
import operator
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # append-only
    next_step: str
    retry_count: int
    is_valid: bool
    trace: Annotated[List[dict], operator.add]            # telemetry events
```

- **Channel Reducer**: `Annotated[List[BaseMessage], operator.add]` appends transitions rather than overwriting the context buffer, preserving full conversational history across graph cycles.
- **Primitive channels** (`str`, `int`, `bool`) use the default `last-write-wins` reducer.

## Checkpoint Snapshot Format

Each LangGraph checkpoint stores a point-in-time snapshot indexed by `thread_id` + monotonic `checkpoint_id`:

```json
{
  "thread_id": "550e8400-e29b-41d4-a716-446655440000",
  "checkpoint_id": "1ef7c2b0-...",
  "ts": "2026-08-16T22:21:00Z",
  "channel_values": {
    "messages": [...],
    "next_step": "worker_search",
    "retry_count": 1,
    "is_valid": true
  }
}
```

## PostgresSaver Setup

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5432/agent_ops"
checkpointer = PostgresSaver.from_conn_string(DB_URI)
checkpointer.setup()  # Creates checkpoint tables on first run

graph = builder.compile(checkpointer=checkpointer)
```

> **Performance**: Writes take 5–15 ms p95 with `psycopg3` and pool size = 20.

## Failure Recovery Loop

```mermaid
graph TD
    A[Supervisor Node] -->|Delegate| B[Worker Tool Call]
    B -->|Schema Error| C{Retry Count < 3}
    C -->|Yes| B
    C -->|No| D[Human-in-the-Loop Fallback]
    D --> A
    B -->|Success| E[Supervisor Re-Evaluation]
    E -->|Complete| F[END]
```

## Thread Resumption

```python
# Resume an interrupted conversation
config = {"configurable": {"thread_id": "existing-thread-id"}}
state = graph.get_state(config)
graph.invoke(None, config=config)  # Continue from last checkpoint
```

## Related

- [[Short-Term State & Checkpointing (Postgres)]]
- [[Failure Recovery & Self-Healing Loops]]
- [[LLM-as-a-Judge Evaluation Framework]]
- [[00 - MOC (Agentic Systems Map)]]

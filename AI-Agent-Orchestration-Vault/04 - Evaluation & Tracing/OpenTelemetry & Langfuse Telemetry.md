---
id: AGENT-0402
title: OpenTelemetry & Langfuse Telemetry
category: Evaluation & Tracing
tags: [opentelemetry, langfuse, tracing, spans, observability, otlp, monitoring]
updated: 2026-08-16
status: verified
---

# OpenTelemetry & Langfuse Telemetry

Production agent systems require distributed tracing to debug latency, trace token costs, and correlate errors across multi-hop graph executions. This note covers the dual-layer stack: **OpenTelemetry** for infrastructure traces and **Langfuse** for LLM-specific spans.

## Langfuse LLM Tracing

Langfuse provides a purpose-built SDK that wraps LLM calls to capture: prompts, completions, token usage, latency, model name, and evaluation scores.

```python
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse(
    public_key="lf-pk-...",
    secret_key="lf-sk-...",
    host="https://cloud.langfuse.com",
)

@observe()  # Auto-captures input/output and latency
async def supervisor_with_tracing(state: AgentState) -> dict:
    langfuse_context.update_current_observation(
        name="supervisor_node",
        metadata={"thread_id": state["thread_id"], "retry": state["retry_count"]},
    )
    # ... node logic
    return result
```

## OpenTelemetry Spans

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup
provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317"))
)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("llm.orchestrator")

# Instrument a graph node
def supervisor_node(state: AgentState) -> dict:
    with tracer.start_as_current_span("supervisor_node") as span:
        span.set_attribute("thread_id", state["thread_id"])
        span.set_attribute("retry_count", state["retry_count"])
        span.set_attribute("model", settings.router_model)
        # ... node logic
        span.set_attribute("route_decision", route)
        return result
```

## Trace Hierarchy

```mermaid
graph TD
    A["Trace: graph_execution (thread_id)"] --> B["Span: supervisor_node"]
    A --> C["Span: worker_search"]
    A --> D["Span: supervisor_node (2nd)"]
    B --> E["Span: llm_call (gpt-4o-mini)"]
    C --> F["Span: tool_call (web_search)"]
    C --> G["Span: llm_call (gpt-4o)"]
    D --> H["Span: llm_call (gpt-4o-mini)"]
```

## Key Metrics to Capture

| Metric | Type | Instrument |
|---|---|---|
| `llm.token.input` | Counter | OTel + Langfuse |
| `llm.token.output` | Counter | OTel + Langfuse |
| `node.latency_ms` | Histogram | OTel |
| `graph.retry_count` | Counter | OTel |
| `evaluation.score` | Gauge | Langfuse |
| `vector.query_ms` | Histogram | OTel |

## Langfuse Score Logging

```python
# After LLM-as-a-Judge evaluation
langfuse.score(
    trace_id=trace_id,
    name="faithfulness",
    value=score.faithfulness,
    comment=score.rationale,
)
```

## Docker Compose (All-in-One Observability)

```yaml
services:
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    ports: ["4317:4317", "4318:4318"]

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports: ["16686:16686"]  # Jaeger UI

  langfuse:
    image: langfuse/langfuse:latest
    ports: ["3000:3000"]
    environment:
      DATABASE_URL: postgresql://postgres:postgres@postgres:5432/langfuse
```

## Related

- [[LLM-as-a-Judge Evaluation Framework]]
- [[LangGraph State Machine & Checkpointing]]
- [[Failure Recovery & Self-Healing Loops]]
- [[00 - MOC (Agentic Systems Map)]]

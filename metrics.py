"""
metrics.py – Prometheus Metrics Registry

Exposes /metrics endpoint with standard Prometheus text format.
Tracks: node call counts, latency histograms, token usage, error rates,
        active threads, rate limit hits, and evaluation scores.

Usage:
    from metrics import metrics
    metrics.inc_node_calls("supervisor")
    metrics.observe_latency("supervisor", 620.5)
    # Then mount: app.add_route("/metrics", metrics.handler)
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Minimal Prometheus text format writer (no prometheus_client dependency)
# ─────────────────────────────────────────────────────────────────────────────

class Counter:
    def __init__(self, name: str, help_text: str, labels: tuple = ()) -> None:
        self.name = name
        self.help = help_text
        self.labels = labels
        self._values: Dict[tuple, float] = defaultdict(float)

    def inc(self, label_values: tuple = (), amount: float = 1.0) -> None:
        self._values[label_values] += amount

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        for lv, val in self._values.items():
            label_str = self._fmt_labels(lv)
            lines.append(f"{self.name}{label_str} {val}")
        return "\n".join(lines)

    def _fmt_labels(self, lv: tuple) -> str:
        if not lv or not self.labels:
            return ""
        pairs = ",".join(f'{k}="{v}"' for k, v in zip(self.labels, lv))
        return "{" + pairs + "}"


class Gauge:
    def __init__(self, name: str, help_text: str, labels: tuple = ()) -> None:
        self.name = name
        self.help = help_text
        self.labels = labels
        self._values: Dict[tuple, float] = defaultdict(float)

    def set(self, value: float, label_values: tuple = ()) -> None:
        self._values[label_values] = value

    def inc(self, label_values: tuple = (), amount: float = 1.0) -> None:
        self._values[label_values] += amount

    def dec(self, label_values: tuple = (), amount: float = 1.0) -> None:
        self._values[label_values] -= amount

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        for lv, val in self._values.items():
            label_str = self._fmt_labels(lv)
            lines.append(f"{self.name}{label_str} {val}")
        return "\n".join(lines)

    def _fmt_labels(self, lv: tuple) -> str:
        if not lv or not self.labels:
            return ""
        pairs = ",".join(f'{k}="{v}"' for k, v in zip(self.labels, lv))
        return "{" + pairs + "}"


class Histogram:
    DEFAULT_BUCKETS = (10, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, float("inf"))

    def __init__(self, name: str, help_text: str, labels: tuple = (), buckets: tuple = DEFAULT_BUCKETS) -> None:
        self.name = name
        self.help = help_text
        self.labels = labels
        self.buckets = buckets
        self._counts: Dict[tuple, List[int]]  = defaultdict(lambda: [0] * len(buckets))
        self._sums:   Dict[tuple, float]       = defaultdict(float)
        self._totals: Dict[tuple, int]         = defaultdict(int)

    def observe(self, value: float, label_values: tuple = ()) -> None:
        self._sums[label_values]   += value
        self._totals[label_values] += 1
        counts = self._counts[label_values]
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                counts[i] += 1

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for lv in self._counts:
            label_str = self._fmt_labels(lv)
            running = 0
            for i, bound in enumerate(self.buckets):
                running += self._counts[lv][i]
                le = "+Inf" if bound == float("inf") else str(bound)
                lines.append(f'{self.name}_bucket{self._fmt_labels(lv, extra={"le": le})} {running}')
            lines.append(f'{self.name}_sum{label_str} {self._sums[lv]}')
            lines.append(f'{self.name}_count{label_str} {self._totals[lv]}')
        return "\n".join(lines)

    def _fmt_labels(self, lv: tuple, extra: Optional[dict] = None) -> str:
        pairs = []
        if lv and self.labels:
            pairs = [f'{k}="{v}"' for k, v in zip(self.labels, lv)]
        if extra:
            pairs += [f'{k}="{v}"' for k, v in extra.items()]
        return ("{" + ",".join(pairs) + "}") if pairs else ""


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator Metrics
# ─────────────────────────────────────────────────────────────────────────────

class OrchestratorMetrics:
    def __init__(self) -> None:
        self._start_time = time.time()

        # Counters
        self.node_calls_total     = Counter("orchestrator_node_calls_total",
                                            "Total node invocations", labels=("node", "status"))
        self.token_total          = Counter("orchestrator_tokens_total",
                                            "Total LLM tokens", labels=("model", "type"))
        self.cost_total           = Counter("orchestrator_cost_usd_total",
                                            "Total estimated cost in USD", labels=("model",))
        self.rate_limit_hits      = Counter("orchestrator_rate_limit_total",
                                            "Total rate limit rejections", labels=("client",))
        self.tool_calls_total     = Counter("orchestrator_tool_calls_total",
                                            "Total tool invocations", labels=("tool", "status"))
        self.eval_total           = Counter("orchestrator_evaluations_total",
                                            "Total evaluations run", labels=("backend",))
        self.ws_connections_total = Counter("orchestrator_ws_connections_total",
                                            "Total WebSocket connections opened")
        self.http_requests_total  = Counter("orchestrator_http_requests_total",
                                            "Total HTTP requests", labels=("method", "endpoint", "status"))

        # Gauges
        self.active_threads       = Gauge("orchestrator_active_threads",
                                         "Currently active graph threads")
        self.active_ws            = Gauge("orchestrator_active_websockets",
                                         "Currently open WebSocket connections")
        self.eval_score           = Gauge("orchestrator_eval_score_latest",
                                         "Latest evaluation overall score", labels=("thread",))

        # Histograms (latency in ms)
        self.node_latency_ms      = Histogram("orchestrator_node_latency_ms",
                                              "Node execution latency in milliseconds", labels=("node",))
        self.tool_latency_ms      = Histogram("orchestrator_tool_latency_ms",
                                              "Tool execution latency in milliseconds", labels=("tool",))
        self.eval_latency_ms      = Histogram("orchestrator_eval_latency_ms",
                                              "Evaluation latency in milliseconds")
        self.http_latency_ms      = Histogram("orchestrator_http_latency_ms",
                                              "HTTP request latency in milliseconds", labels=("endpoint",))

    # ── Convenience Methods ───────────────────────────────────────────────────

    def inc_node_calls(self, node: str, status: str = "success") -> None:
        self.node_calls_total.inc((node, status))

    def observe_node_latency(self, node: str, latency_ms: float) -> None:
        self.node_latency_ms.observe(latency_ms, (node,))

    def inc_tokens(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.token_total.inc((model, "input"),  input_tokens)
        self.token_total.inc((model, "output"), output_tokens)

    def inc_cost(self, model: str, cost_usd: float) -> None:
        self.cost_total.inc((model,), cost_usd)

    def inc_tool_call(self, tool: str, status: str, latency_ms: float) -> None:
        self.tool_calls_total.inc((tool, status))
        self.tool_latency_ms.observe(latency_ms, (tool,))

    def set_eval_score(self, thread_id: str, score: float, latency_ms: float, backend: str) -> None:
        self.eval_score.set(score, (thread_id[:16],))
        self.eval_latency_ms.observe(latency_ms)
        self.eval_total.inc((backend,))

    def render(self) -> str:
        """Render all metrics in Prometheus text exposition format."""
        uptime = Gauge("orchestrator_uptime_seconds", "Gateway uptime in seconds")
        uptime.set(round(time.time() - self._start_time, 2))

        sections = [
            self.node_calls_total.render(),
            self.token_total.render(),
            self.cost_total.render(),
            self.rate_limit_hits.render(),
            self.tool_calls_total.render(),
            self.eval_total.render(),
            self.ws_connections_total.render(),
            self.http_requests_total.render(),
            self.active_threads.render(),
            self.active_ws.render(),
            self.eval_score.render(),
            self.node_latency_ms.render(),
            self.tool_latency_ms.render(),
            self.eval_latency_ms.render(),
            self.http_latency_ms.render(),
            uptime.render(),
        ]
        return "\n\n".join(sections) + "\n"


# Singleton
metrics = OrchestratorMetrics()

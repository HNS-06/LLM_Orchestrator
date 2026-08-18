"""
quota_manager.py – Per-User Token & Cost Quota Management

Tracks cumulative token usage and estimated cost per client_id.
Enforces configurable daily/monthly limits.
Stores state in-memory (always) + optionally persists to Postgres.

Supported models and pricing (USD per 1M tokens, as of 2026):
  gpt-4o-mini:  input $0.15 / output $0.60
  gpt-4o:       input $5.00 / output $15.00
  claude-3-haiku: input $0.25 / output $1.25
  claude-3-5-sonnet: input $3.00 / output $15.00
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Optional

import structlog

log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Pricing Table  (USD per 1M tokens)
# ─────────────────────────────────────────────────────────────────────────────

PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini":          {"input": 0.15,  "output": 0.60},
    "gpt-4o":               {"input": 5.00,  "output": 15.00},
    "claude-3-haiku":       {"input": 0.25,  "output": 1.25},
    "claude-3-5-sonnet":    {"input": 3.00,  "output": 15.00},
    "claude-3-5-haiku":     {"input": 1.00,  "output": 5.00},
    "gemini-1.5-pro":       {"input": 3.50,  "output": 10.50},
    "gemini-1.5-flash":     {"input": 0.35,  "output": 1.05},
    "text-embedding-3-small":{"input": 0.02, "output": 0.0},
}

DEFAULT_PRICING = {"input": 5.00, "output": 15.00}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate USD cost for a model call."""
    pricing = PRICING.get(model, DEFAULT_PRICING)
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Usage Record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UsageRecord:
    client_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    request_count: int = 0
    error_count: int = 0
    last_seen: float = field(default_factory=time.time)
    daily_reset_date: str = field(default_factory=lambda: str(date.today()))

    # Limits (0 = unlimited)
    daily_token_limit: int = 500_000
    monthly_cost_limit_usd: float = 50.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def check_limits(self) -> tuple[bool, str]:
        """Returns (allowed, reason). allowed=False means quota exceeded."""
        today = str(date.today())
        if self.daily_reset_date != today:
            # New day – reset daily counters
            self.input_tokens = 0
            self.output_tokens = 0
            self.daily_reset_date = today

        if self.daily_token_limit > 0 and self.total_tokens >= self.daily_token_limit:
            return False, f"Daily token limit ({self.daily_token_limit:,}) exceeded"
        if self.monthly_cost_limit_usd > 0 and self.total_cost_usd >= self.monthly_cost_limit_usd:
            return False, f"Monthly cost limit (${self.monthly_cost_limit_usd:.2f}) exceeded"
        return True, "ok"

    def to_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "request_count": self.request_count,
            "error_count": self.error_count,
            "last_seen": self.last_seen,
            "daily_reset_date": self.daily_reset_date,
            "limits": {
                "daily_tokens": self.daily_token_limit,
                "monthly_cost_usd": self.monthly_cost_limit_usd,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Thread Cost Record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ThreadCost:
    thread_id: str
    client_id: str
    calls: list = field(default_factory=list)   # list of {model, input, output, cost}

    @property
    def total_input_tokens(self) -> int:
        return sum(c["input_tokens"] for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c["output_tokens"] for c in self.calls)

    @property
    def total_cost_usd(self) -> float:
        return sum(c["cost_usd"] for c in self.calls)

    def add_call(self, model: str, input_tokens: int, output_tokens: int) -> float:
        cost = estimate_cost(model, input_tokens, output_tokens)
        self.calls.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "ts": time.time(),
        })
        return cost

    def to_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "client_id": self.client_id,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "calls": self.calls,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Quota Manager
# ─────────────────────────────────────────────────────────────────────────────

class QuotaManager:
    def __init__(self) -> None:
        self._usage:   Dict[str, UsageRecord] = {}
        self._threads: Dict[str, ThreadCost]  = {}

    # ── Client records ────────────────────────────────────────────────────────

    def get_or_create(self, client_id: str) -> UsageRecord:
        if client_id not in self._usage:
            self._usage[client_id] = UsageRecord(client_id=client_id)
        return self._usage[client_id]

    def check_quota(self, client_id: str) -> tuple[bool, str]:
        record = self.get_or_create(client_id)
        return record.check_limits()

    def record_usage(
        self,
        client_id: str,
        thread_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Record token usage for a client+thread. Returns cost in USD."""
        record = self.get_or_create(client_id)
        record.input_tokens  += input_tokens
        record.output_tokens += output_tokens
        record.request_count += 1
        record.last_seen = time.time()

        # Thread-level tracking
        if thread_id not in self._threads:
            self._threads[thread_id] = ThreadCost(thread_id=thread_id, client_id=client_id)
        cost = self._threads[thread_id].add_call(model, input_tokens, output_tokens)
        record.total_cost_usd += cost

        log.info("quota_usage", client_id=client_id, thread_id=thread_id,
                 model=model, tokens=input_tokens + output_tokens, cost_usd=round(cost, 6))
        return cost

    def record_error(self, client_id: str) -> None:
        self.get_or_create(client_id).error_count += 1

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_client_usage(self, client_id: str) -> Optional[dict]:
        record = self._usage.get(client_id)
        return record.to_dict() if record else None

    def get_thread_cost(self, thread_id: str) -> Optional[dict]:
        tc = self._threads.get(thread_id)
        return tc.to_dict() if tc else None

    def get_all_usage(self) -> list[dict]:
        return [r.to_dict() for r in self._usage.values()]

    def global_stats(self) -> dict:
        all_records = list(self._usage.values())
        return {
            "total_clients": len(all_records),
            "total_threads": len(self._threads),
            "global_input_tokens":  sum(r.input_tokens  for r in all_records),
            "global_output_tokens": sum(r.output_tokens for r in all_records),
            "global_cost_usd":      round(sum(r.total_cost_usd for r in all_records), 4),
            "global_requests":      sum(r.request_count for r in all_records),
            "global_errors":        sum(r.error_count   for r in all_records),
        }


# Singleton
quota_manager = QuotaManager()

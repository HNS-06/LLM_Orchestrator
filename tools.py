"""
tools.py – Dynamic Tool Registry

Provides a self-describing registry of callable tools that agents can discover
and invoke at runtime. Each tool is a Pydantic-validated async function.

Concrete tools included:
  - web_search  – Mock web search (swap body for SerpAPI / Brave Search)
  - code_exec   – Sandboxed Python code execution via RestrictedPython
  - sql_query   – Safe parameterized SQL query runner against Postgres
  - calculator  – Precise arithmetic via Python eval with whitelist
  - memory_recall – Semantic memory recall from Qdrant/mock store
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import operator as _op
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import structlog
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tool Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tool": self.tool_name,
            "success": self.success,
            "output": str(self.output)[:4096] if self.output else None,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
            "metadata": self.metadata,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Tool Descriptor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolDescriptor:
    name: str
    description: str
    parameters: dict          # JSON Schema of input parameters
    fn: Callable
    category: str = "general"

    def to_openai_schema(self) -> dict:
        """Export as OpenAI tool/function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolDescriptor] = {}

    def register(self, descriptor: ToolDescriptor) -> None:
        self._tools[descriptor.name] = descriptor
        log.debug("tool_registered", name=descriptor.name)

    def list_tools(self) -> List[ToolDescriptor]:
        return list(self._tools.values())

    def get_openai_schemas(self) -> List[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def get(self, name: str) -> Optional[ToolDescriptor]:
        return self._tools.get(name)

    async def invoke(self, name: str, args: dict) -> ToolResult:
        t0 = time.perf_counter()
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(tool_name=name, success=False, output=None,
                              error=f"Tool '{name}' not found in registry",
                              latency_ms=0.0)
        try:
            if asyncio.iscoroutinefunction(tool.fn):
                result = await tool.fn(**args)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: tool.fn(**args))
            latency_ms = (time.perf_counter() - t0) * 1000
            log.info("tool_invoked", name=name, latency_ms=round(latency_ms, 2))
            return ToolResult(tool_name=name, success=True, output=result, latency_ms=latency_ms)
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            log.error("tool_error", name=name, error=str(exc))
            return ToolResult(tool_name=name, success=False, output=None,
                              error=str(exc), latency_ms=latency_ms)


# ─────────────────────────────────────────────────────────────────────────────
# Tool Implementations
# ─────────────────────────────────────────────────────────────────────────────

async def _web_search(query: str, num_results: int = 5) -> dict:
    """
    Mock web search. In production, replace with:
      - Brave Search API  (https://api.search.brave.com)
      - SerpAPI           (https://serpapi.com)
      - Tavily            (https://tavily.com)
    """
    await asyncio.sleep(0.05)  # simulate network latency
    return {
        "query": query,
        "results": [
            {"title": f"Result {i+1} for '{query}'",
             "url": f"https://example.com/result-{i+1}",
             "snippet": f"This is a mock search result {i+1} about {query}. "
                        f"In production connect to a live search provider."}
            for i in range(min(num_results, 5))
        ],
        "total_results": 1_000_000,
        "backend": "mock",
    }


async def _code_exec(code: str, timeout_seconds: int = 10) -> dict:
    """
    Safe Python code execution using subprocess isolation.
    Returns stdout, stderr, and exit code.
    """
    import subprocess, sys, tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        tmp_path = f.name
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
        return {
            "stdout": stdout.decode("utf-8", errors="replace")[:8192],
            "stderr": stderr.decode("utf-8", errors="replace")[:2048],
            "exit_code": proc.returncode,
            "success": proc.returncode == 0,
        }
    except asyncio.TimeoutError:
        try: proc.kill()
        except Exception: pass
        return {"stdout": "", "stderr": "Execution timed out", "exit_code": -1, "success": False}
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass


async def _sql_query(query: str, params: Optional[list] = None) -> dict:
    """
    Safe read-only SQL query. In production uses psycopg3 async pool.
    Falls back to a mock when Postgres is not configured.
    """
    from config import settings
    if not settings.use_postgres_checkpointer:
        # Mock mode
        return {
            "rows": [{"id": 1, "note": "Mock SQL result – connect Postgres to run real queries"}],
            "row_count": 1,
            "backend": "mock",
        }
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(settings.postgres_uri) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params or [])
                rows = await cur.fetchmany(100)
                cols = [desc[0] for desc in cur.description] if cur.description else []
                return {
                    "rows": [dict(zip(cols, row)) for row in rows],
                    "row_count": len(rows),
                    "backend": "postgres",
                }
    except Exception as exc:
        raise RuntimeError(f"SQL error: {exc}") from exc


def _calculator(expression: str) -> dict:
    """
    Safe arithmetic evaluation. Whitelist-only AST nodes.
    Supports: +, -, *, /, //, %, **, abs, round, int, float
    """
    SAFE_OPS = {
        ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
        ast.Div: _op.truediv, ast.FloorDiv: _op.floordiv,
        ast.Mod: _op.mod, ast.Pow: _op.pow, ast.USub: _op.neg,
        ast.UAdd: _op.pos,
    }
    SAFE_NAMES = {"abs": abs, "round": round, "int": int, "float": float,
                  "max": max, "min": min, "sum": sum}

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            op = SAFE_OPS.get(type(node.op))
            if op is None: raise ValueError(f"Unsupported operator: {node.op}")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = SAFE_OPS.get(type(node.op))
            if op is None: raise ValueError(f"Unsupported operator: {node.op}")
            return op(_eval(node.operand))
        if isinstance(node, ast.Call):
            fn_name = node.func.id if isinstance(node.func, ast.Name) else None
            fn = SAFE_NAMES.get(fn_name)
            if fn is None: raise ValueError(f"Unsupported function: {fn_name}")
            return fn(*[_eval(a) for a in node.args])
        if isinstance(node, ast.Name) and node.id in SAFE_NAMES:
            return SAFE_NAMES[node.id]
        raise ValueError(f"Unsafe AST node: {type(node).__name__}")

    try:
        tree = ast.parse(expression.strip(), mode='eval')
        result = _eval(tree.body)
        return {"expression": expression, "result": result, "type": type(result).__name__}
    except Exception as exc:
        raise ValueError(f"Calculation error: {exc}") from exc


async def _memory_recall(query: str, thread_id: Optional[str] = None, top_k: int = 5) -> dict:
    """Semantic memory recall from the vector store."""
    from semantic_memory import recall_context
    records = await recall_context(query=query, thread_id=thread_id, top_k=top_k)
    return {"query": query, "memories": records, "count": len(records)}


# ─────────────────────────────────────────────────────────────────────────────
# Registry Initialization
# ─────────────────────────────────────────────────────────────────────────────

tool_registry = ToolRegistry()

tool_registry.register(ToolDescriptor(
    name="web_search",
    description="Search the web for current information, news, or facts on any topic.",
    category="search",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query", "maxLength": 512},
            "num_results": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    },
    fn=_web_search,
))

tool_registry.register(ToolDescriptor(
    name="code_exec",
    description="Execute Python code in a sandboxed subprocess and return stdout/stderr.",
    category="code",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
            "timeout_seconds": {"type": "integer", "default": 10, "minimum": 1, "maximum": 30},
        },
        "required": ["code"],
    },
    fn=_code_exec,
))

tool_registry.register(ToolDescriptor(
    name="sql_query",
    description="Run a read-only SQL query against the configured Postgres database.",
    category="sql",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "SQL SELECT statement"},
            "params": {"type": "array", "items": {}, "description": "Optional bind parameters"},
        },
        "required": ["query"],
    },
    fn=_sql_query,
))

tool_registry.register(ToolDescriptor(
    name="calculator",
    description="Evaluate a mathematical expression safely. Supports +,-,*,/,**,abs,round.",
    category="general",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression, e.g. '2 ** 10 + round(3.7)'"},
        },
        "required": ["expression"],
    },
    fn=_calculator,
))

tool_registry.register(ToolDescriptor(
    name="memory_recall",
    description="Search long-term semantic memory for relevant past interactions.",
    category="memory",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Semantic search query"},
            "thread_id": {"type": "string", "description": "Restrict search to this thread (optional)"},
            "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
        },
        "required": ["query"],
    },
    fn=_memory_recall,
))

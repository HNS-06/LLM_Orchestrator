---
id: AGENT-0301
title: Structured Tool Calling & Pydantic Schemas
category: Tooling & Guardrails
tags: [tool-calling, pydantic, json-mode, schema-validation, openai-tools, langchain]
updated: 2026-08-16
status: verified
---

# Structured Tool Calling & Pydantic Schemas

Reliable agent behavior requires **structured, schema-validated tool interfaces**. Pydantic v2 provides the validation layer; OpenAI's `tool_choice` and LangChain's `@tool` decorator provide the invocation layer.

## Tool Schema Pattern

```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class WebSearchInput(BaseModel):
    query: str = Field(..., description="The search query string", min_length=1, max_length=512)
    num_results: int = Field(default=5, ge=1, le=20, description="Number of results to return")

@tool("web_search", args_schema=WebSearchInput, return_direct=False)
def web_search(query: str, num_results: int = 5) -> str:
    """Search the web for current information on any topic."""
    # ... implementation
    return f"Results for: {query}"
```

## OpenAI Parallel Tool Calls

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(
    tools=[web_search, code_interpreter, sql_executor],
    tool_choice="auto",           # model decides when to call tools
    parallel_tool_calls=True,     # enables concurrent tool execution
)

response = llm_with_tools.invoke("Search for LangGraph benchmarks and summarize the code example")
# Response may contain multiple tool_calls in parallel
```

## Schema Validation with Error Recovery

```python
from pydantic import ValidationError

def safe_tool_invoke(tool_fn, raw_args: dict):
    try:
        validated = tool_fn.args_schema(**raw_args)
        return tool_fn.invoke(validated.model_dump())
    except ValidationError as e:
        # Return structured error for agent self-correction
        return {
            "error": "schema_validation_failed",
            "details": e.errors(),
            "hint": "Correct the input and retry",
        }
```

## JSON Mode Enforcement

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

class RouterDecision(BaseModel):
    route: str
    confidence: float
    rationale: str

parser = JsonOutputParser(pydantic_object=RouterDecision)
chain = router_prompt | llm | parser
decision = chain.invoke({"context": "..."})
# Raises OutputParserException if malformed → triggers retry
```

## Tool Registry Pattern

```python
TOOL_REGISTRY = {
    "web_search": web_search,
    "code_interpreter": code_interpreter,
    "sql_executor": sql_executor,
}

def dispatch_tool(tool_name: str, args: dict) -> str:
    tool = TOOL_REGISTRY.get(tool_name)
    if not tool:
        return f"Unknown tool: {tool_name}"
    return safe_tool_invoke(tool, args)
```

## Related

- [[Failure Recovery & Self-Healing Loops]]
- [[LangGraph State Machine & Checkpointing]]
- [[LLM-as-a-Judge Evaluation Framework]]
- [[00 - MOC (Agentic Systems Map)]]

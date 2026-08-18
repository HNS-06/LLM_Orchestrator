"""
evaluator.py – LLM-as-a-Judge Auto-Evaluation Framework

Automatically scores agent responses after each graph completion.
Publishes scores to the event bus for real-time dashboard display.
Supports: rubric scoring, pairwise preference, per-dimension radar data.
"""

from __future__ import annotations

import time

import structlog
from pydantic import BaseModel, Field

from config import settings

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Score Models
# ─────────────────────────────────────────────────────────────────────────────

class RubricScore(BaseModel):
    """Multi-dimensional quality rubric (0.0 – 1.0 per dimension)."""
    faithfulness:   float = Field(..., ge=0.0, le=1.0, description="Answer grounded in retrieved context")
    relevance:      float = Field(..., ge=0.0, le=1.0, description="Directly addresses the user question")
    completeness:   float = Field(..., ge=0.0, le=1.0, description="No key information missing")
    clarity:        float = Field(..., ge=0.0, le=1.0, description="Well-structured and readable")
    safety:         float = Field(..., ge=0.0, le=1.0, description="No harmful or misleading content")
    overall:        float = Field(..., ge=0.0, le=1.0, description="Holistic quality score")
    rationale:      str   = Field(..., description="Brief explanation of the scores")

    @property
    def weighted_score(self) -> float:
        """Weighted composite: faithfulness×0.3, relevance×0.25, completeness×0.2, clarity×0.15, safety×0.1"""
        return (
            self.faithfulness * 0.30 +
            self.relevance    * 0.25 +
            self.completeness * 0.20 +
            self.clarity      * 0.15 +
            self.safety       * 0.10
        )

    def to_radar_data(self) -> list[dict]:
        """Format for dashboard radar/spider chart."""
        return [
            {"dimension": "Faithfulness",  "score": self.faithfulness,  "weight": 0.30},
            {"dimension": "Relevance",     "score": self.relevance,     "weight": 0.25},
            {"dimension": "Completeness",  "score": self.completeness,  "weight": 0.20},
            {"dimension": "Clarity",       "score": self.clarity,       "weight": 0.15},
            {"dimension": "Safety",        "score": self.safety,        "weight": 0.10},
        ]

    def flag_for_review(self) -> bool:
        return (
            self.safety       < 0.95 or
            self.faithfulness < 0.70 or
            self.overall      < 0.65
        )


class PairwiseResult(BaseModel):
    winner: str = Field(..., description="'A' | 'B' | 'tie'")
    rationale: str


class EvaluationResult(BaseModel):
    thread_id: str
    question: str
    response: str
    score: RubricScore
    flagged: bool
    evaluated_at: float = Field(default_factory=time.time)
    latency_ms: float = 0.0
    backend: str = "mock"   # "llm" | "mock"


# ─────────────────────────────────────────────────────────────────────────────
# Judge Prompts
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are an expert AI response evaluator. Score the RESPONSE strictly
on each dimension from 0.0 to 1.0 using the rubric below. Return ONLY a valid JSON object.

Rubric:
  faithfulness  (0–1): Is the answer grounded in factual context? Penalize hallucinations.
  relevance     (0–1): Does it directly address the question asked?
  completeness  (0–1): Are all key points covered? Penalize shallow answers.
  clarity       (0–1): Is the writing clear, structured, and professional?
  safety        (0–1): Is the content safe and appropriate? 1.0 = fully safe.
  overall       (0–1): Your holistic assessment.
  rationale     (str): One sentence explaining the scores.

JSON schema (respond EXACTLY in this format, no markdown):
{"faithfulness":0.0,"relevance":0.0,"completeness":0.0,"clarity":0.0,"safety":0.0,"overall":0.0,"rationale":""}"""

JUDGE_USER_TEMPLATE = """QUESTION: {question}

CONTEXT: {context}

RESPONSE: {response}"""


# ─────────────────────────────────────────────────────────────────────────────
# Mock Evaluator (no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

def _mock_evaluate(question: str, response: str) -> RubricScore:
    """Heuristic mock evaluator for offline/testing mode."""
    import random
    import re
    word_count = len(response.split())
    has_code   = bool(re.search(r'```|def |class |import ', response))
    has_refs   = bool(re.search(r'https?://|source:|according to', response, re.I))

    faithfulness  = round(min(1.0, 0.6 + has_refs * 0.25 + random.uniform(-0.05, 0.1)), 2)
    relevance     = round(min(1.0, 0.7 + random.uniform(-0.1, 0.2)), 2)
    completeness  = round(min(1.0, 0.5 + min(word_count / 400, 0.4) + random.uniform(-0.05, 0.1)), 2)
    clarity       = round(min(1.0, 0.65 + has_code * 0.15 + random.uniform(-0.05, 0.15)), 2)
    safety        = round(min(1.0, 0.95 + random.uniform(-0.02, 0.05)), 2)
    overall       = round((faithfulness * 0.3 + relevance * 0.25 + completeness * 0.2 + clarity * 0.15 + safety * 0.1), 2)

    return RubricScore(
        faithfulness=faithfulness, relevance=relevance, completeness=completeness,
        clarity=clarity, safety=safety, overall=overall,
        rationale=f"Heuristic mock evaluation. Word count: {word_count}. Weighted: {overall:.2f}.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM Evaluator
# ─────────────────────────────────────────────────────────────────────────────

async def _llm_evaluate(question: str, response: str, context: str = "") -> RubricScore:
    import json
    import re as _re
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=settings.openai_api_key)
    user_msg = JUDGE_USER_TEMPLATE.format(question=question, context=context or "No additional context.", response=response)

    resp = await judge.ainvoke([SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=user_msg)])
    raw = resp.content.strip()

    match = _re.search(r'\{.*\}', raw, _re.DOTALL)
    if not match:
        raise ValueError(f"Judge returned non-JSON: {raw[:200]}")

    data = json.loads(match.group())
    return RubricScore(**data)


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise
# ─────────────────────────────────────────────────────────────────────────────

async def pairwise_preference(question: str, response_a: str, response_b: str) -> PairwiseResult:
    """Compare two responses and return the preferred one."""
    import json
    import re
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    if settings.openai_api_key == "sk-placeholder":
        import random
        return PairwiseResult(winner=random.choice(["A", "B", "tie"]), rationale="Mock pairwise evaluation.")

    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=settings.openai_api_key)
    prompt = (
        f"Given this QUESTION: {question}\n\n"
        f"RESPONSE A:\n{response_a}\n\n"
        f"RESPONSE B:\n{response_b}\n\n"
        f'Which is better? Reply ONLY with JSON: {{"winner":"A"|"B"|"tie","rationale":"..."}}'
    )
    resp = await judge.ainvoke([HumanMessage(content=prompt)])
    match = re.search(r'\{.*\}', resp.content, re.DOTALL)
    data = json.loads(match.group()) if match else {"winner": "tie", "rationale": "Parse error"}
    return PairwiseResult(**data)


# ─────────────────────────────────────────────────────────────────────────────
# Public Evaluator
# ─────────────────────────────────────────────────────────────────────────────

async def evaluate_response(
    thread_id: str,
    question: str,
    response: str,
    context: str = "",
) -> EvaluationResult:
    """
    Auto-evaluate an agent response. Uses LLM judge when OpenAI key is set,
    falls back to heuristic mock otherwise.
    """
    t0 = time.perf_counter()
    backend = "mock"

    try:
        if settings.openai_api_key and settings.openai_api_key != "sk-placeholder":
            score = await _llm_evaluate(question, response, context)
            backend = "llm"
        else:
            score = _mock_evaluate(question, response)
    except Exception as exc:
        log.warning("evaluator_fallback", error=str(exc))
        score = _mock_evaluate(question, response)

    latency_ms = (time.perf_counter() - t0) * 1000
    result = EvaluationResult(
        thread_id=thread_id,
        question=question,
        response=response,
        score=score,
        flagged=score.flag_for_review(),
        latency_ms=round(latency_ms, 2),
        backend=backend,
    )

    # Publish to event bus for dashboard
    from langgraph_engine import event_bus
    await event_bus.publish(thread_id, {
        "type": "evaluation_result",
        "thread_id": thread_id,
        "score": score.model_dump(),
        "radar": score.to_radar_data(),
        "weighted": round(score.weighted_score, 3),
        "flagged": result.flagged,
        "backend": backend,
        "latency_ms": round(latency_ms, 2),
        "ts": time.time(),
    })

    log.info("evaluation_complete", thread_id=thread_id, overall=score.overall, flagged=result.flagged)
    return result

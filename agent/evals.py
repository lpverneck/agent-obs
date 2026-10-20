import os
from typing import Literal
from opentelemetry import metrics
from opentelemetry._logs import get_logger
from pydantic import BaseModel, Field
from llm import OpenAIModelProvider


EVAL_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL") or os.getenv("LLM_MODEL")

# evaluation results are emitted as `gen_ai.evaluation.result` events, following the
# GenAI semantic conventions; they inherit the trace and span ids of the current context
event_logger = get_logger(__name__)

meter = metrics.get_meter(__name__)
score_histogram = meter.create_histogram(
    name="eval.score",
    unit="1",
    description="Scores given by the agent evaluators, between 0 and 1",
    explicit_bucket_boundaries_advisory=[0.0, 0.25, 0.5, 0.75, 1.0],
)


# pydantic data model for a single evaluation result
class Score(BaseModel):
    name: str
    value: float = Field(ge=0.0, le=1.0)
    label: str | None = None
    explanation: str | None = None


# pydantic data model for the LLM judge output
class Verdict(BaseModel):
    """Verdict of the evaluation. Reason first, then pick the label."""

    explanation: str = Field(description="Step by step reasoning behind the label.")
    label: Literal["pass", "partial", "fail"] = Field(description="Final verdict.")


VERDICT_VALUES = {"pass": 1.0, "partial": 0.5, "fail": 0.0}

judge_model = (
    OpenAIModelProvider()
    .create_model(model=EVAL_JUDGE_MODEL, temperature=0.0)
    .with_structured_output(Verdict)
)


def answer_relevance(query: str, answer: str) -> Score:
    """LLM judge: does the answer address the question, regardless of correctness?"""
    verdict = judge_model.invoke(
        "You are grading whether an assistant answer addresses the user question. "
        "Do not judge factual correctness.\n"
        "pass: the answer directly addresses the question.\n"
        "partial: the answer is related but evasive, incomplete or padded with unrelated content.\n"
        "fail: the answer does not address the question.\n\n"
        f"Question: {query}\n"
        f"Assistant answer: {answer}"
    )
    return Score(
        name="answer_relevance",
        value=VERDICT_VALUES[verdict.label],
        label=verdict.label,
        explanation=verdict.explanation,
    )


def evaluate(
    query: str, answer: str, mode: Literal["offline", "online"], **attributes: str
) -> Score:
    """Score the answer relevance and emit the result in the current trace context."""
    score = answer_relevance(query, answer)
    event_attributes = {
        "gen_ai.evaluation.name": score.name,
        "gen_ai.evaluation.score.value": score.value,
        "eval.mode": mode,
        **attributes,
    }
    if score.label:
        event_attributes["gen_ai.evaluation.score.label"] = score.label
    if score.explanation:
        event_attributes["gen_ai.evaluation.explanation"] = score.explanation
    event_logger.emit(
        event_name="gen_ai.evaluation.result",
        body=f"{score.name}={score.value:.2f}",
        attributes=event_attributes,
    )
    score_histogram.record(
        score.value, {"gen_ai.evaluation.name": score.name, "eval.mode": mode}
    )
    return score

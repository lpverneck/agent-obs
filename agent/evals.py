import os
from collections.abc import Callable
from typing import Literal
from langchain_core.messages import BaseMessage, ToolMessage
from opentelemetry import metrics
from opentelemetry._logs import get_logger
from pydantic import BaseModel, Field
from llm import OpenAIModelProvider
from tools import search_knowledge_base


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


# pydantic data model for what is being evaluated: the agent run plus whatever
# reference (offline) or retrieved context (online) is available for it
class EvalSample(BaseModel):
    query: str
    answer: str
    sources: list[str] = Field(default_factory=list)
    context: str | None = None
    expected_answer: str | None = None
    expected_sources: list[str] | None = None


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


def _judge(name: str, prompt: str) -> Score:
    verdict = judge_model.invoke(prompt)
    return Score(
        name=name,
        value=VERDICT_VALUES[verdict.label],
        label=verdict.label,
        explanation=verdict.explanation,
    )


# scorers: each returns None when the sample lacks what it needs to be scored,
# so the same list serves offline experiments (with references) and online traffic
def tool_selection(sample: EvalSample) -> Score | None:
    """Did the agent use the knowledge base exactly when it was expected to?"""
    if sample.expected_sources is None:
        return None
    used_kb, should_use_kb = bool(sample.sources), bool(sample.expected_sources)
    correct = used_kb == should_use_kb
    return Score(
        name="tool_selection",
        value=float(correct),
        label="correct" if correct else "incorrect",
        explanation=f"used knowledge base: {used_kb}, expected: {should_use_kb}",
    )


def source_recall(sample: EvalSample) -> Score | None:
    """Share of the expected knowledge base documents cited by the agent."""
    if not sample.expected_sources:
        return None
    expected = set(sample.expected_sources)
    found = expected & set(sample.sources)
    return Score(
        name="source_recall",
        value=len(found) / len(expected),
        explanation=f"cited {sorted(found)} of expected {sorted(expected)}",
    )


def correctness(sample: EvalSample) -> Score | None:
    """LLM judge: does the answer agree with the reference answer?"""
    if sample.expected_answer is None:
        return None
    return _judge(
        "correctness",
        "You are grading an assistant answer against a reference answer.\n"
        "pass: the answer conveys the same facts as the reference; extra correct detail is fine.\n"
        "partial: the answer is incomplete or partly contradicts the reference.\n"
        "fail: the answer is wrong, contradicts the reference or does not answer.\n\n"
        f"Question: {sample.query}\n"
        f"Reference answer: {sample.expected_answer}\n"
        f"Assistant answer: {sample.answer}",
    )


def answer_relevance(sample: EvalSample) -> Score | None:
    """LLM judge: does the answer address the question, regardless of correctness?"""
    return _judge(
        "answer_relevance",
        "You are grading whether an assistant answer addresses the user question. "
        "Do not judge factual correctness.\n"
        "pass: the answer directly addresses the question.\n"
        "partial: the answer is related but evasive, incomplete or padded with unrelated content.\n"
        "fail: the answer does not address the question.\n\n"
        f"Question: {sample.query}\n"
        f"Assistant answer: {sample.answer}",
    )


def groundedness(sample: EvalSample) -> Score | None:
    """LLM judge: is the answer supported by the documents the agent retrieved?"""
    if sample.context is None:
        return None
    return _judge(
        "groundedness",
        "You are grading whether an assistant answer is supported by the documents it retrieved.\n"
        "pass: every claim in the answer is supported by the documents.\n"
        "partial: some claims are supported, others are not found in the documents.\n"
        "fail: the answer is not supported by, or contradicts, the documents.\n\n"
        f"Question: {sample.query}\n"
        f"Retrieved documents:\n{sample.context}\n\n"
        f"Assistant answer: {sample.answer}",
    )


SCORERS: list[Callable[[EvalSample], Score | None]] = [
    tool_selection,
    source_recall,
    correctness,
    answer_relevance,
    groundedness,
]


def retrieved_context(messages: list[BaseMessage]) -> str | None:
    """Knowledge base results seen by the agent, or None when it did not search."""
    results = [
        message.content
        for message in messages
        if isinstance(message, ToolMessage)
        and message.name == search_knowledge_base.name
    ]
    return "\n".join(results) if results else None


def evaluate(
    sample: EvalSample, mode: Literal["offline", "online"], **attributes: str
) -> list[Score]:
    """Run the applicable scorers and emit each result in the current trace context."""
    scores = []
    for scorer in SCORERS:
        score = scorer(sample)
        if score is None:
            continue
        scores.append(score)
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
    return scores

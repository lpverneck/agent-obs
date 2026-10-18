import logging
import os
import random
from fastapi import BackgroundTasks, FastAPI
from langchain_core.messages import HumanMessage
from opentelemetry import baggage, context, trace
from opentelemetry.processor.baggage import BaggageSpanProcessor
from opentelemetry.sdk.trace import TracerProvider
from pydantic import BaseModel, Field
from agent import Answer, kb_agent
from evals import EvalSample, evaluate, retrieved_context


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# request identifiers propagated as baggage across the trace
BAGGAGE_KEYS = ("session.id", "user.id")

# share of requests scored by the online evaluation, between 0 and 1
EVAL_ONLINE_SAMPLE_RATE = float(os.getenv("EVAL_ONLINE_SAMPLE_RATE") or 0)
# offline experiments are already scored by the runner, so they are not sampled
EVAL_RUNNER_USER_ID = "eval-runner"

# copy the baggage identifiers onto every span this service starts
tracer_provider = trace.get_tracer_provider()
if isinstance(
    tracer_provider, TracerProvider
):  # only set under opentelemetry-instrument
    tracer_provider.add_span_processor(
        BaggageSpanProcessor(lambda key: key in BAGGAGE_KEYS)
    )

tracer = trace.get_tracer(__name__)


# pydantic data model for the agent request
class InvokeRequest(BaseModel):
    query: str = Field(min_length=1, description="Question to ask the agent.")
    session_id: str | None = Field(
        default=None,
        min_length=1,
        description="Session identifier, propagated as baggage across the trace.",
    )
    user_id: str | None = Field(
        default=None,
        min_length=1,
        description="User identifier, propagated as baggage across the trace.",
    )


app = FastAPI()


def evaluate_online(query: str, result: dict, ctx: context.Context) -> None:
    """Score a served answer with the reference-free evaluators, in the request trace."""
    # runs after the response is sent; ctx carries the request span and baggage
    with tracer.start_as_current_span("evaluate kb_agent", context=ctx):
        sample = EvalSample(
            query=query,
            answer=result["answer"].answer,
            sources=result["answer"].sources,
            context=retrieved_context(result["messages"]),
        )
        scores = evaluate(sample, mode="online")
        logger.info(
            "online evaluation: %s",
            ", ".join(f"{score.name}={score.value:.2f}" for score in scores),
        )


@app.post("/invoke")
def invoke_agent(request: InvokeRequest, background_tasks: BackgroundTasks) -> Answer:
    identifiers = {"session.id": request.session_id, "user.id": request.user_id}
    identifiers = {key: value for key, value in identifiers.items() if value}
    ctx = context.get_current()
    for key, value in identifiers.items():
        ctx = baggage.set_baggage(key, value, ctx)
    token = context.attach(ctx)
    # the server span started before the baggage was set, so tag it explicitly
    trace.get_current_span().set_attributes(identifiers)
    try:
        logger.info("agent invoked: query=%s", request.query)
        # session_id in the run metadata becomes gen_ai.conversation.id on the GenAI spans
        config = (
            {"metadata": {"session_id": request.session_id}}
            if request.session_id
            else None
        )
        result = kb_agent.invoke(
            {"messages": [HumanMessage(content=request.query)]}, config=config
        )
        logger.info("agent answered: sources=%s", result["answer"].sources)
        if (
            request.user_id != EVAL_RUNNER_USER_ID
            and random.random() < EVAL_ONLINE_SAMPLE_RATE
        ):
            background_tasks.add_task(
                evaluate_online, request.query, result, context.get_current()
            )
        return result["answer"]
    finally:
        context.detach(token)

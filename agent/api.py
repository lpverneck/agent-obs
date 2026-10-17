import logging
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from opentelemetry import baggage, context, trace
from opentelemetry.processor.baggage import BaggageSpanProcessor
from opentelemetry.sdk.trace import TracerProvider
from pydantic import BaseModel, Field
from agent import Answer, kb_agent


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# request identifiers propagated as baggage across the trace
BAGGAGE_KEYS = ("session.id", "user.id")

# copy the baggage identifiers onto every span this service starts
tracer_provider = trace.get_tracer_provider()
if isinstance(
    tracer_provider, TracerProvider
):  # only set under opentelemetry-instrument
    tracer_provider.add_span_processor(
        BaggageSpanProcessor(lambda key: key in BAGGAGE_KEYS)
    )


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


@app.post("/invoke")
def invoke_agent(request: InvokeRequest) -> Answer:
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
        return result["answer"]
    finally:
        context.detach(token)

import logging
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from agent import Answer, kb_agent


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# pydantic data model for the agent request
class InvokeRequest(BaseModel):
    query: str = Field(min_length=1, description="Question to ask the agent.")


app = FastAPI()


@app.post("/invoke")
def invoke_agent(request: InvokeRequest) -> Answer:
    logger.info("agent invoked: query=%s", request.query)
    result = kb_agent.invoke({"messages": [HumanMessage(content=request.query)]})
    logger.info("agent answered: sources=%s", result["answer"].sources)
    return result["answer"]

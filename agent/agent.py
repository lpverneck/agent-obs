import logging
from typing import Literal
from IPython.display import Image, display
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import PydanticToolsParser
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from opentelemetry import baggage
from pydantic import BaseModel, Field
from llm import OpenAIModelProvider
from tools import toolbox


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# pydantic data model for the final answer to the user
class Answer(BaseModel):
    """Final answer to the employee. Call this once you are ready to respond."""

    answer: str = Field(description="Answer to the employee's question.")
    sources: list[str] = Field(
        default_factory=list,
        description="IDs of the knowledge base articles used in the answer. Empty when the knowledge base was not used.",
    )


# agent state definition
class AgentState(MessagesState):
    answer: Answer | None


# output parser for the agent answer
answer_parser = PydanticToolsParser(tools=[Answer], first_tool_only=True)

# create the agent model with tools
model = OpenAIModelProvider().create_model()
model_with_tools = model.bind_tools(
    [*toolbox, Answer], tool_choice="any", parallel_tool_calls=False
)  # tool_choice="any" forces a tool call every turn, either search the KB or deliver the Answer

# system prompt for the agent: IT support assistant for the company employees
assistant_system_message = SystemMessage(
    content=(
        "You are the company's IT Helpdesk assistant, helping employees with their devices, "
        "accounts, software and network access.\n"
        "- For IT questions, use `search_knowledge_base` and base your answer on the articles found; "
        "skip it for greetings or off-topic requests.\n"
        "- Reply in the same language as the employee's message, with clear numbered steps.\n"
        "- Suggest contacting the IT Helpdesk when the issue is not covered or persists.\n"
        "- Always finish by calling `Answer`, listing the IDs of the articles you relied on."
    )
)


# nodes defining the agent's behavior
def assistant(state: AgentState):
    return {
        "messages": [
            model_with_tools.invoke([assistant_system_message, *state["messages"]])
        ]
    }


def respond(state: AgentState):
    last_message = state["messages"][-1]
    # reply to the Answer tool call so the history stays valid for follow-up turns
    tool_message = ToolMessage(
        content="Answer delivered.", tool_call_id=last_message.tool_calls[0]["id"]
    )
    # end of the agent flow: confirm the request identifiers reached it via baggage
    logger.info(
        "agent flow finished: session.id=%s user.id=%s",
        baggage.get_baggage("session.id"),
        baggage.get_baggage("user.id"),
    )
    return {"answer": answer_parser.invoke(last_message), "messages": [tool_message]}


# edges
def route(state: AgentState) -> Literal["tools", "respond"]:
    tool_call = state["messages"][-1].tool_calls[0]
    return "respond" if tool_call["name"] == Answer.__name__ else "tools"


# build the agent state graph
builder = StateGraph(AgentState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(toolbox))
builder.add_node("respond", respond)
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", route)
builder.add_edge("tools", "assistant")
builder.add_edge("respond", END)
kb_agent = builder.compile()


# run the agent in a loop
if __name__ == "__main__":
    query = input("Ask the agent: ")
    result = kb_agent.invoke({"messages": [HumanMessage(content=query)]})

    for message in result["messages"]:
        message.pretty_print()

    print(result["answer"])

"""LangGraph workflow composing job management and documentation search.

Demonstrates a supervisor agent that delegates to specialized sub-agents:
- Job Agent: manages DIRAC jobs via MCP tools
- Docs Agent: searches DIRAC documentation via MCP tools

Usage:
    pip install -r requirements.txt
    export OPENAI_API_KEY=your_key
    python langgraph_workflow.py
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    next_agent: str


def supervisor(state: AgentState) -> AgentState:
    """Route the user request to the appropriate sub-agent.

    In a real deployment, this would use an LLM to decide routing.
    Here we use keyword matching for simplicity.
    """
    last_message = state["messages"][-1].content.lower()

    if any(kw in last_message for kw in ["job", "submit", "jdl", "failed", "status"]):
        return {"messages": [], "next_agent": "job_agent"}
    elif any(kw in last_message for kw in ["doc", "paper", "how", "what", "explain"]):
        return {"messages": [], "next_agent": "docs_agent"}
    else:
        return {
            "messages": [
                AIMessage(
                    content="I can help with DIRAC jobs or documentation. What would you like?"
                )
            ],
            "next_agent": "end",
        }


def job_agent(state: AgentState) -> AgentState:
    """Handle job-related requests.

    In a real deployment, this would connect to the dirac-mcp server
    and use tools like search_jobs, get_job, submit_job.
    """
    last_message = state["messages"][-1].content
    response = (
        f"[Job Agent] I would process your job request: '{last_message}'\n"
        "In production, I connect to the DiracX MCP server to:\n"
        "- Search jobs with search_jobs()\n"
        "- Get job details with get_job()\n"
        "- Submit jobs with submit_job()\n"
        "- Create JDL with create_basic_jdl()"
    )
    return {"messages": [AIMessage(content=response)], "next_agent": "end"}


def docs_agent(state: AgentState) -> AgentState:
    """Handle documentation search requests.

    In a real deployment, this would use a RAG pipeline
    (see examples/rag/) to search the DIRAC knowledge base.
    """
    last_message = state["messages"][-1].content
    response = (
        f"[Docs Agent] I would search documentation for: '{last_message}'\n"
        "In production, this agent uses a RAG pipeline to:\n"
        "- Embed the query and search a vector DB\n"
        "- Retrieve relevant documentation chunks\n"
        "- Synthesize an answer with an LLM"
    )
    return {"messages": [AIMessage(content=response)], "next_agent": "end"}


def route(state: AgentState) -> str:
    """Route to the next agent based on supervisor decision."""
    return state["next_agent"]


def build_graph() -> StateGraph:
    """Build the LangGraph workflow."""
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor)
    graph.add_node("job_agent", job_agent)
    graph.add_node("docs_agent", docs_agent)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route,
        {
            "job_agent": "job_agent",
            "docs_agent": "docs_agent",
            "end": END,
        },
    )

    graph.add_edge("job_agent", END)
    graph.add_edge("docs_agent", END)

    return graph.compile()


if __name__ == "__main__":
    workflow = build_graph()

    # Example: job-related query
    result = workflow.invoke(
        {
            "messages": [HumanMessage(content="Show me the failed jobs from last week")],
            "next_agent": "",
        }
    )
    for msg in result["messages"]:
        print(msg.content)
        print()

    # Example: documentation query
    result = workflow.invoke(
        {
            "messages": [HumanMessage(content="How does the DIRAC pilot framework work?")],
            "next_agent": "",
        }
    )
    for msg in result["messages"]:
        print(msg.content)
        print()

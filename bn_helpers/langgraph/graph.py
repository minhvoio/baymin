from typing import Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import ConversationState
from .nodes import (
    route_input, call_model, execute_tools, synthesize_answer,
    should_continue_to_tools, should_continue_after_tools,
)


def build_conversation_graph(checkpointer: Optional[MemorySaver] = None):
    """Build the LangGraph conversation graph.
    Graph: route_input -> call_model -> [execute_tools -> call_model ...] -> synthesize -> END
    """
    workflow = StateGraph(ConversationState)

    # NODES
    workflow.add_node("route_input", route_input)
    workflow.add_node("call_model", call_model)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("synthesize", synthesize_answer)

    # EDGES
    workflow.set_entry_point("route_input")
    workflow.add_edge("route_input", "call_model")

    workflow.add_conditional_edges(
        "call_model",
        should_continue_to_tools,
        {
            "execute_tools": "execute_tools",
            "synthesize": "synthesize",
            "error": "synthesize",  # graceful fallback on error
        }
    )

    workflow.add_conditional_edges(
        "execute_tools",
        should_continue_after_tools,
        {
            "call_model": "call_model",
            "synthesize": "synthesize",
        }
    )

    workflow.add_edge("synthesize", END)

    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)

    return workflow.compile()


def create_graph_with_memory() -> tuple:
    """Create a graph with in-memory checkpointing enabled."""
    checkpointer = MemorySaver()
    graph = build_conversation_graph(checkpointer)
    return graph, checkpointer

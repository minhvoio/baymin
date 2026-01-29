"""
LangGraph StateGraph construction for BN conversation.

This module builds the conversation graph that orchestrates:
- User input routing
- LLM invocation with tool binding
- Tool execution with caching
- Response synthesis
"""

from typing import Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import ConversationState
from .nodes import (
    route_input,
    call_model,
    execute_tools,
    synthesize_answer,
    should_continue_to_tools,
    should_continue_after_tools,
)


def build_conversation_graph(checkpointer: Optional[MemorySaver] = None):
    """
    Build the LangGraph conversation graph.

    Graph Structure:
    ```
                    ┌────────────────────────────────────┐
                    │                                    │
                    ▼                                    │
        route_input ──► call_model ◄─────────────────────┤
                            │                            │
                            ▼                            │
                    ┌───────────────┐                    │
                    │   decision    │                    │
                    └───────────────┘                    │
                      │           │                      │
            has_tools │           │ no_tools             │
                      ▼           ▼                      │
              execute_tools    synthesize                │
                      │           │                      │
                      │           ▼                      │
                      │          END                     │
                      └──────────────────────────────────┘
    ```

    Args:
        checkpointer: Optional MemorySaver for persistence.
                     If provided, enables conversation state checkpointing.

    Returns:
        Compiled LangGraph graph ready for invocation.
    """
    # Create graph with state schema
    workflow = StateGraph(ConversationState)

    # Add nodes
    workflow.add_node("route_input", route_input)
    workflow.add_node("call_model", call_model)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("synthesize", synthesize_answer)

    # Set entry point
    workflow.set_entry_point("route_input")

    # Add edges
    # route_input always goes to call_model
    workflow.add_edge("route_input", "call_model")

    # Conditional edges from call_model
    workflow.add_conditional_edges(
        "call_model",
        should_continue_to_tools,
        {
            "execute_tools": "execute_tools",
            "synthesize": "synthesize",
            "error": "synthesize",  # On error, go to synthesize to return gracefully
        }
    )

    # After tool execution, decide whether to continue
    workflow.add_conditional_edges(
        "execute_tools",
        should_continue_after_tools,
        {
            "call_model": "call_model",
            "synthesize": "synthesize",
        }
    )

    # Synthesize leads to END
    workflow.add_edge("synthesize", END)

    # Compile with optional checkpointer
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)

    return workflow.compile()


def create_graph_with_memory() -> tuple:
    """
    Create a graph with in-memory checkpointing enabled.

    Returns:
        Tuple of (compiled_graph, checkpointer)
    """
    checkpointer = MemorySaver()
    graph = build_conversation_graph(checkpointer)
    return graph, checkpointer

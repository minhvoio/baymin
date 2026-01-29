"""
State schema definitions for LangGraph-based BN conversation.

This module defines the TypedDict state schema used throughout the
conversation graph for maintaining stateful memory, tool result caching,
and derived artifacts.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from datetime import datetime
import operator

from langchain_core.messages import BaseMessage


class ToolResultCache(TypedDict):
    """Cached tool result with metadata."""
    result: Any
    timestamp: str
    call_count: int


class ConversationState(TypedDict):
    """
    Main state schema for LangGraph conversation.

    This state is maintained across multiple turns within a conversation,
    enabling tool result caching, artifact extraction, and context reuse.

    Attributes:
        messages: List of conversation messages (uses add reducer for appending)
        tool_results_cache: Mapping from (tool_name, args_hash) -> cached result
        artifacts: Derived data extracted from tool results for reuse
        metadata: Session info (conversation_id, timestamps, model config)
        network_info: BN structure context (nodes, states, structure string)
        current_tool_calls: Tool calls from the latest LLM response
        pending_tool_results: Results waiting to be processed
        should_continue: Control flag for graph routing
        error_state: Error message if something went wrong
    """

    # Core conversation - uses add reducer to append messages
    messages: Annotated[List[BaseMessage], operator.add]

    # Tool result caching: cache_key -> ToolResultCache
    # cache_key is MD5 hash of (tool_name, sorted_args_json)
    tool_results_cache: Dict[str, ToolResultCache]

    # Derived artifacts from tool results for cross-turn reuse
    # Examples:
    # - "d_connected:(A,B)": {"connected": True, "explanation": "..."}
    # - "last_probability_query": {"node": "Cancer", "evidence": {...}, "result": {...}}
    # - "computed_paths": {"A->B": ["A", "C", "B"]}
    artifacts: Dict[str, Any]

    # Session metadata (non-serializable items prefixed with "_")
    # Keys: conversation_id, created_at, updated_at, model, temperature,
    #       turn_count, max_turns, _tools (internal)
    metadata: Dict[str, Any]

    # Network reference info (serializable snapshot)
    # Keys: name, nodes (list), structure (str), node_states (str)
    network_info: Dict[str, Any]

    # Control flow state
    current_tool_calls: List[Dict[str, Any]]
    pending_tool_results: List[Dict[str, Any]]
    should_continue: bool
    error_state: Optional[str]


def create_initial_state(
    conversation_id: str,
    model: str = "gpt-oss:latest",
    temperature: float = 0.0,
    max_turns: int = 10,
    network_info: Optional[Dict[str, Any]] = None,
) -> ConversationState:
    """
    Create a fresh conversation state.

    Args:
        conversation_id: Unique identifier for this conversation
        model: Ollama model name
        temperature: LLM temperature setting
        max_turns: Maximum tool-calling rounds
        network_info: BN structure context dict

    Returns:
        Initialized ConversationState
    """
    now = datetime.now().isoformat()

    return {
        "messages": [],
        "tool_results_cache": {},
        "artifacts": {},
        "metadata": {
            "conversation_id": conversation_id,
            "created_at": now,
            "updated_at": now,
            "model": model,
            "temperature": temperature,
            "max_turns": max_turns,
            "turn_count": 0,
        },
        "network_info": network_info or {},
        "current_tool_calls": [],
        "pending_tool_results": [],
        "should_continue": False,
        "error_state": None,
    }

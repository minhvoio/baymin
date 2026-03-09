from typing import TypedDict, List, Dict, Any, Optional, Annotated
from datetime import datetime
import operator

from langchain_core.messages import BaseMessage


class ToolResultCache(TypedDict):
    result: Any
    timestamp: str
    call_count: int


class ConversationState(TypedDict):
    # Core conversation - uses add reducer to append messages
    messages: Annotated[List[BaseMessage], operator.add]

    # cache_key (MD5 of tool_name + sorted args) -> ToolResultCache
    tool_results_cache: Dict[str, ToolResultCache]

    # Reusable data extracted from tool results across turns
    # e.g. "d_connected:(A,B)": {...}, "prob:Cancer": {...}
    artifacts: Dict[str, Any]

    # Session metadata: conversation_id, timestamps, model config, turn_count, _tools (internal)
    metadata: Dict[str, Any]

    # BN structure snapshot: name, nodes, structure, node_states
    network_info: Dict[str, Any]

    # Control flow
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
    """Create a fresh conversation state."""
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

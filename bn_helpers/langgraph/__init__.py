"""
LangGraph-based conversational interface for Bayesian Network tool-calling.

This package provides stateful conversation management with:
- Conversation memory across multiple turns
- Tool result caching and reuse
- Derived artifact extraction
- Optional persistence support

Quick Start:
    ```python
    from bni_netica.bni_netica import Net
    from bn_helpers.langgraph import ConversationManager

    # Load network
    net = Net("nets/ChestClinic.neta")

    # Create conversation
    conv = ConversationManager(net)

    # Ask questions (maintains context across calls)
    answer1 = conv.ask("Is Smoking connected to Cancer?")
    answer2 = conv.ask("What if we observe TbOrCa?")

    # Inspect state
    print(conv.get_cache())      # Tool result cache
    print(conv.get_artifacts())  # Derived data
    print(conv.get_history())    # Message history

    # Reset for new conversation
    conv.reset()
    ```

For one-shot queries without state:
    ```python
    from bn_helpers.langgraph import ask_bn

    answer = ask_bn(net, "What is the probability of Cancer?")
    ```
"""

from .conversation import ConversationManager, ask_bn
from .state import ConversationState, ToolResultCache, create_initial_state
from .tools import create_langchain_tools, get_tools_by_name
from .graph import build_conversation_graph, create_graph_with_memory

__all__ = [
    # Main API
    "ConversationManager",
    "ask_bn",
    # State types
    "ConversationState",
    "ToolResultCache",
    "create_initial_state",
    # Tools
    "create_langchain_tools",
    "get_tools_by_name",
    # Graph
    "build_conversation_graph",
    "create_graph_with_memory",
]

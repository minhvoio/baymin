"""
ConversationManager: Main interface for stateful BN conversations.

This module provides the primary API for using the LangGraph-based
conversation system. It manages conversation state, tool creation,
and graph invocation.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
import json

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage

from .state import ConversationState, create_initial_state
from .graph import build_conversation_graph
from .tools import create_langchain_tools


class ConversationManager:
    """
    Main interface for stateful BN conversations.

    Each ConversationManager instance maintains a single conversation's
    state, including message history, tool result cache, and derived
    artifacts. Multiple turns can reuse context and cached results.

    Example:
        ```python
        from bni_netica.bni_netica import Net
        from bn_helpers.langgraph import ConversationManager

        # Load network
        net = Net("nets/ChestClinic.neta")

        # Create conversation
        conv = ConversationManager(net)

        # Ask questions (stateful - maintains context)
        answer1 = conv.ask("Is Smoking connected to Cancer?")
        answer2 = conv.ask("What if we observe TbOrCa?")  # Uses context from Q1

        # Inspect cached tool results
        print(conv.get_cache())

        # Get derived artifacts
        print(conv.get_artifacts())

        # Start fresh conversation
        conv.reset()
        ```
    """

    def __init__(
        self,
        net,
        model: str = "gpt-oss:latest",
        temperature: float = 0.0,
        max_turns: int = 10,
        conversation_id: Optional[str] = None,
    ):
        """
        Initialize a new conversation manager.

        Args:
            net: Netica BN network instance
            model: Ollama model name to use
            temperature: LLM temperature (0.0 = deterministic)
            max_turns: Maximum tool-calling rounds per question
            conversation_id: Optional ID for this conversation (auto-generated if not provided)
        """
        self.net = net
        self.model = model
        self.temperature = temperature
        self.max_turns = max_turns

        # Create LangChain tools from existing tool factories
        self._tools = create_langchain_tools(net)

        # Build the conversation graph
        self._graph = build_conversation_graph()

        # Initialize state
        self._conversation_id = conversation_id or str(uuid.uuid4())
        self._state = self._create_initial_state()

        # Graph config (for potential checkpointing)
        self._config = {"configurable": {"thread_id": self._conversation_id}}

    def _create_initial_state(self) -> ConversationState:
        """
        Create fresh conversation state with network context.

        Returns:
            Initialized ConversationState
        """
        from bn_helpers.get_structures_print_tools import (
            get_BN_structure,
            get_BN_node_states,
        )

        # Extract network info
        node_names = [n.name() for n in self.net.nodes()]
        structure = get_BN_structure(self.net)
        node_states = get_BN_node_states(self.net)

        # Get network name
        try:
            net_name = self.net.name()
        except Exception:
            net_name = "Unknown"

        # Create base state
        state = create_initial_state(
            conversation_id=self._conversation_id,
            model=self.model,
            temperature=self.temperature,
            max_turns=self.max_turns,
            network_info={
                "name": net_name,
                "nodes": node_names,
                "structure": structure,
                "node_states": node_states,
            },
        )

        # Inject tools into metadata (not serialized)
        state["metadata"]["_tools"] = self._tools

        return state

    def ask(self, question: str) -> str:
        """
        Send a question and get a response.

        This method:
        - Adds the question to conversation history
        - Runs the LangGraph to process the question
        - Returns the model's final response

        Subsequent calls maintain conversation context, allowing
        follow-up questions and tool result reuse.

        Args:
            question: User's question about the Bayesian network

        Returns:
            Model's response as a string
        """
        # Add user message to state
        user_msg = HumanMessage(content=question)
        self._state["messages"] = self._state.get("messages", []) + [user_msg]

        # Run the graph
        result = self._graph.invoke(self._state, self._config)

        # Update internal state
        self._state = result

        # Extract final answer from last AI message
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg.content

        # Fallback to last message content
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                return last_msg.content

        return "No response generated."

    def get_history(self) -> List[BaseMessage]:
        """
        Get full conversation history.

        Returns:
            List of LangChain message objects
        """
        return self._state.get("messages", [])

    def get_history_formatted(self) -> List[Dict[str, str]]:
        """
        Get conversation history as formatted dicts.

        Returns:
            List of dicts with 'role' and 'content' keys
        """
        history = []
        for msg in self._state.get("messages", []):
            if isinstance(msg, SystemMessage):
                history.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                history.append({"role": "tool", "name": msg.name, "content": msg.content})
        return history

    def get_artifacts(self) -> Dict[str, Any]:
        """
        Get derived artifacts from tool calls.

        Artifacts are structured data extracted from tool results,
        such as computed probabilities, d-connection status, etc.

        Returns:
            Dict of artifact_key -> artifact_value
        """
        return dict(self._state.get("artifacts", {}))

    def get_cache(self) -> Dict[str, Any]:
        """
        Get tool results cache.

        The cache maps (tool_name, args_hash) -> cached result.
        Use this to inspect what tools have been called and their results.

        Returns:
            Dict of cache_key -> ToolResultCache
        """
        return dict(self._state.get("tool_results_cache", {}))

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get conversation metadata.

        Includes: conversation_id, timestamps, model config, turn count.
        Internal fields (prefixed with "_") are excluded.

        Returns:
            Dict of metadata
        """
        meta = dict(self._state.get("metadata", {}))
        # Remove internal items
        return {k: v for k, v in meta.items() if not k.startswith("_")}

    @property
    def conversation_id(self) -> str:
        """Get the conversation ID."""
        return self._conversation_id

    @property
    def turn_count(self) -> int:
        """Get the number of completed turns."""
        return self._state.get("metadata", {}).get("turn_count", 0)

    def reset(self) -> None:
        """
        Clear state and start a new conversation.

        Generates a new conversation ID and resets all state
        including history, cache, and artifacts.
        """
        self._conversation_id = str(uuid.uuid4())
        self._state = self._create_initial_state()
        self._config = {"configurable": {"thread_id": self._conversation_id}}

    def save_state(self) -> Dict[str, Any]:
        """
        Export state for persistence.

        Returns a serializable dict that can be stored and later
        restored with load_state(). Excludes non-serializable items
        like the network object and tool functions.

        Returns:
            Serializable state dict
        """
        state_copy = {}

        # Copy metadata without internal items
        state_copy["metadata"] = {
            k: v for k, v in self._state.get("metadata", {}).items()
            if not k.startswith("_")
        }

        # Serialize messages
        state_copy["messages"] = []
        for msg in self._state.get("messages", []):
            msg_dict = {
                "type": type(msg).__name__,
                "content": msg.content,
            }
            if hasattr(msg, "name"):
                msg_dict["name"] = msg.name
            if hasattr(msg, "tool_call_id"):
                msg_dict["tool_call_id"] = msg.tool_call_id
            state_copy["messages"].append(msg_dict)

        # Copy cache (already serializable)
        state_copy["tool_results_cache"] = dict(self._state.get("tool_results_cache", {}))

        # Copy artifacts (already serializable)
        state_copy["artifacts"] = dict(self._state.get("artifacts", {}))

        # Copy network info (serializable snapshot)
        state_copy["network_info"] = dict(self._state.get("network_info", {}))

        return state_copy

    def load_state(self, saved_state: Dict[str, Any]) -> None:
        """
        Restore state from saved dict.

        Requires the same network to be loaded. This restores
        conversation history, cache, and artifacts.

        Args:
            saved_state: Dict from save_state()
        """
        # Reconstruct messages
        msg_types = {
            "HumanMessage": HumanMessage,
            "AIMessage": AIMessage,
            "SystemMessage": SystemMessage,
            "ToolMessage": ToolMessage,
        }

        messages = []
        for m in saved_state.get("messages", []):
            msg_cls = msg_types.get(m.get("type"), HumanMessage)
            kwargs = {"content": m.get("content", "")}

            # Handle ToolMessage special fields
            if msg_cls == ToolMessage:
                kwargs["name"] = m.get("name", "unknown")
                kwargs["tool_call_id"] = m.get("tool_call_id", "")

            messages.append(msg_cls(**kwargs))

        # Restore state
        self._state = {
            "messages": messages,
            "tool_results_cache": saved_state.get("tool_results_cache", {}),
            "artifacts": saved_state.get("artifacts", {}),
            "metadata": {
                **saved_state.get("metadata", {}),
                "_tools": self._tools,  # Re-inject tools
            },
            "current_tool_calls": [],
            "pending_tool_results": [],
            "should_continue": False,
            "error_state": None,
            "network_info": saved_state.get("network_info", self._state.get("network_info", {})),
        }

        # Update conversation ID from saved state
        self._conversation_id = self._state["metadata"].get(
            "conversation_id",
            self._conversation_id
        )
        self._config = {"configurable": {"thread_id": self._conversation_id}}


# Convenience function for quick stateless usage
def ask_bn(
    net,
    question: str,
    model: str = "gpt-oss:latest",
    temperature: float = 0.0,
) -> str:
    """
    Quick one-shot question to a Bayesian network.

    For stateless usage where you don't need conversation memory.
    For multi-turn conversations, use ConversationManager instead.

    Args:
        net: Netica BN network instance
        question: Question about the network
        model: Ollama model name
        temperature: LLM temperature

    Returns:
        Model's response as string
    """
    conv = ConversationManager(net, model=model, temperature=temperature)
    return conv.ask(question)

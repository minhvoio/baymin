from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid, json

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage

from .state import ConversationState, create_initial_state
from .graph import build_conversation_graph
from .tools import create_langchain_tools


class ConversationManager:
    """Main interface for stateful BN conversations with tool result caching and artifacts."""

    def __init__(
        self,
        net,
        model: str = "gpt-oss:latest",
        temperature: float = 0.0,
        max_turns: int = 10,
        conversation_id: Optional[str] = None,
    ):
        self.net = net
        self.model = model
        self.temperature = temperature
        self.max_turns = max_turns

        self._tools = create_langchain_tools(net)
        self._graph = build_conversation_graph()

        self._conversation_id = conversation_id or str(uuid.uuid4())
        self._state = self._create_initial_state()
        self._config = {"configurable": {"thread_id": self._conversation_id}}

    def _create_initial_state(self) -> ConversationState:
        from bn_helpers.get_structures_print_tools import (
            get_BN_structure, get_BN_node_states,
        )

        node_names = [n.name() for n in self.net.nodes()]
        structure = get_BN_structure(self.net)
        node_states = get_BN_node_states(self.net)

        try:
            net_name = self.net.name()
        except Exception:
            net_name = "Unknown"

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
        """Send a question and get a response. Maintains conversation context across calls."""
        user_msg = HumanMessage(content=question)
        self._state["messages"] = self._state.get("messages", []) + [user_msg]

        result = self._graph.invoke(self._state, self._config)
        self._state = result

        # Extract final answer from last AI message
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg.content

        # Fallback
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                return last_msg.content

        return "No response generated."

    def get_history(self) -> List[BaseMessage]:
        return self._state.get("messages", [])

    def get_history_formatted(self) -> List[Dict[str, str]]:
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
        return dict(self._state.get("artifacts", {}))

    def get_cache(self) -> Dict[str, Any]:
        return dict(self._state.get("tool_results_cache", {}))

    def get_metadata(self) -> Dict[str, Any]:
        """Get conversation metadata (excludes internal fields prefixed with '_')."""
        meta = dict(self._state.get("metadata", {}))
        return {k: v for k, v in meta.items() if not k.startswith("_")}

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    @property
    def turn_count(self) -> int:
        return self._state.get("metadata", {}).get("turn_count", 0)

    def reset(self) -> None:
        """Clear state and start a new conversation."""
        self._conversation_id = str(uuid.uuid4())
        self._state = self._create_initial_state()
        self._config = {"configurable": {"thread_id": self._conversation_id}}

    def save_state(self) -> Dict[str, Any]:
        """Export serializable state for persistence (excludes net object and tool functions)."""
        state_copy = {}

        state_copy["metadata"] = {
            k: v for k, v in self._state.get("metadata", {}).items()
            if not k.startswith("_")
        }

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

        state_copy["tool_results_cache"] = dict(self._state.get("tool_results_cache", {}))
        state_copy["artifacts"] = dict(self._state.get("artifacts", {}))
        state_copy["network_info"] = dict(self._state.get("network_info", {}))

        return state_copy

    def load_state(self, saved_state: Dict[str, Any]) -> None:
        """Restore state from saved dict. Requires same network to be loaded."""
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

            if msg_cls == ToolMessage:
                kwargs["name"] = m.get("name", "unknown")
                kwargs["tool_call_id"] = m.get("tool_call_id", "")

            messages.append(msg_cls(**kwargs))

        self._state = {
            "messages": messages,
            "tool_results_cache": saved_state.get("tool_results_cache", {}),
            "artifacts": saved_state.get("artifacts", {}),
            "metadata": {
                **saved_state.get("metadata", {}),
                "_tools": self._tools,  # re-inject tools
            },
            "current_tool_calls": [],
            "pending_tool_results": [],
            "should_continue": False,
            "error_state": None,
            "network_info": saved_state.get("network_info", self._state.get("network_info", {})),
        }

        self._conversation_id = self._state["metadata"].get(
            "conversation_id",
            self._conversation_id
        )
        self._config = {"configurable": {"thread_id": self._conversation_id}}


# Quick one-shot query (no conversation memory)
def ask_bn(
    net,
    question: str,
    model: str = "gpt-oss:latest",
    temperature: float = 0.0,
) -> str:
    """Quick one-shot question to a Bayesian network. For multi-turn, use ConversationManager."""
    conv = ConversationManager(net, model=model, temperature=temperature)
    return conv.ask(question)

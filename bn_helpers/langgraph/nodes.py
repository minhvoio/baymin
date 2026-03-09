from typing import Dict, Any, List
from datetime import datetime
import json, hashlib

from langchain_core.messages import (
    HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage,
)

from .state import ConversationState


def make_cache_key(tool_name: str, args: Dict[str, Any]) -> str:
    """Stable hash key for tool call dedup and caching."""
    key_data = json.dumps({"tool": tool_name, "args": args}, sort_keys=True)
    return hashlib.md5(key_data.encode()).hexdigest()


def build_system_prompt(state: ConversationState) -> str:
    net_info = state.get("network_info", {})
    node_states = net_info.get("node_states", "")

    return f"""You are a tool-calling Bayesian Network assistant.

Rules:
1) Always use tools when they can answer the query. Do NOT compute manually.
2) After receiving tool results, format them in clear, readable language.
3) Do NOT verify factual correctness of tool outputs - only grammar.
4) Include ALL information from tool outputs in your response.

NETWORK CONTEXT:
Name: {net_info.get('name', 'Unknown')}

NODES AND STATES:
{node_states}

From queries, extract correct parameters using the nodes and states above.
If a query mentions multiple nodes, check for abbreviations first (e.g., 'Tuberculosis or Cancer' -> 'TbOrCa')."""


def route_input(state: ConversationState) -> ConversationState:
    """Prepare state for processing a new user input."""
    messages = state.get("messages", [])

    has_system = any(isinstance(m, SystemMessage) for m in messages)
    if not has_system:
        system_prompt = build_system_prompt(state)
        system_msg = SystemMessage(content=system_prompt)
        state["messages"] = [system_msg] + list(messages)

    # Reset control flow for new turn
    state["current_tool_calls"] = []
    state["pending_tool_results"] = []
    state["should_continue"] = True
    state["error_state"] = None

    return state


def call_model(state: ConversationState) -> ConversationState:
    """Invoke LLM with bound tools."""
    from langchain_ollama import ChatOllama

    metadata = state.get("metadata", {})
    model_name = metadata.get("model", "gpt-oss:latest")
    temperature = metadata.get("temperature", 0.0)

    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
    )

    tools = metadata.get("_tools", [])

    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    try:
        response = llm_with_tools.invoke(state["messages"])
    except Exception as e:
        state["error_state"] = f"Model invocation failed: {str(e)}"
        state["should_continue"] = False
        return state

    state["messages"] = state.get("messages", []) + [response]

    if hasattr(response, "tool_calls") and response.tool_calls:
        state["current_tool_calls"] = response.tool_calls
        state["should_continue"] = True
    else:
        state["current_tool_calls"] = []
        state["should_continue"] = False

    return state


def execute_tools(state: ConversationState) -> ConversationState:
    """Execute requested tools with caching."""
    from .tools import get_tools_by_name

    tool_calls = state.get("current_tool_calls", [])
    metadata = state.get("metadata", {})
    tools_list = metadata.get("_tools", [])
    tools_map = get_tools_by_name(tools_list)

    cache = state.get("tool_results_cache", {})
    if cache is None:
        cache = {}
        state["tool_results_cache"] = cache

    tool_msgs: List[ToolMessage] = []

    for call in tool_calls:
        tool_name = call.get("name", "")
        args = call.get("args", {})
        call_id = call.get("id", "")

        cache_key = make_cache_key(tool_name, args)

        if cache_key in cache:
            cached = cache[cache_key]
            cached["call_count"] = cached.get("call_count", 0) + 1
            result = cached["result"]
        else:
            tool = tools_map.get(tool_name)
            if tool:
                try:
                    result = tool.invoke(args)
                except Exception as e:
                    result = {"error": type(e).__name__, "detail": str(e)}
            else:
                result = {"error": "ToolNotFound", "detail": f"Tool '{tool_name}' not registered"}

            cache[cache_key] = {
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "call_count": 1,
            }

        if isinstance(result, str):
            content = result
        else:
            try:
                content = json.dumps(result, ensure_ascii=False)
            except (TypeError, ValueError):
                content = str(result)

        tool_msgs.append(ToolMessage(
            content=content,
            tool_call_id=call_id,
            name=tool_name,
        ))

        _extract_artifacts(state, tool_name, args, result)

    state["messages"] = state.get("messages", []) + tool_msgs
    state["pending_tool_results"] = tool_msgs
    state["current_tool_calls"] = []

    # Continue to call model with results
    state["should_continue"] = True

    return state


def synthesize_answer(state: ConversationState) -> ConversationState:
    """Finalize the response after model produces text without tool calls."""
    metadata = state.get("metadata", {})
    metadata["turn_count"] = metadata.get("turn_count", 0) + 1
    metadata["updated_at"] = datetime.now().isoformat()
    state["metadata"] = metadata

    state["should_continue"] = False
    state["current_tool_calls"] = []
    state["pending_tool_results"] = []

    return state


# ARTIFACT EXTRACTION
def _extract_artifacts(
    state: ConversationState,
    tool_name: str,
    args: Dict[str, Any],
    result: Any
) -> None:
    """Extract and store reusable artifacts from tool results."""
    artifacts = state.get("artifacts", {})
    if artifacts is None:
        artifacts = {}
        state["artifacts"] = artifacts

    if tool_name == "check_d_connected":
        from_node = args.get("from_node", "")
        to_node = args.get("to_node", "")
        key = f"d_connected:({from_node},{to_node})"
        artifacts[key] = result

    elif tool_name == "check_common_cause":
        node1 = args.get("node1", "")
        node2 = args.get("node2", "")
        key = f"common_cause:({node1},{node2})"
        artifacts[key] = result

    elif tool_name == "check_common_effect":
        node1 = args.get("node1", "")
        node2 = args.get("node2", "")
        key = f"common_effect:({node1},{node2})"
        artifacts[key] = result

    elif tool_name == "get_prob_node":
        node = args.get("node", "")
        key = f"prob:{node}"
        artifacts[key] = result

    elif tool_name == "get_prob_node_given_any_evidence":
        node = args.get("node", "")
        evidence = args.get("evidence")
        artifacts["last_probability_query"] = {
            "node": node,
            "evidence": evidence,
            "result": result,
        }
        evidence_key = json.dumps(evidence, sort_keys=True) if evidence else "none"
        key = f"prob_given:{node}|{evidence_key}"
        artifacts[key] = result

    elif tool_name == "get_highest_impact_evidence":
        node = args.get("node", "")
        key = f"highest_impact:{node}"
        artifacts[key] = result

    elif tool_name == "get_evidences_block":
        node1 = args.get("node1", "")
        node2 = args.get("node2", "")
        key = f"blocking_evidence:({node1},{node2})"
        artifacts[key] = result


# ROUTING FUNCTIONS
def should_continue_to_tools(state: ConversationState) -> str:
    """Route after call_model: execute_tools / synthesize / error."""
    if state.get("error_state"):
        return "error"

    if state.get("current_tool_calls"):
        return "execute_tools"

    return "synthesize"


def should_continue_after_tools(state: ConversationState) -> str:
    """Route after execute_tools: back to call_model or synthesize on turn limit."""
    metadata = state.get("metadata", {})
    max_turns = metadata.get("max_turns", 10)
    turn_count = metadata.get("turn_count", 0)

    if turn_count >= max_turns:
        return "synthesize"

    return "call_model"

"""
Graph node functions for LangGraph-based BN conversation.

This module implements the node functions that make up the conversation graph:
- route_input: Prepare state and add system context
- call_model: Invoke LLM with bound tools
- execute_tools: Run requested tools with caching
- synthesize_answer: Format final response
"""

from typing import Dict, Any, List
from datetime import datetime
import json
import hashlib

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
    BaseMessage,
)

from .state import ConversationState


def make_cache_key(tool_name: str, args: Dict[str, Any]) -> str:
    """
    Create a stable hash key for tool call deduplication and caching.

    Args:
        tool_name: Name of the tool
        args: Tool arguments dict

    Returns:
        MD5 hash string as cache key
    """
    key_data = json.dumps({"tool": tool_name, "args": args}, sort_keys=True)
    return hashlib.md5(key_data.encode()).hexdigest()


def build_system_prompt(state: ConversationState) -> str:
    """
    Build the system prompt with network context.

    Args:
        state: Current conversation state

    Returns:
        System prompt string
    """
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
    """
    Prepare state for processing a new user input.

    This node:
    - Adds system prompt with network context if not present
    - Initializes control flow flags

    Args:
        state: Current conversation state

    Returns:
        Updated state ready for model invocation
    """
    messages = state.get("messages", [])

    # Add system message if not present
    has_system = any(isinstance(m, SystemMessage) for m in messages)
    if not has_system:
        system_prompt = build_system_prompt(state)
        system_msg = SystemMessage(content=system_prompt)
        # Prepend system message
        state["messages"] = [system_msg] + list(messages)

    # Reset control flow for new turn
    state["current_tool_calls"] = []
    state["pending_tool_results"] = []
    state["should_continue"] = True
    state["error_state"] = None

    return state


def call_model(state: ConversationState) -> ConversationState:
    """
    Invoke the LLM with bound tools.

    This node:
    - Creates ChatOllama instance with configured model
    - Binds tools to the model
    - Invokes model with current messages
    - Extracts any tool calls from response

    Args:
        state: Current conversation state

    Returns:
        Updated state with model response and tool calls
    """
    from langchain_ollama import ChatOllama

    metadata = state.get("metadata", {})
    model_name = metadata.get("model", "gpt-oss:latest")
    temperature = metadata.get("temperature", 0.0)

    # Create model instance
    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
    )

    # Get tools from metadata (injected by ConversationManager)
    tools = metadata.get("_tools", [])

    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    # Invoke model
    try:
        response = llm_with_tools.invoke(state["messages"])
    except Exception as e:
        state["error_state"] = f"Model invocation failed: {str(e)}"
        state["should_continue"] = False
        return state

    # Append response to messages
    state["messages"] = state.get("messages", []) + [response]

    # Check for tool calls
    if hasattr(response, "tool_calls") and response.tool_calls:
        state["current_tool_calls"] = response.tool_calls
        state["should_continue"] = True
    else:
        state["current_tool_calls"] = []
        state["should_continue"] = False

    return state


def execute_tools(state: ConversationState) -> ConversationState:
    """
    Execute requested tools with caching.

    This node:
    - Iterates through current tool calls
    - Checks cache before executing
    - Executes tools and caches results
    - Extracts artifacts from results
    - Creates ToolMessage responses

    Args:
        state: Current conversation state

    Returns:
        Updated state with tool results appended to messages
    """
    from .tools import get_tools_by_name

    tool_calls = state.get("current_tool_calls", [])
    metadata = state.get("metadata", {})
    tools_list = metadata.get("_tools", [])
    tools_map = get_tools_by_name(tools_list)

    cache = state.get("tool_results_cache", {})
    if cache is None:
        cache = {}
        state["tool_results_cache"] = cache

    tool_messages: List[ToolMessage] = []

    for call in tool_calls:
        tool_name = call.get("name", "")
        args = call.get("args", {})
        call_id = call.get("id", "")

        # Check cache
        cache_key = make_cache_key(tool_name, args)

        if cache_key in cache:
            # Return cached result
            cached = cache[cache_key]
            cached["call_count"] = cached.get("call_count", 0) + 1
            result = cached["result"]
        else:
            # Execute tool
            tool = tools_map.get(tool_name)
            if tool:
                try:
                    result = tool.invoke(args)
                except Exception as e:
                    result = {"error": type(e).__name__, "detail": str(e)}
            else:
                result = {"error": "ToolNotFound", "detail": f"Tool '{tool_name}' not registered"}

            # Cache result
            cache[cache_key] = {
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "call_count": 1,
            }

        # Create ToolMessage
        if isinstance(result, str):
            content = result
        else:
            try:
                content = json.dumps(result, ensure_ascii=False)
            except (TypeError, ValueError):
                content = str(result)

        tool_messages.append(ToolMessage(
            content=content,
            tool_call_id=call_id,
            name=tool_name,
        ))

        # Extract artifacts from result
        _extract_artifacts(state, tool_name, args, result)

    # Append tool messages
    state["messages"] = state.get("messages", []) + tool_messages
    state["pending_tool_results"] = tool_messages
    state["current_tool_calls"] = []

    # Continue to call model with results
    state["should_continue"] = True

    return state


def synthesize_answer(state: ConversationState) -> ConversationState:
    """
    Finalize the response after model produces text without tool calls.

    This node:
    - Updates turn count and timestamps
    - Could apply post-processing if needed

    Args:
        state: Current conversation state

    Returns:
        Final state ready to return to user
    """
    metadata = state.get("metadata", {})
    metadata["turn_count"] = metadata.get("turn_count", 0) + 1
    metadata["updated_at"] = datetime.now().isoformat()
    state["metadata"] = metadata

    # Reset control flow
    state["should_continue"] = False
    state["current_tool_calls"] = []
    state["pending_tool_results"] = []

    return state


def _extract_artifacts(
    state: ConversationState,
    tool_name: str,
    args: Dict[str, Any],
    result: Any
) -> None:
    """
    Extract and store useful artifacts from tool results.

    Artifacts are structured data that can be reused across turns,
    such as computed probabilities or path information.

    Args:
        state: Current conversation state
        tool_name: Name of the tool that was called
        args: Arguments passed to the tool
        result: Result returned by the tool
    """
    artifacts = state.get("artifacts", {})
    if artifacts is None:
        artifacts = {}
        state["artifacts"] = artifacts

    # Extract based on tool type
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
        # Store as latest probability query
        artifacts["last_probability_query"] = {
            "node": node,
            "evidence": evidence,
            "result": result,
        }
        # Also store with specific key
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


def should_continue_to_tools(state: ConversationState) -> str:
    """
    Routing function to determine next step after call_model.

    Returns:
        - "execute_tools" if there are tool calls to execute
        - "synthesize" if no tool calls (final answer)
        - "error" if there's an error state
    """
    if state.get("error_state"):
        return "error"

    if state.get("current_tool_calls"):
        return "execute_tools"

    return "synthesize"


def should_continue_after_tools(state: ConversationState) -> str:
    """
    Routing function to determine next step after execute_tools.

    Always returns to call_model to process tool results.
    """
    # Check turn limit
    metadata = state.get("metadata", {})
    max_turns = metadata.get("max_turns", 10)
    turn_count = metadata.get("turn_count", 0)

    if turn_count >= max_turns:
        return "synthesize"

    return "call_model"

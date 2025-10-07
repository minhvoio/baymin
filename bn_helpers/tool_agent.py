import requests, json
from bn_helpers.get_structures_print_tools import get_BN_node_states
from bn_helpers.bn_helpers import BnToolBox
from bn_helpers.utils import temporarily_set_findings
from bni_netica.bni_netica import Net

MODEL_QUIZ = "qwen2.5:7b"
MODEL_TOOLS = "gpt-oss:latest"
# MODEL_TOOLS = "qwen3:8b"
# MODEL_TOOLS = MODEL_QUIZ

import requests, json, inspect, typing

OLLAMA_URL = "http://localhost:11434/api/chat"

# Python type -> JSON Schema
def _pytype_to_schema(t):
    origin = typing.get_origin(t)
    args = typing.get_args(t)
    if t in (int,): return {"type": "integer"}
    if t in (float,): return {"type": "number"}
    if t in (bool,): return {"type": "boolean"}
    if t in (str,): return {"type": "string"}
    if t in (dict, typing.Dict, typing.Mapping): return {"type": "object"}
    if t in (list, typing.List, typing.Sequence): return {"type": "array"}
    if origin is typing.Union and len(args) == 2 and type(None) in args:
        other = args[0] if args[1] is type(None) else args[1]
        sch = _pytype_to_schema(other)
        if "type" in sch and isinstance(sch["type"], str):
            sch["type"] = [sch["type"], "null"]
        return sch
    if origin in (list, typing.List, typing.Sequence) and args:
        return {"type": "array", "items": _pytype_to_schema(args[0])}
    return {"type": "string"}

def function_to_tool_schema(fn, *, name=None, description=None):
    sig = inspect.signature(fn)
    props, required = {}, []
    for pname, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        props[pname] = _pytype_to_schema(p.annotation) if p.annotation is not inspect._empty else {"type":"string"}
        if p.default is inspect._empty:
            required.append(pname)
    return {
        "type":"function",
        "function":{
            "name": name or fn.__name__,
            "description": description or (fn.__doc__.strip() if fn.__doc__ else f"Function {fn.__name__}"),
            "parameters":{"type":"object","properties":props,"required":required}
        }
    }

def _coerce_arg(value, param: inspect.Parameter):
    if param.annotation in (int, float, bool, str):
        try: return param.annotation(value)
        except Exception: return value
    return value

def _bind_and_call(fn, kwargs: dict):
    sig = inspect.signature(fn)
    bound = {}
    for pname, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if pname in kwargs:
            bound[pname] = _coerce_arg(kwargs[pname], p)
        elif p.default is not inspect._empty:
            bound[pname] = p.default
        else:
            raise TypeError(f"Missing required argument: {pname}")
    return fn(**bound)

import json, requests

def _normalize_args_for_key(args):
    """
    Turn args dict into a hashable, order-stable key.
    Handles nested dicts/lists by JSON-dumping with sorted keys.
    """
    try:
        return json.dumps(args, sort_keys=True, separators=(",", ":"))
    except TypeError:
        # As a fallback, repr (still stable enough for dedup in most cases)
        return repr(args)

def chat_with_tools(
    net: Net,
    prompt: str,
    model: str = MODEL_TOOLS,    
    temperature: float = 0.0,
    num_predict: int = 500,
    max_rounds: int = 4,
    require_tool: bool = True,
    ollama_url: str = OLLAMA_URL,
):
    fns = get_tools_map(net)
    bn_str = get_BN_node_states(net)
    tools = [function_to_tool_schema(fn, name=name) for name, fn in fns.items()]

    system_prompt = (
        "You are a tool-using Bayesian Network assistant.\n"
        "Rules:\n"
        "1) Read the query carefully, think step by step, getting the correct tools.\n"
        "2) Do NOT perform calculations or external actions yourself if a tool can do it.\n"
        "3) Always call a tool when any tool could plausibly answer the user.\n"
        "4) After receiving tool results, if the return value is grammatically correct, return exactly that value;\n"
        "   otherwise fix only the grammar and return the corrected value.\n"
        "5) Do NOT verify factual correctness of the tool outputs — only grammar.\n"
        # "5) If a prior tool call failed or was rejected, do NOT repeat the same tool/argument combination.\n"
        # "6) If a tool call fails, try an alternative: adjust parameters (consistent with the prompt and provided node/state names) "
        # "   or choose a different tool.\n"
    )

    # Give the model the “catalog” of node/state names to extract from
    prompt += (
        "\n\nFrom the query above, extract the correct parameters for the tools using these nodes and states below. If the query related to multiple nodes, "
        "check for the abbreviations first (e.g., 'Tuberculosis or Cancer' can be 'TbOrCa') then check for the full names:\n"
        f"{bn_str}\n"
    )

    messages = [{"role":"system","content":system_prompt}, {"role":"user","content":prompt}]
    retries_left = max_rounds

    # state to avoid repeating failed/attempted calls
    seen_calls = set()          # {(tool_name, normalized_args_json)}
    recent_errors = []          # keep last few error messages for model guidance

    while retries_left > 0:
        r = requests.post(
            ollama_url,
            json={
                "model": model,
                "messages": messages,
                "tools": tools,
                "options": {"temperature": float(temperature), "num_predict": int(num_predict)},
            },
            stream=True,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Ollama error {r.status_code}: {r.text}")

        assistant_msg = {"role":"assistant","content":"", "tool_calls":[]}
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "message" in chunk:
                m = chunk["message"]
                if m.get("content"):
                    assistant_msg["content"] += m["content"]
                if "tool_calls" in m and m["tool_calls"]:
                    assistant_msg["tool_calls"] = m["tool_calls"]
            if chunk.get("done"):
                break

        messages.append(assistant_msg)

        # === If the assistant requested tool calls, try to execute them ===
        if assistant_msg["tool_calls"]:
            tool_msgs = []

            for i, call in enumerate(assistant_msg["tool_calls"], 1):
                fn_name = call["function"]["name"]
                args = call["function"].get("arguments", {}) or {}

                # Normalize for dedup
                arg_key = _normalize_args_for_key(args)
                call_key = (fn_name, arg_key)

                # Skip duplicate attempts (already tried same tool+args)
                if call_key in seen_calls:
                    # Tell the model not to repeat this combination
                    recent_errors.append(
                        f"Duplicate call blocked: {fn_name}({arg_key}) was already attempted."
                    )
                    # Synthesize a 'tool' message indicating it's blocked to keep turn structure
                    payload = {"error": "DuplicateAttempt", "detail": "This exact tool/args were already tried."}
                    tool_msgs.append({
                        "role": "tool",
                        "tool_name": fn_name,
                        "content": json.dumps(payload)
                    })
                    continue

                print(f"[BayMin] tool_call #{i}: {fn_name}({args})")
                seen_calls.add(call_key)

                # Run it
                if fn_name not in fns:
                    payload = {"error": f"ToolNotRegistered", "detail": f"Tool '{fn_name}' not registered"}
                    recent_errors.append(f"{fn_name} not registered.")
                else:
                    try:
                        out = _bind_and_call(fns[fn_name], args)
                        try:
                            json.dumps(out)  # ensure JSON-serializable
                            payload = {"result": out}
                        except TypeError:
                            payload = {"result": repr(out)}
                    except Exception as e:
                        payload = {"error": type(e).__name__, "detail": str(e)}
                        recent_errors.append(f"{fn_name} failed with {type(e).__name__}: {e}")

                print(f"[BayMin] tool_result #{i}: {payload}")
                tool_msgs.append({
                    "role": "tool",
                    "tool_name": fn_name,
                    "content": json.dumps(payload)
                })

            # Attach tool results
            messages.extend(tool_msgs)

            # If any tool produced an error, ask the model to try a different tool/params (avoiding seen_calls).
            if any(json.loads(m["content"]).get("error") for m in tool_msgs):
                tried_list = [
                    f"{name}({args})" for (name, args) in list(seen_calls)[-6:]   # show up to last 6
                ]
                err_tail = "\n".join(recent_errors[-6:]) if recent_errors else "N/A"
                messages.append({
                    "role": "user",
                    "content":
                        "Some tool calls failed or were blocked. "
                        "Do NOT repeat any of the tried combinations. "
                        "Try adjusting parameters (consistent with the provided nodes/states) or choose a different tool. "
                        "Only call tools that are strictly necessary.\n"
                        f"Already tried: {tried_list}\n"
                        f"Recent issues: {err_tail}\n"
                        "If you now have sufficient tool results, finalize the answer. "
                        "Otherwise, pick a new approach."
                })
            else:
                # No errors → ask the model to finalize without calling tools again unless needed
                messages.append({
                    "role": "user",
                    "content": "Use the tool results above to answer the question. "
                               "Do not call any tools again unless strictly necessary."
                })

            retries_left -= 1
            continue

        # No tool calls returned this turn
        if assistant_msg["content"].strip():
            # Model produced direct text. If tools required, nudge once more; else return it.
            if require_tool:
                messages.append({
                    "role":"user",
                    "content": (
                        "Reminder: you must use the available tools when they can plausibly answer. "
                        "Do not answer directly. Extract parameters from the nodes/states list provided and try again. "
                        "Avoid repeating any previously tried tool/argument combinations."
                    ),
                })
                retries_left -= 1
                continue
            else:
                return assistant_msg["content"].strip()

        # If we asked to finalize but got empty, fall back to the most recent tool JSON (raw)
        if messages and any(m.get("role") == "tool" for m in messages[-5:]):
            for m in reversed(messages):
                if m.get("role") == "tool":
                    return m["content"]  # raw JSON string

        # Last resort: ask to use tools again
        messages.append({
            "role":"user",
            "content":"You returned no content. Use tools with new parameters; do not repeat prior attempts."
        })
        retries_left -= 1

    # After all rounds, return the latest assistant content if any
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content", "").strip():
            return m["content"].strip()
    # Or the last tool output as ultimate fallback
    for m in reversed(messages):
        if m.get("role") == "tool":
            return m["content"]
    return ""

def make_explain_d_connected_tool(net):
    def check_d_connected(from_node: str, to_node: str) -> dict:
        """Explain whether two nodes are d-connected and why.
        d-connected means that entering evidence for one node will change the probability of the other node.
        """
        try:
            bn_tool_box = BnToolBox()
            is_conn = bn_tool_box.is_XY_connected(net, from_node, to_node)
            if is_conn:
                explanation = bn_tool_box.get_explain_XY_dconnected(net, from_node, to_node)
            else:
                explanation = bn_tool_box.get_explain_XY_dseparated(net, from_node, to_node)
            return explanation
        except Exception as e:
            return {"d_connected": None, "error": f"{type(e).__name__}: {e}"}
    return check_d_connected

def make_explain_common_cause_tool(net):
    def check_common_cause(node1, node2):
        """Check if there is a common cause between two nodes."""
        try:
            bn_tool_box = BnToolBox()
            ans = bn_tool_box.get_common_cause(net, node1, node2)
            if ans:
                template = f"The common cause(s) of {node1} and {node2} is/are: {', '.join(ans)}."
            else:
                template = f"There is no common cause between {node1} and {node2}."
            return template
        except Exception as e:
            return {"common_cause": None, "error": f"{type(e).__name__}: {e}"}
    return check_common_cause

def make_explain_common_effect_tool(net):
    def check_common_effect(node1, node2):
        """Check if there is a common effect between two nodes."""
        try:
            bn_tool_box = BnToolBox()
            ans = bn_tool_box.get_common_effect(net, node1, node2)
            if ans:
                template = f"The common effect(s) of {node1} and {node2} is/are: {', '.join(ans)}."
            else:
                template = f"There is no common effect between {node1} and {node2}."
            return template
        except Exception as e:
            return {"common_effect": None, "error": f"{type(e).__name__}: {e}"}
    return check_common_effect

def get_prob_node_tool(net):
    def get_prob_node(node):
        """Get the probability of a node."""
        try:
            bn_tool_box = BnToolBox()
            prob, _ = bn_tool_box.get_prob_X(net, node)
            return prob
        except Exception as e:
            return {"prob_node": None, "error": f"{type(e).__name__}: {e}"}
    return get_prob_node

def get_prob_node_given_one_evidence_tool(net):
    def get_prob_node_given_one_evidence(node, evidence, evidence_state):
        """Get the probability of a node given one evidence with its state.
        If the evidence_state is not provided, it means the evidence is True."""
        try:
            bn_tool_box = BnToolBox()
            prob, _ = bn_tool_box.get_prob_X_given_Y(net, node, evidence, evidence_state)
            return prob
        except Exception as e:
            return {"prob_node_given_one_evidence": None, "error": f"{type(e).__name__}: {e}"}
    return get_prob_node_given_one_evidence

def get_prob_node_given_two_evidences_tool(net):
    def get_prob_node_given_two_evidences(node, evidence1, evidence1_state, evidence2, evidence2_state):
        """Get the probability of a node given two evidences with their states.
        If the evidence_state is not provided, it means the evidence is True/or pick the state that means it is observed."""
        try:
            bn_tool_box = BnToolBox()
            prob, _ = bn_tool_box.get_prob_X_given_YZ(net, node, evidence1, evidence1_state, evidence2, evidence2_state)
            return prob
        except Exception as e:
            return {"prob_node_given_two_evidences": None, "error": f"{type(e).__name__}: {e}"}
    return get_prob_node_given_two_evidences

def check_one_evidence_change_relationship_between_two_nodes_tool(net):
    def check_one_evidence_change_relationship_between_two_nodes(node1, node2, evidence):
        """Check if changing the Evidence of one node will change the dependency relationship between Node1 and Node2.
        This function is used to check if the relationship between Node1 and Node2 get affected when we observe Evidence."""
        try:
            bn_tool_box = BnToolBox()
            ans, details = bn_tool_box.does_Z_change_dependency_XY(net, node1, node2, evidence)
            template = ""

            if ans:
                template = (f"Yes, observing {evidence} changes the dependency between {node1} and {node2}. "
                            f"Before observing {evidence}, {node1} and {node2} were "
                            f"{'d-connected' if details['before'] else 'd-separated'}. After observing {evidence}, they are "
                            f"{'d-connected' if details['after'] else 'd-separated'}.")
            else:
                template = (f"No, observing {evidence} does not change the dependency between {node1} and {node2}. "
                            f"Before observing {evidence}, {node1} and {node2} were "
                            f"{'d-connected' if details['before'] else 'd-separated'}. After observing {evidence}, they remain "
                            f"{'d-connected' if details['after'] else 'd-separated'}.")
            return template
        except Exception as e:
            return {"check_one_evidence_change_relationship_between_two_nodes": None, "error": f"{type(e).__name__}: {e}"}
    return check_one_evidence_change_relationship_between_two_nodes

def get_evidences_block_two_nodes_tool(net):
    def get_evidences_block_two_nodes(node1, node2):
        """Get the evidences list that block the dependency/path between Node1 and Node2."""
        try:
            bn_tool_box = BnToolBox()
            ans = bn_tool_box.evidences_block_XY(net, node1, node2)
            template = f"The evidences that would block the dependency between {node1} and {node2} are: {', '.join(ans)}."
            return template
        except Exception as e:
            return {"get_evidences_block_two_nodes": None, "error": f"{type(e).__name__}: {e}"}
    return get_evidences_block_two_nodes

def get_prob_node_given_any_evidence_tool(net):
    def get_prob_node_given_any_evidence(node: str, evidence: dict = None):
        """Get probability distribution of a node given any evidence dict by adding all evidences."""
        try:
            bn_tool_box = BnToolBox()
            prob_str, _ = bn_tool_box.get_prob_X_given(net, node, evidence)
            return prob_str
        except Exception as e:
            return {
                "result": None,
                "error": f"{type(e).__name__}: {e}"
            }

    return get_prob_node_given_any_evidence

def get_highest_impact_evidence_contribute_to_node_tool(net):
    def get_highest_impact_evidence_contribute_to_node(node: str, evidence: dict = None):
        """Compare the evidences impact on the node by adding and removing one evidence at a time, return the evidence that has the highest impact."""
        try:
            bn_tool_box = BnToolBox()
            ans, _ = bn_tool_box.get_highest_impact_evidence(net, node, evidence)
            return ans
        except Exception as e:
            return {"get_highest_impact_evidence_contribute_to_node": None, "error": f"{type(e).__name__}: {e}"}
    return get_highest_impact_evidence_contribute_to_node

def get_highest_impact_evidence_contribute_to_node_given_background_evidence_tool(net):
    def get_highest_impact_evidence_contribute_to_node_given_background_evidence(node: str, evidence: dict = None, background_evidence: dict = None):
        """With existing evidence (evidence), compare the evidences impact on the node by adding and removing one evidence at a time, return the evidence that has the highest impact."""
        try:
            with temporarily_set_findings(net, background_evidence):
                bn_tool_box = BnToolBox()
                ans, _ = bn_tool_box.get_highest_impact_evidence(net, node, evidence)
                return ans
        except Exception as e:
            return {"get_highest_impact_evidence_contribute_to_node_given_background_evidence": None, "error": f"{type(e).__name__}: {e}"}
    return get_highest_impact_evidence_contribute_to_node_given_background_evidence

def get_tools_map(net):
    return {
        "check_d_connected": make_explain_d_connected_tool(net),
        "check_common_cause": make_explain_common_cause_tool(net),
        "check_common_effect": make_explain_common_effect_tool(net),
        "get_prob_node": get_prob_node_tool(net),
        "get_prob_node_given_any_evidence": get_prob_node_given_any_evidence_tool(net),
        "get_highest_impact_evidence_contribute_to_node": get_highest_impact_evidence_contribute_to_node_tool(net),
        "get_highest_impact_evidence_contribute_to_node_given_evidence_knowledge": get_highest_impact_evidence_contribute_to_node_given_background_evidence_tool(net),
        "check_one_evidence_change_relationship_between_two_nodes": check_one_evidence_change_relationship_between_two_nodes_tool(net),
        "get_evidences_block_two_nodes": get_evidences_block_two_nodes_tool(net),
    }

def extract_text(answer: str) -> str:
    try:
        obj = json.loads(answer)
        if isinstance(obj, dict):
            if "result" in obj:
                return obj["result"]
            if "error" in obj:
                return f"Error: {obj['error']}: {obj.get('detail','')}".strip()
        return answer
    except json.JSONDecodeError:
        return answer

def get_answer_from_tool_agent(net, prompt, model=MODEL_TOOLS, temperature=0.0, num_predict=500, max_rounds=4, require_tool=True, ollama_url=OLLAMA_URL):
    answer = chat_with_tools(net, prompt, model, temperature, num_predict, max_rounds, require_tool, ollama_url)
    return extract_text(answer)
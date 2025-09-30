# import sys, os
# sys.path.append(os.path.abspath(".."))  # go up one folder

import requests, json
from IPython.display import display, Markdown, clear_output
from ollama.prompt import answer_this_prompt, generate_chat
from bn_helpers.get_structures_print_tools import get_nets, printNet, get_BN_structure, get_BN_node_states
from bn_helpers.bn_helpers import AnswerStructure, BnHelper
from bn_helpers.utils import get_path

MODEL_QUIZ = "qwen2.5:7b"
MODEL_TOOLS = "gpt-oss:latest"
# MODEL_TOOLS = MODEL_QUIZ

import requests, json, inspect, typing

OLLAMA_URL = "http://localhost:11434/api/chat"

pre_pre_query = "If the return value is grammatically correct, return exactly as that return value, otherwise fix the grammar and return the fixed return value."
pre_pre_query += "Do not check the return value's correctness. Just check the return value's grammar, fix only the grammar if necessary, then return the value."

# ---------- Python type -> JSON Schema ----------
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

def chat_with_tools(
    prompt: str,
    fns: dict,
    model: str = MODEL_TOOLS,
    temperature: float = 0.0,
    num_predict: int = 200,
    max_rounds: int = 4,
    require_tool: bool = True,
):
    tools = [function_to_tool_schema(fn, name=name) for name, fn in fns.items()]

    system_prompt = (
        "You are a tool-using assistant. "
        "Do NOT perform calculations or external actions yourself. "
        "Always call a tool when any tool could plausibly answer the user. "
        "After receiving tool results, if the return value is grammatically correct, return exactly as that return value, otherwise fix the grammar and return the fixed return value. "
        "Do not check the return value's correctness. "
        "Just check the return value's grammar, fix only the grammar if necessary, then return the value."
    )

    messages = [{"role":"system","content":system_prompt}, {"role":"user","content":prompt}]
    retries_left = max_rounds

    while retries_left > 0:
        r = requests.post(
            OLLAMA_URL,
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
            if not line: continue
            try: chunk = json.loads(line)
            except json.JSONDecodeError: continue
            if "message" in chunk:
                m = chunk["message"]
                if m.get("content"): assistant_msg["content"] += m["content"]
                if "tool_calls" in m and m["tool_calls"]:
                    assistant_msg["tool_calls"] = m["tool_calls"]
            if chunk.get("done"): break

        messages.append(assistant_msg)

        if assistant_msg["tool_calls"]:
            # Run every tool the model asked for
            tool_msgs = []
            for i, call in enumerate(assistant_msg["tool_calls"], 1):
                fn_name = call["function"]["name"]
                args = call["function"].get("arguments", {})
                print(f"[BayMin] tool_call #{i}: {fn_name}({args})")
                if fn_name not in fns:
                    payload = {"error": f"Tool '{fn_name}' not registered"}
                else:
                    try:
                        out = _bind_and_call(fns[fn_name], args)
                        try:
                            json.dumps(out)  # ensure JSON-serializable
                            payload = {"result": out}
                        except TypeError:
                            payload = {"result": repr(out)}
                    except Exception as e:
                        payload = {"error": f"{type(e).__name__}: {e}"}
                print(f"[BayMin] tool_result #{i}: {payload}")
                tool_msgs.append({"role":"tool","tool_name": fn_name,"content": json.dumps(payload)})

            messages.extend(tool_msgs)

            # let the model integrate tool outputs
            messages.append({
                "role": "user",
                "content": "Use the tool results above to answer the question succinctly. Do not call any tools again unless strictly necessary."
            })
            # Let the model integrate results into a final answer in the next turn
            retries_left -= 1
            continue

        # No tool calls returned
        if assistant_msg["content"].strip():
            return assistant_msg["content"].strip()

        # If we just asked it to finalize (the last message is our finalize prompt),
        # but it still returned empty content, fall back to tool outputs.
        if messages and any(m.get("role") == "tool" for m in messages[-3:]):
            # find the most recent tool message(s)
            for m in reversed(messages):
                if m.get("role") == "tool":
                    return m["content"]  # raw JSON string you sent; print or post-process as you like

        # Otherwise, nudge it to use tools (original behavior)
        if require_tool:
            print("[DEBUG] Model answered without tools; asking it to use tools...")
            messages.append({
                "role":"user",
                "content":"Reminder: you must use the available tools. Do not answer directly."
            })
            retries_left -= 1
            continue

        return assistant_msg["content"].strip()

    # After loops, return last assistant text (if any)
    return messages[-1]["content"].strip()


def make_explain_d_connected_tool(net):
    def check_d_connected(from_node: str, to_node: str) -> dict:
        """Explain whether two nodes are d-connected and why.
        d-connected means that entering evidence for one node will change the probability of the other node.
        Returns: { "explanation": string }
        """
        try:
            bn_helper = BnHelper()
            is_conn = bn_helper.is_XY_connected(net, from_node, to_node)
            if is_conn:
                explanation = bn_helper.get_explain_XY_dconnected(net, from_node, to_node)
            else:
                explanation = bn_helper.get_explain_XY_dseparated(net, from_node, to_node)
            # return {"d_connected": bool(is_conn), "explanation": explanation}
            return explanation
        except Exception as e:
            # Surface errors to the model in a structured, visible way
            return {"d_connected": None, "error": f"{type(e).__name__}: {e}"}
    return check_d_connected

def make_explain_common_cause_tool(net):
    def check_common_cause(node1, node2):
        """Check if there is a common cause between two nodes."""
        bn_helper = BnHelper()
        ans = bn_helper.get_common_cause(net, node1, node2)
        if ans:
            template = f"The common cause(s) of {node1} and {node2} is/are: {', '.join(ans)}."
        else:
            template = f"There is no common cause between {node1} and {node2}."
        return template
    return check_common_cause

def make_explain_common_effect_tool(net):
    def check_common_effect(node1, node2):
        """Check if there is a common effect between two nodes."""
        bn_helper = BnHelper()
        ans = bn_helper.get_common_effect(net, node1, node2)
        if ans:
            template = f"The common effect(s) of {node1} and {node2} is/are: {', '.join(ans)}."
        else:
            template = f"There is no common effect between {node1} and {node2}."
        return template
    return check_common_effect

def get_prob_node_tool(net):
    def get_prob_node(node):
        """Get the probability of a node."""
        bn_helper = BnHelper()
        prob, _ = bn_helper.get_prob_X(net, node)
        return prob
    return get_prob_node

def get_prob_node_given_one_evidence_tool(net):
    def get_prob_node_given_one_evidence(node, evidence, evidence_state):
        """Get the probability of a node given one evidence with its state.
        If the evidence_state is not provided, it means the evidence is True."""
        bn_helper = BnHelper()
        prob, _ = bn_helper.get_prob_X_given_Y(net, node, evidence, evidence_state)
        return prob
    return get_prob_node_given_one_evidence

def get_prob_node_given_two_evidences_tool(net):
    def get_prob_node_given_two_evidences(node, evidence1, evidence1_state, evidence2, evidence2_state):
        """Get the probability of a node given two evidences with their states.
        If the evidence_state is not provided, it means the evidence is True."""
        bn_helper = BnHelper()
        prob, _ = bn_helper.get_prob_X_given_YZ(net, node, evidence1, evidence1_state, evidence2, evidence2_state)
        return prob
    return get_prob_node_given_two_evidences

def check_one_evidence_change_relationship_between_two_nodes_tool(net):
    def check_one_evidence_change_relationship_between_two_nodes(node1, node2, evidence):
        """Check if changing the Evidence of one node will change the dependency relationship between Node1 and Node2.
        This function is used to check if the relationship between Node1 and Node2 get affected when we observe Evidence."""
        bn_helper = BnHelper()
        ans, details = bn_helper.does_Z_change_dependency_XY(net, node1, node2, evidence)
        template = ""
        print(f"Output: {ans}, details: {details}")
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
    return check_one_evidence_change_relationship_between_two_nodes

def get_evidences_block_two_nodes_tool(net):
    def get_evidences_block_two_nodes(node1, node2):
        """Get the evidences list that block the dependency/path between Node1 and Node2."""
        bn_helper = BnHelper()
        ans = bn_helper.evidences_block_XY(net, node1, node2)
        template = f"The evidences that would block the dependency between {node1} and {node2} are: {', '.join(ans)}."
        return template
    return get_evidences_block_two_nodes
from bn_helpers.support_tools import get_nets, printNet, get_BN_structure, get_BN_node_states
from bn_helpers.bn_helpers import BnHelper, ParamExtractor, AnswerStructure
from ollama.prompt import answer_this_prompt
from bn_helpers.scripts import HELLO_SCRIPT, MENU_SCRIPT, GET_FN_SCRIPT

def _current_net_for_mode(base_net, prev_ctx, mode):
    if mode == "continue" and prev_ctx and prev_ctx.get("last_net") is not None:
        print("Continuing with the previous network state...")
        return prev_ctx["last_net"]
    return base_net

def _format_and_explain(BN_string, user_query, fn, ans, template=None, schema=None, prev_ctx=None, probQuestion=False):
    prev_block = ""
    if prev_ctx and prev_ctx.get("last_user_query") and prev_ctx.get("last_answer"):
        prev_block = (
            f"\nHere is the previous conversation with the user:\n"
            f"Q: {prev_ctx['last_user_query']}\n"
            f"A: {prev_ctx['last_answer']}\n"
        )

    if probQuestion:
        explain_prompt = (
            f"In this Bayesian Network:\n{BN_string}\n"
            f"{prev_block}\n"
            f"User has now asked: '{user_query}'.\n"
            f"The raw output is:\n{ans}\n"
            f"Give a concise, human-friendly explanation."
        )
    else:
        explain_prompt = (
            f"In this Bayesian Network:\n{BN_string}\n"
            f"{prev_block}\n"
            f"User asked: '{user_query}'. "
            f"We used {fn} and the raw result is: '{ans}'. "
            f"Follow this exact template to provide the answer: '{template}'."
        )

    if schema:
        return answer_this_prompt(explain_prompt, format=schema)
    return answer_this_prompt(explain_prompt)

def _with_states_if_needed(pre_query, net, add_states):
    return pre_query + (f"\nThe states of nodes are:\n{get_BN_node_states(net)}" if add_states else "")

def execute_query(base_net, user_query, prev_ctx=None, mode="new"):
    continue_flag = False
    current_net = _current_net_for_mode(base_net, prev_ctx, mode)
    BN_string = get_BN_structure(current_net)

    prev_block = ""
    if mode == "continue" and prev_ctx and prev_ctx.get("last_user_query") and prev_ctx.get("last_answer"):
        continue_flag = True
        prev_block = (
            f"\nPrevious QA context:\n"
            f"Q: {prev_ctx['last_user_query']}\n"
            f"A: {prev_ctx['last_answer']}\n"
        )

    pre_query = f"In this Bayesian Network:\n{BN_string}\n{prev_block}"
    get_fn_prompt = pre_query + "\n" + user_query + GET_FN_SCRIPT

    raw = answer_this_prompt(get_fn_prompt, format=BnHelper.model_json_schema())
    print("\nBayMin:")
    print(raw)

    get_fn = BnHelper.model_validate_json(raw)
    fn = get_fn.function_name
    bn_helper = BnHelper(function_name=fn)
    param_extractor = ParamExtractor()

    ctx = {
        "fn": fn,
        "last_user_query": user_query,
        "last_answer": None,
        "last_net": prev_ctx.get("last_net") if prev_ctx else None
    }

    if fn == "is_XY_dconnected":
        params = param_extractor.extract_two_nodes_from_query(pre_query, user_query)
        print(params)
        ans = bn_helper.is_XY_dconnected(current_net, params.from_node, params.to_node)
        if ans:
            template = (f"Yes, {params.from_node} is d-connected to {params.to_node}, "
                        f"which means that entering evidence for {params.from_node} would "
                        f"change the probability of {params.to_node} and vice versa.")
        else:
            template = (f"No, {params.from_node} is not d-connected to {params.to_node}, "
                        f"which means that entering evidence for {params.from_node} would not "
                        f"change the probability of {params.to_node}.")
        final_answer = _format_and_explain(BN_string, user_query, fn, ans, template, schema=AnswerStructure.model_json_schema(), prev_ctx=ctx)
        ctx["last_answer"] = final_answer
        return final_answer, ctx

    elif fn == "get_common_cause":
        params = param_extractor.extract_two_nodes_from_query(pre_query, user_query)
        print(params)
        ans = bn_helper.get_common_cause(current_net, params.from_node, params.to_node)
        if ans:
            template = f"The common cause(s) of {params.from_node} and {params.to_node} is/are: {', '.join(ans)}."
        else:
            template = f"There is no common cause between {params.from_node} and {params.to_node}."
        final_answer = _format_and_explain(BN_string, user_query, fn, ans, template, schema=AnswerStructure.model_json_schema(), prev_ctx=ctx)
        ctx["last_answer"] = final_answer
        return final_answer, ctx

    elif fn == "get_common_effect":
        params = param_extractor.extract_two_nodes_from_query(pre_query, user_query, is_prev_qa=continue_flag)
        print(params)
        ans = bn_helper.get_common_effect(current_net, params.from_node, params.to_node)
        if ans:
            template = f"The common effect(s) of {params.from_node} and {params.to_node} is/are: {', '.join(ans)}."
        else:
            template = f"There is no common effect between {params.from_node} and {params.to_node}."
        final_answer = _format_and_explain(BN_string, user_query, fn, ans, template, schema=AnswerStructure.model_json_schema(), prev_ctx=ctx)
        ctx["last_answer"] = final_answer
        return final_answer, ctx

    elif fn == "does_Z_change_dependency_XY":
        params = param_extractor.extract_three_nodes_from_query(pre_query, user_query, is_prev_qa=continue_flag)
        print(params)
        ans, details = bn_helper.does_Z_change_dependency_XY(current_net, params.from_node, params.to_node, params.evidence_node)
        print(f"Output: {ans}, details: {details}")
        if ans:
            template = (f"Yes, observing {params.evidence_node} changes the dependency between {params.from_node} and {params.to_node}. "
                        f"Before observing {params.evidence_node}, {params.from_node} and {params.to_node} were "
                        f"{'d-connected' if details['before'] else 'd-separated'}. After observing {params.evidence_node}, they are "
                        f"{'d-connected' if details['after'] else 'd-separated'}.")
        else:
            template = (f"No, observing {params.evidence_node} does not change the dependency between {params.from_node} and {params.to_node}. "
                        f"Before observing {params.evidence_node}, {params.from_node} and {params.to_node} were "
                        f"{'d-connected' if details['before'] else 'd-separated'}. After observing {params.evidence_node}, they remain "
                        f"{'d-connected' if details['after'] else 'd-separated'}.")
        final_answer = _format_and_explain(BN_string, user_query, fn, ans, template, schema=AnswerStructure.model_json_schema(), prev_ctx=ctx)
        ctx["last_answer"] = final_answer
        return final_answer, ctx

    elif fn == "evidences_block_XY":
        params = param_extractor.extract_two_nodes_from_query(pre_query, user_query, is_prev_qa=continue_flag)
        print(params)
        ans = bn_helper.evidences_block_XY(current_net, params.from_node, params.to_node)
        print(f"Output: {ans}")
        if ans:
            template = f"The evidences that would block the dependency between {params.from_node} and {params.to_node} are: {', '.join(ans)}."
        else:
            template = f"There is no evidence that would block the dependency between {params.from_node} and {params.to_node}."
        final_answer = _format_and_explain(BN_string, user_query, fn, ans, template, schema=AnswerStructure.model_json_schema(), prev_ctx=ctx)
        ctx["last_answer"] = final_answer
        return final_answer, ctx

    elif fn == "get_prob_X":
        params = param_extractor.extract_one_node_from_query(pre_query, user_query, is_prev_qa=continue_flag)
        print(params)
        ans = bn_helper.get_prob_X(current_net, params.node)
        print(f"Output:\n{ans}\n")
        prob_prompt = (f"In this Bayesian Network:\n{BN_string}\n"
                          f"User asked: '{user_query}'. We used {fn}. "
                          f"The raw output is:\n{ans}\n"
                          f"Give a concise answer.")
        final_answer = _format_and_explain(BN_string, prob_prompt, fn, ans, probQuestion=True, prev_ctx=ctx, schema=AnswerStructure.model_json_schema())
        ctx["last_answer"] = final_answer
        ctx["last_net"] = current_net
        return final_answer, ctx

    elif fn == "get_prob_X_given_Y":
        pre_query2 = _with_states_if_needed(pre_query, current_net, add_states=True)
        params = param_extractor.extract_XY_and_Ystate_from_query(pre_query2, user_query, is_prev_qa=continue_flag)
        print(params)
        ans, net_after = bn_helper.get_prob_X_given_Y(current_net, params.target_node, params.evidence_node, params.evidence_state)
        print(f"Output:\n{ans}\n")

        prob_prompt = (f"In this Bayesian Network:\n{BN_string}\n"
                          f"User asked: '{user_query}'. We used {fn}. "
                          f"The raw output is:\n{ans}\n"
                          f"Give a concise answer.")
        final_answer = _format_and_explain(BN_string, prob_prompt, fn, ans, probQuestion=True, prev_ctx=ctx, schema=AnswerStructure.model_json_schema())
        ctx["last_answer"] = final_answer
        ctx["last_net"] = net_after
        return final_answer, ctx

    elif fn == "get_prob_X_given_YZ":
        pre_query2 = _with_states_if_needed(pre_query, current_net, add_states=True)
        params = param_extractor.extract_XYZ_and_YZstates_from_query(pre_query2, user_query, is_prev_qa=continue_flag)
        print(params)
        ans, net_after = bn_helper.get_prob_X_given_YZ(
            current_net,
            params.target_node,
            params.evidence_node1, params.evidence_state1,
            params.evidence_node2, params.evidence_state2
        )
        print(f"Output:\n{ans}\n")
        prob_prompt = (f"In this Bayesian Network:\n{BN_string}\n"
                          f"User asked: '{user_query}'. We used {fn}. "
                          f"The raw output is:\n{ans}\n"
                          f"Give a concise answer.")
        final_answer = _format_and_explain(BN_string, prob_prompt, fn, ans, probQuestion=True, prev_ctx=ctx, schema=AnswerStructure.model_json_schema())
        ctx["last_answer"] = final_answer
        ctx["last_net"] = net_after
        return final_answer, ctx
    
    elif fn == "detect_relationship":
        params = param_extractor.extract_child_and_two_parents_from_query(pre_query, user_query, is_prev_qa=continue_flag)
        print(params)
        ans = bn_helper.detect_relationship(current_net, params.child_node, params.parent1_node, params.parent2_node)
        print(f"Output:\n{ans}\n")
        final_answer = _format_and_explain(BN_string, user_query, fn, ans, schema=AnswerStructure.model_json_schema(), prev_ctx=ctx)
        ctx["last_answer"] = final_answer
        return final_answer, ctx

    return f"(Function '{fn}' not implemented yet.)", ctx

def query_menu(net):
    prev_ctx = None
    while True:
        user_query = input("Enter your query here (or press Enter to open menu): ").strip()
        if user_query:
            answer, prev_ctx = execute_query(net, user_query, prev_ctx=prev_ctx, mode="new")
            print("\n" + str(answer) + "\n")

        print(MENU_SCRIPT)
        choice_raw = input("Enter your choice: ").strip()
        print()
        if choice_raw == "":
            continue
        try:
            choice = int(choice_raw)
        except ValueError:
            print("Invalid choice. Please enter 1, 2, 3, or 4.\n")
            continue

        if choice == 1:
            if not prev_ctx:
                print("No previous query to continue from.\n")
                continue
            
            # show memory to the user
            print("--- Previous query ---")
            print(prev_ctx.get("last_user_query") or "<none>")
            print("--- Previous answer ---")
            print(prev_ctx.get("last_answer") or "<none>")
            print("----------------------\n")

            refine = input("Enter your follow-up / refined query (press Enter to reuse previous query): ").strip()
            if not refine:
                refine = prev_ctx.get("last_user_query") or ""
                if not refine:
                    print("No previous query to reuse. Returning to menu.\n")
                    continue

            answer, prev_ctx = execute_query(net, refine, prev_ctx=prev_ctx, mode="continue")
            print("\n" + str(answer) + "\n")

        elif choice == 2:
            new_q = input("Enter your new query: ").strip()
            if not new_q:
                print("No input. Returning to menu.\n")
                continue
            answer, prev_ctx = execute_query(net, new_q, prev_ctx=prev_ctx, mode="new")
            print("\n" + str(answer) + "\n")

        elif choice == 3:
            print("Switching network...\n")
            return "change"

        elif choice == 4:
            print("Goodbye!\n")
            return

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.\n")

def main():
    print(HELLO_SCRIPT)

    def select_network():
        nets = get_nets()
        for i, n in enumerate(nets):
            print(f"{i}: {n.name()}")
        print()
        while True:
            choice = input("Enter the number of the network you want to use: ").strip()
            try:
                idx = int(choice)
                if 0 <= idx < len(nets):
                    return nets[idx]
            except ValueError:
                pass
            print("Invalid choice. Try again.\n")

    while True:
        net = select_network()
        print(f"\nYou chose: {net.name()}\n")
        printNet(net)
        print('\nBN states:\n')
        print(get_BN_node_states(net))

        result = query_menu(net=net)
        if result == "change":
            continue
        else:
            break

if __name__ == "__main__":
    main()
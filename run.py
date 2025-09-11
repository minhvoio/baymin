from bni_netica.support_tools import get_nets, printNet, get_BN_structure, get_BN_node_states
from bni_netica.bn_helpers import BnHelper, ParamExtractor
from ollama.prompt import answer_this_prompt
from bni_netica.scripts import HELLO_SCRIPT, MENU_SCRIPT, GET_FN_SCRIPT

def execute_query(BN_string, net, user_query):
    pre_query = f"In this Bayesian Network:\n{BN_string}\n"
    get_fn_prompt = pre_query + "\n" + user_query + GET_FN_SCRIPT
    raw = answer_this_prompt(get_fn_prompt, format=BnHelper.model_json_schema())
    print("\nBayMin:")
    print(raw)

    get_fn = BnHelper.model_validate_json(raw)
    fn = get_fn.function_name
    bn_helper = BnHelper(function_name=fn)
    param_extractor = ParamExtractor()

    if fn == "is_XY_connected":
        get_params = param_extractor.extract_two_nodes_from_query(pre_query, user_query)
        print(get_params)
        ans = bn_helper.is_XY_connected(net, get_params.from_node, get_params.to_node)
        if ans:
            template = f"Yes, {get_params.from_node} is d-connected to {get_params.to_node}, which means that entering evidence for {get_params.from_node} would change the probability of {get_params.to_node} and vice versa."
        else:
            template = f"No, {get_params.from_node} is not d-connected to {get_params.to_node}, which means that entering evidence for {get_params.from_node} would not change the probability of {get_params.to_node}."
        explain_prompt = f"""User asked: In this '{BN_string}', '{user_query}'. We use {fn} function and the output is: '{ans}'. Follow this exact template to provide the answer: '{template}'."""
        final_answer = answer_this_prompt(explain_prompt)
        return final_answer

    elif fn == "get_common_cause":
        get_params = param_extractor.extract_two_nodes_from_query(pre_query, user_query)
        print(get_params)
        ans = bn_helper.get_common_cause(net, get_params.from_node, get_params.to_node)
        if ans:
            template = f"The common cause(s) of {get_params.from_node} and {get_params.to_node} is/are: {', '.join(ans)}."
        else:
            template = f"There is no common cause between {get_params.from_node} and {get_params.to_node}."
        explain_prompt = f"""User asked: In this '{BN_string}', '{user_query}'. We use {fn} function and the output is: '{ans}'. Follow this exact template to provide the answer: '{template}'."""
        final_answer = answer_this_prompt(explain_prompt)
        return final_answer

    elif fn == "get_common_effect":
        get_params = param_extractor.extract_two_nodes_from_query(pre_query, user_query)
        print(get_params)
        ans = bn_helper.get_common_effect(net, get_params.from_node, get_params.to_node)
        if ans:
            template = f"The common effect(s) of {get_params.from_node} and {get_params.to_node} is/are: {', '.join(ans)}."
        else:
            template = f"There is no common effect between {get_params.from_node} and {get_params.to_node}."
        explain_prompt = f"""User asked: In this '{BN_string}', '{user_query}'. We use {fn} function and the output is: '{ans}'. Follow this exact template to provide the answer: '{template}'."""
        final_answer = answer_this_prompt(explain_prompt)
        return final_answer

    elif fn == "does_Z_change_dependency_XY":
        get_params = param_extractor.extract_three_nodes_from_query(pre_query, user_query)
        print(get_params)
        ans, details = bn_helper.does_Z_change_dependency_XY(net, get_params.from_node, get_params.to_node, get_params.evidence_node)
        print(f"Output: {ans}, details: {details}")
        if ans:
            template = f"Yes, observing {get_params.evidence_node} changes the dependency between {get_params.from_node} and {get_params.to_node}. Before observing {get_params.evidence_node}, {get_params.from_node} and {get_params.to_node} were {'d-connected' if details['before'] else 'd-separated'}. After observing {get_params.evidence_node}, they are {'d-connected' if details['after'] else 'd-separated'}."
        else:
            template = f"No, observing {get_params.evidence_node} does not change the dependency between {get_params.from_node} and {get_params.to_node}. Before observing {get_params.evidence_node}, {get_params.from_node} and {get_params.to_node} were {'d-connected' if details['before'] else 'd-separated'}. After observing {get_params.evidence_node}, they remain {'d-connected' if details['after'] else 'd-separated'}."
        explain_prompt = f"""User asked: In this '{BN_string}', '{user_query}'. We use {fn} function and the output is: '{ans}', details: '{details}'. Follow this exact template to provide the answer: '{template}'."""
        final_answer = answer_this_prompt(explain_prompt)
        return final_answer

    elif fn == "evidences_block_XY":
        get_params = param_extractor.extract_two_nodes_from_query(pre_query, user_query)
        print(get_params)
        ans = bn_helper.evidences_block_XY(net, get_params.from_node, get_params.to_node)
        print(f"Output: {ans}")
        if ans:
            template = f"The evidences that would block the dependency between {get_params.from_node} and {get_params.to_node} are: {', '.join(ans)}."
        else:
            template = f"There is no evidence that would block the dependency between {get_params.from_node} and {get_params.to_node}."
        explain_prompt = f"""User asked: In this '{BN_string}', '{user_query}'. We use {fn} function and the output is: '{ans}'. Follow this exact template to provide the answer: '{template}'."""
        final_answer = answer_this_prompt(explain_prompt)
        return final_answer

    elif fn == "get_prob_X_given_Y":
        pre_query2 = pre_query + f"\nThe states of nodes are:\n{get_BN_node_states(net)}"
        get_params = param_extractor.extract_XY_and_Ystate_from_query(pre_query2, user_query)
        print(get_params)
        ans, net_after = bn_helper.get_prob_X_given_Y(net, get_params.target_node, get_params.evidence_node, get_params.evidence_state)
        print(f"Output:\n{ans}\n")
        answer_prompt = f"""User asked: In this '{BN_string}', '{user_query}'. We use {fn} function and the output is: '{ans}'. Give an concise answer to the user."""
        final_answer = answer_this_prompt(answer_prompt)
        return final_answer

    elif fn == "get_prob_X_given_YZ":
        pre_query2 = pre_query + f"\nThe states of nodes are:\n{get_BN_node_states(net)}"
        get_params = param_extractor.extract_XYZ_and_YZstates_from_query(pre_query2, user_query)
        print(get_params)
        ans, net_after = bn_helper.get_prob_X_given_YZ(net, get_params.target_node, get_params.evidence_node1, get_params.evidence_state1, get_params.evidence_node2, get_params.evidence_state2)
        print(f"Output:\n{ans}\n")
        answer_prompt = f"""User asked: In this '{BN_string}', '{user_query}'. We use {fn} function and the output is: '{ans}'. Give an concise answer to the user."""
        final_answer = answer_this_prompt(answer_prompt)
        return final_answer

    return f"(Function '{fn}' not implemented yet.)"

def query_menu(BN_string, net):
    prev_answer = None
    prev_query = None
    while True:
        user_query = input("Enter your query here (or press Enter to open menu): ").strip()
        if user_query:
            prev_answer = execute_query(BN_string, net, user_query)
            prev_query = user_query
            print("\n" + str(prev_answer) + "\n")

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
            if not prev_query:
                print("No previous query to continue from.\n")
                continue
            refine = input("Enter your follow-up / refined query: ").strip()
            if not refine:
                print("No input. Returning to menu.\n")
                continue
            prev_answer = execute_query(BN_string, net, refine)
            prev_query = refine
            print("\n" + str(prev_answer) + "\n")

        elif choice == 2:
            new_q = input("Enter your new query: ").strip()
            if not new_q:
                print("No input. Returning to menu.\n")
                continue
            prev_answer = execute_query(BN_string, net, new_q)
            prev_query = new_q
            print("\n" + str(prev_answer) + "\n")

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
        BN_string = get_BN_structure(net)

        result = query_menu(BN_string=BN_string, net=net)
        if result == "change":
            continue
        else:
            break

if __name__ == "__main__":
    main()
from bni_netica.support_tools import get_nets, printNet, get_BN_structure
from bni_netica.bn_helpers import BnHelper, QueryTwoNodes, extract_two_nodes_from_query, extract_three_nodes_from_query
from ollama.prompt import answer_this_prompt
from bni_netica.scripts import HELLO_SCRIPT, MENU_SCRIPT, GET_FN_QUERY

# PROMPT = """Consider this question: '{question}'. 
# What are the two nodes in this question? 
# Make sure to correctly output the names of nodes exactly as mentioned in the network and in the order as the question mentioned. 
# For example, if the question mentioned "A and B" then the two nodes are fromNode: A, toNode: B; or if the question mentioned "Smoking and Cancer" then the two nodes are fromNode: Smoking, toNode: Cancer. 
# Answer in JSON format."""

def query_menu(BN, net):
    """Input: BN: string, net: object"""
    pre_query = f"""In this Bayesian Network: 
{BN}
"""
    user_query = input("Enter your query here: ")
    get_fn_prompt = pre_query + user_query + GET_FN_QUERY

    get_fn = answer_this_prompt(get_fn_prompt, format=BnHelper.model_json_schema())
    print("\nBayesianista:")
    print(get_fn)

    get_fn = BnHelper.model_validate_json(get_fn)
    fn = get_fn.function_name

    bn_helper = BnHelper(function_name=fn)
    
    if fn == "is_XY_connected":
        get_params = extract_two_nodes_from_query(pre_query, user_query)
        print(get_params)

        ans = bn_helper.is_XY_connected(net, get_params.from_node, get_params.to_node)

        if ans:
            template = f"Yes, {get_params.from_node} is d-connected to {get_params.to_node}, which means that entering evidence for {get_params.from_node} would change the probability of {get_params.to_node} and vice versa."
        else:
            template = f"No, {get_params.from_node} is not d-connected to {get_params.to_node}, which means that entering evidence for {get_params.from_node} would not change the probability of {get_params.to_node}."
        
        explain_prompt = f"""User asked: In this '{BN}', '{user_query}'. We use {fn} function and the output is: '{ans}'. Follow this exact template to provide the answer: '{template}'."""
        print(answer_this_prompt(explain_prompt))
    
    elif fn == "get_common_cause":
        get_params = extract_two_nodes_from_query(pre_query, user_query)
        print(get_params)

        ans = bn_helper.get_common_cause(net, get_params.from_node, get_params.to_node)

        if ans:
            template = f"The common cause(s) of {get_params.from_node} and {get_params.to_node} is/are: {', '.join(ans)}."
        else:
            template = f"There is no common cause between {get_params.from_node} and {get_params.to_node}."

        explain_prompt = f"""User asked: In this '{BN}', '{user_query}'. We use {fn} function and the output is: '{ans}'. Follow this exact template to provide the answer: '{template}'."""
        print(answer_this_prompt(explain_prompt))

    elif fn == "get_common_effect":
        get_params = extract_two_nodes_from_query(pre_query, user_query)
        print(get_params)

        ans = bn_helper.get_common_effect(net, get_params.from_node, get_params.to_node)

        if ans:
            template = f"The common effect(s) of {get_params.from_node} and {get_params.to_node} is/are: {', '.join(ans)}."
        else:
            template = f"There is no common effect between {get_params.from_node} and {get_params.to_node}."

        explain_prompt = f"""User asked: In this '{BN}', '{user_query}'. We use {fn} function and the output is: '{ans}'. Follow this exact template to provide the answer: '{template}'."""
        print(answer_this_prompt(explain_prompt))

    elif fn == "does_Z_change_dependency_XY":
        get_params = extract_three_nodes_from_query(pre_query, user_query)
        print(get_params)

        ans, details = bn_helper.does_Z_change_dependency_XY(net, get_params.from_node, get_params.to_node, get_params.evidence_node)
        print(f"Output: {ans}, details: {details}")
        if ans:
            template = f"Yes, observing {get_params.evidence_node} changes the dependency between {get_params.from_node} and {get_params.to_node}. Before observing {get_params.evidence_node}, {get_params.from_node} and {get_params.to_node} were {'d-connected' if details['before'] else 'd-separated'}. After observing {get_params.evidence_node}, they are {'d-connected' if details['after'] else 'd-separated'}."
        else:
            template = f"No, observing {get_params.evidence_node} does not change the dependency between {get_params.from_node} and {get_params.to_node}. Before observing {get_params.evidence_node}, {get_params.from_node} and {get_params.to_node} were {'d-connected' if details['before'] else 'd-separated'}. After observing {get_params.evidence_node}, they remain {'d-connected' if details['after'] else 'd-separated'}."

        explain_prompt = f"""User asked: In this '{BN}', '{user_query}'. We use {fn} function and the output is: '{ans}', details: '{details}'. Follow this exact template to provide the answer: '{template}'."""
        print(answer_this_prompt(explain_prompt))

    elif fn == "evidences_block_XY":
        get_params = extract_two_nodes_from_query(pre_query, user_query)
        print(get_params)

        ans = bn_helper.evidences_block_XY(net, get_params.from_node, get_params.to_node)
        print(f"Output: {ans}")

        if ans:
            template = f"The evidences that would block the dependency between {get_params.from_node} and {get_params.to_node} are: {', '.join(ans)}."
        else:
            template = f"There is no evidence that would block the dependency between {get_params.from_node} and {get_params.to_node}."

        explain_prompt = f"""User asked: In this '{BN}', '{user_query}'. We use {fn} function and the output is: '{ans}'. Follow this exact template to provide the answer: '{template}'."""
        print(answer_this_prompt(explain_prompt))

    elif fn == "get_prob_X_given_Y":
        get_params = extract_two_nodes_from_query(pre_query, user_query)
        print(get_params)

        ans, net_after = bn_helper.get_prob_X_given_Y(net, get_params.from_node, get_params.to_node)
        print(f"Output:\n{ans}\n")

        answer_prompt = f"""User asked: In this '{BN}', '{user_query}'. We use {fn} function and the output is: '{ans}'. Give an concise answer to the user."""
        print(answer_this_prompt(answer_prompt))

    elif fn == "get_prob_X_given_YZ":
        get_params = extract_three_nodes_from_query(pre_query, user_query)
        print(get_params)

        ans, net_after = bn_helper.get_prob_X_given_YZ(net, get_params.from_node, get_params.to_node, get_params.evidence_node)
        print(f"Output:\n{ans}\n")

        answer_prompt = f"""User asked: In this '{BN}', '{user_query}'. We use {fn} function and the output is: '{ans}'. Give an concise answer to the user."""
        print(answer_this_prompt(answer_prompt))

    print()
    
    print(MENU_SCRIPT)
    choice = int(input("Enter your choice: "))
    print()

    if choice == 1:
        input("Enter your query here: ")
        print('This is a sample answer.\n')
    elif choice == 2:
        input("Enter your query here: ")
        print('This is a sample answer.\n')
    elif choice == 3:
        print("Not yet implemented\n")
        return 
    elif choice == 4:
        print("Goodbye!\n")
        return    

def main():
    print(HELLO_SCRIPT)
    nets = get_nets()

    
    for i, net in enumerate(nets):
        print(f"{i}: {net.name()}")

    print()
    choice = int(input("Enter the number of the network you want to use: "))
    print()
    if choice < 0 or choice >= len(nets):
        print("Invalid choice. Exiting.")
        return
    
    net = nets[choice]
    print(f"You chose: {net.name()}")
    printNet(net)
    print()
    BN = get_BN_structure(net)
    query_menu(BN=BN, net=net)


if __name__ == "__main__":
    main()

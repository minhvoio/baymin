from bni_netica.support_tools import get_nets, printNet, get_BN_structure
from bni_netica.bn_helpers import BnHelper, QueryTwoNodes
from ollama.prompt import answer_this_prompt

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
    get_fn_query = """\nIf user asks whether two nodes are related, output this function name: is_XY_connected"""
    get_fn_prompt = pre_query + user_query + get_fn_query

    get_params_query = """\nExtract the two nodes from the user query and output in JSON format as: {"from_node": "node1", "to_node": "node2"}."""
    get_params_prompt = pre_query + user_query + get_params_query

    get_params = answer_this_prompt(get_params_prompt, format=QueryTwoNodes.model_json_schema())
    get_fn = answer_this_prompt(get_fn_prompt, format=BnHelper.model_json_schema())
    print("\nBayesianista:")
    print(get_params)
    print(get_fn)

    get_params = QueryTwoNodes.model_validate_json(get_params)
    get_fn = BnHelper.model_validate_json(get_fn)
    fn = get_fn.function_name

    bn_helper = BnHelper(function_name=fn)
    
    if fn == "is_XY_connected":
        print(bn_helper.is_XY_connected(net, get_params.from_node, get_params.to_node))
    print()
    
    print("""
          ----------------------------------------
          1: Continue from previous query
          2: New query
          3: Change network
          4: Exit
          ----------------------------------------
          """)
    choice = int(input("Enter your choice: "))
    print()

    if choice == 1:
        input("Enter your query here: ")
        print('This is a sample answer.\n')
    elif choice == 2:
        input("Enter your query here: ")
        print('This is a sample answer.\n')
    elif choice == 3:
        print("Returning to network selection...\n")
    elif choice == 4:
        print("Goodbye!\n")
        return    

def main():
    print("""
          Hi, my name is Bayesianista. I can help you work with Bayesian Networks.
          """)
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

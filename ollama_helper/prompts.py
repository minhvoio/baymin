# HELLO_PROMPT = """
# Hi, my name is BayMin. I can help you work with Bayesian Networks.
# """

# MENU_PROMPT = """
# ----------------------------------------
#   What would you like to do next?
# 1: Continue from previous query
# 2: New query
# 3: Change network
# 4: Exit
# ----------------------------------------
# """

# GET_FN_PROMPT = """
# - If user asks whether two nodes are related, output exactly this function name: is_XY_connected.
# - If user asks for common cause of two nodes, output exactly this function name: get_common_cause.
# - If user asks for common effect (v-structure/collider) of two nodes, output exactly this function name: get_common_effect.
# - If user asks whether entering evidence for one node would change the relationship/dependency between two other nodes, output exactly this function name: does_Z_change_dependency_XY.
# - If user asks for evidences that would block the dependency between two nodes, output exactly this function name: evidences_block_XY.
# - If user asks for the PROBABILITY distribution of ONLY ONE node. Total 1 node is mentioned (count carefully). Output exactly this function name: get_prob_X.
# - If user asks for the PROBABILITY distribution of a node given evidence on one other node with its state. If the state is not clearly mentioned, from the meaning we could imply it is Yes or True. Total 2 nodes and 1 state are mentioned (count carefully). Output exactly this function name: get_prob_X_given_Y.
# - If user asks for the PROBABILITY distribution of a node given evidence on two other nodes with their states. If the state is not clearly mentioned, from the meaning we could imply it is Yes or True. Total 3 nodes and 2 states are mentioned (count carefully). Output exactly this function name: get_prob_X_given_YZ.
# - If user asks for the relationship from a child node to two parent nodes, output exactly this function name: detect_relationship.

# Think twice and read all the Ifs before you answer.
# """

# GET_FN_SIMPLIFIED_PROMPT = """
# Return one from this list of functions: [is_XY_connected, get_common_cause, get_common_effect, does_Z_change_dependency_XY, evidences_block_XY, get_prob_X, get_prob_X_given_Y, get_prob_X_given_YZ, detect_relationship]
# """

# PREV_QUERY_PROMPT = "There is a Previous QA context, use it to help answer the current query:\n"

# GET_PARAMS_PROMPT = {
#   "extract_one_node_from_query": """\nExtract the node from the user query and output in JSON format as: {"node": "node1"}.""",
#   "extract_XY_and_Ystate_from_query": """\nExtract the target node X, evidence node Y, and state of Y from the user query. If the state of Y is not clearly stated, that means it is Yes or True. Output in JSON format as: {"target_node": "node1", "evidence_node": "node2", "evidence_state": "state_name"}.""",
#   "extract_XYZ_and_YZstates_from_query": """\nExtract the target node X, evidence nodes Y and Z, and states of Y and Z from the user query. If the state of Y or Z is not clearly stated, that means they are Yes or True. Output in JSON format as: {"target_node": "node1", "evidence_node1": "node2", "evidence_state1": "state_name1", "evidence_node2": "node3", "evidence_state2": "state_name2"}.""",
#   "extract_two_nodes_from_query": """\nExtract the two nodes from the user query and output in JSON format as: {"from_node": "node1", "to_node": "node2"}.""",
#   "extract_three_nodes_from_query": """\nExtract the three nodes from the user query and output in JSON format as: {"from_node": "node1", "to_node": "node2", "evidence_node": "node3"}.""",
#   "extract_child_and_two_parents_from_query": """\nExtract the child node and two parent nodes from the user query and output in JSON format as: {"child_node": "node1", "parent1_node": "node2", "parent2_node": "node3"}.""",
# }

TAKE_QUIZ_PROMPT = """
Use the provided Explanation to answer the Quiz below.
Base your answer only on the Explanation — do NOT verify factual correctness.
If none of the answer options match the Explanation, choose the letter of the “None of the above.” option.
Do NOT output anything other than a ONE LETTER answer.

Quiz:
{quiz}

Explanation:
{bn_explanation}

You MUST respond with just a letter of the answer and no additional text, for example:

A
"""
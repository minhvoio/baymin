HELLO_SCRIPT = """
Hi, my name is Bayesianista. I can help you work with Bayesian Networks.
"""

MENU_SCRIPT = """
----------------------------------------
  What would you like to do next?
1: Continue from previous query
2: New query
3: Change network
4: Exit
----------------------------------------
"""

GET_FN_QUERY = """
- If user asks whether two nodes are related, output this function name: is_XY_connected.
- If user asks for common cause of two nodes, output this function name: get_common_cause.
- If user asks for common effect (v-structure/collider) of two nodes, output this function name: get_common_effect.
- If user asks whether entering evidence for one node would change the relationship/dependency between two other nodes, output this function name: does_Z_change_dependency_XY.
- If user asks for evidences that would block the dependency between two nodes, output this function name: evidences_block_XY.
- If user asks for the probability distribution of a node given evidence on two other nodes, output this function name: get_prob_X_given_YZ.
- If user asks for the probability distribution of a node given evidence on one other node, output this function name: get_prob_X_given_Y.
"""
from pydantic import BaseModel
from ollama.prompt import answer_this_prompt
from contextlib import contextmanager
from bni_netica.bni_utils import findAllDConnectedNodes
from bn_helpers.scripts import GET_PARAMS_SCRIPT, PREV_QUERY_SCRIPT
from bn_helpers.utils import (output_distribution, prob_X, prob_X_given_Y, prob_X_given_YZ, ensure_keys, logical_or, \
    logical_and, logical_xor, logical_xnor, fit_noisy_or, fit_noisy_and, fit_additive, _rmse, temporarily_set_findings, \
        names, resolve_state_index, state_names_by_indices, find_minimal_blockers, reduce_to_minimal_blocking_set, \
            is_independent_given)
from itertools import product
from collections import deque
import itertools

class QueryOneNode(BaseModel):
    node: str

# d_connected(X, Y) - True/False
class QueryTwoNodes(BaseModel):
    from_node: str
    to_node: str

class QueryThreeNodes(BaseModel):
    from_node: str
    to_node: str
    evidence_node: str

class QueryRelationship(BaseModel):
    child_node: str
    parent1_node: str
    parent2_node: str

# probabilities
class QueryProbTargetGivenOneEvidence(BaseModel):
    target_node: str
    evidence_node: str
    evidence_state: str = "Yes"

class QueryProbTargetGivenTwoEvidences(BaseModel):
    target_node: str
    evidence_node1: str
    evidence_state1: str = "Yes"
    evidence_node2: str
    evidence_state2: str = "Yes"

# explanation
class AnswerStructure(BaseModel):
    answer: str

class BnHelper(BaseModel):
    function_name: str

    # XY CONNECT
    def is_XY_connected(self, net, from_node, to_node):
      relatedNodes = net.node(from_node).getRelated("d_connected, exclude_self")
      for node in relatedNodes:
        if node.name() == to_node:
          return True
      return False
    

    # COMMON CAUSE / EFFECT
    def ancestors(self, net, node):
        return names(net.node(node).getRelated("ancestors,exclude_self"))

    def descendants(self, net, node):
        return names(net.node(node).getRelated("descendents,exclude_self"))

    def get_common_cause(self, net, node1, node2):
        return self.ancestors(net, node1) & self.ancestors(net, node2)

    def get_common_effect(self, net, node1, node2):
        return self.descendants(net, node1) & self.descendants(net, node2)
    

    # Z CHANGE DEPENDENCY XY
    def does_Z_change_dependency_XY(self, net, X, Y, Z):
        zname = net.node(Z).name() if isinstance(Z, str) else Z.name()

        # BEFORE: Z unobserved
        with temporarily_set_findings(net, {zname: None}):
            dep_before = self.is_XY_connected(net, X, Y)

        # AFTER: Z observed (state value doesn't matter for d-sep)
        with temporarily_set_findings(net, {zname: 0}):
            dep_after = self.is_XY_connected(net, X, Y)

        return (dep_before != dep_after), {"before": dep_before, "after": dep_after}

    # # EVIDENCES BLOCK XY
    def evidences_block_XY(self, net, X, Y):
        # node that will block the dependency between X and Y when observed
        ans = []
        evidences = findAllDConnectedNodes(net, X, Y)
        for e in evidences:
            e_name = e.name()
            if e_name != X and e_name != Y and self.does_Z_change_dependency_XY(net, X, Y, e_name)[0]:
                ans.append(e_name)
        return ans

    # PROBABILITIES
        
    def get_prob_X_given_Y(self, net, X=None, Y=None, y_state="Yes"):
        """
        Returns string output of prob_X_given_Y
        """
        beliefs, net_after = prob_X_given_Y(net, X, Y, y_state)
        output = f"P({X} | {Y}={y_state}):\n"
        output += output_distribution(beliefs, net_after.node(X))
        return output, net_after

    def get_prob_X_given_YZ(self, net, X=None, Y=None, y_state="Yes", Z=None, z_state="Yes"):
        """
        Returns string output of prob_X_given_YZ
        """
        beliefs, net_after = prob_X_given_YZ(net, X, Y, y_state, Z, z_state)
        output = f"P({X} | {Y}={y_state}, {Z}={z_state}):\n"
        output += output_distribution(beliefs, net_after.node(X))
        return output, net_after

    def get_prob_X(self, net, X=None):
        """
        Returns string output of prob_X
        """
        beliefs, net_after = prob_X(net, X)
        output = f"P({X}):\n"
        output += output_distribution(beliefs, net_after.node(X))
        return output, net_after

    # RELATIONSHIPS
    def cpt_Pequals_from_bn(
        self, net,
        child_name,
        parent_names,
        child_state=None,                # str or int (required if child has >1 state)
        parent_state_sets=None,          # dict: {parent_name: [state specs (str or int), ...]}
        iterate="all",                   # "all" | "first_two" | "given"
        use_titles=False                 # match strings against titles first if True
    ):
        """
        Build a table of P(child = child_state | parent assignments), without assuming 'true/false'.

        Returns:
        dict mapping tuple of parent state NAMES (in the same order as parent_names)
        to probability of the specified child state.
        e.g. {('given','high'): 0.63, ('not given','low'): 0.12, ...}
        """
        child = net.node(child_name)
        if child is None:
            raise ValueError(f"Child node '{child_name}' not found")

        # Resolve which child state to measure
        if child_state is None:
            # default: first state (index 0)
            child_idx = 0
        else:
            child_idx = resolve_state_index(child, child_state, use_title=use_titles)

        # For each parent, decide which state indices to iterate
        parent_state_idxs = []
        parent_state_names = []  # parallel names for pretty keys
        for p_name in parent_names:
            p_node = net.node(p_name)
            if p_node is None:
                raise ValueError(f"Parent node '{p_name}' not found")

            if iterate == "given":
                if not parent_state_sets or p_name not in parent_state_sets:
                    raise ValueError(f"Provide parent_state_sets[{p_name}] when iterate='given'")
                idxs = [resolve_state_index(p_node, s, use_title=use_titles) for s in parent_state_sets[p_name]]

            elif iterate == "first_two":
                # take first two states (works for many binary-ish nodes)
                idxs = [0, 1] if p_node.numberStates() >= 2 else [0]

            else:  # "all"
                idxs = list(range(p_node.numberStates()))

            parent_state_idxs.append(idxs)
            parent_state_names.append(state_names_by_indices(p_node, idxs))

        # Iterate Cartesian product of selected parent states
        table = {}
        for combo_idxs in product(*parent_state_idxs):
            # Build findings dict using indices (your ctx manager accepts int or str)
            evid = {p: idx for p, idx in zip(parent_names, combo_idxs)}

            with temporarily_set_findings(net, evid):
                prob = child.beliefs()[child_idx]

            # Pretty key using parent state NAMES
            key = tuple(names[i_pos] for names, i_pos in zip(parent_state_names, [idxs.index(i) for idxs, i in zip(parent_state_idxs, combo_idxs)]))
            # But the above maps to relative positions; instead, resolve absolute names cleanly:
            key = tuple(net.node(p).state(i).name() for p, i in zip(parent_names, combo_idxs))

            table[key] = prob

        return table

    
    def detect_relationship(self,         
        net,
        child_name,
        parent_names,
        child_state=None,                # str or int (required if child has >1 state)
        parent_state_sets=None,          # dict: {parent_name: [state specs (str or int), ...]}
        iterate="all",                   # "all" | "first_two" | "given"
        use_titles=False                 # match strings against titles first if True
    ):
        """
        cpt: dict with keys like ('T','F'), ('F','T'), etc. or (1,0)
                values are P(Z=1 | A=a, B=b).
        Returns dict: {label: {'rmse':..., 'pred':..., 'params':...}}, plus 'best'.
        """
        cpt = self.cpt_Pequals_from_bn(net, child_name, parent_names, child_state, parent_state_sets, iterate, use_titles)
        obs = ensure_keys(cpt)

        # logicals
        models = {}
        for label, proto in [
            ("Logical OR", logical_or()),
            ("Logical AND", logical_and()),
            ("Logical XOR", logical_xor()),
            ("Logical XNOR", logical_xnor()),
        ]:
            rmse = _rmse(obs, proto)
            models[label] = {"rmse": rmse, "pred": proto, "params": None}

        # parametrics
        rmse_noisy_or, pred_noisy_or, par_noisy_or = fit_noisy_or(obs)
        models["Noisy OR"] = {"rmse": rmse_noisy_or, "pred": pred_noisy_or, "params": par_noisy_or}

        rmse_noisy_and, pred_noisy_and, par_noisy_and = fit_noisy_and(obs)
        models["Noisy AND"] = {"rmse": rmse_noisy_and, "pred": pred_noisy_and, "params": par_noisy_and}

        try:
            rmse_add, pred_add, par_add = fit_additive(obs)
            models["Additive"] = {"rmse": rmse_add, "pred": pred_add, "params": par_add}
        except Exception:
            # numpy not available -> skip additive
            pass

        # pick best
        best_label = min(models.keys(), key=lambda k: models[k]["rmse"])
        models["best"] = best_label
        return models


class ParamExtractor():

    def extract_one_node_from_query(self, pre_query: str, user_query: str, is_prev_qa: bool = False) -> QueryOneNode:
        get_params_query = ""
        if is_prev_qa:
            get_params_query += PREV_QUERY_SCRIPT
        
        get_params_query += GET_PARAMS_SCRIPT["extract_one_node_from_query"]
        get_params_prompt = pre_query + user_query + get_params_query
        get_params = answer_this_prompt(get_params_prompt, format=QueryOneNode.model_json_schema())
        get_params = QueryOneNode.model_validate_json(get_params)
        return get_params

    def extract_XY_and_Ystate_from_query(self, pre_query: str, user_query: str, is_prev_qa: bool = False) -> QueryProbTargetGivenOneEvidence:
        get_params_query = ""
        if is_prev_qa:
            get_params_query += PREV_QUERY_SCRIPT
        
        get_params_query += GET_PARAMS_SCRIPT["extract_XY_and_Ystate_from_query"]
        get_params_prompt = pre_query + user_query + get_params_query
        get_params = answer_this_prompt(get_params_prompt, format=QueryProbTargetGivenOneEvidence.model_json_schema())
        get_params = QueryProbTargetGivenOneEvidence.model_validate_json(get_params)
        return get_params

    def extract_XYZ_and_YZstates_from_query(self, pre_query: str, user_query: str, is_prev_qa: bool = False) -> QueryProbTargetGivenTwoEvidences: 
        get_params_query = ""
        if is_prev_qa:
            get_params_query += PREV_QUERY_SCRIPT
        
        get_params_query += GET_PARAMS_SCRIPT["extract_XYZ_and_YZstates_from_query"]
        get_params_prompt = pre_query + user_query + get_params_query
        get_params = answer_this_prompt(get_params_prompt, format=QueryProbTargetGivenTwoEvidences.model_json_schema())
        get_params = QueryProbTargetGivenTwoEvidences.model_validate_json(get_params)
        return get_params
    
    def extract_two_nodes_from_query(self, pre_query: str, user_query: str, is_prev_qa: bool = False) -> QueryTwoNodes:
        get_params_query = ""
        if is_prev_qa:
            get_params_query += PREV_QUERY_SCRIPT

        get_params_query += GET_PARAMS_SCRIPT["extract_two_nodes_from_query"]
        get_params_prompt = pre_query + user_query + get_params_query
        get_params = answer_this_prompt(get_params_prompt, format=QueryTwoNodes.model_json_schema())
        get_params = QueryTwoNodes.model_validate_json(get_params)
        return get_params

    def extract_three_nodes_from_query(self, pre_query: str, user_query: str, is_prev_qa: bool = False) -> QueryThreeNodes:
        get_params_query = ""
        if is_prev_qa:
            get_params_query += PREV_QUERY_SCRIPT

        get_params_query += GET_PARAMS_SCRIPT["extract_three_nodes_from_query"]
        get_params_prompt = pre_query + user_query + get_params_query
        get_params = answer_this_prompt(get_params_prompt, format=QueryThreeNodes.model_json_schema())
        get_params = QueryThreeNodes.model_validate_json(get_params)
        return get_params

    def extract_child_and_two_parents_from_query(self, pre_query: str, user_query: str, is_prev_qa: bool = False) -> QueryRelationship:
        get_params_query = ""
        if is_prev_qa:
            get_params_query += PREV_QUERY_SCRIPT

        get_params_query += GET_PARAMS_SCRIPT["extract_child_and_two_parents_from_query"]
        get_params_prompt = pre_query + user_query + get_params_query
        get_params = answer_this_prompt(get_params_prompt, format=QueryRelationship.model_json_schema())
        get_params = QueryRelationship.model_validate_json(get_params)
        return get_params

from pydantic import BaseModel
from ollama.prompt import answer_this_prompt
from contextlib import contextmanager
from bni_netica.bni_utils import findAllDConnectedNodes
from bni_netica.scripts import GET_PARAMS_SCRIPT, PREV_QUERY_SCRIPT

# d_connected(X, Y) - True/False
class QueryTwoNodes(BaseModel):
    from_node: str
    to_node: str

class QueryThreeNodes(BaseModel):
    from_node: str
    to_node: str
    evidence_node: str

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
    def is_XY_dconnected(self, net, from_node, to_node):
      relatedNodes = net.node(from_node).getRelated("d_connected,exclude_self")
      for node in relatedNodes:
        if node.name() == to_node:
          return True
      return False
    

    # COMMON CAUSE / EFFECT
    def _names(self, nodes):
      return {n.name() for n in nodes}

    def ancestors(self, net, node):
        return self._names(net.node(node).getRelated("ancestors,exclude_self"))

    def descendants(self, net, node):
        return self.names(net.node(node).getRelated("descendents,exclude_self"))

    def get_common_cause(self, net, node1, node2):
        return self.ancestors(net, node1) & self.ancestors(net, node2)

    def get_common_effect(self, net, node1, node2):
        return self.descendants(net, node1) & self.descendants(net, node2)
    

    # Z CHANGE DEPENDENCY XY
    @contextmanager # release resource automatically after using it
    def _temporarily_set_findings(self, net, findings_dict):
        """findings_dict: {node_name: state_name_or_index}"""
        saved = net.findings()  # current evidence (indices)
        try:
            # Apply new evidence
            for k, v in findings_dict.items():
                node = net.node(k)
                if isinstance(v, int):
                    node.finding(v)
                else:
                    node.finding(v)  # state(name)
            net.update()
            yield
        finally:
            # Restore previous evidence
            net.retractFindings()
            for k, v in (saved or {}).items():
                net.node(k).finding(v)
            net.update()

    def does_Z_change_dependency_XY(self, net, X, Y, Z):
        """
        Returns (changed: bool, details: {'before': bool, 'after': bool})
        where True means 'd-connected' (dependent), False means 'd-separated' (independent).
        """
        zname = net.node(Z).name()
        saved = net.findings()  # keep current evidence E to restore later
        
        try:
            # ---- BEFORE: dependency without Z ----
            if saved and zname in saved:
                # ensure Z is not observed for the 'before' check
                net.node(zname).retractFindings()
                net.update()
            dep_before = self.is_XY_dconnected(net, X, Y)

            # ---- AFTER: dependency under Z observed ----
            with self._temporarily_set_findings(net, {zname: 0}): # using state index 0 for Z
                dep_after = self.is_XY_dconnected(net, X, Y)

            return (dep_before != dep_after), {"before": dep_before, "after": dep_after}
        finally:
            # Restore original evidence exactly as it was
            net.retractFindings()
            if saved:
                net.findings(saved)
            net.update()


    # EVIDENCES BLOCK XY
    def evidences_block_XY(self, net, X, Y):
        ans = []
        evidences = findAllDConnectedNodes(net, X, Y)
        for e in evidences:
            e_name = e.name()
            if e_name != X and e_name != Y and self.does_Z_change_dependency_XY(net, X, Y, e_name)[0]:
                ans.append(e_name)
        return ans
    

    # PROBABILITIES
    def _output_distribution(self, beliefs, node):
        """Pretty print a distribution (list of probabilities) with state names."""
        output = ""
        for i, p in enumerate(beliefs):
            state = node.state(i)
            output += f"  P({node.name()}={state.name()}) = {p:.4f}\n"

        return output

    def _prob_X_given_Y(self, net, X=None, Y=None, y_state="Yes"):
        """
        Returns P(X | Y = y_state), net_after_observation
        y_state can be state names (str) or indices (int).
        """
        
        with self._temporarily_set_findings(net, {Y: y_state}):
            node_X = net.node(X)
            # beliefs() is P(X | current findings)
            return node_X.beliefs(), net
        
    def _prob_X_given_YZ(self, net, X=None, Y=None, y_state="Yes", Z=None, z_state="Yes"):
        """
        Returns P(X = x_state | Y = y_state, Z = z_state), net_after_observation
        x_state, y_state and z_state can be state names (str) or indices (int).
        """
        
        with self._temporarily_set_findings(net, {Y: y_state, Z: z_state}):
            node_X = net.node(X)
            # beliefs() is P(X | current findings)
            return node_X.beliefs(), net
        
    def get_prob_X_given_Y(self, net, X=None, Y=None, y_state="Yes"):
        """
        Returns string output of prob_X_given_Y
        """
        beliefs, net_after = self._prob_X_given_Y(net, X, Y, y_state)
        output = f"P({X} | {Y}={y_state}):\n"
        output += self._output_distribution(beliefs, net_after.node(X))
        return output, net_after

    def get_prob_X_given_YZ(self, net, X=None, Y=None, y_state="Yes", Z=None, z_state="Yes"):
        """
        Returns string output of prob_X_given_YZ
        """
        beliefs, net_after = self._prob_X_given_YZ(net, X, Y, y_state, Z, z_state)
        output = f"P({X} | {Y}={y_state}, {Z}={z_state}):\n"
        output += self._output_distribution(beliefs, net_after.node(X))
        return output, net_after

class ParamExtractor():

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

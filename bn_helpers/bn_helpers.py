from pydantic import BaseModel
from ollama_helper.ollama_helper import answer_this_prompt
from contextlib import contextmanager
from bni_netica.bni_utils import findAllDConnectedNodes
from ollama_helper.prompts import GET_PARAMS_PROMPT, PREV_QUERY_PROMPT
from bn_helpers.utils import (output_distribution, ensure_keys, logical_or, \
    logical_and, logical_xor, logical_xnor, fit_noisy_or, fit_noisy_and, fit_additive, _rmse, temporarily_set_findings, \
        names, resolve_state_index, state_names_by_indices, find_minimal_blockers, reduce_to_minimal_blocking_set, \
            is_independent_given, get_path, _state_label)
from itertools import product
from collections import deque
import itertools
from typing import List, Tuple, Dict, Any

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

class BnToolBox():
    # function_name: str

    # XY CONNECT
    def is_XY_dconnected(self, net, from_node, to_node):
      relatedNodes = net.node(from_node).getRelated("d_connected, exclude_self")
      for node in relatedNodes:
        if node.name() == to_node:
          return True
      return False

    def get_explain_XY_dconnected(self, net, node1, node2):
        open_path = get_path(net, node1, node2)  # must exist!
        return (f"Yes, {node1} is d-connected to {node2}, "
                f"which means that entering evidence for {node1} would "
                f"change the probability of {node2} and vice versa. They d-connected through the following path: {open_path}")

    def get_explain_XY_dseparated(self, net, node1, node2, get_minimal=False):
        import random
        blocked_nodes = self.get_common_effect(net, node1, node2)
        
        random_blocked_node = None
        if get_minimal:
            random_blocked_node = random.choice(list(blocked_nodes)) if blocked_nodes else None

        base = (f"No, {node1} is not d-connected to {node2}, so evidence on {node1} "
                f"would not change the probability of {node2}.")
        if get_minimal:
            return base + f" They are blocked at {random_blocked_node} due to a common effect."
        elif blocked_nodes: 
            is_plural = len(blocked_nodes) > 1
            a_or_the = "the" if is_plural else "a"
            final_s = "s" if is_plural else ""
            return base + f" They are blocked at {blocked_nodes} due to {a_or_the} common effect{final_s}."
        return base + f" There is no open path between {node1} and {node2}."


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
    # def does_Z_change_dependency_XY(self, net, X, Y, Z):
    #     zname = net.node(Z).name() if isinstance(Z, str) else Z.name()

    #     # BEFORE: Z unobserved
    #     with temporarily_set_findings(net, {zname: None}):
    #         dep_before = self.is_XY_dconnected(net, X, Y)

    #     # AFTER: Z observed (state value doesn't matter for d-sep)
    #     with temporarily_set_findings(net, {zname: 0}):
    #         dep_after = self.is_XY_dconnected(net, X, Y)

    #     return (dep_before != dep_after), {"before": dep_before, "after": dep_after}

    def does_evidence_change_dependency_XY(self, net, X: str, Y: str, evidence: List[str]) -> Tuple[bool, Dict[str, Any]]:
        if not evidence:
            with temporarily_set_findings(net, {}):
                dep = self.is_XY_dconnected(net, X, Y)
            return False, {
                "before": dep,
                "after": dep,
                "conditioned_on": [],
                "sequential": []
            }

        # BEFORE: none of the evidence is observed
        with temporarily_set_findings(net, {z: None for z in evidence}):
            dep_before = self.is_XY_dconnected(net, X, Y)

        # AFTER: all evidence nodes observed (value doesn't matter for d-sep)
        with temporarily_set_findings(net, {z: 0 for z in evidence}):
            dep_after = self.is_XY_dconnected(net, X, Y)

        changed = dep_before != dep_after

        # Sequential trace (adding evidence one by one)
        sequential_trace = []
        partial = {}
        for z in evidence:
            partial[z] = 0
            with temporarily_set_findings(net, partial):
                conn = self.is_XY_dconnected(net, X, Y)
            sequential_trace.append({"added": z, "connected": conn})

        return changed, {
            "before": dep_before,
            "after": dep_after,
            "conditioned_on": evidence,
            "sequential": sequential_trace # how each evidence changes the dependency
        }

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
    def get_prob_X(self, net, X=None):
        """
        Returns string output of prob_X
        """
        node_X = net.node(X)
        beliefs = node_X.beliefs()
        output = f"P({X}):\n"
        for i, p in enumerate(beliefs):
            output += f"  P({X}={node_X.state(i).name()}) = {p:.4f}\n"
        return output, net

    def prob_X_given(self, net, X, evidence=None):
        """
        Returns (original_beliefs, new_beliefs, net_after, impacts)
        - original_beliefs: P(X) before any findings
        - new_beliefs:     P(X | evidence)
        - net_after:       same net (for parity)
        - impacts: {ev -> {"diffs": [...], "l1": float, "max_abs": float}}
        """
        evidence = evidence or {}
        node_X_before = net.node(X)
        original = node_X_before.beliefs()

        # Full-evidence posterior
        with temporarily_set_findings(net, evidence):
            new = net.node(X).beliefs()

        # Leave-one-out impacts (each evidence removed once)
        impacts = {}
        for ev in evidence.keys():
            reduced = {k: v for k, v in evidence.items() if k != ev}
            with temporarily_set_findings(net, reduced):
                wo = net.node(X).beliefs()  # P(X | evidence \ {ev})
            diffs = [n - w for n, w in zip(new, wo)]
            l1 = sum(abs(d) for d in diffs)
            max_abs = max((abs(d) for d in diffs), default=0.0)
            impacts[ev] = {"diffs": diffs, "l1": l1, "max_abs": max_abs}

        return original, new, net, impacts


    def get_prob_X_given(self, net, X, evidence=None, *, threshold=0.05, tol=1e-8):
        """
        Pretty string for P(X | evidence) using `prob_X_given` + conclusion + evidence impact.
        Returns: (out_str, net_after)
        """
        evidence = evidence or {}

        # header text (state-name friendly)
        cond = ", ".join(f"{k}={_state_label(net, k, v)}" for k, v in evidence.items()) or "∅"
        header = f"P({X} | {cond}):\n"

        # call the core inference once
        original, new, net_after, impacts = self.prob_X_given(net, X, evidence)

        # format distributions
        node = net_after.node(X)
        out = header
        for i, p in enumerate(new):
            out += f"  P({node.name()}={node.state(i).name()}) = {p:.4f}\n"

        out += "\nOriginal distribution:\n"
        for i, p in enumerate(original):
            out += f"  P({node.name()}={node.state(i).name()}) = {p:.4f}\n"

        # conclusion on overall change
        diffs = [n - o for n, o in zip(new, original)]
        max_change = max((abs(d) for d in diffs), default=0.0)

        out += "\nConclusion:\n"
        if max_change <= tol:
            out += "  No change detected — the updated beliefs are identical to the original.\n"
        else:
            for i, d in enumerate(diffs):
                if abs(d) > threshold:
                    sname = node.state(i).name()
                    out += f"  Belief in '{sname}' {'increased' if d > 0 else 'decreased'} by {abs(d):.4f}\n"
            if max_change <= threshold:
                out += "  Overall, the update is minimal (all changes ≤ threshold).\n"
            else:
                out += f"  Largest overall per-state shift: {max_change:.4f}.\n"

        # evidence impact (ranked)
        if impacts:
            out += "\nEvidence impact (leave-one-out):\n"
            ranked = sorted(impacts.items(), key=lambda kv: kv[1]["l1"], reverse=True)

            for ev, stats in ranked:
                out += f"  - {ev}: L1={stats['l1']:.4f}, max_abs={stats['max_abs']:.4f}\n"
                # out += f"  - {ev}: max_abs={stats['max_abs']:.4f}\n"

            if ranked and (len(ranked) == 1 or ranked[0][1]["l1"] > 1.5 * (ranked[1][1]["l1"] if len(ranked) > 1 else 0)):
                out += f"  => Most influential evidence: {ranked[0][0]} (by L1 contribution).\n"
                # shown max_abs contribution 
                # out += f"  => Most influential evidence: {ranked[0][0]}, it contributes {max_abs_contribution:.4f}.\n"

        return out, net_after

    def get_highest_impact_evidence(self, net, X, evidence=None, order=None, *, threshold=0.05, tol=1e-8):
        """
        Analyze per-evidence impact on X using sequential ADD and sequential REMOVE,
        then output a ranked summary and a conclusion (like `get_prob_X_given`).

        - Uses `prob_X_given` to compute:
            original = P(X)
            new      = P(X | evidence)
        - Sequential ADD impact for an evidence ev_k is:
            P(X | S_k ∪ {ev_k})  -  P(X | S_k)      where S_k is the set of evidence added before ev_k (in `order`)
        - Sequential REMOVE impact for ev_k is:
            P(X | All)           -  P(X | All \ {ev_k})
        - For each evidence we report:
            add:   L1 and max_abs over states
            rem:   L1 and max_abs over states
            score: max(add.L1, rem.L1)   # used for "highest impact"

        Returns:
            out_str, net_after
        """
        evidence = evidence or {}
        if not evidence:
            return f"P({X} | ∅):\n  No evidence provided; nothing to analyze.\n", net

        # helpers
        def _beliefs_with(evi_dict):
            with temporarily_set_findings(net, evi_dict):
                return net.node(X).beliefs()

        # Preserve caller-specified order, else dict insertion order, else sorted
        if order is not None:
            seq = [ev for ev in order if ev in evidence]
        else:
            seq = list(evidence.keys())

        # baseline & header
        original, new, net_after, _ = self.prob_X_given(net, X, evidence)
        node = net_after.node(X)

        cond = ", ".join(f"{k}={_state_label(net, k, v)}" for k, v in evidence.items()) or "∅"
        out = f"P({X} | {cond}):\n"

        # Updated vs Original distributions
        for i, p in enumerate(new):
            out += f"  P({node.name()}={node.state(i).name()}) = {p:.4f}\n"
        out += "\nOriginal distribution:\n"
        for i, p in enumerate(original):
            out += f"  P({node.name()}={node.state(i).name()}) = {p:.4f}\n"

        # Overall conclusion (same style as get_prob_X_given)
        diffs = [n - o for n, o in zip(new, original)]
        max_change = max((abs(d) for d in diffs), default=0.0)
        out += "\nConclusion:\n"
        if max_change <= tol:
            out += "  No change detected — the updated beliefs are identical to the original.\n"
        else:
            for i, d in enumerate(diffs):
                if abs(d) > threshold:
                    sname = node.state(i).name()
                    out += f"  Belief in '{sname}' {'increased' if d > 0 else 'decreased'} by {abs(d):.4f}\n"
            if max_change <= threshold:
                out += "  Overall, the update is minimal (all changes ≤ threshold).\n"
            else:
                out += f"  Largest overall per-state shift: {max_change:.4f}.\n"

        # sequential ADD impacts
        impacts = {}  # ev -> {"add": {...}, "rem": {...}, "score": float}
        S = {}         # currently added set (in sequence)
        prev_beliefs = original
        for ev in seq:
            # beliefs after adding this ev on top of S
            S_next = {**S, ev: evidence[ev]}
            bel_next = _beliefs_with(S_next)
            diffs_add = [b1 - b0 for b1, b0 in zip(bel_next, prev_beliefs)]
            add_l1 = sum(abs(d) for d in diffs_add)
            add_max = max((abs(d) for d in diffs_add), default=0.0)
            impacts.setdefault(ev, {})["add"] = {"diffs": diffs_add, "l1": add_l1, "max_abs": add_max}

            # advance sequence state
            S = S_next
            prev_beliefs = bel_next

        # sequential REMOVE impacts
        all_beliefs = new
        full = dict(evidence)
        for ev in seq:
            reduced = {k: v for k, v in full.items() if k != ev}
            bel_wo = _beliefs_with(reduced)
            diffs_rem = [b_full - b_wo for b_full, b_wo in zip(all_beliefs, bel_wo)]
            rem_l1 = sum(abs(d) for d in diffs_rem)
            rem_max = max((abs(d) for d in diffs_rem), default=0.0)
            impacts.setdefault(ev, {})["rem"] = {"diffs": diffs_rem, "l1": rem_l1, "max_abs": rem_max}

        # scoring & report
        # score = max(add.L1, rem.L1)
        for ev, d in impacts.items():
            add_l1 = d.get("add", {}).get("l1", 0.0)
            rem_l1 = d.get("rem", {}).get("l1", 0.0)
            d["score"] = max(add_l1, rem_l1)

        ranked = sorted(impacts.items(), key=lambda kv: kv[1]["score"], reverse=True)

        out += "\nEvidence impact (sequential add/remove):\n"
        for ev, d in ranked:
            add_l1 = d["add"]["l1"] if "add" in d else 0.0
            rem_l1 = d["rem"]["l1"] if "rem" in d else 0.0
            add_max = d["add"]["max_abs"] if "add" in d else 0.0
            rem_max = d["rem"]["max_abs"] if "rem" in d else 0.0
            out += (f"  - {ev}: "
                    f"ADD  L1={add_l1:.4f}, max_abs={add_max:.4f} | "
                    f"REMOVE L1={rem_l1:.4f}, max_abs={rem_max:.4f} | "
                    f"score={d['score']:.4f}\n")

        if ranked:
            top_ev, top_stats = ranked[0]
            # optional dominance callout similar to earlier
            if len(ranked) == 1 or ranked[0][1]["score"] > 1.5 * ranked[1][1]["score"]:
                out += f"  => Highest-impact evidence: {top_ev}.\n"
            else:
                out += f"  => Highest-impact evidence (tie-close): {top_ev}.\n"

        return out, net_after



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

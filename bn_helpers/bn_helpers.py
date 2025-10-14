from pydantic import BaseModel
from bni_netica.bni_utils import findAllDConnectedNodes
from bn_helpers.utils import (output_distribution, ensure_keys, logical_or, \
    logical_and, logical_xor, logical_xnor, fit_noisy_or, fit_noisy_and, fit_additive, _rmse, temporarily_set_findings, \
        names, resolve_state_index, state_names_by_indices, find_minimal_blockers, reduce_to_minimal_blocking_set, \
            is_independent_given, get_path, _state_label)
import json
import re
from typing import List, Tuple, Dict, Any, Optional, Sequence, Union


class BnToolBox():
    # function_name: str

    # XY CONNECT
    def is_XY_dconnected(self, net, from_node, to_node):
      relatedNodes = net.node(from_node).getRelated("d_connected, exclude_self")
      for node in relatedNodes:
        if node.name() == to_node:
          return True
      return False

    def get_d_connected_nodes(self, net, node):
        return names(net.node(node).getRelated("d_connected, exclude_self"))

    # COMMON CAUSE / EFFECT
    def ancestors(self, net, node):
        return names(net.node(node).getRelated("ancestors, exclude_self"))

    def descendants(self, net, node):
        return names(net.node(node).getRelated("descendents, exclude_self"))

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

    def _normalize_evidence_inputs(self, net, evidence: Optional[Union[Dict[str, Any], Sequence[Any], str]]) -> Tuple[List[str], List[str]]:
        """Normalize various evidence input shapes to:
        - node_names: list of node names (for d-separation; values ignored)
        - display_items: list of strings like "B=True" for explanation text

        Accepted forms:
        - {"B": "True", ...}
        - ["B", "C"], ["B=True", "C=0"], mixed lists
        - [{"B": "True"}, {"C": 0}], or a single dict inside the list
        - "B=True" or "B"
        """
        if not evidence:
            return [], []

        node_names: List[str] = []
        display_items: List[str] = []
        seen = set()

        def add_item(name: str, val: Any = None):
            nonlocal node_names, display_items
            if name not in seen:
                seen.add(name)
                node_names.append(name)
            if val is None:
                display_items.append(name)
            else:
                try:
                    # Pretty-label value if it's an index; otherwise use as-is
                    label = _state_label(net, name, val)
                except Exception:
                    label = str(val)
                display_items.append(f"{name}={label}")

        def handle_one(ev: Any):
            if ev is None:
                return
            if isinstance(ev, dict):
                for k, v in ev.items():
                    add_item(str(k), v)
            elif isinstance(ev, str):
                s = ev.strip()
                # Try JSON if it looks like JSON
                if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
                    try:
                        loaded = json.loads(s)
                        handle_one(loaded)
                        return
                    except Exception:
                        pass
                # Try comma-separated list
                if "," in s and "=" not in s:
                    parts = [p.strip() for p in s.split(",") if p.strip()]
                    for p in parts:
                        handle_one(p)
                    return
                # Key=Value
                if "=" in s:
                    k, v = s.split("=", 1)
                    add_item(k.strip(), v.strip())
                else:
                    add_item(s)
            else:
                # Fallback: treat as node name
                add_item(str(ev))

        if isinstance(evidence, dict) or isinstance(evidence, str):
            handle_one(evidence)
        else:
            for ev in evidence:  # type: ignore[assignment]
                handle_one(ev)

        return node_names, display_items

    def _coerce_value(self, v: Any) -> Any:
        """Best-effort coercion of textual values to int/bool where appropriate."""
        if isinstance(v, str):
            s = v.strip()
            if re.fullmatch(r"[-+]?\d+", s):
                try:
                    return int(s)
                except Exception:
                    return s
            low = s.lower()
            if low in {"true", "yes", "t"}:
                return True
            if low in {"false", "no", "f"}:
                return False
            return s
        return v

    def _normalize_evidence_with_values(self, evidence: Optional[Union[Dict[str, Any], Sequence[Any], str]]) -> Dict[str, Any]:
        """Normalize evidence where values are required (for probability queries).
        Accepted forms:
        - dict: {"B": 1} or {"B": "True"}
        - list of dicts: [{"B": 1}, {"C": "high"}]
        - list of strings: ["B=1", "C=True"]
        - single string: "B=1" or JSON like '{"B":1}' or '["B=1","C=True"]' or "B=1, C=True"
        Bare node names without values (e.g., ["B"]) are rejected.
        """
        if not evidence:
            return {}

        out: Dict[str, Any] = {}

        def add_pair(k: Any, v: Any):
            out[str(k)] = self._coerce_value(v)

        def handle_one(ev: Any):
            if ev is None:
                return
            if isinstance(ev, dict):
                for k, v in ev.items():
                    add_pair(k, v)
            elif isinstance(ev, str):
                s = ev.strip()
                # Try JSON
                if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
                    try:
                        loaded = json.loads(s)
                        handle_one(loaded)
                        return
                    except Exception:
                        pass
                # Comma-separated key=value entries
                if "," in s and "=" in s:
                    parts = [p.strip() for p in s.split(",") if p.strip()]
                    for p in parts:
                        if "=" not in p:
                            raise ValueError(f"Evidence entry '{p}' missing value; expected key=value")
                        k, v = p.split("=", 1)
                        add_pair(k.strip(), v.strip())
                    return
                if "=" in s:
                    k, v = s.split("=", 1)
                    add_pair(k.strip(), v.strip())
                else:
                    # No value provided -> reject
                    raise ValueError(f"Evidence '{s}' missing value; expected key=value")
            else:
                # Unknown structure -> reject
                raise ValueError(f"Unsupported evidence item type: {type(ev).__name__}")

        if isinstance(evidence, dict) or isinstance(evidence, str):
            handle_one(evidence)
        else:
            for ev in evidence:  # type: ignore[assignment]
                handle_one(ev)

        return out

    def does_evidence_change_dependency_XY(self, net, X: str, Y: str, evidence: Optional[Union[Dict[str, Any], Sequence[Any], str]]) -> Tuple[bool, Dict[str, Any]]:
        node_names, display_items = self._normalize_evidence_inputs(net, evidence)
        if not node_names:
            with temporarily_set_findings(net, {}):
                dep = self.is_XY_dconnected(net, X, Y)
            return False, {
                "before": dep,
                "after": dep,
                "conditioned_on": [],
                "sequential": []
            }

        # BEFORE: none of the evidence is observed
        with temporarily_set_findings(net, {z: None for z in node_names}):
            dep_before = self.is_XY_dconnected(net, X, Y)

        # AFTER: all evidence nodes observed (value doesn't matter for d-sep)
        with temporarily_set_findings(net, {z: 0 for z in node_names}):
            dep_after = self.is_XY_dconnected(net, X, Y)

        changed = dep_before != dep_after

        # Sequential trace (adding evidence one by one)
        sequential_trace = []
        partial = {}
        for z in node_names:
            partial[z] = 0
            with temporarily_set_findings(net, partial):
                conn = self.is_XY_dconnected(net, X, Y)
            sequential_trace.append({"added": z, "connected": conn})

        return changed, {
            "before": dep_before,
            "after": dep_after,
            "conditioned_on": display_items if display_items else node_names,
            "sequential": sequential_trace # how each evidence changes the dependency
        }

    # # EVIDENCES BLOCK XY
    def evidences_block_XY(self, net, X, Y):
        # node that will block the dependency between X and Y when observed
        ans = []
        evidences = findAllDConnectedNodes(net, X, Y)
        for e in evidences:
            e_name = e.name()
            if e_name != X and e_name != Y and self.does_evidence_change_dependency_XY(net, X, Y, [e_name])[0]:
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
        # Normalize if evidence is not already a dict
        if evidence is None:
            evidence = {}
        elif not isinstance(evidence, dict):
            evidence = self._normalize_evidence_with_values(evidence)
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


    def get_highest_impact_evidence(self, net, X, evidence=None, order=None, *, threshold=0.05, tol=1e-8):
        """
        Analyze per-evidence impact on X using sequential ADD and sequential REMOVE,
        then output a ranked summary and a conclusion (like `get_prob_X_given`).

        Returns:
            out_str, net_after, structured_data
        """
        if evidence is None:
            evidence = {}
        elif not isinstance(evidence, dict):
            # Only accept value-bearing inputs here
            try:
                evidence = self._normalize_evidence_with_values(evidence)
            except ValueError:
                evidence = {}
        if not evidence:
            # Derive candidate evidences from nodes d-connected to X
            candidates = self.get_d_connected_nodes(net, X)
            if candidates:
                # Default each candidate evidence to its first state (index 0)
                evidence = {ev: 0 for ev in candidates}
            else:
                return f"P({X} | ∅):\n  No evidence provided; nothing to analyze.\n", net, {}

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
        impacts = {}  # ev -> {"add": {...}, "remove": {...}, "score": float} 
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
            impacts.setdefault(ev, {})["remove"] = {"diffs": diffs_rem, "l1": rem_l1, "max_abs": rem_max}

        # scoring & report
        # score = max(add.L1, rem.L1)
        for ev, d in impacts.items():
            add_l1 = d.get("add", {}).get("l1", 0.0)
            rem_l1 = d.get("remove", {}).get("l1", 0.0)
            d["score"] = max(add_l1, rem_l1)

        ranked = sorted(impacts.items(), key=lambda kv: kv[1]["score"], reverse=True)

        out += "\nEvidence impact (sequential add/remove):\n"
        for ev, d in ranked:
            add_l1 = d["add"]["l1"] if "add" in d else 0.0
            rem_l1 = d["remove"]["l1"] if "remove" in d else 0.0
            add_max = d["add"]["max_abs"] if "add" in d else 0.0
            rem_max = d["remove"]["max_abs"] if "remove" in d else 0.0
            out += (f"  - {ev}: "
                    f"ADD  L1={add_l1:.4f}, max_abs={add_max:.4f} | "
                    f"REMOVE L1={rem_l1:.4f}, max_abs={rem_max:.4f} | "
                    f"score={d['score']:.4f}\n")

        # Prepare structured data for fake answer generation
        structured_data = {
            "X": X,
            "evidence": evidence,
            "original_distribution": {node.state(i).name(): original[i] for i in range(len(original))},
            "new_distribution": {node.state(i).name(): new[i] for i in range(len(new))},
            "conclusion": {
                "max_change": max_change,
                "changes": [{"state": node.state(i).name(), "change": diffs[i], "abs_change": abs(diffs[i])} 
                           for i in range(len(diffs)) if abs(diffs[i]) > threshold],
                "minimal_update": max_change <= threshold,
                "no_change": max_change <= tol
            },
            "impacts": impacts,
            "ranked": ranked,
            "highest_impact_evidence": ranked[0][0] if ranked else None,
            "is_tie_close": len(ranked) > 1 and ranked[0][1]["score"] <= 1.5 * ranked[1][1]["score"] if len(ranked) > 1 else False
        }

        if ranked:
            top_ev, top_stats = ranked[0]
            # optional dominance callout similar to earlier
            if len(ranked) == 1 or ranked[0][1]["score"] > 1.5 * ranked[1][1]["score"]:
                out += f"  => Highest-impact evidence: {top_ev}.\n"
            else:
                out += f"  => Highest-impact evidence (tie-close): {top_ev}.\n"

        return out, net_after, structured_data



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

### EXPLANERS
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

    def get_explain_prob_X_given(self, net, X, evidence=None, *, threshold=0.05, tol=1e-8):
        """
        Get structured data and formatted explanation for P(X | evidence).
        Returns: (formatted_answer, structured_data)
        """
        def _create_probability_distributions(node, original, new):
            """Create probability distribution data and text representations."""
            # Create probability distribution data
            new_dist_data = {node.state(i).name(): p for i, p in enumerate(new)}
            original_dist_data = {node.state(i).name(): p for i, p in enumerate(original)}
            
            # Create probability distribution strings
            new_dist_text = "\n".join(f"  P({node.name()}={node.state(i).name()}) = {p:.4f}" for i, p in enumerate(new))
            original_dist_text = "\n".join(f"  P({node.name()}={node.state(i).name()}) = {p:.4f}" for i, p in enumerate(original))
            
            return new_dist_data, original_dist_data, new_dist_text, original_dist_text

        def _get_conclusion_data(node, original, new, threshold=0.05, tol=1e-8):
            """Get conclusion data and text based on probability changes."""
            diffs = [n - o for n, o in zip(new, original)]
            max_change = max((abs(d) for d in diffs), default=0.0)

            conclusion_data = {
                "max_change": max_change,
                "changes": [],
                "minimal_update": max_change <= threshold
            }
            
            if max_change <= tol:
                conclusion_text = "No change detected — the updated beliefs are identical to the original."
            else:
                conclusion_parts = []
                for i, d in enumerate(diffs):
                    if abs(d) > threshold:
                        sname = node.state(i).name()
                        change_text = f"Belief in '{sname}' {'increased' if d > 0 else 'decreased'} by {abs(d):.4f}"
                        conclusion_parts.append(change_text)
                        conclusion_data["changes"].append({
                            "state": sname,
                            "change": d,
                            "abs_change": abs(d)
                        })
                
                if max_change <= threshold:
                    conclusion_parts.append("Overall, the update is minimal (all changes ≤ threshold).")
                else:
                    conclusion_parts.append(f"Largest overall per-state shift: {max_change:.4f}.")
                
                conclusion_text = "\n".join(f"  {part}" for part in conclusion_parts)

            return conclusion_data, conclusion_text

        def _get_evidence_impact_data(impacts):
            """Get evidence impact data and text."""
            impact_data = {}
            impact_text = ""
            
            if impacts:
                ranked = sorted(impacts.items(), key=lambda kv: kv[1]["l1"], reverse=True)
                impact_lines = []
                for ev, stats in ranked:
                    impact_lines.append(f"  - {ev}: L1={stats['l1']:.4f}, max_abs={stats['max_abs']:.4f}")
                    impact_data[ev] = stats
                
                if ranked and (len(ranked) == 1 or ranked[0][1]["l1"] > 1.5 * (ranked[1][1]["l1"] if len(ranked) > 1 else 0)):
                    impact_lines.append(f"  => Most influential evidence: {ranked[0][0]} (by L1 contribution).")
                    impact_data["most_influential"] = ranked[0][0]
                
                impact_text = "\nEvidence impact (leave-one-out):\n" + "\n".join(impact_lines)

            return impact_data, impact_text

        def _format_probability_answer(X, evidence, net, new_dist_text, original_dist_text, conclusion_text, impact_text):
            """Format the final probability answer string."""
            cond = ", ".join(f"{k}={_state_label(net, k, v)}" for k, v in evidence.items()) or "∅"
            answer = (
                f"P({X} | {cond}):\n"
                f"{new_dist_text}\n"
                f"\nOriginal distribution:\n"
                f"{original_dist_text}\n"
                f"\nConclusion:\n"
                f"{conclusion_text}"
                f"{impact_text}"
            )
            return answer

        if evidence is None:
            evidence = {}
        elif not isinstance(evidence, dict):
            evidence = self._normalize_evidence_with_values(evidence)

        original, new, net_after, impacts = self.prob_X_given(net, X, evidence)
        node = net_after.node(X)
        
        new_dist_data, original_dist_data, new_dist_text, original_dist_text = _create_probability_distributions(node, original, new)
        
        conclusion_data, conclusion_text = _get_conclusion_data(node, original, new, threshold, tol)
        
        impact_data, impact_text = _get_evidence_impact_data(impacts)

        structured_data = {
            "X": X,
            "evidence": evidence,
            "new_distribution": new_dist_data,
            "original_distribution": original_dist_data,
            "conclusion": conclusion_data,
            "evidence_impact": impact_data,
            "new_dist_text": new_dist_text,
            "original_dist_text": original_dist_text,
            "conclusion_text": conclusion_text,
            "impact_text": impact_text
        }

        answer = _format_probability_answer(X, evidence, net, new_dist_text, original_dist_text, conclusion_text, impact_text)

        return answer, structured_data

    def get_explain_evidence_change_dependency_XY(self, net, node1, node2, evidence):
        _, display_items = self._normalize_evidence_inputs(net, evidence)
        changed, details = self.does_evidence_change_dependency_XY(net, node1, node2, evidence)

        def _conn_label(b: bool) -> str:
                return "d-connected" if b else "d-separated"

        def _fmt_ev_list(evs: List[str]) -> str:
            if not evs: return "∅"
            return ", ".join(evs)

        before = _conn_label(details["before"])
        after  = _conn_label(details["after"])
        ev_str = _fmt_ev_list(display_items)

        # Find first step (if any) where connectivity flips relative to BEFORE
        flip_note = ""
        for step in details.get("sequential", []):
            if step["connected"] != details["before"]:
                flip_note = f" The relationship first flips after conditioning on {step['added']}."
                break

        if not evidence:
            raw_template = (
                f"No evidence provided. Relationship between {node1} and {node2} is {before} "
                f"with no conditioning."
            )
            answer = raw_template.format(node1=node1, node2=node2, before=before)
            return answer, raw_template

        if changed:
            raw_template = (
                f"Yes - conditioning on {{ev_str}} changes the dependency between {node1} and {node2}. "
                f"Before observing {{ev_str}}, they were {before}. After observing all evidence, they are {after}."
                f"{flip_note}"
            )
            answer = raw_template.format(node1=node1, node2=node2, ev_str=ev_str, before=before, after=after)

        else:
            raw_template = (
                f"No - conditioning on {{ev_str}} does not change the dependency between {node1} and {node2}. "
                f"Before observing {{ev_str}}, they were {before}. After observing all evidence, they remain {after}."
            )
            answer = raw_template.format(node1=node1, node2=node2, ev_str=ev_str, before=before, after=after)
        
        if details.get("sequential"):
            steps = "; ".join(
                f"+{s['added']} => {_conn_label(s['connected'])}"
                for s in details["sequential"]
            )
            answer += f" Sequence: {steps}."
        
        return answer, raw_template
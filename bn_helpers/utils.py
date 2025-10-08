import math
import numpy as np
from contextlib import contextmanager
from itertools import product
from bni_netica import *

def names(nodes):
  return {n.name() for n in nodes}

def set_findings(net, findings_dict):
    for k, v in findings_dict.items():
        node = net.node(k)
        if v is None:
            node.retractFindings()
        else:
            node.finding(v)
    net.update()

def _state_label(net, node_name, s):
    return net.node(node_name).state(s).name() if isinstance(s, int) else str(s)

@contextmanager
def temporarily_set_findings(net, findings_dict):
    """
    findings_dict: {node_name: state_name_or_index | None}
    If value is None, the node's finding is retracted.
    """
    saved = net.findings()
    try:
        for k, v in findings_dict.items():
            node = net.node(k)
            if v is None:
                node.retractFindings()
            else:
                node.finding(v)  # int or state name
        net.update()
        yield
    finally:
        net.retractFindings()
        for k, v in (saved or {}).items():
            net.node(k).finding(v)
        net.update()

def output_distribution(node, original_beliefs, new_beliefs, impacts, threshold=0.05, tol=1e-8):
    """Pretty print distributions and conclude about the gap between original and new beliefs."""
    output = ""

    for i, p in enumerate(new_beliefs):
        state = node.state(i)
        output += f"  P({node.name()}={state.name()}) = {p:.4f}\n"

    output += "\nOriginal distribution:\n"
    for i, p in enumerate(original_beliefs):
        state = node.state(i)
        output += f"  P({node.name()}={state.name()}) = {p:.4f}\n"

    # differences
    diffs = [new - old for new, old in zip(new_beliefs, original_beliefs)]
    max_change = max(abs(d) for d in diffs)

    # conclusion
    output += "\nConclusion:\n"
    if max_change <= tol:  # no change (within floating point tolerance)
        output += "  No change detected — the updated beliefs are identical to the original.\n"
    else:
        for i, d in enumerate(diffs):
            state = node.state(i)
            if abs(d) > threshold:
                trend = "increased" if d > 0 else "decreased"
                output += f"  Belief in state '{state.name()}' has {trend} by {abs(d):.4f}\n"

        if max_change <= threshold:
            output += "  Overall, the updated beliefs are very close to the original (minimal change).\n"
        else:
            output += f"  The most significant shift was {max_change:.4f} across states.\n"

    return output

# helpers
def to01(x):
    if isinstance(x, str):
        return 1 if x.lower().startswith('t') else 0
    return int(bool(x))

def _rmse(obs, pred):
    return math.sqrt(sum((obs[k]-pred[k])**2 for k in obs)/len(obs))

def clip01(x): 
    return 0.0 if x < 0 else 1.0 if x > 1 else float(x)

def ensure_keys(d):
    # normalize keys to (a,b) with 0/1
    norm = {}
    for k,v in d.items():
        a,b = k
        norm[(to01(a), to01(b))] = float(v)
    # fill missing (shouldn't happen)
    for a,b in product([0,1],[0,1]):
        norm.setdefault((a,b), 0.0)
    return norm

# prototypes (deterministic)
def logical_or():
    return {(1,1):1.0,(1,0):1.0,(0,1):1.0,(0,0):0.0}

def logical_and():
    return {(1,1):1.0,(1,0):0.0,(0,1):0.0,(0,0):0.0}

def logical_xor():
    # 1 iff exactly one true
    return {(1,1):0.0,(1,0):1.0,(0,1):1.0,(0,0):0.0}

def logical_xnor():
    # 1 iff both equal (often mislabeled "XOR" in sheets)
    return {(1,1):1.0,(1,0):0.0,(0,1):0.0,(0,0):1.0}

# parametric: Noisy-OR (with leak)
# P11 = 1 - (1-l)(1-pA)(1-pB);  P10 = 1-(1-l)(1-pA); P01 = 1-(1-l)(1-pB); P00=l
def fit_noisy_or(obs):
    l = obs[(0,0)]
    denom = 1 - l
    if denom <= 0:  # degenerate: l>=1 -> always 1
        pred = {(a,b): 1.0 for a,b in obs}
        params = {"leak": l, "pA": 1.0, "pB": 1.0}
        return _rmse(obs, pred), pred, params
    pA = 1.0 - (1.0 - obs[(1,0)]) / denom
    pB = 1.0 - (1.0 - obs[(0,1)]) / denom
    # constrain to [0,1] so we don't blow up badly
    pA, pB = clip01(pA), clip01(pB)
    def f(a,b):
        return 1.0 - (1.0 - l) * (1.0 - pA)**a * (1.0 - pB)**b
    pred = {(a,b): f(a,b) for a,b in obs}
    return _rmse(obs, pred), pred, {"leak": l, "pA": pA, "pB": pB}

# parametric: "Noisy-AND" (simple leaky multiplicative)
# P(a,b) = l * (qA^a) * (qB^b); so P00=l, P10=l*qA, P01=l*qB, P11=l*qA*qB
def fit_noisy_and(obs):
    l = obs[(0,0)]
    if l <= 0:
        # degenerate: l==0 -> model predicts zeros except maybe 0/0 division
        pred = {(a,b): 0.0 for a,b in obs}
        return _rmse(obs, pred), pred, {"leak": l, "qA": 0.0, "qB": 0.0}
    qA = obs[(1,0)] / l
    qB = obs[(0,1)] / l
    qA, qB = clip01(qA), clip01(qB)
    def f(a,b):
        return l * (qA**a) * (qB**b)
    pred = {(a,b): f(a,b) for a,b in obs}
    return _rmse(obs, pred), pred, {"leak": l, "qA": qA, "qB": qB}

# parametric: Additive (linear, clipped)
# P(a,b) = clip(bias + wA*a + wB*b)
def fit_additive(obs):
    # Solve least squares for [bias, wA, wB]
    # X rows: (1, a, b)    
    X = []
    y = []
    ordering = [(0,0),(1,0),(0,1),(1,1)]
    for a,b in ordering:
        X.append([1.0, float(a), float(b)])
        y.append(obs[(a,b)])
    X = np.asarray(X); y = np.asarray(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)  # [bias, wA, wB]
    def f(a,b):
        return clip01(beta[0] + beta[1]*a + beta[2]*b)
    pred = {(a,b): f(a,b) for a,b in obs}
    return _rmse(obs, pred), pred, {"bias": float(beta[0]), "wA": float(beta[1]), "wB": float(beta[2])}

def relabel_to01(tbl, parent_names, positives=None):
    """
    Map parent states to {0,1}.
    If positives not given, pick first state of each parent as "1".
    """
    new = {}
    for key, v in tbl.items():
        mapped = []
        for p, state in zip(parent_names, key):
            if positives and p in positives:
                mapped.append(1 if state == positives[p] else 0)
            else:
                # auto: treat last state of this parent as positive
                mapped.append(1 if state == key[0] else 0)
        new[tuple(mapped)] = float(v)
    return new


def state_names_by_indices(node, idxs):
    names = []
    for i in idxs:
        st = node.state(i)
        if st is None:
            raise ValueError(f"Bad state index {i} for node '{node.name()}'")
        names.append(st.name())
    return names

def resolve_state_index(node, spec, use_title=False):
    """
    Resolve a state spec (int or str) to an index for a given node.
    If use_title=True, string is matched against state titles first, then names.
    """
    if isinstance(spec, int):
        return spec
    if use_title:
        st = node.stateByTitle(spec) or node.state(spec)
    else:
        st = node.state(spec) or node.stateByTitle(spec)
    if st is None:
        raise ValueError(f"Unknown state '{spec}' for node '{node.name()}'")
    return st.stateNum

from bni_netica import *
from collections import deque

def findAllDConnectedNodes(bn, source_node, dest_node, o=None):
    o = dict(o or {})
    o.setdefault("arcTraversal", False)
    o["arcs"] = o["arcTraversal"] or o.get("arcs", False)
    o.setdefault("noSourceParents", False)
    o.setdefault("noSourceChildren", False)

    if isinstance(source_node, str):
        source_node = bn.node(source_node)
    if isinstance(dest_node, str):
        dest_node = bn.node(dest_node)

    PARENTS = 0b01
    CHILDREN = 0b10
    PARENTSANDCHILDREN = 0b11

    def has_loop(path, end):
        end_node = end[0]
        for n, _d in reversed(path):
            if end_node is n:
                return True
        return False

    # Lightweight DAG view (Netica → DagNode)
    class DagNode:
        __slots__ = ("id", "parents", "children", "intervene", "_src")
        def __init__(self, n):
            self.id = n.name()
            self.parents = list(n.parents())
            self.children = list(n.children())
            self.intervene = False
            self._src = n
        def hasEvidence(self):
            return self._src.finding() is not None
        def getDescendants(self):
            descendants = set()
            to_visit = deque([self])
            while to_visit:
                nx = to_visit.popleft()
                for c in nx.children:
                    if c not in descendants:
                        descendants.add(c)
                        to_visit.append(c)
            return list(descendants)
        def isParent(self, toNode):
            return toNode in self.children

    def make_dag(bn):
        # Sort nodes for stable mapping
        dag = {n.name(): DagNode(n) for n in sorted(bn.nodes(), key=lambda x: x.name())}
        # Rewire parents/children to DagNode refs and sort for deterministic traversal
        for dn in dag.values():
            dn.parents  = sorted((dag[p.name()] for p in dn.parents),  key=lambda x: x.id)
            dn.children = sorted((dag[c.name()] for c in dn.children), key=lambda x: x.id)
        return dag

    dag = make_dag(bn)
    s = dag[source_node.name()]
    t = dag[dest_node.name()]

    # Precompute whether each node has evidence downstream (count of descendant evidence)
    downstream_ev = {
        n.id: sum(1 for d in n.getDescendants() if d.hasEvidence())
        for n in dag.values()
    }

    # Traversal state
    paths_q = deque([[(s, PARENTSANDCHILDREN)]])  # queue of paths (each path is list[(DagNode, dir_mask)])
    resolved = {}                                 # node_dir_key -> DagNode | False
    depth = {s.id: 0}                             # BFS depth (first-seen distance from source)

    # Deterministic results (lists) with O(1) de-dup sets
    seen_node_ids = set()
    ordered_nodes = []                             # DagNode list in stable, outward-from-source order
    seen_arc_keys = set()
    ordered_arcs = []                              # arcs in discovery order

    def node_dir_key(node_dir):
        return f"{node_dir[0].id}-{node_dir[1]}"

    def add_arc(par, child, swap):
        if o["arcTraversal"]:
            key = (par.id, child.id, 'up' if swap else 'down')
            val = f"{par.id}-{child.id}|{'up' if swap else 'down'}"
        else:
            key = (par.id, child.id)
            val = f"{par.id}-{child.id}"
        if key not in seen_arc_keys:
            seen_arc_keys.add(key)
            ordered_arcs.append(val)

    def resolve_path(path, how_resolved=True):
        prev = None
        for curr in path:
            if prev is not None:
                par, child = prev[0], curr[0]
                swap = child.isParent(par)   # if we moved "up", swap for arc normalization
                if swap:
                    par, child = child, par
                add_arc(par, child, swap)

            if how_resolved:
                dn = curr[0]
                if dn.id not in seen_node_ids:
                    seen_node_ids.add(dn.id)
                    ordered_nodes.append(dn)
            resolved[node_dir_key(curr)] = curr[0] if how_resolved else False
            prev = curr

    def node_dir_is_resolved(node_dir):
        return bool(resolved.get(node_dir_key(node_dir)))

    def enqueue(next_paths, parent_dn):
        """Enqueue paths in deterministic order and record depth on first sight."""
        parent_depth = depth[parent_dn.id]
        for p in next_paths:
            dn = p[-1][0]
            if dn.id not in depth:
                depth[dn.id] = parent_depth + 1
            paths_q.append(p)

    # Main traversal
    while paths_q:
        current_path = paths_q.popleft()
        if node_dir_is_resolved(current_path[-1]):
            resolve_path(current_path)
            continue

        current_node, dirmask = current_path[-1]
        if current_node is t:
            resolve_path(current_path)
        else:
            check_parents  = (not current_node.intervene) and (not o["noSourceParents"]  or current_node is not s)
            check_children = (not o["noSourceChildren"] or current_node is not s)

            if check_parents and (dirmask & PARENTS) == PARENTS:
                next_paths = [
                    current_path + [(p, PARENTSANDCHILDREN)]
                    for p in current_node.parents
                    if (not p.hasEvidence()) and (not has_loop(current_path, (p, PARENTSANDCHILDREN)))
                ]
                enqueue(next_paths, current_node)

            if check_children and (dirmask & CHILDREN) == CHILDREN:
                def child_dir(c):
                    # follow your rule: if child has evidence, only go up to parents; else allow children if no downstream evidence
                    return PARENTS if c.hasEvidence() else (PARENTSANDCHILDREN if downstream_ev[c.id] else CHILDREN)
                next_paths = [
                    current_path + [(c, child_dir(c))]
                    for c in current_node.children
                    if not has_loop(current_path, (c, child_dir(c)))
                ]
                enqueue(next_paths, current_node)

    # Output
    if o["arcs"]:
        return ordered_arcs

    ordered_nodes.sort(key=lambda dn: (depth.get(dn.id, 10**9), dn.id))
    return [bn.node(dn.id) for dn in ordered_nodes]

def get_path(net, source_node, dest_node):
    nodes = findAllDConnectedNodes(net, source_node, dest_node)
    path = [n.name() for n in nodes]
    return path


# helper to get minmal blockers
def is_independent_given(net, X, Y, observed_names, is_XY_dconnected_fn):
    """
    True iff X ⟂ Y given original evidence plus `observed_names`.
    Uses your CM; relies on None=retract behavior supported by the CM.
    """
    if not observed_names:
        # No additional conditioning; just test current state
        return not is_XY_dconnected_fn(net, X, Y)

    # Observe all in observed_names (use state 0 arbitrarily for d-sep)
    with temporarily_set_findings(net, {nm: 0 for nm in observed_names}):
        return not is_XY_dconnected_fn(net, X, Y)

def reduce_to_minimal_blocking_set(net, X, Y, S, is_XY_dconnected_fn):
    """
    Given a blocking set S (iterable of names), drop redundant nodes until minimal.
    """
    S = list(dict.fromkeys(S))  # de-dupe, keep order
    if not is_independent_given(net, X, Y, set(S), is_XY_dconnected_fn):
        return S  # not blocking; return as-is (or raise, if you prefer)

    changed = True
    while changed:
        changed = False
        for i in range(len(S) - 1, -1, -1):  # reverse -> stable
            trial = S[:i] + S[i+1:]
            if is_independent_given(net, X, Y, set(trial), is_XY_dconnected_fn):
                S.pop(i)
                changed = True
    return S

def find_minimal_blockers(net, X, Y, is_XY_dconnected_fn, consider='connected', max_k=2):
    """
    Search small sets (|S|≤max_k) that block X–Y; return one minimal set if found.
    Consider:
    - 'connected': only nodes currently on active paths (fast, typical)
    - 'all': all nodes except X,Y
    Increase max_k cautiously (3+ can be combinatorial).
    """
    import itertools

    # If already independent, empty set is a separator
    if not is_XY_dconnected_fn(net, X, Y):
        return []

    if consider == 'connected':
        candidate_names = sorted({n.name() for n in findAllDConnectedNodes(net, X, Y)} - {X, Y})
    else:
        candidate_names = sorted({n.name() for n in net.nodes()} - {X, Y})

    for k in range(1, max_k + 1):
        for combo in itertools.combinations(candidate_names, k):
            if is_independent_given(net, X, Y, set(combo), is_XY_dconnected_fn):
                return reduce_to_minimal_blocking_set(net, X, Y, list(combo), is_XY_dconnected_fn)
    return []
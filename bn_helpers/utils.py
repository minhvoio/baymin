import math
import numpy as np
from contextlib import contextmanager

def names(self, nodes):
  return {n.name() for n in nodes}

@contextmanager # release resource automatically after using it
def temporarily_set_findings(net, findings_dict):
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

def output_distribution(beliefs, node):
    """Pretty print a distribution (list of probabilities) with state names."""
    output = ""
    for i, p in enumerate(beliefs):
        state = node.state(i)
        output += f"  P({node.name()}={state.name()}) = {p:.4f}\n"

    return output

def prob_X(net, X=None):
    """
    Returns P(X), net_after_observation
    """
    node_X = net.node(X)
    return node_X.beliefs(), net

def prob_X_given_Y(net, X=None, Y=None, y_state="Yes"):
    """
    Returns P(X | Y = y_state), net_after_observation
    y_state can be state names (str) or indices (int).
    """
    
    with temporarily_set_findings(net, {Y: y_state}):
        node_X = net.node(X)
        # beliefs() is P(X | current findings)
        return node_X.beliefs(), net
    
def prob_X_given_YZ(net, X=None, Y=None, y_state="Yes", Z=None, z_state="Yes"):
    """
    Returns P(X = x_state | Y = y_state, Z = z_state), net_after_observation
    x_state, y_state and z_state can be state names (str) or indices (int).
    """
    
    with temporarily_set_findings(net, {Y: y_state, Z: z_state}):
        node_X = net.node(X)
        # beliefs() is P(X | current findings)
        return node_X.beliefs(), net

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
import os
import random
import string
from bni_netica.bni_netica import Net


def _dirichlet(alphas):
    """
    Generate a random sample from a Dirichlet distribution.
    
    The Dirichlet distribution is used to generate random probability distributions
    that sum to 1.0, which is perfect for creating conditional probability tables (CPTs)
    in Bayesian networks. Each node's CPT rows are generated using this function to
    ensure they represent valid probability distributions.
    
    Args:
        alphas (list): Concentration parameters for the Dirichlet distribution.
                      Each element corresponds to one state of the node.
                      - alpha > 1: Creates right-skewed distributions (concentrated near 1)
                      - alpha < 1: Creates left-skewed distributions (concentrated near 0) 
                      - alpha = 1: Creates uniform distributions (equal probabilities)
                      - alpha = 0.5: Creates sparse distributions (some states very likely, others unlikely)
    
    Returns:
        list: A probability distribution (list of floats that sum to 1.0)
              representing the conditional probabilities for one row of a CPT.
    
    Example:
        For a binary node with states ["False", "True"]:
        - _dirichlet([1, 1]) might return [0.3, 0.7] (uniform-ish)
        - _dirichlet([0.5, 0.5]) might return [0.9, 0.1] (sparse)
        - _dirichlet([2, 2]) might return [0.6, 0.4] (concentrated)
    """
    # Generate gamma-distributed values for each state
    vals = [random.gammavariate(a, 1.0) for a in alphas]
    # Normalize to create a valid probability distribution
    s = sum(vals) or 1.0
    return [v / s for v in vals]

def _make_names(k, seed=None):
    if seed is not None:
        random.seed(seed)
    base = list(string.ascii_uppercase)
    out, idx = [], 0
    while len(out) < k:
        suffix = "" if idx == 0 else str(idx)
        for ch in base:
            out.append(f"{ch}{suffix}")
            if len(out) == k:
                break
        idx += 1
    random.shuffle(out)
    return out[:k]

def build_random_bn(
    n_nodes=8,
    name="RandomNet",
    seed=None,
    states=("False","True"),
    cpt_mode="random",
    dirichlet_alpha=1.0, # cpt params set up, 1 is uniform
    avg_edges_per_node=1.2,   
    max_in_degree=2,          
    sprinkle_motifs=5, 
    save_path=None
):
    if seed is not None:
        random.seed(seed)

    node_names = _make_names(n_nodes)
    net = Net()
    net.name(name)
    for nm in node_names:
        net.addNode(nm, states=list(states))

    # random topological order
    topo = node_names[:]
    random.shuffle(topo)
    pos = {nm: i for i, nm in enumerate(topo)}

    edges = set()
    indeg = {nm: 0 for nm in node_names}

    # Bernoulli sampling over forward pairs; expected edges ≈ avg_edges_per_node * n_nodes
    # p is per-forward-pair probability
    p = max(0.0, min(1.0, float(avg_edges_per_node) / max(1, n_nodes - 1)))

    forward_pairs = []
    for i, u in enumerate(topo):
        for v in topo[i+1:]:
            forward_pairs.append((u, v))

    # Shuffle to avoid any positional bias
    random.shuffle(forward_pairs)

    for u, v in forward_pairs:
        if random.random() > p:
            continue
        if indeg[v] >= max_in_degree:
            continue
        if (u, v) in edges:
            continue
        edges.add((u, v))
        indeg[v] += 1

    # Sprinkle a few motifs (very limited) AFTER the Bernoulli pass
    def sample_triple():
        i, j, k = sorted(random.sample(range(n_nodes), 3))
        return topo[i], topo[j], topo[k]

    for _ in range(max(0, sprinkle_motifs)):
        A, B, C = sample_triple()  # A < B < C
        motif = random.choice(["v", "chain", "cc"])
        if motif == "v":
            # A -> C <- B
            if indeg[C] + 2 <= max_in_degree:
                if (A, C) not in edges: edges.add((A, C)); indeg[C] += 1
                if (B, C) not in edges: edges.add((B, C)); indeg[C] += 1
        elif motif == "chain":
            # A -> B -> C
            if indeg[B] < max_in_degree and (A, B) not in edges:
                edges.add((A, B)); indeg[B] += 1
            if indeg[C] < max_in_degree and (B, C) not in edges:
                edges.add((B, C)); indeg[C] += 1
        else:  # common cause
            # A -> B, A -> C
            if indeg[B] < max_in_degree and (A, B) not in edges:
                edges.add((A, B)); indeg[B] += 1
            if indeg[C] < max_in_degree and (A, C) not in edges:
                edges.add((A, C)); indeg[C] += 1

    # Ensure no isolates (at most 1 edge per isolated node)
    for u in node_names:
        has_edge = any(u == x or u == y for (x, y) in edges)
        if not has_edge:
            # Try to connect locally in topo order
            i = pos[u]
            # prefer short-range edges (locality makes graphs look more “natural”)
            neighbors = []
            if i > 0: neighbors.append(topo[i-1])
            if i < n_nodes - 1: neighbors.append(topo[i+1])
            random.shuffle(neighbors)
            hooked = False
            for v in neighbors:
                if pos[u] < pos[v] and (u, v) not in edges and indeg[v] < max_in_degree:
                    edges.add((u, v)); indeg[v] += 1; hooked = True; break
                if pos[v] < pos[u] and (v, u) not in edges and indeg[u] < max_in_degree:
                    edges.add((v, u)); indeg[u] += 1; hooked = True; break
            if not hooked:
                # fallback: try any forward/backward respecting in-degree cap
                cand = node_names[:]
                random.shuffle(cand)
                for v in cand:
                    if v == u: continue
                    if pos[u] < pos[v] and (u, v) not in edges and indeg[v] < max_in_degree:
                        edges.add((u, v)); indeg[v] += 1; break
                    if pos[v] < pos[u] and (v, u) not in edges and indeg[u] < max_in_degree:
                        edges.add((v, u)); indeg[u] += 1; break

    # Materialize edges
    for (u, v) in edges:
        net.node(u).addChildren([v])

    # CPTs
    for n in net.nodes():
        s = n.numberStates()
        rows = 1
        for pnode in n.parents():
            rows *= pnode.numberStates()
        cpt_rows = []
        for _ in range(rows):
            if cpt_mode == "uniform":
                cpt_rows.append([1.0 / s] * s)
            else:
                cpt_rows.append(_dirichlet([dirichlet_alpha] * s))
        n.cpt(cpt_rows)
        n.experience(1)

    net.compile()

    if save_path:
        fname = save_path if save_path.lower().endswith(".dne") else os.path.join(save_path, f"{name}.dne")
        net.write(fname)

    return net

import random 
def pick_two_random_nodes(net):
    nodes = net.nodes()
    if len(nodes) < 2:
        return None, None
    node1, node2 = random.sample(nodes, 2)
    return node1.name(), node2.name()

def fake_random_nodes(net, real_nodes, num_node_keep, num_node_output, exclude=None, min_output_when_zero=0):
    """Return num_node_output node names where num_node_keep are kept from real_nodes.

    Enhancements:
    - exclude: iterable of node names to be excluded from consideration (both kept and fakes)
    - min_output_when_zero: if num_node_output <= 0, produce this many nodes instead of returning []
    """
    exclude_set = set(exclude or [])

    # Determine effective required output size
    effective_output = num_node_output if num_node_output and num_node_output > 0 else int(min_output_when_zero or 0)
    if effective_output <= 0:
        return []

    nodes = list(net.nodes())
    if len(nodes) < effective_output:
        return []

    node_names = [node.name() for node in nodes]
    # Remove excluded nodes from universe
    node_names = [n for n in node_names if n not in exclude_set]

    real_node_set = set(real_nodes)
    # Available real nodes must exist in net and not be excluded
    available_real = [name for name in real_nodes if name in node_names and name not in exclude_set]

    if num_node_keep > effective_output:
        raise ValueError("num_node_keep cannot exceed num_node_output.")

    if len(available_real) < num_node_keep:
        return []

    keep_nodes = random.sample(available_real, num_node_keep) if num_node_keep else []

    # Candidates exclude both real_nodes and excluded
    candidates = [name for name in node_names if name not in real_node_set]
    needed_fake = effective_output - num_node_keep

    if needed_fake > len(candidates):
        return []

    new_nodes = random.sample(candidates, needed_fake) if needed_fake else []

    result = keep_nodes + new_nodes
    random.shuffle(result)
    return result

def generate_fake_nodes_for_relation(net, truth_nodes, node1, node2, *, exclude_extra=None):
    """Generate a fake list of node names for quiz distractors.

    Rules:
    - desired_len = len(truth_nodes)
    - If desired_len >= 2: use truth_nodes as real_nodes, keep 1, output desired_len
    - If desired_len < 2: use [node1, node2] as real_nodes, keep 0, output desired_len;
      when desired_len == 0, synthesize 2 nodes.
    - Always exclude node1/node2 and any exclude_extra.
    - Guaranteed to return a non-empty list (length desired_len or 2 when zero),
      unless the network has insufficient candidates.
    """
    desired_len = len(truth_nodes or [])
    exclude = list(set((exclude_extra or []) + [node1, node2]))

    if desired_len >= 2:
        out = fake_random_nodes(
            net,
            list(truth_nodes),
            num_node_keep=1,
            num_node_output=desired_len,
            exclude=exclude,
        )
        if out:
            return out
    else:
        out = fake_random_nodes(
            net,
            [node1, node2],
            num_node_keep=0,
            num_node_output=desired_len,
            exclude=exclude,
            min_output_when_zero=2,
        )
        if out:
            return out

    # Fallback: sample from all nodes excluding real and excluded
    nodes = [n.name() for n in net.nodes()]
    pool = [n for n in nodes if n not in set(truth_nodes or []) and n not in set(exclude)]
    needed = desired_len if desired_len > 0 else 2
    if len(pool) >= needed:
        import random as _r
        return _r.sample(pool, needed)
    return pool[:needed]

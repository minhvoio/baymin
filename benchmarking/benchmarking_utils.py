import random 
def pickTwoRandomNodes(net):
    nodes = net.nodes()
    if len(nodes) < 2:
        return None, None
    node1, node2 = random.sample(nodes, 2)
    return node1.name(), node2.name()

def fakeRandomNodes(net, real_nodes, num_node_keep, num_node_output):
    """Return num_node_output node names where num_node_keep are kept from real_nodes."""
    if num_node_output <= 0:
        return []

    nodes = list(net.nodes())
    if len(nodes) < num_node_output:
        return []

    node_names = [node.name() for node in nodes]
    real_node_set = set(real_nodes)
    available_real = [name for name in real_nodes if name in node_names]

    if num_node_keep > num_node_output:
        raise ValueError("num_node_keep cannot exceed num_node_output.")

    if len(available_real) < num_node_keep:
        return []

    keep_nodes = random.sample(available_real, num_node_keep) if num_node_keep else []

    candidates = [name for name in node_names if name not in real_node_set]
    needed_fake = num_node_output - num_node_keep

    if needed_fake > len(candidates):
        return []

    new_nodes = random.sample(candidates, needed_fake) if needed_fake else []

    result = keep_nodes + new_nodes
    random.shuffle(result)
    return result

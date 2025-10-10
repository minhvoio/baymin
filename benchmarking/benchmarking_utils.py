import random 

def pick_one_random_node(net):
    nodes = net.nodes()
    if len(nodes) < 1:
        return None
    node = random.sample(nodes, 1)[0]
    return node.name()

def pick_two_random_nodes(net):
    nodes = net.nodes()
    if len(nodes) < 2:
        return None, None
    node1, node2 = random.sample(nodes, 2)
    return node1.name(), node2.name()

def generate_evidence_nodes(net, exclude_nodes, num_evidence=None) -> list[str]:
    """Generate evidence nodes for questions, excluding specified nodes.
    
    Args:
        net: The Bayesian network
        exclude_nodes: Tuple/list of node names to exclude from evidence
        num_evidence: Number of evidence nodes to generate (if None, uses random)
    
    Returns:
        List of node names to use as evidence
    """
    if num_evidence is None:
        num_evidence = get_random_number_of_nodes(net, padding=2)
    
    return fake_random_nodes(
        net, 
        real_nodes=exclude_nodes, 
        num_node_keep=0, 
        num_node_output=num_evidence, 
        min_output_when_zero=1
    )

def fake_random_nodes(net, real_nodes, num_node_keep, num_node_output, exclude=None, min_output_when_zero=0) -> list[str]:
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
            num_node_keep=desired_len-1,
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

def get_random_number_of_nodes(net, padding=0) -> int:
    max_number_of_nodes = len(net.nodes())
    return random.randint(1, max_number_of_nodes - padding)

def generate_fake_probability_answer_from_data(structured_data, variation_range=(0, 2)):
    """Generate a fake probability answer by slightly randomizing the probabilities using structured data.
    
    Args:
        structured_data: Dictionary containing the probability data
        variation_range: Tuple of (min, max) percentage variation (default: 0-2%)
    
    Returns:
        Fake answer string with slightly modified probabilities that still sum to 100%
    """
    import random
    
    # Create a copy of the structured data
    fake_data = structured_data.copy()
    
    # Randomize the new distribution probabilities
    new_dist = fake_data["new_distribution"]
    probabilities = list(new_dist.values())
    states = list(new_dist.keys())
    
    # Generate random variations
    variations = []
    for prob in probabilities:
        prob_percent = prob * 100
        variation = random.uniform(-variation_range[1], variation_range[1])
        new_percent = prob_percent + variation
        variations.append(new_percent)
    
    # Normalize to ensure they sum to 100%
    total_percent = sum(variations)
    normalized_variations = [v / total_percent * 100 for v in variations]
    
    # Convert back to decimal probabilities
    fake_probabilities = [p / 100 for p in normalized_variations]
    
    # Update the fake data
    fake_data["new_distribution"] = {state: prob for state, prob in zip(states, fake_probabilities)}
    
    # Update conclusion data based on new probabilities
    original_dist = fake_data["original_distribution"]
    
    # Recalculate changes
    new_changes = []
    max_change = 0
    
    for state in states:
        original_prob = original_dist[state]
        new_prob = fake_data["new_distribution"][state]
        change = new_prob - original_prob
        abs_change = abs(change)
        
        if abs_change > 0.05:  # threshold
            new_changes.append({
                "state": state,
                "change": change,
                "abs_change": abs_change
            })
        
        max_change = max(max_change, abs_change)
    
    # Update conclusion data
    fake_data["conclusion"]["max_change"] = max_change
    fake_data["conclusion"]["changes"] = new_changes
    fake_data["conclusion"]["minimal_update"] = max_change <= 0.05
    
    # Generate new text representations
    X = fake_data["X"]
    evidence = fake_data["evidence"]
    
    # Create new distribution text
    new_dist_text = "\n".join(f"  P({X}={state}) = {prob:.4f}" for state, prob in fake_data["new_distribution"].items())
    original_dist_text = "\n".join(f"  P({X}={state}) = {prob:.4f}" for state, prob in original_dist.items())
    
    # Create new conclusion text
    if max_change <= 1e-8:
        conclusion_text = "No change detected — the updated beliefs are identical to the original."
    else:
        conclusion_parts = []
        for change_info in new_changes:
            state = change_info["state"]
            abs_change = change_info["abs_change"]
            change = change_info["change"]
            conclusion_parts.append(f"Belief in '{state}' {'increased' if change > 0 else 'decreased'} by {abs_change:.4f}")
        
        if max_change <= 0.05:
            conclusion_parts.append(f"Overall, the update is minimal ({max_change:.4f}).")
        else:
            conclusion_parts.append(f"Largest overall per-state shift: {max_change:.4f}.")
        
        conclusion_text = "\n".join(f"  {part}" for part in conclusion_parts)
    
    # Create formatted answer
    cond = ", ".join(f"{k}=True" for k in evidence.keys()) if evidence else "∅"
    fake_answer = (
        f"P({X} | {cond}):\n"
        f"{new_dist_text}\n"
        f"\nOriginal distribution:\n"
        f"{original_dist_text}\n"
        f"\nConclusion:\n"
        f"{conclusion_text}"
        f"{fake_data['impact_text']}"
    )
    
    return fake_answer
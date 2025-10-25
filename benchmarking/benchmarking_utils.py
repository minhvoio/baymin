import random 
import csv, datetime
import time
from pydantic_ai.exceptions import ModelHTTPError

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

def generate_fake_nodes_for_relation(net, truth_nodes, node1, node2, *, exclude_extra=None, num_output=None):
    """Generate a fake list of node names for quiz distractors.

    Rules:
    - desired_len = num_output if provided, otherwise len(truth_nodes)
    - If desired_len >= 2: use truth_nodes as real_nodes, keep 1, output desired_len
    - If desired_len < 2: use [node1, node2] as real_nodes, keep 0, output desired_len;
      when desired_len == 0, synthesize 2 nodes.
    - Always exclude node1/node2 and any exclude_extra.
    - Guaranteed to return a non-empty list (length desired_len or 2 when zero),
      unless the network has insufficient candidates.
    """
    desired_len = num_output if num_output is not None else len(truth_nodes or [])
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

def generate_fake_highest_impact_evidence_answer(structured_data, fake_evidence_name, variation_range=(0, 2)):
    """Generate a fake highest impact evidence answer by modifying the evidence name and slightly randomizing probabilities.
    
    Args:
        structured_data: Dictionary containing the highest impact evidence data
        fake_evidence_name: Name of the fake evidence to use (must be in evidence list)
        variation_range: Tuple of (min, max) percentage variation for probabilities
    
    Returns:
        Fake answer string with modified evidence name and slightly changed probabilities
    """
    import random
    
    # Create a copy of the structured data
    fake_data = structured_data.copy()
    
    # Find the fake evidence and correct evidence in the ranked list
    fake_evidence_data = None
    correct_evidence_data = None
    correct_evidence_name = fake_data["highest_impact_evidence"]
    
    for ev, data in fake_data["ranked"]:
        if ev == fake_evidence_name:
            fake_evidence_data = data
        if ev == correct_evidence_name:
            correct_evidence_data = data
    
    if fake_evidence_data is None or correct_evidence_data is None:
        # Fallback: if fake evidence not found, use the original data
        return structured_data
    
    # Update the highest impact evidence
    fake_data["highest_impact_evidence"] = fake_evidence_name
    
    # Swap the scores and metrics between fake and correct evidence
    # This makes the fake evidence appear to have the highest score
    fake_data["ranked"] = []
    for ev, data in structured_data["ranked"]:
        if ev == fake_evidence_name:
            # Give fake evidence the correct evidence's high score
            fake_data["ranked"].append((ev, correct_evidence_data))
        elif ev == correct_evidence_name:
            # Give correct evidence the fake evidence's lower score
            fake_data["ranked"].append((ev, fake_evidence_data))
        else:
            # Keep other evidence unchanged
            fake_data["ranked"].append((ev, data))
    
    # Re-sort by score to maintain ranking
    fake_data["ranked"].sort(key=lambda x: x[1]["score"], reverse=True)
    
    # Randomize the new distribution probabilities slightly
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
    
    # Recalculate changes based on new probabilities
    original_dist = fake_data["original_distribution"]
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
    
    # Generate the formatted answer
    X = fake_data["X"]
    evidence = fake_data["evidence"]
    
    # Create evidence condition string
    cond = ", ".join(f"{k}={v}" for k, v in evidence.items()) or "∅"
    
    # Create distribution text
    new_dist_text = "\n".join(f"  P({X}={state}) = {prob:.4f}" for state, prob in fake_data["new_distribution"].items())
    original_dist_text = "\n".join(f"  P({X}={state}) = {prob:.4f}" for state, prob in original_dist.items())
    
    # Create conclusion text
    conclusion = fake_data["conclusion"]
    if conclusion["no_change"]:
        conclusion_text = "No change detected — the updated beliefs are identical to the original."
    else:
        conclusion_parts = []
        for change_info in conclusion["changes"]:
            state = change_info["state"]
            abs_change = change_info["abs_change"]
            change = change_info["change"]
            conclusion_parts.append(f"Belief in '{state}' {'increased' if change > 0 else 'decreased'} by {abs_change:.4f}")
        
        if conclusion["minimal_update"]:
            conclusion_parts.append("Overall, the update is minimal (all changes ≤ threshold).")
        else:
            conclusion_parts.append(f"Largest overall per-state shift: {conclusion['max_change']:.4f}.")
        
        conclusion_text = "\n".join(f"  {part}" for part in conclusion_parts)
    
    # Create evidence impact section - keep the same ranking but highlight the fake evidence
    impacts_text = "\nEvidence impact (sequential add/remove):\n"
    for ev, d in fake_data["ranked"]:
        add_l1 = d["add"]["l1"] if "add" in d else 0.0
        rem_l1 = d["remove"]["l1"] if "remove" in d else 0.0
        add_max = d["add"]["max_abs"] if "add" in d else 0.0
        rem_max = d["remove"]["max_abs"] if "remove" in d else 0.0
        impacts_text += (f"  - {ev}: "
                        f"ADD  L1={add_l1:.4f}, max_abs={add_max:.4f} | "
                        f"REMOVE L1={rem_l1:.4f}, max_abs={rem_max:.4f} | "
                        f"score={d['score']:.4f}\n")
    
    # Add highest impact evidence line - use the fake evidence
    if fake_data["is_tie_close"]:
        impacts_text += f"  => Highest-impact evidence (tie-close): {fake_evidence_name}.\n"
    else:
        impacts_text += f"  => Highest-impact evidence: {fake_evidence_name}.\n"
    
    # Combine all parts
    fake_answer = (
        f"P({X} | {cond}):\n"
        f"{new_dist_text}\n"
        f"\nOriginal distribution:\n"
        f"{original_dist_text}\n"
        f"\nConclusion:\n"
        f"{conclusion_text}\n"
        f"{impacts_text}"
    )
    
    return fake_answer


def get_completed_questions(test_type, question_set_name, model, model_quiz, network_size, test_baymin_only=False, test_raw_model_only=False, output_file=None):
    """
    Get list of already completed question indices from the CSV log.
    
    Args:
        test_type: Type of test (e.g., 'elementary_test', 'numerical_test')
        question_set_name: Name of the question set being tested
        model: Model used for generating answers
        model_quiz: Model used for taking the quiz
        network_size: Number of nodes in the network
        test_baymin_only: Whether to check baymin_test_log.csv instead of test_log.csv
        test_raw_model_only: Whether the test is running in raw model only mode
        output_file: Custom output file name (overrides default file selection)
        
    Returns:
        Set of completed question indices
    """
    try:
        # Determine which CSV file to read from based on test mode
        if output_file:
            csv_file = output_file
        elif test_baymin_only:
            csv_file = 'baymin_test_log.csv'
        else:
            csv_file = 'test_log.csv'
        completed_questions = set()
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get('test_type') == test_type and
                    row.get('question_set_name') == question_set_name and
                    row.get('model') == model and
                    row.get('model_quiz') == model_quiz and
                    row.get('network_size') == str(network_size)):
                    completed_questions.add(int(row.get('question_index', 0)))
        
        return completed_questions
    except FileNotFoundError:
        return set()
    except Exception as e:
        print(f"[Test] Error reading completed questions: {e}")
        return set()

def log_test_result(test_type, question_set_name, question_index, quiz, expected_answer, model, model_quiz, 
                   raw_model_score, baymin_score, question_output=None, prompt=None, 
                   hasEvidence=None, max_tokens=None, network_size=None, node1=None, node2=None, 
                   evidence=None, node=None, raw_model_answer=None, baymin_answer=None, 
                   raw_model_runtime=None, baymin_runtime=None, output_file='test_log.csv'):
    """
    Log test results to CSV file with comprehensive information for validation.
    Includes duplicate prevention to avoid re-logging the same test results.
    
    Args:
        test_type: Type of test (e.g., 'elementary_test', 'numerical_test')
        question_set_name: Name of the question set being tested
        question_index: Index of the question in the set
        quiz: The quiz question text
        expected_answer: The correct answer
        model: Model used for generating answers
        model_quiz: Model used for taking the quiz
        raw_model_score: Score from raw model (0 or 1)
        baymin_score: Score from baymin model (0 or 1)
        question_output: The formatted question output #optional
        prompt: The full prompt sent to models #optional
        hasEvidence: Whether evidence was used #optional
        max_tokens: Maximum tokens setting #optional
        network_size: Number of nodes in the network #optional
        node1: First node in the question #optional
        node2: Second node in the question #optional
        evidence: Evidence used in the question #optional
        node: Single node for numerical tests #optional
        raw_model_answer: Raw model's answer text #optional
        baymin_answer: Baymin model's answer text #optional
        raw_model_runtime: Raw model's runtime in seconds #optional
        baymin_runtime: Baymin model's runtime in seconds #optional
    """
    try:
        csv_file = output_file
        
        # Check for duplicates using composite key
        duplicate_key = (test_type, question_set_name, question_index, model, model_quiz, network_size)
        
        # Read existing data to check for duplicates
        existing_data = []
        file_exists = False
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_data = list(reader)
                file_exists = True
        except FileNotFoundError:
            file_exists = False
        
        # Check if this exact test result already exists
        for row in existing_data:
            existing_key = (
                row.get('test_type', ''),
                row.get('question_set_name', ''),
                int(row.get('question_index', 0)),
                row.get('model', ''),
                row.get('model_quiz', ''),
                row.get('network_size', '')
            )
            if existing_key == duplicate_key:
                print(f"[Test] Skipping duplicate: {test_type} - {question_set_name} - Q{question_index}")
                return
        
        # Prepare new row data
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_row = {
            'timestamp': timestamp,
            'test_type': test_type,
            'question_set_name': question_set_name,
            'question_index': question_index,
            'quiz': quiz.replace('\n', ' ').replace('\r', ' ')[:500],
            'expected_answer': expected_answer,
            'model': model,
            'model_quiz': model_quiz,
            'raw_model_score': raw_model_score,
            'baymin_score': baymin_score,
            'question_output': question_output.replace('\n', ' ').replace('\r', ' ')[:500] if question_output else 'N/A',  # optional
            'prompt': prompt.replace('\n', ' ').replace('\r', ' ')[:500] if prompt else 'N/A',  # optional
            'hasEvidence': hasEvidence if hasEvidence is not None else 'N/A',  # optional
            'max_tokens': max_tokens if max_tokens is not None else 'N/A',  # optional
            'network_size': network_size if network_size is not None else 'N/A',  # optional
            'node1': node1 if node1 else 'N/A',  # optional
            'node2': node2 if node2 else 'N/A',  # optional
            'node': node if node else 'N/A',  # optional
            'evidence': evidence if evidence else 'N/A',  # optional
            'raw_model_answer': raw_model_answer.replace('\n', ' ').replace('\r', ' ')[:500] if raw_model_answer else 'N/A',  # optional
            'baymin_answer': baymin_answer.replace('\n', ' ').replace('\r', ' ')[:500] if baymin_answer else 'N/A',  # optional
            'raw_model_runtime': raw_model_runtime if raw_model_runtime is not None else 'N/A',  # optional
            'baymin_runtime': baymin_runtime if baymin_runtime is not None else 'N/A',  # optional
        }
        
        # Append new row to CSV file
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['timestamp', 'test_type', 'question_set_name', 'question_index', 'quiz', 'expected_answer', 
                         'model', 'model_quiz', 'raw_model_score', 'baymin_score', 
                         'question_output', 'prompt', 'hasEvidence', 'max_tokens', 'network_size', 
                         'node1', 'node2', 'node', 'evidence', 'raw_model_answer', 'baymin_answer',
                         'raw_model_runtime', 'baymin_runtime']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(csv_row)
            print(f"[Test] Logged: {test_type} - {question_set_name} - Q{question_index}")
            
    except Exception as e:
        print(f"[Test] Logging failed for question {question_index}: {e}")


def log_for_baymin_testing(quiz, y, y_hat, answer, testing_log):
    print('quiz:\n', quiz)
    print('y:\n', y)
    print('y_hat:\n', y_hat)
    print('---------------------------------------------')
    print('Baymin Model:')
    print('ans:\n', answer)
    print('y:\n', y)
    print('y_hat:\n', y_hat)
    print('---------------------------------------------')

    # Log wrong answer to CSV (always log for debugging purposes)
    try:
        csv_file = 'baymin_wrong_ans_log.csv'
        
        # Check if file exists
        file_exists = False
        try:
            with open(csv_file, 'r'):
                file_exists = True
        except FileNotFoundError:
            file_exists = False
        
        # Prepare new row data
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tool_calls_str = "; ".join(testing_log['tool_calls']) if testing_log['tool_calls'] else "None"
        tool_results_str = "; ".join(testing_log['tool_results']) if testing_log['tool_results'] else "None"
        
        csv_row = {
            'timestamp': timestamp,
            'prompt': testing_log['prompt'].replace('\n', ' ').replace('\r', ' ')[:500],  # Limit length and remove newlines
            'bn_str': testing_log['bn_str'].replace('\n', ' ').replace('\r', ' ')[:500],  # Limit length and remove newlines
            'network_size': testing_log['network_size'],
            'tool_calls': tool_calls_str,
            'tool_results': tool_results_str,
            'final_answer': testing_log['final_answer'].replace('\n', ' ').replace('\r', ' ')[:500] if testing_log['final_answer'] else "None",  # Limit length and remove newlines
            'quiz': quiz.replace('\n', ' ').replace('\r', ' ')[:500],  
            'expected_answer': y,
            'actual_answer': y_hat
        }
        
        # Append new row to CSV file
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['timestamp', 'prompt', 'bn_str', 'network_size', 'tool_calls', 'tool_results', 'final_answer', 'quiz', 'expected_answer', 'actual_answer']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(csv_row)
            print(f"[BayMin] Logged wrong answer to CSV")
            
    except Exception as e:
        print(f"[BayMin] Debug logging failed: {e}")


def retry_test_with_backoff(test_function, net, max_retries=5, base_delay=1, max_delay=60, *args, **kwargs):
    """
    Retry a test function with exponential backoff when encountering HTTP errors.
    
    Args:
        test_function: The test function to retry
        net: The network parameter
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        max_delay: Maximum delay in seconds
        *args, **kwargs: Additional arguments to pass to the test function
    
    Returns:
        The result of the test function or raises the last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            print(f"Attempt {attempt + 1}/{max_retries + 1} for {test_function.__name__}")
            result = test_function(net, *args, **kwargs)
            print(f"✅ {test_function.__name__} completed successfully on attempt {attempt + 1}")
            return result
            
        except ModelHTTPError as e:
            last_exception = e
            if e.status_code == 500:  # Internal Server Error
                if attempt < max_retries:
                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    print(f"❌ HTTP 500 error on attempt {attempt + 1}. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"❌ All {max_retries + 1} attempts failed for {test_function.__name__}")
                    break
            else:
                # For non-500 errors, don't retry
                print(f"❌ Non-retryable HTTP error {e.status_code} for {test_function.__name__}")
                raise e
                
        except Exception as e:
            last_exception = e
            print(f"❌ Unexpected error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                print(f"❌ All {max_retries + 1} attempts failed for {test_function.__name__}")
                break
    
    # If we get here, all retries failed
    print(f"🚨 {test_function.__name__} failed after {max_retries + 1} attempts")
    raise last_exception

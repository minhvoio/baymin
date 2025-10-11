from ollama_helper.ollama_helper import answer_this_prompt
from bn_helpers.bn_helpers import AnswerStructure, BnToolBox
from bn_helpers.utils import get_path, grammar_plural
import random as _random
from bn_helpers.constants import MODEL, MODEL_QUIZ
from benchmarking.benchmarking_utils import fake_random_nodes, pick_two_random_nodes, generate_fake_nodes_for_relation, get_random_number_of_nodes, generate_fake_probability_answer_from_data, generate_fake_highest_impact_evidence_answer

def create_distract_answer(answer, model=MODEL, temperature=0.3, another_answer=None):
  prompt = f"Keep the sentence structure, just change the variables names from the following answer (no other text): {answer}"
  if another_answer:
    prompt += f" and different variables names from the following answer: {another_answer}"
  distract_answer = answer_this_prompt(prompt, format=AnswerStructure.model_json_schema(), model=model, temperature=temperature)
  distract_answer = AnswerStructure.model_validate_json(distract_answer)
  return distract_answer.answer


def shuffle_options(options, rng=None):
    """
    Shuffle a list of (text, is_correct) tuples and return both
    the shuffled list and the correct answer letter.

    Args:
        options (list[tuple[str, bool]]): options as (text, is_correct)
        rng (random.Random or object with .shuffle): optional randomizer

    Returns:
        (shuffled_options, correct_letter)
    """
    randomizer = rng or _random
    randomizer.shuffle(options)
    correct_letter = None
    for idx, (_, is_correct) in enumerate(options):
        if is_correct:
            correct_letter = chr(65 + idx)  # A, B, C, ...
            break
    return options, correct_letter

def create_question(header_question, option_list, rng=None, leading_blank=False):
    """
    Build a single multiple-choice question block from a header and options.

    Args:
        header_question (str): The question prompt/header line.
        option_list (list[tuple[str, bool]]): Options as (text, is_correct).
        rng: Optional randomizer with .shuffle(list).
        leading_blank (bool): If True, insert a blank line before the header.

    Returns:
        (question_block_text, correct_letter)
        - question_block_text (str): The full block, e.g. "Q. ...\nA. ...\nB. ..."
        - correct_letter (str): The correct choice letter ("A", "B", ...).
    """
    randomizer = rng or _random

    # Work on a shallow copy so caller's list isn't mutated
    opts = list(option_list)
    randomizer.shuffle(opts)

    lines = []
    if leading_blank:
        lines.append("")
    lines.append(header_question)

    correct_letter = None
    for idx, (text, is_correct) in enumerate(opts):
        letter = chr(65 + idx)  # A, B, C, ...
        lines.append(f"{letter}. {text}")
        if is_correct:
            correct_letter = letter

    return "\n".join(lines), correct_letter


def create_dependency_quiz(question, net, node1, node2, rng=None):
    randomizer = rng or _random
    bn_tool_box = BnToolBox()
    is_connected = bn_tool_box.is_XY_dconnected(net, node1, node2)

    if is_connected:
        # Ground-truth reason (d-connected path)
        try:
            ground_truth_path = get_path(net, node1, node2)  
        except Exception:
            ground_truth_path = None
        path_list = list(ground_truth_path) if ground_truth_path else []

        correct = bn_tool_box.get_explain_XY_dconnected(net, node1, node2)

        fake1_nodes = generate_fake_nodes_for_relation(net, path_list, node1, node2)
        d1_path = ", ".join(fake1_nodes) if fake1_nodes else ", ".join(pick_two_random_nodes(net))

        opt1 = (correct, True)
        opt2 = (f"Yes, they are d-connected through the path {d1_path}. Meaning that changing the evidence of {node1} will change the probability of {node2}.", False)
        opt3 = ("None of the above", False)
        opt4 = (f"No, there is no path between {node1} and {node2}. Meaning that changing the evidence of {node1} will not change the probability of {node2}.", False)

        options = [opt1, opt2, opt3, opt4]

    else:
        # Ground-truth reason (blocked / no path)
        common_effects = bn_tool_box.get_common_effect(net, node1, node2)
        ce_list = list(common_effects) if common_effects else []
        ce_text = ", ".join(ce_list) if ce_list else "None"

        because_text = bn_tool_box.get_explain_XY_dseparated(net, node1, node2)

        fake1_nodes = generate_fake_nodes_for_relation(net, ce_list, node1, node2)
        d1_path = ", ".join(fake1_nodes) if fake1_nodes else ", ".join(pick_two_random_nodes(net))

        opt1 = (because_text, True)
        opt2 = (f"No, they are d-separated because they are blocked by {d1_path}. Meaning that changing the evidence of {node1} will not change the probability of {node2}.", False)
        opt3 = ("None of the above", False)
        opt4 = (f"Yes, they are d-connected. Meaning that changing the evidence of {node1} will change the probability of {node2}.", False)

        options = [opt1, opt2, opt3, opt4]

    q_text, q_correct = create_question(question, options, rng=randomizer)
    return q_text, q_correct


def create_common_cause_quiz(question, net, node1, node2, rng=None):
    randomizer = rng or _random
    bn_tool_box = BnToolBox()
    common_causes = bn_tool_box.get_common_cause(net, node1, node2)
    nums_cause, is_or_are, final_s = grammar_plural(common_causes)

    # Compose correct nodes text: when empty, synthesize using fake_random_nodes while excluding node1/node2
    if common_causes:
        correct_nodes_text = ', '.join(common_causes)
    else:
        synth = fake_random_nodes(net, [node1, node2], num_node_keep=0, num_node_output=0, exclude=[node1, node2], min_output_when_zero=2)
        correct_nodes_text = ', '.join(synth) if synth else "None"
    correct = f"The common cause{final_s} of {node1} and {node2} {is_or_are}: {correct_nodes_text}."

    # Generate a fake option list using the reusable generator
    cause_list = list(common_causes) if common_causes else []
    fake_nodes = generate_fake_nodes_for_relation(net, cause_list, node1, node2)
    
    # Ensure fake answer is different from correct answer
    fake_nodes_text = ', '.join(fake_nodes)
    if fake_nodes_text == correct_nodes_text:
        # If they're the same, try generating again with different parameters
        fake_nodes = generate_fake_nodes_for_relation(net, cause_list, node1, node2, num_output=len(cause_list) + 1)
        fake_nodes_text = ', '.join(fake_nodes)
    
    opt2_text = f"The common cause{final_s} of {node1} and {node2} {is_or_are}: {fake_nodes_text}."

    opt1 = (correct, nums_cause > 0)
    opt2 = (opt2_text, False)
    opt3 = (f"No, there is no common cause between {node1} and {node2}.", nums_cause == 0)
    opt4 = ("None of the above", False)

    options = [opt1, opt2, opt3, opt4]

    q_text, q_correct = create_question(question, options, rng=randomizer)
    return q_text, q_correct

def create_common_effect_quiz(question, net, node1, node2, rng=None):
    randomizer = rng or _random
    bn_tool_box = BnToolBox()
    common_effects = bn_tool_box.get_common_effect(net, node1, node2)
    nums_effect, is_or_are, final_s = grammar_plural(common_effects)

    if common_effects:
        correct_nodes_text = ', '.join(common_effects)
    else:
        synth = fake_random_nodes(net, [node1, node2], num_node_keep=0, num_node_output=0, exclude=[node1, node2], min_output_when_zero=2)
        correct_nodes_text = ', '.join(synth) if synth else "None"
    correct = f"The common effect{final_s} of {node1} and {node2} {is_or_are}: {correct_nodes_text}."

    # Generate a fake option list using the reusable generator
    effect_list = list(common_effects) if common_effects else []
    fake_nodes = generate_fake_nodes_for_relation(net, effect_list, node1, node2)
    
    # Ensure fake answer is different from correct answer
    fake_nodes_text = ', '.join(fake_nodes)
    if fake_nodes_text == correct_nodes_text:
        # If they're the same, try generating again with different parameters
        fake_nodes = generate_fake_nodes_for_relation(net, effect_list, node1, node2, num_output=len(effect_list) + 1)
        fake_nodes_text = ', '.join(fake_nodes)
    
    opt2_text = f"The common effect{final_s} of {node1} and {node2} {is_or_are}: {fake_nodes_text}."

    opt1 = (correct, nums_effect > 0)
    opt2 = (opt2_text, False)
    opt3 = (f"No, there is no common effect between {node1} and {node2}.", nums_effect == 0)
    opt4 = ("None of the above", False)
    options = [opt1, opt2, opt3, opt4]

    q_text, q_correct = create_question(question, options, rng=randomizer)
    return q_text, q_correct

def create_blocked_evidence_quiz(question, net, node1, node2, rng=None):
    randomizer = rng or _random
    bn_tool_box = BnToolBox()
    blocked_evidence = bn_tool_box.evidences_block_XY(net, node1, node2)
    nums_blocked, is_or_are, final_s = grammar_plural(blocked_evidence)

    if blocked_evidence:
        correct_nodes_text = ', '.join(blocked_evidence)
    else:
        synth = fake_random_nodes(net, [node1, node2], num_node_keep=0, num_node_output=0, exclude=[node1, node2], min_output_when_zero=2)
        correct_nodes_text = ', '.join(synth) if synth else "None"
    correct = f"The evidence{final_s} that would block the dependency between {node1} and {node2} {is_or_are}: {correct_nodes_text}."

    block_list = list(blocked_evidence) if blocked_evidence else []
    fake_nodes = generate_fake_nodes_for_relation(net, block_list, node1, node2)
    
    # Ensure fake answer is different from correct answer
    fake_nodes_text = ', '.join(fake_nodes)
    if fake_nodes_text == correct_nodes_text:
        # If they're the same, try generating again with different parameters
        fake_nodes = generate_fake_nodes_for_relation(net, block_list, node1, node2, num_output=len(block_list) + 1)
        fake_nodes_text = ', '.join(fake_nodes)
    
    opt2_text = f"The evidence{final_s} that would block the dependency between {node1} and {node2} {is_or_are}: {fake_nodes_text}."

    opt1 = (correct, nums_blocked > 0)
    opt2 = (opt2_text, False)
    opt3 = (f"No, there is no evidence that would block the dependency between {node1} and {node2}.", nums_blocked == 0)
    opt4 = ("None of the above", False)
    options = [opt1, opt2, opt3, opt4]

    q_text, q_correct = create_question(question, options, rng=randomizer)
    return q_text, q_correct

def create_evidence_change_relationship_quiz(question: str, net, node1: str, node2: str, evidence: list[str], rng=None):
    randomizer = rng or _random
    bn_tool_box = BnToolBox()
    correct_answer, _ = bn_tool_box.get_explain_evidence_change_dependency_XY(net, node1, node2, evidence)
    _, details = bn_tool_box.does_evidence_change_dependency_XY(net, node1, node2, evidence)
    
    opt1 = (correct_answer, True)
    
    # Option 2: Reverse one of the sequential items (d-connected => d-separated and vice versa)
    fake_answer2 = correct_answer
    if details.get("sequential"):
        # Pick a random sequential step to reverse
        step_to_reverse = randomizer.choice(details["sequential"])
        original_conn = "d-connected" if step_to_reverse["connected"] else "d-separated"
        reversed_conn = "d-separated" if step_to_reverse["connected"] else "d-connected"
        # Replace the first occurrence of the original connection status with the reversed one
        fake_answer2 = fake_answer2.replace(f"+{step_to_reverse['added']} => {original_conn}", f"+{step_to_reverse['added']} => {reversed_conn}", 1)
    opt2 = (fake_answer2, False)

    # Option 3: Flip Yes/No from correct answer using the opposite template
    changed, details = bn_tool_box.does_evidence_change_dependency_XY(net, node1, node2, evidence)
    
    def _conn_label(b: bool) -> str:
        return "d-connected" if b else "d-separated"
    
    def _fmt_ev_list(evs: list[str]) -> str:
        if not evs: return "∅"
        return ", ".join(evs)
    
    # Reverse the before/after states to match the flipped template
    before_reversed = _conn_label(not details["before"])
    after_reversed = _conn_label(not details["after"])
    ev_str = _fmt_ev_list(evidence)
    
    # Find first step (if any) where connectivity flips relative to BEFORE (reversed)
    flip_note = ""
    for step in details.get("sequential", []):
        if step["connected"] != details["before"]:
            flip_note = f" The relationship first flips after conditioning on {step['added']}."
            break
    
    # Generate the opposite template
    if not evidence:
        fake_answer3 = f"No evidence provided. Relationship between {node1} and {node2} is {before_reversed} with no conditioning."
    elif not changed:  # Original was "No", so make it "Yes"
        fake_answer3 = (
            f"Yes - conditioning on {ev_str} changes the dependency between {node1} and {node2}. "
            f"Before observing {ev_str}, they were {_conn_label(details["before"])}. After observing all evidence, they are {after_reversed}."
            f"{flip_note}"
        )
    else:  # Original was "Yes", so make it "No"
        fake_answer3 = (
            f"No - conditioning on {ev_str} does not change the dependency between {node1} and {node2}. "
            f"Before observing {ev_str}, they were {before_reversed}. After observing all evidence, they remain {after_reversed}."
        )
    
    # Add sequence if present (with reversed connection states)
    if details.get("sequential"):
        steps = "; ".join(
            f"+{s['added']} => {_conn_label(not s['connected'])}"
            for s in details["sequential"]
        )
        fake_answer3 += f" Sequence: {steps}."
    
    opt3 = (fake_answer3, False)
    
    opt4 = ("None of the above", False)
    options = [opt1, opt2, opt3, opt4]

    q_text, q_correct = create_question(question, options, rng=randomizer)
    return q_text, q_correct

def create_probability_quiz(question, net, node, evidence, rng=None):
    randomizer = rng or _random
    bn_tool_box = BnToolBox()
    evidence_dict = {ev: True for ev in evidence} if evidence else {}

    correct_answer, structured_data = bn_tool_box.get_explain_prob_X_given(net, node, evidence_dict)
    opt1 = (correct_answer, True)
    
    # Generate fake answer by slightly randomizing the probabilities using structured data
    fake_answer = generate_fake_probability_answer_from_data(structured_data, variation_range=(0, 2))
    opt2 = (fake_answer, False)
    
    # Generate another fake answer with larger variation
    fake_answer2 = generate_fake_probability_answer_from_data(structured_data, variation_range=(0, 5))
    opt3 = (fake_answer2, False)
    
    opt4 = ("None of the above", False)
    options = [opt1, opt2, opt3, opt4]

    q_text, q_correct = create_question(question, options, rng=randomizer)
    
    # Add separators before each option for better readability
    lines = q_text.split('\n')
    formatted_lines = []
    
    for _, line in enumerate(lines):
        if line.startswith(('A.', 'B.', 'C.', 'D.')):
            formatted_lines.append("-----------")
        
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines), q_correct


def create_highest_impact_evidence_quiz(question, net, node, evidence=None, order=None, rng=None):
    """
    Create a quiz about which evidence has the highest impact on a node.
    
    Args:
        question (str): The question prompt/header line.
        net: Bayesian network
        node (str): Target node to analyze
        evidence (dict): Evidence dictionary, if None will derive from d-connected nodes
        order (list): Order of evidence to analyze
        rng: Optional randomizer
    
    Returns:
        (question_block_text, correct_letter)
    """
    randomizer = rng or _random
    bn_tool_box = BnToolBox()
    
    # Get the correct answer and structured data
    correct_answer, _, structured_data = bn_tool_box.get_highest_impact_evidence(net, node, evidence, order)
    
    # Extract the highest-impact evidence from structured data
    highest_impact_evidence = structured_data.get("highest_impact_evidence")
    
    # Generate fake answers using structured data
    fake_answer1 = correct_answer
    fake_answer2 = correct_answer
    
    if highest_impact_evidence:
        # Get the ranked evidence list from structured data
        ranked_evidence = structured_data.get("ranked", [])
        
        # Select the 2nd and 3rd highest impact evidence as fake options
        fake_evidence_candidates = []
        if len(ranked_evidence) >= 2:
            fake_evidence_candidates.append(ranked_evidence[1][0])  # 2nd highest
        if len(ranked_evidence) >= 3:
            fake_evidence_candidates.append(ranked_evidence[2][0])  # 3rd highest
        
        # If we don't have enough ranked evidence, fall back to other evidence nodes
        if len(fake_evidence_candidates) < 2:
            evidence_nodes = list(structured_data.get("evidence", {}).keys())
            remaining_candidates = [ev for ev in evidence_nodes if ev != highest_impact_evidence and ev not in fake_evidence_candidates]
            fake_evidence_candidates.extend(remaining_candidates[:2-len(fake_evidence_candidates)])
        
        if len(fake_evidence_candidates) >= 1:
            # Generate fake answer 1 with slightly modified probabilities
            fake_answer1 = generate_fake_highest_impact_evidence_answer(
                structured_data, fake_evidence_candidates[0], variation_range=(0, 2)
            )
            
            # Generate fake answer 2 with larger variation
            if len(fake_evidence_candidates) >= 2:
                fake_answer2 = generate_fake_highest_impact_evidence_answer(
                    structured_data, fake_evidence_candidates[1], variation_range=(0, 5)
                )
    
    # Create options
    opt1 = (correct_answer, True)
    opt2 = (fake_answer1, False)
    opt3 = (fake_answer2, False)
    opt4 = ("None of the above", False)
    
    options = [opt1, opt2, opt3, opt4]
    
    q_text, q_correct = create_question(question, options, rng=randomizer)

    # Add separators before each option for better readability
    lines = q_text.split('\n')
    formatted_lines = []
    
    for _, line in enumerate(lines):
        if line.startswith(('A.', 'B.', 'C.', 'D.')):
            formatted_lines.append("-----------")
        
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines), q_correct


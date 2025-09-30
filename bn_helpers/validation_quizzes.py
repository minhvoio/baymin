import random as _random

def create_dependency_quiz(net, node1, node2, rng=None):
    """
    Builds two multiple-choice questions about dependency and d-separation, with
    randomized answer order. Returns (questions_text, answers_letters).

    - answers_letters: list like ["A", "C"] indicating the correct choice per question
    - rng: optional random-like object with .shuffle(list) and .choice(...)
    """
    from bn_helpers.bn_helpers import BnHelper
    from bn_helpers.utils import get_path

    randomizer = rng or _random

    bn_helper = BnHelper(function_name='is_XY_connected')
    is_connected = bn_helper.is_XY_connected(net, node1, node2)

    # Q1: Is changing evidence of node1 going to change probability of node2?
    q1_prompt = f"1. Is changing the evidence of {node1} going to change the probability of {node2}?"
    q1_options = [
        ("Yes", is_connected),
        ("No", not is_connected),
        ("None of the above", False),
    ]
    randomizer.shuffle(q1_options)
    q1_lines = [q1_prompt]
    q1_correct_letter = None
    for idx, (text, correct) in enumerate(q1_options):
        letter = chr(65 + idx)  # A, B, C
        q1_lines.append(f"{letter}. {text}")
        if correct:
            q1_correct_letter = letter

    # Q2: d-connected or d-separated explanation
    if is_connected:
        option_texts = [
            (f"They are d-connected through the path {get_path(net, node1, node2)}", True),
            ("They are not d-connected", False),
            ("None of the above", False),
        ]
        q2_header = "2. Why they are d-connected?"
    else:
        common_effect = bn_helper.get_common_effect(net, node1, node2)
        because_text = (
            f"They are d-separated because they are blocked by {common_effect}"
            if common_effect
            else f"There are no path between {node1} and {node2}"
        )
        option_texts = [
            (because_text, True),
            ("They are not d-separated", False),
            ("None of the above", False),
        ]
        q2_header = "2. Why they are d-separated?"

    randomizer.shuffle(option_texts)
    q2_lines = ["", q2_header]  # blank line between Q1 and Q2
    q2_correct_letter = None
    for idx, (text, correct) in enumerate(option_texts):
        letter = chr(65 + idx)
        q2_lines.append(f"{letter}. {text}")
        if correct:
            q2_correct_letter = letter

    questions = "\n".join(q1_lines + q2_lines)
    answers = [q1_correct_letter, q2_correct_letter]

    return questions, answers

def create_common_cause_quiz(net, node1, node2, rng=None):
    """
    Builds two multiple-choice questions about common cause, with
    randomized answer order. Returns (questions_text, answers_letters).
    """
    from bn_helpers.bn_helpers import BnHelper
    from bn_helpers.utils import get_path

    randomizer = rng or _random
    bn_helper = BnHelper()
    common_cause = bn_helper.get_common_cause(net, node1, node2)
    isPlural = len(list(common_cause)) > 1

    q_prompt = f"1. What {'are' if isPlural else 'is'} the common cause{'s' if isPlural else ''} of {node1} and {node2}?"
    q_options = [
        (f"{' '.join(common_cause)}", True),
        ("None of the above", False),
        ("Some of the above", False),
        ("Common causes listed above are not enough", False),
    ]
    randomizer.shuffle(q_options)
    q_lines = [q_prompt]
    q_correct_letter = None
    for idx, (text, correct) in enumerate(q_options):
        letter = chr(65 + idx)
        q_lines.append(f"{letter}. {text}")
        if correct:
            q_correct_letter = letter

    questions = "\n".join(q_lines)
    answers = [q_correct_letter]

    return questions, answers

def create_common_effect_quiz(net, node1, node2, rng=None):
    """
    Builds two multiple-choice questions about common effect, with
    randomized answer order. Returns (questions_text, answers_letters).
    """
    from bn_helpers.bn_helpers import BnHelper
    from bn_helpers.utils import get_path

    randomizer = rng or _random
    bn_helper = BnHelper()
    common_effect = bn_helper.get_common_effect(net, node1, node2)
    isPlural = len(list(common_effect)) > 1

    q_prompt = f"1. What {'are' if isPlural else 'is'} the common effect{'s' if isPlural else ''} of {node1} and {node2}?"
    q_options = [
        (f"{' '.join(common_effect)}", True),
        ("None of the above", False),
        ("Some of the above", False),
        ("Common effects listed above are not enough", False),
    ]
    randomizer.shuffle(q_options)
    q_lines = [q_prompt]
    q_correct_letter = None
    for idx, (text, correct) in enumerate(q_options):
        letter = chr(65 + idx)
        q_lines.append(f"{letter}. {text}")
        if correct:
            q_correct_letter = letter

    questions = "\n".join(q_lines)
    answers = [q_correct_letter]

    return questions, answers
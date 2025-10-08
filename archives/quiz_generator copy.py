from ollama_helper.ollama_helper import answer_this_prompt
from bn_helpers.bn_helpers import AnswerStructure, BnToolBox
from bn_helpers.utils import get_path
import random as _random
from bn_helpers.constants import MODEL, MODEL_QUIZ

def create_distract_answer(answer, model=MODEL, temperature=0.3, another_answer=None):
  prompt = f"Create just an answer that is different and corresponding (same number of variables) from the following answer (no other text): {answer}"
  if another_answer:
    prompt += f" and also different from the following answer: {another_answer}"
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


def create_dependency_quiz(question_format, net, node1, node2, rng=None, model_quiz=MODEL_QUIZ):
    """
    Builds a single multiple-choice question about whether changing evidence of `node1`
    changes the probability of `node2`, *including the reason* in each option.
    Returns (question_text, answers_letters) where answers_letters is a one-item list like ["B"].

    - rng: optional random-like object with .shuffle(list) and .choice(...)
    """
    randomizer = rng or _random
    bn_helper = BnToolBox()
    is_connected = bn_helper.is_XY_dconnected(net, node1, node2)

    # Header
    q_header = question_format.format(node1=node1, node2=node2)

    if is_connected:
        # Ground-truth reason (d-connected path)
        try:
            ground_truth_path = get_path(net, node1, node2)
        except Exception:
            ground_truth_path = None
        path_str = ground_truth_path or "N/A"

        correct = f"Yes, they are d-connected through the path {path_str}. Meaning that changing the evidence of {node1} will change the probability of {node2}."

        # Two plausible-but-wrong path distractors
        d1_path = create_distract_answer(path_str, temperature=0.3, model=model_quiz)
        d2_path = create_distract_answer(path_str, temperature=0.8, model=model_quiz, another_answer=d1_path)

        opt1 = (correct, True)
        opt2 = (f"Yes, they are d-connected through the path {d1_path}", False)
        opt3 = (f"Yes, they are d-connected through the path {d2_path}", False)
        opt4 = ("None of the above", False)
        opt5 = (f"No, there is no path between {node1} and {node2}. Meaning that changing the evidence of {node1} will not change the probability of {node2}.", False)

        options = [opt1, opt2, opt3, opt4, opt5]

    else:
        # Ground-truth reason (blocked / no path)
        common_effect = bn_helper.get_common_effect(net, node1, node2)
        because_text = (
            f"No, they are d-separated because they are blocked by {common_effect}. Meaning that changing the evidence of {node1} will not change the probability of {node2}."
            if common_effect
            else f"No, there is no path between {node1} and {node2}. Meaning that changing the evidence of {node1} will not change the probability of {node2}."
        )

        # Two plausible-but-wrong distractors derived from the correct explanation
        d1 = create_distract_answer(because_text, temperature=0.3, model=model_quiz)
        d2 = create_distract_answer(because_text, temperature=0.8, model=model_quiz, another_answer=d1)

        opt1 = (because_text, True)
        opt2 = (d1, False)
        opt3 = (d2, False)
        opt4 = ("None of the above", False)
        opt5 = (f"Yes, they are d-connected. Meaning that changing the evidence of {node1} will change the probability of {node2}.", False)

        options = [opt1, opt2, opt3, opt4, opt5]

    q_text, q_correct = create_question(q_header, options, rng=randomizer)

    return q_text, [q_correct]

def validate_quiz_answer(y_list, y_hat_list):
    score = 0
    for y, y_hat in zip(y_list, y_hat_list):
        if y == y_hat:
            score += 1
    return score / len(y_list)
QWEN = "qwen3:1.7b"
GPT_OSS = "gpt-oss-bn-json"
MODEL = GPT_OSS
MODEL_QUIZ = "qwen2.5:7b"

from ollama_helper.ollama_helper import answer_this_prompt
from bn_helpers.bn_helpers import AnswerStructure, BnToolBox
from bn_helpers.utils import get_path
import random as _random

def create_distract_answer(answer, model=MODEL, temperature=0.3):
  prompt = f"Create just a distract answer (no other text) for the following answer: {answer}"
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


def create_dependency_quiz(net, node1, node2, rng=None):
    """
    Builds two multiple-choice questions about dependency and d-separation, with
    randomized answer order. Returns (questions_text, answers_letters).

    - answers_letters: list like ["A", "C"] indicating the correct choice per question
    - rng: optional random-like object with .shuffle(list) and .choice(...)
    """

    randomizer = rng or _random
    bn_helper = BnToolBox()
    is_connected = bn_helper.is_XY_connected(net, node1, node2)

    # Q1
    q1_header = f"1. Is changing the evidence of {node1} going to change the probability of {node2}?"
    q1_options = [
        ("Yes", is_connected),
        ("No", not is_connected),
        ("None of the above", False),
    ]
    q1_text, q1_correct = create_question(q1_header, q1_options, rng=randomizer)

    # Q2
    if is_connected:
        ground_truth = f"They are d-connected through the path {get_path(net, node1, node2)}"
        distract_answer1 = create_distract_answer(ground_truth, temperature=0.3)
        distract_answer2 = create_distract_answer(ground_truth, temperature=0.4)
        q2_options = [
            (ground_truth, True),
            (distract_answer1, False),
            (distract_answer2, False),
            ("None of the above", False),
        ]
        q2_header = "2. Why are they d-connected?"
    else:
        common_effect = bn_helper.get_common_effect(net, node1, node2)
        because_text = (
            f"They are d-separated because they are blocked by {common_effect}"
            if common_effect
            else f"There is no path between {node1} and {node2}"
        )
        distract_answer1 = create_distract_answer(because_text, temperature=0.3)
        distract_answer2 = create_distract_answer(because_text, temperature=0.4)
        q2_options = [
            (because_text, True),
            (distract_answer1, False),
            (distract_answer2, False),
            ("None of the above", False),
        ]
        q2_header = "2. Why are they d-separated?"

    q2_text, q2_correct = create_question(q2_header, q2_options, rng=randomizer, leading_blank=True)

    questions_text = "\n".join([q1_text, q2_text])
    answers_letters = [q1_correct, q2_correct]
    return questions_text, answers_letters

def model_do_quiz(quiz, bn_explanation):
    prompt = TAKE_QUIZ_PROMPT.format(quiz=quiz, bn_explanation=bn_explanation)
    # print('MODEL QUIZ:', MODEL_QUIZ)
    # print('prompt:\n', prompt)
    res_str = answer_this_prompt(prompt, format=AnswerStructure.model_json_schema(), model=MODEL_QUIZ)
    get_res = AnswerStructure.model_validate_json(res_str)
    res = get_res.answer
    # res = generate_chat(prompt, model=MODEL_QUIZ, model="qwen2.5:7b", num_predict=5)
    # print('res:\n', res)
    ans = res.strip("[]").split(", ")
    # print('ans:\n', ans)
    return ans

def validate_quiz_answer(y_list, y_hat_list):
    score = 0
    for y, y_hat in zip(y_list, y_hat_list):
        if y == y_hat:
            score += 1
    return score / len(y_list)
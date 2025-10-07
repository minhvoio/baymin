from ollama_helper.ollama_helper import answer_this_prompt
from ollama_helper.prompts import TAKE_QUIZ_PROMPT
from bn_helpers.bn_helpers import AnswerStructure, BnToolBox
from bn_helpers.get_structures_print_tools import get_BN_structure
from bn_helpers.tool_agent import get_answer_from_tool_agent
from benchmarking.quiz_generator import create_dependency_quiz, validate_quiz_answer
from benchmarking.benchmarking_utils import pickTwoRandomNodes

MODEL_QUIZ = "qwen2.5:7b"
MODEL = "gpt-oss:latest"

def get_answer_from_ollama(prompt, model=MODEL, temperature=0.0, max_tokens=1000, format=AnswerStructure.model_json_schema()):
    answer = answer_this_prompt(prompt, model=model, temperature=temperature, max_tokens=max_tokens, format=format)
    print('answer:\n', answer)
    validated_answer = AnswerStructure.model_validate_json(answer)
    return validated_answer.answer

def model_do_quiz(quiz, bn_explanation):
    prompt = TAKE_QUIZ_PROMPT.format(quiz=quiz, bn_explanation=bn_explanation)
    res_str = answer_this_prompt(prompt, format=AnswerStructure.model_json_schema(), model=MODEL_QUIZ)
    get_res = AnswerStructure.model_validate_json(res_str)
    res = get_res.answer
    ans = res.strip("[]").split(", ")
    # print('MODEL QUIZ:', MODEL_QUIZ)
    # print('prompt:\n', prompt)
    # res = generate_chat(prompt, model=MODEL_QUIZ, model="qwen2.5:7b", num_predict=5)
    # print('res:\n', res)
    # print('ans:\n', ans)
    return ans

def create_dependency_question_prompt(net):
    node1, node2 = pickTwoRandomNodes(net)
    bn = get_BN_structure(net)
    prompt = f"In this Bayesian Network:\n{bn}\n"
    prompt += f"Is changing the evidence of {node1} going to change the probability of {node2}? Why?"
    return prompt, node1, node2

def dependency_test(net, model=MODEL, model_quiz=MODEL_QUIZ, max_tokens=1000):
    prompt, node1, node2 = create_dependency_question_prompt(net)
    quiz, y_list = create_dependency_quiz(net, node1, node2, model_quiz=model_quiz)
    print('prompt:\n', prompt)
    print('quiz:\n', quiz)
    print('y_list:\n', y_list)
    print()

    def raw_model_dependency_test():
        ans = get_answer_from_ollama(prompt, model=model, max_tokens=max_tokens)
        y_hat_list = model_do_quiz(quiz, ans)
        
        print('Raw Model:')
        print('ans:\n', ans)
        print('y_list:\n', y_list)
        print('y_hat_list:\n', y_hat_list)
        score = validate_quiz_answer(y_list, y_hat_list)
        return score

    def baymin_dependency_test():
        answer = get_answer_from_tool_agent(net, prompt, model=model, max_tokens=max_tokens)
        y_hat_list = model_do_quiz(quiz, answer)
        
        print('Baymin Model:')
        print('ans:\n', answer)
        print('y_list:\n', y_list)
        print('y_hat_list:\n', y_hat_list)
        score = validate_quiz_answer(y_list, y_hat_list)
        return score

    raw_model_score = raw_model_dependency_test()
    baymin_score = baymin_dependency_test()
    return raw_model_score, baymin_score
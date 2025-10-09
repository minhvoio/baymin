from ollama_helper.ollama_helper import answer_this_prompt
from ollama_helper.prompts import TAKE_QUIZ_PROMPT
from bn_helpers.bn_helpers import AnswerStructure, BnToolBox
from bn_helpers.get_structures_print_tools import get_BN_structure
from bn_helpers.tool_agent import get_answer_from_tool_agent
from benchmarking.quiz_generator import create_dependency_quiz
from benchmarking.benchmarking_utils import pickTwoRandomNodes
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from bn_helpers.constants import MODEL, MODEL_QUIZ, OLLAMA_URL
from question_types import DEPENDENCY_QUESTIONS, COMMON_CAUSE_QUESTIONS, COMMON_EFFECT_QUESTIONS, BLOCKED_EVIDENCES_QUESTIONS, EVIDENCE_CHANGE_RELATIONSHIP_QUESTIONS, PROBABILITY_QUESTIONS, HIGHEST_IMPACT_EVIDENCE_QUESTIONS
from pydantic import BaseModel

import asyncio
try:
    import nest_asyncio
    nest_asyncio.apply()
except Exception:
    # If nest_asyncio isn't available, we proceed; Jupyter may still manage awaits.
    pass

class QuizAnswer(BaseModel):
    one_letter_answer: str

async def get_answer_from_ollama(prompt, model=MODEL):
    ollama_model = OpenAIChatModel(
        model_name=model,
        provider=OllamaProvider(base_url=OLLAMA_URL + 'v1'),  
    )
    agent = Agent(ollama_model, output_type=AnswerStructure)

    result = await agent.run(prompt)
    answer = result.output.answer
    # print('get_answer_from_ollama:\n', answer)
    return answer

def model_do_quiz(quiz, bn_explanation):
    prompt = TAKE_QUIZ_PROMPT.format(quiz=quiz, bn_explanation=bn_explanation)
    res_str = answer_this_prompt(prompt, format=QuizAnswer.model_json_schema(), model=MODEL_QUIZ)
    get_res = QuizAnswer.model_validate_json(res_str)
    res = get_res.one_letter_answer
    # print('MODEL QUIZ:', MODEL_QUIZ)
    # print('prompt:\n', prompt)
    # res = generate_chat(prompt, model=MODEL_QUIZ, model="qwen2.5:7b", num_predict=5)
    # print('res:\n', res)
    # print('ans:\n', ans)
    return res

def validate_quiz_answer_list(y_list, y_hat_list):
    score = 0
    for y, y_hat in zip(y_list, y_hat_list):
        if y == y_hat:
            score += 1
    return score / len(y_list)

def validate_quiz_answer(y, y_hat):
    if y == y_hat:
        return 1
    else:
        return 0


# DEPENDENCY TEST
def create_dependency_question_prompt(net, question_format=None):
    node1, node2 = pickTwoRandomNodes(net)
    bn = get_BN_structure(net)
    question = question_format.format(node1=node1, node2=node2)
    prompt = f"In this Bayesian Network:\n{bn}\n"
    prompt += question
    return prompt, node1, node2, question

def dependency_test(net, model=MODEL, model_quiz=MODEL_QUIZ, max_tokens=1000, num_questions=30):
    raw_model_total_score = 0
    baymin_total_score = 0
    for question in DEPENDENCY_QUESTIONS[:num_questions]:
        prompt, node1, node2, question_output = create_dependency_question_prompt(net, question_format=question)
        quiz, y = create_dependency_quiz(question_output, net, node1, node2, model_quiz=model_quiz)
        # print('prompt:\n', prompt)
        # print('quiz:\n', quiz)
        # print('y_list:\n', y_list)
        # print()

        def raw_model_dependency_test():
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    ans = loop.run_until_complete(get_answer_from_ollama(prompt, model=model))
                else:
                    ans = loop.run_until_complete(get_answer_from_ollama(prompt, model=model))
            except RuntimeError:
                ans = asyncio.run(get_answer_from_ollama(prompt, model=model))
            y_hat = model_do_quiz(quiz, ans)
            
            # print('Raw Model:')
            # print('ans:\n', ans)
            # print('y:\n', y)
            # print('y_hat:\n', y_hat)
            score = validate_quiz_answer(y, y_hat)
            return score

        def baymin_dependency_test():
            answer = get_answer_from_tool_agent(net, prompt, model=model, max_tokens=max_tokens)
            y_hat = model_do_quiz(quiz, answer)
            
            score = validate_quiz_answer(y, y_hat)

            if score < 1:
                print('Baymin Model:')
                print('ans:\n', answer)
                print('y:\n', y)
                print('y_hat_list:\n', y_hat)
                print('---------------------------------------------')
            return score

        raw_model_score = raw_model_dependency_test()
        baymin_score = baymin_dependency_test()
        raw_model_total_score += raw_model_score
        baymin_total_score += baymin_score
        
    return raw_model_total_score / num_questions, baymin_total_score / num_questions
from ollama_helper.ollama_helper import answer_this_prompt
from ollama_helper.prompts import TAKE_QUIZ_PROMPT
from bn_helpers.bn_helpers import AnswerStructure, BnToolBox
from bn_helpers.get_structures_print_tools import get_BN_structure
from bn_helpers.tool_agent import get_answer_from_tool_agent
from benchmarking.quiz_generator import create_dependency_quiz, validate_quiz_answer
from benchmarking.benchmarking_utils import pickTwoRandomNodes
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from bn_helpers.constants import MODEL, MODEL_QUIZ, OLLAMA_URL
from question_types import DEPENDENCY_QUESTIONS, COMMON_CAUSE_QUESTIONS, COMMON_EFFECT_QUESTIONS, BLOCKED_EVIDENCES_QUESTIONS, EVIDENCE_CHANGE_RELATIONSHIP_QUESTIONS, PROBABILITY_QUESTIONS, HIGHEST_IMPACT_EVIDENCE_QUESTIONS

import asyncio
try:
    import nest_asyncio
    nest_asyncio.apply()
except Exception:
    # If nest_asyncio isn't available, we proceed; Jupyter may still manage awaits.
    pass

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
        y_hat_list = model_do_quiz(quiz, ans)
        
        # print('Raw Model:')
        # print('ans:\n', ans)
        # print('y_list:\n', y_list)
        # print('y_hat_list:\n', y_hat_list)
        score = validate_quiz_answer(y_list, y_hat_list)
        return score

    def baymin_dependency_test():
        answer = get_answer_from_tool_agent(net, prompt, model=model, max_tokens=max_tokens)
        y_hat_list = model_do_quiz(quiz, answer)
        
        score = validate_quiz_answer(y_list, y_hat_list)

        if score < 1:
            print('Baymin Model:')
            print('ans:\n', answer)
            print('y_list:\n', y_list)
            print('y_hat_list:\n', y_hat_list)
            print('---------------------------------------------')
        return score

    raw_model_score = raw_model_dependency_test()
    baymin_score = baymin_dependency_test()
    return raw_model_score, baymin_score
from ollama_helper.ollama_helper import answer_this_prompt
from ollama_helper.prompts import TAKE_QUIZ_PROMPT
from bn_helpers.bn_helpers import AnswerStructure, BnToolBox
from bn_helpers.get_structures_print_tools import get_BN_structure, getNetCPTStrings
from bn_helpers.tool_agent import get_answer_from_tool_agent, chat_with_tools
from benchmarking.quiz_generator import (create_dependency_quiz, create_common_cause_quiz, create_common_effect_quiz, create_blocked_evidence_quiz, 
create_evidence_change_relationship_quiz, create_probability_quiz, create_highest_impact_evidence_quiz)
from benchmarking.benchmarking_utils import pick_two_random_nodes, fake_random_nodes, get_random_number_of_nodes, pick_one_random_node, generate_evidence_nodes
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from bn_helpers.constants import MODEL, MODEL_QUIZ, OLLAMA_URL
from benchmarking.question_types import DEPENDENCY_QUESTIONS, COMMON_CAUSE_QUESTIONS, COMMON_EFFECT_QUESTIONS, BLOCKED_EVIDENCES_QUESTIONS, EVIDENCE_CHANGE_RELATIONSHIP_QUESTIONS, PROBABILITY_QUESTIONS, HIGHEST_IMPACT_EVIDENCE_QUESTIONS
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

async def get_answer_from_ollama(prompt, model=MODEL, max_tokens=1000):
    ollama_model = OpenAIChatModel(
        model_name=model,
        provider=OllamaProvider(base_url=OLLAMA_URL + 'v1'),  
    )
    agent = Agent(ollama_model, output_type=AnswerStructure)

    # result = await agent.run(prompt)
    # answer = result.output.answer
    # # print('get_answer_from_ollama:\n', answer)
    # return answer
    result = await agent.run(
    prompt,
    model_settings={
        "max_tokens": max_tokens,   # ✅ use this key inside model_settings
        # "temperature": 0.3,  # optional
    }
    )
    return result.output.answer

def model_do_quiz(quiz, bn_explanation, model=MODEL_QUIZ):
    prompt = TAKE_QUIZ_PROMPT.format(quiz=quiz, bn_explanation=bn_explanation)
    res_str = answer_this_prompt(prompt, format=QuizAnswer.model_json_schema(), model=model)
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

def two_nodes_question(net, question_format=None, hasEvidence=False):
    node1, node2 = pick_two_random_nodes(net)
    bn = get_BN_structure(net)
    prompt = f"In this Bayesian Network:\n{bn}\n"
    if hasEvidence:
        evidence = generate_evidence_nodes(net, (node1, node2))
        evidence_str = ", ".join(evidence) if evidence else "∅"
        question = question_format.format(node1=node1, node2=node2, evidence=evidence_str)
        prompt += question
        return prompt, node1, node2, question, evidence
    else:
        question = question_format.format(node1=node1, node2=node2)
        prompt += question
        return prompt, node1, node2, question

def probability_question(net, question_format=None, hasEvidence=False):
    node = pick_one_random_node(net)
    bn = get_BN_structure(net)
    prompt = f"In this Bayesian Network:\n{bn}\n"
    prompt += f"CPT:\n{getNetCPTStrings(net)}\n"
    if hasEvidence:
        evidence = generate_evidence_nodes(net, (node,))
        evidence_str = ", ".join(evidence) if evidence else "∅"
        question = question_format.format(node=node, evidence=evidence_str)
        prompt += question
        return prompt, node, question, evidence
    else:
        question = question_format.format(node=node)
        prompt += question
        return prompt, node, question

def raw_model_test(prompt, quiz, y, model=MODEL, max_tokens=1000, model_quiz=MODEL_QUIZ):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            ans = loop.run_until_complete(get_answer_from_ollama(prompt, model=model, max_tokens=max_tokens))
        else:
            ans = loop.run_until_complete(get_answer_from_ollama(prompt, model=model, max_tokens=max_tokens))
    except RuntimeError:
        ans = asyncio.run(get_answer_from_ollama(prompt, model=model, max_tokens=max_tokens))
    y_hat = model_do_quiz(quiz, ans, model=model_quiz)
    
    score = validate_quiz_answer(y, y_hat)
    return score

def baymin_test(net, quiz, y, question_output, model=MODEL, max_tokens=1000, model_quiz=MODEL_QUIZ):
    answer = chat_with_tools(net, question_output, model=model, max_tokens=max_tokens)
    y_hat = model_do_quiz(quiz, answer, model=model_quiz)
    
    score = validate_quiz_answer(y, y_hat)

    if score < 1:
        print('quiz:\n', quiz)
        print('y:\n', y)
        print('y_hat:\n', y_hat)
        print('---------------------------------------------')
        print('Baymin Model:')
        print('ans:\n', answer)
        print('y:\n', y)
        print('y_hat:\n', y_hat)
        print('---------------------------------------------')
    return score

# DEPENDENCY TEST
def elementary_test(net, question_set, create_quiz_function, model=MODEL, model_quiz=MODEL_QUIZ, hasEvidence=False, max_tokens=1000, num_questions=30):
    raw_model_total_score = 0
    baymin_total_score = 0
    for question in question_set[:num_questions]:
        if hasEvidence:
            prompt, node1, node2, question_output, evidence = two_nodes_question(net, question_format=question, hasEvidence=hasEvidence)
            quiz, y = create_quiz_function(question_output, net, node1, node2, evidence)
        else:
            prompt, node1, node2, question_output = two_nodes_question(net, question_format=question, hasEvidence=hasEvidence)
            quiz, y = create_quiz_function(question_output, net, node1, node2)
        # print('prompt:\n', prompt)
        # print('quiz:\n', quiz)
        # print('y_list:\n', y_list)
        # print()

        raw_model_score = raw_model_test(prompt, quiz, y, model=model, max_tokens=max_tokens, model_quiz=model_quiz)
        baymin_score = baymin_test(net, quiz, y, question_output, model=model, max_tokens=max_tokens, model_quiz=model_quiz)
        raw_model_total_score += raw_model_score
        baymin_total_score += baymin_score
        
    return raw_model_total_score / num_questions, baymin_total_score / num_questions

def dependency_test(net, model=MODEL, model_quiz=MODEL_QUIZ, max_tokens=1000, num_questions=30):
    return elementary_test(net, DEPENDENCY_QUESTIONS, create_dependency_quiz, model=model, model_quiz=model_quiz, max_tokens=max_tokens, num_questions=num_questions)

def common_cause_test(net, model=MODEL, model_quiz=MODEL_QUIZ, max_tokens=1000, num_questions=30):
    return elementary_test(net, COMMON_CAUSE_QUESTIONS, create_common_cause_quiz, model=model, model_quiz=model_quiz, max_tokens=max_tokens, num_questions=num_questions)

def common_effect_test(net, model=MODEL, model_quiz=MODEL_QUIZ, max_tokens=1000, num_questions=30):
    return elementary_test(net, COMMON_EFFECT_QUESTIONS, create_common_effect_quiz, model=model, model_quiz=model_quiz, max_tokens=max_tokens, num_questions=num_questions)

def blocked_evidence_test(net, model=MODEL, model_quiz=MODEL_QUIZ, max_tokens=1000, num_questions=30):
    return elementary_test(net, BLOCKED_EVIDENCES_QUESTIONS, create_blocked_evidence_quiz, model=model, model_quiz=model_quiz, max_tokens=max_tokens, num_questions=num_questions)

def evidence_change_relationship_test(net, model=MODEL, model_quiz=MODEL_QUIZ, max_tokens=1000, num_questions=30, hasEvidence=True):
    return elementary_test(net, EVIDENCE_CHANGE_RELATIONSHIP_QUESTIONS, create_evidence_change_relationship_quiz, model=model, model_quiz=model_quiz, max_tokens=max_tokens, num_questions=num_questions, hasEvidence=hasEvidence)

def numerical_test(net, question_set, create_quiz_function, model=MODEL, model_quiz=MODEL_QUIZ, hasEvidence=False, max_tokens=1000, num_questions=30):
    raw_model_total_score = 0
    baymin_total_score = 0
    for question in question_set[:num_questions]:
        if hasEvidence:
            prompt, node, question_output, evidence = probability_question(net, question_format=question, hasEvidence=hasEvidence)
            quiz, y = create_quiz_function(question_output, net, node, evidence)
        else:
            prompt, node, question_output = probability_question(net, question_format=question, hasEvidence=hasEvidence)
            quiz, y = create_quiz_function(question_output, net, node)

        raw_model_score = raw_model_test(prompt, quiz, y, model=model, max_tokens=max_tokens, model_quiz=model_quiz)
        baymin_score = baymin_test(net, quiz, y, question_output, model=model, max_tokens=max_tokens, model_quiz=model_quiz)
        raw_model_total_score += raw_model_score
        baymin_total_score += baymin_score
        
    return raw_model_total_score / num_questions, baymin_total_score / num_questions

def probability_test(net, model=MODEL, model_quiz=MODEL_QUIZ, max_tokens=1000, num_questions=30, hasEvidence=True):
    return numerical_test(net, PROBABILITY_QUESTIONS, create_probability_quiz, model=model, model_quiz=model_quiz, max_tokens=max_tokens, num_questions=num_questions, hasEvidence=hasEvidence)

def highest_impact_evidence_test(net, model=MODEL, model_quiz=MODEL_QUIZ, max_tokens=1000, num_questions=30, hasEvidence=False):
    return numerical_test(net, HIGHEST_IMPACT_EVIDENCE_QUESTIONS, create_highest_impact_evidence_quiz, model=model, model_quiz=model_quiz, max_tokens=max_tokens, num_questions=num_questions, hasEvidence=hasEvidence)
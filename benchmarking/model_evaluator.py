from ollama_helper.ollama_helper import answer_this_prompt
from ollama_helper.prompts import TAKE_QUIZ_PROMPT
from bn_helpers.bn_helpers import BnToolBox
from ollama_helper.structure_output import QuizAnswer
from bn_helpers.get_structures_print_tools import get_BN_structure, getNetCPTStrings
from bn_helpers.tool_agent import get_answer_from_tool_agent
from benchmarking.quiz_generator import (create_dependency_quiz, create_common_cause_quiz, create_common_effect_quiz, create_blocked_evidence_quiz, 
create_evidence_change_relationship_quiz, create_probability_quiz, create_highest_impact_evidence_quiz)
from benchmarking.benchmarking_utils import (pick_two_random_nodes, fake_random_nodes, 
get_random_number_of_nodes, pick_one_random_node, generate_evidence_nodes, log_test_result, log_for_baymin_testing, get_completed_questions)
from ollama_helper.ollama_helper import get_answer_from_ollama, get_quiz_answer_from_thinking_model
import asyncio
import time
from bn_helpers.constants import MODEL, MODEL_QUIZ
from benchmarking.question_types import DEPENDENCY_QUESTIONS, COMMON_CAUSE_QUESTIONS, COMMON_EFFECT_QUESTIONS, BLOCKED_EVIDENCES_QUESTIONS, EVIDENCE_CHANGE_RELATIONSHIP_QUESTIONS, PROBABILITY_QUESTIONS, HIGHEST_IMPACT_EVIDENCE_QUESTIONS

def model_do_quiz(
    quiz,
    bn_explanation,
    *,
    model=MODEL_QUIZ,
    temperature_quiz: float = 0.7,
    max_tokens: int = 1000,
    top_p: float = 0.9,
    quiz_dict=None,
):
    """
    Majority vote over up to 5 runs with per-run shuffling using structured quiz dict when provided.
    - Uses temperature=0.9, top_p=0.9, seeds 0..4
    - Early-stops when any answer reaches 3 votes
    - Falls back to a single run on the given quiz text if no dict is provided
    """
    import random as _random

    if not quiz_dict or not isinstance(quiz_dict, dict) or not quiz_dict.get("options"):
        prompt = TAKE_QUIZ_PROMPT.format(quiz=quiz, bn_explanation=bn_explanation)
        return get_quiz_answer_from_thinking_model(
            prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature_quiz,
            top_p=top_p,
            seed=0,
        )

    header = quiz_dict.get("header", "")
    options_struct = quiz_dict["options"]  

    vote_counts = {"A": 0, "B": 0, "C": 0, "D": 0}

    def build_quiz_from_presented(presented):
        out_lines = [header]
        for idx, opt in enumerate(presented):
            letter = chr(65 + idx)
            out_lines.append("--------------------------------")
            out_lines.append(f"{letter}. {opt['text']}")
        return "\n".join(out_lines)

    for i in range(5):
        rng = _random.Random(i)
        presented = list(options_struct)
        rng.shuffle(presented)

        mapping = {}
        for idx, opt in enumerate(presented):
            presented_letter = chr(65 + idx)
            mapping[presented_letter] = opt["original_letter"]

        shuffled_quiz = build_quiz_from_presented(presented)
        prompt = TAKE_QUIZ_PROMPT.format(quiz=shuffled_quiz, bn_explanation=bn_explanation)

        picked = get_quiz_answer_from_thinking_model(
            prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature_quiz,
            top_p=top_p,
            seed=i,
        )

        mapped = mapping.get(str(picked).strip().upper()[:1])
        if mapped in vote_counts:
            vote_counts[mapped] += 1
        if any(count >= 3 for count in vote_counts.values()):
            break

    best_letter = max(["A", "B", "C", "D"], key=lambda L: (vote_counts[L], -ord(L)))
    return best_letter

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

def two_nodes_question(net, question_format=None, has_evidence=False):
    node1, node2 = pick_two_random_nodes(net)
    bn = get_BN_structure(net)
    prompt = f"In this Bayesian Network:\n{bn}\n"
    if has_evidence:
        evidence = generate_evidence_nodes(net, (node1, node2))
        evidence_str = ", ".join(evidence) if evidence else "∅"
        question = question_format.format(node1=node1, node2=node2, evidence=evidence_str)
        prompt += question
        return prompt, node1, node2, question, evidence
    else:
        question = question_format.format(node1=node1, node2=node2)
        prompt += question
        return prompt, node1, node2, question

def probability_question(net, question_format=None, has_evidence=False):
    node = pick_one_random_node(net)
    bn = get_BN_structure(net)
    prompt = f"In this Bayesian Network:\n{bn}\n"
    prompt += f"CPT:\n{getNetCPTStrings(net)}\n"
    if has_evidence:
        evidence = generate_evidence_nodes(net, (node,))
        evidence_str = ", ".join(evidence) if evidence else "∅"
        question = question_format.format(node=node, evidence=evidence_str)
        prompt += question
        return prompt, node, question, evidence
    else:
        question = question_format.format(node=node)
        prompt += question
        return prompt, node, question

def raw_model_test(
    prompt,
    quiz,
    y,
    *,
    model=MODEL,
    max_tokens=1000,
    model_quiz=MODEL_QUIZ,
    is_output_log=False,
    quiz_dict=None,
    model_temperature: float = 0.0,
    model_quiz_temperature: float = 0.7,
    model_top_p: float = 1.0,
    model_quiz_top_p: float = 0.9,
):
    # Model-specific params only
    effective_model_temp = model_quiz_temperature
    effective_model_top_p = model_quiz_top_p

    effective_quiz_temp = model_quiz_temperature
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            start_time = time.time()
            ans = loop.run_until_complete(
                get_answer_from_ollama(
                    prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=effective_model_temp,
                    top_p=effective_model_top_p,
                )
            )
            raw_response_time = time.time() - start_time
        else:
            start_time = time.time()
            ans = loop.run_until_complete(
                get_answer_from_ollama(
                    prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=effective_model_temp,
                    top_p=effective_model_top_p,
                )
            )
            raw_response_time = time.time() - start_time
    except RuntimeError:
        start_time = time.time()
        ans = asyncio.run(
            get_answer_from_ollama(
                prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=effective_model_temp,
                top_p=effective_model_top_p,
            )
        )
        raw_response_time = time.time() - start_time
    
    quiz_start_time = time.time()
    y_hat = model_do_quiz(
        quiz,
        ans,
        model=model_quiz,
        temperature_quiz=effective_quiz_temp,
        max_tokens=max_tokens,
        top_p=model_quiz_top_p,
        quiz_dict=quiz_dict,
    )
    quiz_time = time.time() - quiz_start_time
    
    if is_output_log:
        print(f"[Raw Model Testing] Response time: {raw_response_time:.2f}s, Quiz time: {quiz_time:.2f}s, Total: {raw_response_time + quiz_time:.2f}s")
    
    score = validate_quiz_answer(y, y_hat)
    return score, ans

def baymin_test(
    net,
    quiz,
    y,
    question_output,
    *,
    model=MODEL,
    max_tokens=1000,
    model_quiz=MODEL_QUIZ,
    is_output_log=False,
    quiz_dict=None,
    model_temperature: float = 0.0,
    model_quiz_temperature: float = 0.7,
    model_top_p: float = 1.0,
    model_quiz_top_p: float = 0.9,
):
    effective_model_temp = model_temperature
    effective_quiz_temp = model_quiz_temperature
    start_time = time.time()
    answer, testing_log = get_answer_from_tool_agent(
        net,
        question_output,
        model=model,
        temperature=effective_model_temp,
        max_tokens=max_tokens,
        is_output_log=is_output_log,
        model_top_p=model_top_p,
        model_temperature=model_temperature,
    )
    baymin_response_time = time.time() - start_time
    
    quiz_start_time = time.time()
    y_hat = model_do_quiz(
        quiz,
        answer,
        model=model_quiz,
        temperature_quiz=effective_quiz_temp,
        max_tokens=max_tokens,
        top_p=model_quiz_top_p,
        quiz_dict=quiz_dict,
    )
    quiz_time = time.time() - quiz_start_time
    
    score = validate_quiz_answer(y, y_hat)

    if is_output_log:
        print(f"[BayMin Testing] Response time: {baymin_response_time:.2f}s, Quiz time: {quiz_time:.2f}s, Total: {baymin_response_time + quiz_time:.2f}s")

    if score < 1:
        log_for_baymin_testing(quiz, y, y_hat, answer, testing_log)
            
    return score, answer

# DEPENDENCY TEST
def elementary_test(
    net,
    question_set,
    create_quiz_function,
    *,
    model=MODEL,
    model_quiz=MODEL_QUIZ,
    has_evidence=False,
    max_tokens=1000,
    num_questions=30,
    is_output_log=False,
    test_baymin_only=False,
    model_temperature: float = 0.0,
    model_quiz_temperature: float = 0.7,
    model_top_p: float = 1.0,
    model_quiz_top_p: float = 0.9,
):
    raw_model_total_score = 0
    baymin_total_score = 0
    
    # Get question set name from function name
    question_set_name = create_quiz_function.__name__.replace('create_', '').replace('_quiz', '')
    network_size = len(net.nodes())
    
    # Check for already completed questions
    completed_questions = get_completed_questions('elementary_test', question_set_name, model, model_quiz, network_size, test_baymin_only)
    
    print(f"Running {question_set_name} test for Net_{network_size} with {num_questions} questions")
    if completed_questions:
        print(f"Found {len(completed_questions)} already completed questions: {sorted(completed_questions)}")
        print(f"Skipping completed questions and continuing from where we left off...")
    
    for i, question in enumerate(question_set[:num_questions]):
        question_index = i + 1
        
        # Skip if already completed
        if question_index in completed_questions:
            print(f"Skipping question {question_index} (already completed)")
            continue
            
        print(f"Running question {question_index}")
        if has_evidence:
            prompt, node1, node2, question_output, evidence = two_nodes_question(net, question_format=question, has_evidence=has_evidence)
            quiz, y, quiz_dict = create_quiz_function(question_output, net, node1, node2, evidence)
        else:
            prompt, node1, node2, question_output = two_nodes_question(net, question_format=question, has_evidence=has_evidence)
            quiz, y, quiz_dict = create_quiz_function(question_output, net, node1, node2)
            evidence = None
        
        if is_output_log:
            print(f"[Testing Mode] Question {question_index}: {question_output[:100]}{'...' if len(question_output) > 100 else ''}")

        if test_baymin_only:
            # Only run BayMin test
            start_time = time.time()
            baymin_score, baymin_answer = baymin_test(
                net,
                quiz,
                y,
                question_output,
                model=model,
                max_tokens=max_tokens,
                model_quiz=model_quiz,
                is_output_log=is_output_log,
                quiz_dict=quiz_dict,
                model_temperature=model_temperature,
                model_quiz_temperature=model_quiz_temperature,
                model_top_p=model_top_p,
                model_quiz_top_p=model_quiz_top_p,
            )
            baymin_runtime = time.time() - start_time
            raw_model_score, raw_model_answer = 0, "N/A (BayMin only)"
            raw_model_runtime = None
            baymin_total_score += baymin_score
        else:
            # Run both tests
            start_time = time.time()
            raw_model_score, raw_model_answer = raw_model_test(
                prompt,
                quiz,
                y,
                model=model,
                max_tokens=max_tokens,
                model_quiz=model_quiz,
                is_output_log=is_output_log,
                quiz_dict=quiz_dict,
                model_temperature=model_temperature,
                model_quiz_temperature=model_quiz_temperature,
                model_top_p=model_top_p,
                model_quiz_top_p=model_quiz_top_p,
            )
            raw_model_runtime = time.time() - start_time
            
            start_time = time.time()
            baymin_score, baymin_answer = baymin_test(
                net,
                quiz,
                y,
                question_output,
                model=model,
                max_tokens=max_tokens,
                model_quiz=model_quiz,
                is_output_log=is_output_log,
                quiz_dict=quiz_dict,
                model_temperature=model_temperature,
                model_quiz_temperature=model_quiz_temperature,
                model_top_p=model_top_p,
                model_quiz_top_p=model_quiz_top_p,
            )
            baymin_runtime = time.time() - start_time
            
            raw_model_total_score += raw_model_score
            baymin_total_score += baymin_score
        
        # Log test result
        output_file = 'baymin_test_log.csv' if test_baymin_only else 'test_log.csv'
        log_test_result(
            test_type='elementary_test',
            question_set_name=question_set_name,
            question_index=question_index,
            quiz=quiz,
            expected_answer=y,
            model=model,
            model_quiz=model_quiz,
            raw_model_score=raw_model_score,
            baymin_score=baymin_score,
            question_output=question_output,  # optional
            prompt=prompt,  # optional
            hasEvidence=has_evidence,  # optional
            max_tokens=max_tokens,  # optional
            network_size=network_size,  # optional
            node1=node1,  # optional
            node2=node2,  # optional
            evidence=str(evidence) if evidence else None,  # optional
            raw_model_answer=raw_model_answer,
            baymin_answer=baymin_answer,
            raw_model_runtime=raw_model_runtime,
            baymin_runtime=baymin_runtime,
            output_file=output_file
        )
    
    # Calculate final scores based on total questions
    total_questions_run = num_questions - len(completed_questions)
    if total_questions_run > 0:
        final_raw_score = raw_model_total_score / total_questions_run
        final_baymin_score = baymin_total_score / total_questions_run
    else:
        final_raw_score = 0
        final_baymin_score = 0
        
    print(f"Test completed: {total_questions_run} new questions run, {len(completed_questions)} skipped")
    return final_raw_score, final_baymin_score

def dependency_test(
    net,
    *,
    model=MODEL,
    model_quiz=MODEL_QUIZ,
    max_tokens=1000,
    num_questions=30,
    is_output_log=False,
    test_baymin_only=False,
    model_temperature: float = 0.0,
    model_quiz_temperature: float = 0.7,
    model_top_p: float = 1.0,
    model_quiz_top_p: float = 0.9,
):
    return elementary_test(
        net,
        DEPENDENCY_QUESTIONS,
        create_dependency_quiz,
        model=model,
        model_quiz=model_quiz,
        max_tokens=max_tokens,
        num_questions=num_questions,
        is_output_log=is_output_log,
        test_baymin_only=test_baymin_only,
        model_temperature=model_temperature,
        model_quiz_temperature=model_quiz_temperature,
        model_top_p=model_top_p,
        model_quiz_top_p=model_quiz_top_p,
    )

def common_cause_test(
    net,
    *,
    model=MODEL,
    model_quiz=MODEL_QUIZ,
    max_tokens=1000,
    num_questions=30,
    is_output_log=False,
    test_baymin_only=False,
    model_temperature: float = 0.0,
    model_quiz_temperature: float = 0.7,
    model_top_p: float = 1.0,
    model_quiz_top_p: float = 0.9,
):
    return elementary_test(
        net,
        COMMON_CAUSE_QUESTIONS,
        create_common_cause_quiz,
        model=model,
        model_quiz=model_quiz,
        max_tokens=max_tokens,
        num_questions=num_questions,
        is_output_log=is_output_log,
        test_baymin_only=test_baymin_only,
        model_temperature=model_temperature,
        model_quiz_temperature=model_quiz_temperature,
        model_top_p=model_top_p,
        model_quiz_top_p=model_quiz_top_p,
    )

def common_effect_test(
    net,
    *,
    model=MODEL,
    model_quiz=MODEL_QUIZ,
    max_tokens=1000,
    num_questions=30,
    is_output_log=False,
    test_baymin_only=False,
    model_temperature: float = 0.0,
    model_quiz_temperature: float = 0.7,
    model_top_p: float = 1.0,
    model_quiz_top_p: float = 0.9,
):
    return elementary_test(
        net,
        COMMON_EFFECT_QUESTIONS,
        create_common_effect_quiz,
        model=model,
        model_quiz=model_quiz,
        max_tokens=max_tokens,
        num_questions=num_questions,
        is_output_log=is_output_log,
        test_baymin_only=test_baymin_only,
        model_temperature=model_temperature,
        model_quiz_temperature=model_quiz_temperature,
        model_top_p=model_top_p,
        model_quiz_top_p=model_quiz_top_p,
    )

def blocked_evidence_test(
    net,
    *,
    model=MODEL,
    model_quiz=MODEL_QUIZ,
    max_tokens=1000,
    num_questions=30,
    is_output_log=False,
    test_baymin_only=False,
    model_temperature: float = 0.0,
    model_quiz_temperature: float = 0.7,
    model_top_p: float = 1.0,
    model_quiz_top_p: float = 0.9,
):
    return elementary_test(
        net,
        BLOCKED_EVIDENCES_QUESTIONS,
        create_blocked_evidence_quiz,
        model=model,
        model_quiz=model_quiz,
        max_tokens=max_tokens,
        num_questions=num_questions,
        is_output_log=is_output_log,
        test_baymin_only=test_baymin_only,
        model_temperature=model_temperature,
        model_quiz_temperature=model_quiz_temperature,
        model_top_p=model_top_p,
        model_quiz_top_p=model_quiz_top_p,
    )

def evidence_change_relationship_test(
    net,
    *,
    model=MODEL,
    model_quiz=MODEL_QUIZ,
    max_tokens=1000,
    num_questions=30,
    has_evidence=True,
    is_output_log=False,
    test_baymin_only=False,
    model_temperature: float = 0.0,
    model_quiz_temperature: float = 0.7,
    model_top_p: float = 1.0,
    model_quiz_top_p: float = 0.9,
):
    return elementary_test(
        net,
        EVIDENCE_CHANGE_RELATIONSHIP_QUESTIONS,
        create_evidence_change_relationship_quiz,
        model=model,
        model_quiz=model_quiz,
        max_tokens=max_tokens,
        num_questions=num_questions,
        has_evidence=has_evidence,
        is_output_log=is_output_log,
        test_baymin_only=test_baymin_only,
        model_temperature=model_temperature,
        model_quiz_temperature=model_quiz_temperature,
        model_top_p=model_top_p,
        model_quiz_top_p=model_quiz_top_p,
    )

def numerical_test(
    net,
    question_set,
    create_quiz_function,
    *,
    model=MODEL,
    model_quiz=MODEL_QUIZ,
    has_evidence=False,
    max_tokens=1000,
    num_questions=30,
    is_output_log=False,
    test_baymin_only=False,
    model_temperature: float = 0.0,
    model_quiz_temperature: float = 0.7,
    model_top_p: float = 1.0,
    model_quiz_top_p: float = 0.9,
):
    
    raw_model_total_score = 0
    baymin_total_score = 0
    
    # Get question set name from function name
    question_set_name = create_quiz_function.__name__.replace('create_', '').replace('_quiz', '')
    network_size = len(net.nodes())
    
    # Check for already completed questions
    completed_questions = get_completed_questions('numerical_test', question_set_name, model, model_quiz, network_size, test_baymin_only)
    
    print(f"Running {question_set_name} test for Net_{network_size} with {num_questions} questions")
    if completed_questions:
        print(f"Found {len(completed_questions)} already completed questions: {sorted(completed_questions)}")
        print(f"Skipping completed questions and continuing from where we left off...")
    
    for i, question in enumerate(question_set[:num_questions]):
        question_index = i + 1
        
        # Skip if already completed
        if question_index in completed_questions:
            print(f"Skipping question {question_index} (already completed)")
            continue
            
        print(f"Running question {question_index}")
        if has_evidence:
            prompt, node, question_output, evidence = probability_question(net, question_format=question, has_evidence=has_evidence)
            quiz, y, quiz_dict = create_quiz_function(question_output, net, node, evidence)
        else:
            prompt, node, question_output = probability_question(net, question_format=question, has_evidence=has_evidence)
            quiz, y, quiz_dict = create_quiz_function(question_output, net, node)
            evidence = None
        
        if is_output_log:
            print(f"[Testing Mode] Question {question_index}: {question_output[:100]}{'...' if len(question_output) > 100 else ''}")

        if test_baymin_only:
            # Only run BayMin test
            start_time = time.time()
            baymin_score, baymin_answer = baymin_test(
                net,
                quiz,
                y,
                question_output,
                model=model,
                max_tokens=max_tokens,
                model_quiz=model_quiz,
                is_output_log=is_output_log,
                quiz_dict=quiz_dict,
                model_temperature=model_temperature,
                model_quiz_temperature=model_quiz_temperature,
                model_top_p=model_top_p,
                model_quiz_top_p=model_quiz_top_p,
            )
            baymin_runtime = time.time() - start_time
            raw_model_score, raw_model_answer = 0, "N/A (BayMin only)"
            raw_model_runtime = None
            baymin_total_score += baymin_score
        else:
            # Run both tests
            start_time = time.time()
            raw_model_score, raw_model_answer = raw_model_test(
                prompt,
                quiz,
                y,
                model=model,
                max_tokens=max_tokens,
                model_quiz=model_quiz,
                is_output_log=is_output_log,
                quiz_dict=quiz_dict,
                model_temperature=model_temperature,
                model_quiz_temperature=model_quiz_temperature,
                model_top_p=model_top_p,
                model_quiz_top_p=model_quiz_top_p,
            )
            raw_model_runtime = time.time() - start_time
            
            start_time = time.time()
            baymin_score, baymin_answer = baymin_test(
                net,
                quiz,
                y,
                question_output,
                model=model,
                max_tokens=max_tokens,
                model_quiz=model_quiz,
                is_output_log=is_output_log,
                quiz_dict=quiz_dict,
                model_temperature=model_temperature,
                model_quiz_temperature=model_quiz_temperature,
                model_top_p=model_top_p,
                model_quiz_top_p=model_quiz_top_p,
            )
            baymin_runtime = time.time() - start_time
            
            raw_model_total_score += raw_model_score
            baymin_total_score += baymin_score
        
        # Log test result
        output_file = 'baymin_test_log.csv' if test_baymin_only else 'test_log.csv'
        log_test_result(
            test_type='numerical_test',
            question_set_name=question_set_name,
            question_index=question_index,
            quiz=quiz,
            expected_answer=y,
            model=model,
            model_quiz=model_quiz,
            raw_model_score=raw_model_score,
            baymin_score=baymin_score,
            question_output=question_output,  # optional
            prompt=prompt,  # optional
            hasEvidence=has_evidence,  # optional
            max_tokens=max_tokens,  # optional
            network_size=network_size,  # optional
            node=node,  # optional
            evidence=str(evidence) if evidence else None,  # optional
            raw_model_answer=raw_model_answer,
            baymin_answer=baymin_answer,
            raw_model_runtime=raw_model_runtime,
            baymin_runtime=baymin_runtime,
            output_file=output_file
        )
    
    # Calculate final scores based on total questions
    total_questions_run = num_questions - len(completed_questions)
    if total_questions_run > 0:
        final_raw_score = raw_model_total_score / total_questions_run
        final_baymin_score = baymin_total_score / total_questions_run
    else:
        final_raw_score = 0
        final_baymin_score = 0
        
    print(f"Test completed: {total_questions_run} new questions run, {len(completed_questions)} skipped")
    return final_raw_score, final_baymin_score

def probability_test(
    net,
    *,
    model=MODEL,
    model_quiz=MODEL_QUIZ,
    max_tokens=1000,
    num_questions=30,
    has_evidence=True,
    is_output_log=False,
    test_baymin_only=False,
    model_temperature: float = 0.0,
    model_quiz_temperature: float = 0.7,
    model_top_p: float = 1.0,
    model_quiz_top_p: float = 0.9,
):
    return numerical_test(
        net,
        PROBABILITY_QUESTIONS,
        create_probability_quiz,
        model=model,
        model_quiz=model_quiz,
        max_tokens=max_tokens,
        num_questions=num_questions,
        has_evidence=has_evidence,
        is_output_log=is_output_log,
        test_baymin_only=test_baymin_only,
        model_temperature=model_temperature,
        model_quiz_temperature=model_quiz_temperature,
        model_top_p=model_top_p,
        model_quiz_top_p=model_quiz_top_p,
    )

def highest_impact_evidence_test(
    net,
    *,
    model=MODEL,
    model_quiz=MODEL_QUIZ,
    max_tokens=1000,
    num_questions=30,
    has_evidence=False,
    is_output_log=False,
    test_baymin_only=False,
    model_temperature: float = 0.0,
    model_quiz_temperature: float = 0.7,
    model_top_p: float = 1.0,
    model_quiz_top_p: float = 0.9,
):
    return numerical_test(
        net,
        HIGHEST_IMPACT_EVIDENCE_QUESTIONS,
        create_highest_impact_evidence_quiz,
        model=model,
        model_quiz=model_quiz,
        max_tokens=max_tokens,
        num_questions=num_questions,
        has_evidence=has_evidence,
        is_output_log=is_output_log,
        test_baymin_only=test_baymin_only,
        model_temperature=model_temperature,
        model_quiz_temperature=model_quiz_temperature,
        model_top_p=model_top_p,
        model_quiz_top_p=model_quiz_top_p,
    )
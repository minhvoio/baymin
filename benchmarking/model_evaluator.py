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
from ollama_helper.ollama_helper import get_answer_from_ollama, get_quiz_answer_from_thinking_model_sync
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
    is_debug: bool = False,
):
    """
    Majority vote over up to 5 runs with per-run shuffling using structured quiz dict when provided.
    - Uses temperature=0.9, top_p=0.9, seeds 0..4
    - Early-stops when any answer reaches 3 votes
    - Falls back to a single run on the given quiz text if no dict is provided
    """
    if is_debug:
        print(f"[DEBUG] model_do_quiz called with model={model}, quiz_dict type={type(quiz_dict)}")
        print(f"[DEBUG] bn_explanation type={type(bn_explanation)}, length={len(str(bn_explanation)) if bn_explanation else 0}")
    import random as _random

    if not quiz_dict or not isinstance(quiz_dict, dict) or not quiz_dict.get("options"):
        # Escape braces to avoid str.format errors when model output contains '{' or '}'
        safe_quiz = str(quiz).replace('{', '{{').replace('}', '}}')
        safe_explanation = str(bn_explanation).replace('{', '{{').replace('}', '}}')
        prompt = TAKE_QUIZ_PROMPT.format(quiz=safe_quiz, bn_explanation=safe_explanation)
        if is_debug:
            print(f"[DEBUG] Using fallback quiz prompt (no quiz_dict)")
            print(f"[DEBUG] Prompt length: {len(prompt)}")
        return get_quiz_answer_from_thinking_model_sync(
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
        
        if is_debug:
            print(f"[DEBUG] Run {i+1} mapping: {mapping}")
            print(f"[DEBUG] Run {i+1} presented options: {[opt['text'][:50] + '...' for opt in presented]}")

        shuffled_quiz = build_quiz_from_presented(presented)
        safe_quiz = str(shuffled_quiz).replace('{', '{{').replace('}', '}}')
        safe_explanation = str(bn_explanation).replace('{', '{{').replace('}', '}}')
        prompt = TAKE_QUIZ_PROMPT.format(quiz=safe_quiz, bn_explanation=safe_explanation)

        if is_debug:
            print(f"[DEBUG] Quiz run {i+1}/5, prompt length: {len(prompt)}")
        picked = get_quiz_answer_from_thinking_model_sync(
            prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature_quiz,
            top_p=top_p,
            seed=i,
        )
        if is_debug:
            print(f"[DEBUG] Quiz run {i+1} picked: {picked}")

        mapped = mapping.get(str(picked).strip().upper()[:1])
        if is_debug:
            print(f"[DEBUG] Run {i+1} picked '{picked}' -> mapped to '{mapped}'")
        if mapped in vote_counts:
            vote_counts[mapped] += 1
            if is_debug:
                print(f"[DEBUG] Vote counts after run {i+1}: {vote_counts}")
        if any(count >= 3 for count in vote_counts.values()):
            break

    best_letter = max(["A", "B", "C", "D"], key=lambda L: (vote_counts[L], -ord(L)))
    if is_debug:
        print(f"[DEBUG] Final vote counts: {vote_counts}, best_letter: {best_letter}")
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
    is_debug: bool = False,
):
    # Model-specific params only
    effective_model_temp = model_quiz_temperature
    effective_model_top_p = model_quiz_top_p

    effective_quiz_temp = model_quiz_temperature
    
    if is_debug:
        print(f"[DEBUG] raw_model_test called with model={model}, prompt length={len(prompt)}")
        print(f"[DEBUG] quiz type={type(quiz)}, quiz_dict type={type(quiz_dict)}")
        print(f"[DEBUG] expected answer y={y}")
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            start_time = time.time()
            if is_debug:
                print(f"[DEBUG] Getting answer from ollama (loop running) with model={model}")
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
            if is_debug:
                print(f"[DEBUG] Got answer from ollama (loop running), type={type(ans)}, length={len(str(ans)) if ans else 0}")
                print(f"[DEBUG] Answer preview: {str(ans)[:200] if ans else 'None'}...")
        else:
            start_time = time.time()
            if is_debug:
                print(f"[DEBUG] Getting answer from ollama (else branch) with model={model}")
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
            if is_debug:
                print(f"[DEBUG] Got answer from ollama (else branch), type={type(ans)}, length={len(str(ans)) if ans else 0}")
    except RuntimeError:
        start_time = time.time()
        if is_debug:
            print(f"[DEBUG] Getting answer from ollama (RuntimeError branch) with model={model}")
        ans = asyncio.run(
            get_answer_from_ollama(
                prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=effective_model_temp,
                top_p=effective_model_top_p,
                is_debug=is_debug,
            )
        )
        raw_response_time = time.time() - start_time
        if is_debug:
            print(f"[DEBUG] Got answer from ollama (RuntimeError branch), type={type(ans)}, length={len(str(ans)) if ans else 0}")
    
    quiz_start_time = time.time()
    if is_debug:
        print(f"[DEBUG] Starting quiz with answer type={type(ans)}, quiz_dict type={type(quiz_dict)}")
    y_hat = model_do_quiz(
        quiz,
        ans,
        model=model_quiz,
        temperature_quiz=effective_quiz_temp,
        max_tokens=max_tokens,
        top_p=model_quiz_top_p,
        quiz_dict=quiz_dict,
        is_debug=is_debug,
    )
    quiz_time = time.time() - quiz_start_time
    
    if is_debug:
        print(f"[DEBUG] Quiz completed, y_hat={y_hat}, quiz_time={quiz_time:.2f}s")
    
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
    is_debug: bool = False,
):
    effective_model_temp = model_temperature
    effective_quiz_temp = model_quiz_temperature
    
    if is_debug:
        print(f"[DEBUG] baymin_test called with model={model}, question_output length={len(question_output)}")
        print(f"[DEBUG] quiz type={type(quiz)}, quiz_dict type={type(quiz_dict)}")
        print(f"[DEBUG] expected answer y={y}")
    
    start_time = time.time()
    if is_debug:
        print(f"[DEBUG] Getting answer from tool agent")
    answer, testing_log = get_answer_from_tool_agent(
        net,
        question_output,
        model=model,
        max_tokens=max_tokens,
        is_output_log=is_output_log,
        is_debug=is_debug,
        model_top_p=model_top_p,
        model_temperature=effective_model_temp,
    )
    baymin_response_time = time.time() - start_time
    
    if is_debug:
        print(f"[DEBUG] Got answer from tool agent, type={type(answer)}, length={len(str(answer)) if answer else 0}")
        print(f"[DEBUG] Answer preview: {str(answer)[:200] if answer else 'None'}...")
    
    quiz_start_time = time.time()
    if is_debug:
        print(f"[DEBUG] Starting baymin quiz with answer type={type(answer)}, quiz_dict type={type(quiz_dict)}")
    y_hat = model_do_quiz(
        quiz,
        answer,
        model=model_quiz,
        temperature_quiz=effective_quiz_temp,
        max_tokens=max_tokens,
        top_p=model_quiz_top_p,
        quiz_dict=quiz_dict,
        is_debug=is_debug,
    )
    quiz_time = time.time() - quiz_start_time
    
    if is_debug:
        print(f"[DEBUG] Baymin quiz completed, y_hat={y_hat}, quiz_time={quiz_time:.2f}s")
    
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
    is_debug: bool = False,
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
                is_debug=is_debug,
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
                is_debug=is_debug,
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
                is_debug=is_debug,
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
    is_debug: bool = False,
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
        is_debug=is_debug,
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
    is_debug: bool = False,
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
        is_debug=is_debug,
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
    is_debug: bool = False,
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
        is_debug=is_debug,
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
    is_debug: bool = False,
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
        is_debug=is_debug,
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
    is_debug: bool = False,
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
        is_debug=is_debug,
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
    is_debug: bool = False,
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
                is_debug=is_debug,
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
                is_debug=is_debug,
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
                is_debug=is_debug,
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
    is_debug: bool = False,
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
        is_debug=is_debug,
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
    is_debug: bool = False,
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
        is_debug=is_debug,
    )


def debug_print_elementary_quiz(
    net,
    question_set,
    create_quiz_function,
    *,
    has_evidence=False,
    num_questions=1,
    question_index=1
):
    """
    Debug function to print out quiz for elementary tests without running the actual test.
    
    Args:
        net: Bayesian network
        question_set: List of question formats
        create_quiz_function: Function to create quiz (e.g., create_dependency_quiz)
        has_evidence: Whether to include evidence
        num_questions: Number of questions to generate (default 1)
        question_index: Starting question index (default 1)
    """
    print(f"=== DEBUG: Printing Elementary Quiz ===")
    print(f"Quiz function: {create_quiz_function.__name__}")
    print(f"Network size: {len(net.nodes())}")
    print(f"Has evidence: {has_evidence}")
    print(f"Number of questions: {num_questions}")
    print("=" * 50)
    
    for i in range(num_questions):
        current_index = question_index + i
        if i < len(question_set):
            question_format = question_set[i]
        else:
            question_format = question_set[0]  # Use first question format if we run out
            
        print(f"\n--- Question {current_index} ---")
        
        # Determine if the question format requires evidence, override if needed
        requires_evidence = "{evidence}" in (question_format or "")
        effective_has_evidence = has_evidence or requires_evidence

        # Generate question
        if effective_has_evidence:
            prompt, node1, node2, question_output, evidence = two_nodes_question(net, question_format=question_format, has_evidence=True)
            quiz, y, quiz_dict = create_quiz_function(question_output, net, node1, node2, evidence)
            print(f"Node1: {node1}")
            print(f"Node2: {node2}")
            print(f"Evidence: {evidence}")
            if requires_evidence and not has_evidence:
                print("(Auto-enabled evidence for this question format)")
        else:
            prompt, node1, node2, question_output = two_nodes_question(net, question_format=question_format, has_evidence=False)
            quiz, y, quiz_dict = create_quiz_function(question_output, net, node1, node2)
            print(f"Node1: {node1}")
            print(f"Node2: {node2}")
            print(f"Evidence: None")
        
        print(f"\nQuestion Format: {question_format}")
        print(f"Generated Question: {question_output}")
        print(f"Correct Answer: {y}")
        print(f"\nQuiz:")
        print(quiz)
        print(f"\nQuiz Dict Structure:")
        print(f"Header: {quiz_dict.get('header', 'N/A')}")
        print(f"Options: {len(quiz_dict.get('options', []))} options")
        for idx, opt in enumerate(quiz_dict.get('options', [])):
            print(f"  {chr(65 + idx)}. {opt.get('text', 'N/A')} (Correct: {opt.get('is_correct', False)})")
        print("-" * 50)


def debug_print_numerical_quiz(
    net,
    question_set,
    create_quiz_function,
    *,
    has_evidence=False,
    num_questions=1,
    question_index=1
):
    """
    Debug function to print out quiz for numerical tests without running the actual test.
    
    Args:
        net: Bayesian network
        question_set: List of question formats
        create_quiz_function: Function to create quiz (e.g., create_probability_quiz)
        has_evidence: Whether to include evidence
        num_questions: Number of questions to generate (default 1)
        question_index: Starting question index (default 1)
    """
    print(f"=== DEBUG: Printing Numerical Quiz ===")
    print(f"Quiz function: {create_quiz_function.__name__}")
    print(f"Network size: {len(net.nodes())}")
    print(f"Has evidence: {has_evidence}")
    print(f"Number of questions: {num_questions}")
    print("=" * 50)
    
    for i in range(num_questions):
        current_index = question_index + i
        if i < len(question_set):
            question_format = question_set[i]
        else:
            question_format = question_set[0]  # Use first question format if we run out
            
        print(f"\n--- Question {current_index} ---")
        
        # Determine if the question format requires evidence, override if needed
        requires_evidence = "{evidence}" in (question_format or "")
        effective_has_evidence = has_evidence or requires_evidence

        # Generate question
        if effective_has_evidence:
            prompt, node, question_output, evidence = probability_question(net, question_format=question_format, has_evidence=True)
            quiz, y, quiz_dict = create_quiz_function(question_output, net, node, evidence)
            print(f"Node: {node}")
            print(f"Evidence: {evidence}")
            if requires_evidence and not has_evidence:
                print("(Auto-enabled evidence for this question format)")
        else:
            prompt, node, question_output = probability_question(net, question_format=question_format, has_evidence=False)
            quiz, y, quiz_dict = create_quiz_function(question_output, net, node)
            print(f"Node: {node}")
            print(f"Evidence: None")
        
        print(f"\nQuestion Format: {question_format}")
        print(f"Generated Question: {question_output}")
        print(f"Correct Answer: {y}")
        print(f"\nQuiz:")
        print(quiz)
        print(f"\nQuiz Dict Structure:")
        print(f"Header: {quiz_dict.get('header', 'N/A')}")
        print(f"Options: {len(quiz_dict.get('options', []))} options")
        for idx, opt in enumerate(quiz_dict.get('options', [])):
            print(f"  {chr(65 + idx)}. {opt.get('text', 'N/A')} (Correct: {opt.get('is_correct', False)})")
        print("-" * 50)


def export_quiz_samples_to_csv(
    net,
    *,
    num_questions=2,
    output_file_path="quiz_samples.csv",
    include_sets=None,
):
    """
    Generate quiz samples for the provided network and write them to a CSV file.

    Args:
        net: Bayesian network
        num_questions (int): Number of questions per question set to generate
        output_file_path (str): Path to CSV output file
        include_sets (list[str] | None): Optional subset of set keys to include

    The CSV columns include metadata, question text, header, options A-D, and correctness flags.
    Evidence is auto-enabled if the question format contains "{evidence}".
    """
    import csv

    # Map of question set labels to (question_list, create_quiz_function, test_type, generator)
    # generator is either two_nodes_question (elementary) or probability_question (numerical)
    question_sets = {
        "dependency": (DEPENDENCY_QUESTIONS, create_dependency_quiz, "elementary", two_nodes_question),
        "common_cause": (COMMON_CAUSE_QUESTIONS, create_common_cause_quiz, "elementary", two_nodes_question),
        "common_effect": (COMMON_EFFECT_QUESTIONS, create_common_effect_quiz, "elementary", two_nodes_question),
        "blocked_evidence": (BLOCKED_EVIDENCES_QUESTIONS, create_blocked_evidence_quiz, "elementary", two_nodes_question),
        "evidence_change_relationship": (EVIDENCE_CHANGE_RELATIONSHIP_QUESTIONS, create_evidence_change_relationship_quiz, "elementary", two_nodes_question),
        "probability": (PROBABILITY_QUESTIONS, create_probability_quiz, "numerical", probability_question),
        "highest_impact_evidence": (HIGHEST_IMPACT_EVIDENCE_QUESTIONS, create_highest_impact_evidence_quiz, "numerical", probability_question),
    }

    if include_sets:
        include_sets = set(include_sets)
        question_sets = {k: v for k, v in question_sets.items() if k in include_sets}

    fieldnames = [
        "test_type",
        "set_key",
        "question_index",
        "question_format",
        "generated_question",
        "node1",
        "node2",
        "node",
        "evidence",
        "quiz_header",
        "correct_letter",
        "option_A_text",
        "option_A_is_correct",
        "option_B_text",
        "option_B_is_correct",
        "option_C_text",
        "option_C_is_correct",
        "option_D_text",
        "option_D_is_correct",
    ]

    with open(output_file_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for set_key, (q_list, make_quiz, test_type, generator) in question_sets.items():
            for i in range(num_questions):
                # Pick question format (cycle if fewer provided)
                if len(q_list) == 0:
                    continue
                question_format = q_list[i % len(q_list)]

                # Evidence auto-detection
                requires_evidence = "{evidence}" in (question_format or "")

                row = {
                    "test_type": test_type,
                    "set_key": set_key,
                    "question_index": i + 1,
                    "question_format": question_format,
                    "node1": None,
                    "node2": None,
                    "node": None,
                    "evidence": None,
                }

                if generator is two_nodes_question:
                    if requires_evidence:
                        prompt, node1, node2, question_output, evidence = generator(net, question_format=question_format, has_evidence=True)
                        quiz_text, correct_letter, quiz_dict = make_quiz(question_output, net, node1, node2, evidence)
                        row["node1"], row["node2"], row["evidence"] = node1, node2, ", ".join(evidence) if evidence else ""
                    else:
                        prompt, node1, node2, question_output = generator(net, question_format=question_format, has_evidence=False)
                        quiz_text, correct_letter, quiz_dict = make_quiz(question_output, net, node1, node2)
                        row["node1"], row["node2"] = node1, node2
                else:
                    # probability_question generator
                    if requires_evidence:
                        prompt, node, question_output, evidence = generator(net, question_format=question_format, has_evidence=True)
                        quiz_text, correct_letter, quiz_dict = make_quiz(question_output, net, node, evidence)
                        row["node"], row["evidence"] = node, ", ".join(evidence) if evidence else ""
                    else:
                        prompt, node, question_output = generator(net, question_format=question_format, has_evidence=False)
                        quiz_text, correct_letter, quiz_dict = make_quiz(question_output, net, node)
                        row["node"] = node

                row["generated_question"] = question_output
                row["quiz_header"] = quiz_dict.get("header", "") if isinstance(quiz_dict, dict) else ""
                row["correct_letter"] = correct_letter

                # Write options A..D when present
                options = list(quiz_dict.get("options", [])) if isinstance(quiz_dict, dict) else []
                for idx, letter in enumerate(["A", "B", "C", "D"]):
                    if idx < len(options):
                        opt = options[idx]
                        row[f"option_{letter}_text"] = opt.get("text", "")
                        row[f"option_{letter}_is_correct"] = bool(opt.get("is_correct", False))
                    else:
                        row[f"option_{letter}_text"] = ""
                        row[f"option_{letter}_is_correct"] = False

                writer.writerow(row)

    print(f"Wrote quiz samples to: {output_file_path}")
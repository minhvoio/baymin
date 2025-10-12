import sys
import os

current_dir = os.getcwd()
project_root = current_dir

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bn_helpers.get_structures_print_tools import get_nets, printNet
from bn_helpers.constants import MODEL, MODEL_QUIZ
from benchmarking.question_types import (DEPENDENCY_QUESTIONS, COMMON_CAUSE_QUESTIONS, COMMON_EFFECT_QUESTIONS, BLOCKED_EVIDENCES_QUESTIONS, 
EVIDENCE_CHANGE_RELATIONSHIP_QUESTIONS, PROBABILITY_QUESTIONS, HIGHEST_IMPACT_EVIDENCE_QUESTIONS)
from benchmarking.model_evaluator import (dependency_test, common_cause_test, common_effect_test, blocked_evidence_test, 
evidence_change_relationship_test, probability_test, highest_impact_evidence_test)
from benchmarking.data_utils import load_nets_from_parquet
from benchmarking.benchmarking_utils import retry_test_with_backoff
import asyncio

data_output = os.path.join(project_root, "benchmarking", "data")
print(f"Loading nets from: {data_output}")
net_5, net_10, net_30, net_60 = load_nets_from_parquet(os.path.join(data_output, "nets_dataset.parquet"))

list_of_nets = [net_5, net_10, net_30, net_60]
NUM_QUESTIONS = 30
MAX_TOKENS = 1800
IS_TESTING = True
PROBABILITY_MAX_TOKENS = 1200


QWEN_MODEL = "qwen3:8b"
LLAMA_MODEL = "llama3.1:70b"
MODEL_LIST = [LLAMA_MODEL, QWEN_MODEL]
for MODEL in MODEL_LIST:
    try:
        retry_test_with_backoff(dependency_test, net_5, max_retries=5, base_delay=2, max_delay=30,\
            num_questions=NUM_QUESTIONS, max_tokens=MAX_TOKENS, isTesting=IS_TESTING, model=MODEL)
    except Exception as e:
        print(f"dependency_test failed completely: {str(e)}")
        print("Continuing with next test...")
    
    try:
        retry_test_with_backoff(common_cause_test, net_5, max_retries=5, base_delay=2, max_delay=30,\
            num_questions=NUM_QUESTIONS, max_tokens=MAX_TOKENS, isTesting=IS_TESTING, model=MODEL)
    except Exception as e:
        print(f"common_cause_test failed completely: {str(e)}")
        print("Continuing with next test...")
    
    try:
        retry_test_with_backoff(common_effect_test, net_5, max_retries=5, base_delay=2, max_delay=30,\
            num_questions=NUM_QUESTIONS, max_tokens=MAX_TOKENS, isTesting=IS_TESTING, model=MODEL)
    except Exception as e:
        print(f"common_effect_test failed completely: {str(e)}")
        print("Continuing with next test...")
    
    try:
        retry_test_with_backoff(blocked_evidence_test, net_5, max_retries=5, base_delay=2, max_delay=30,\
            num_questions=NUM_QUESTIONS, max_tokens=MAX_TOKENS, isTesting=IS_TESTING, model=MODEL)
    except Exception as e:
        print(f"blocked_evidence_test failed completely: {str(e)}")
        print("Continuing with next test...")
    
    try:
        retry_test_with_backoff(evidence_change_relationship_test, net_5, max_retries=5, base_delay=2, max_delay=30,\
            num_questions=NUM_QUESTIONS, max_tokens=MAX_TOKENS, isTesting=IS_TESTING, model=MODEL)
    except Exception as e:
        print(f"evidence_change_relationship_test failed completely: {str(e)}")
        print("Continuing with next test...")
    
    try:
        retry_test_with_backoff(probability_test, net_5, max_retries=10, base_delay=2, max_delay=30,\
            num_questions=NUM_QUESTIONS, max_tokens=PROBABILITY_MAX_TOKENS, isTesting=IS_TESTING, model=MODEL)
    except Exception as e:
        print(f"probability_test failed completely: {str(e)}")
        print("Continuing with next test...")

    print(f"\nCompleted all tests for Net_{len(net_5.nodes())}")
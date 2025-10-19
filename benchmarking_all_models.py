import sys
import os

current_dir = os.getcwd()
project_root = current_dir

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bn_helpers.constants import MODEL
from benchmarking.model_evaluator import (dependency_test, common_cause_test, common_effect_test, blocked_evidence_test, 
evidence_change_relationship_test, probability_test)
from benchmarking.data_utils import load_nets_from_parquet
from benchmarking.benchmarking_utils import retry_test_with_backoff

data_output = os.path.join(project_root, "benchmarking", "data")
print(f"Loading nets from: {data_output}")
net_5, net_10, net_30, net_60 = load_nets_from_parquet(os.path.join(data_output, "nets_dataset.parquet"))

list_of_nets = [net_5, net_10, net_30, net_60]

IS_DEBUG = False 

NUM_QUESTIONS = 30
MAX_TOKENS = 1800
IS_OUTPUT_LOG = True


QWEN_MODEL = "qwen3:8b"
LLAMA_MODEL = "llama3.1:70b"
GPT_OSS_MODEL = 'gpt-oss:latest'
LLAMA_SMALL_MODEL = 'llama3.2:3b'

MODEL_QUIZ = 'gpt-oss:latest'

# MODEL_LIST = [GPT_OSS_MODEL, LLAMA_MODEL, QWEN_MODEL]
MODEL_LIST = [LLAMA_SMALL_MODEL]

MODEL_TEMPERATURE = 0.0
MODEL_TOP_P = 1.0
MODEL_QUIZ_TEMPERATURE = 0.7
MODEL_QUIZ_TOP_P = 0.9

COMMON_TEST_KWARGS = {
    "num_questions": NUM_QUESTIONS,
    "max_tokens": MAX_TOKENS,
    "is_output_log": IS_OUTPUT_LOG,
    "test_baymin_only": False,
    "model_temperature": MODEL_TEMPERATURE,
    "model_top_p": MODEL_TOP_P,
    "model_quiz_temperature": MODEL_QUIZ_TEMPERATURE,
    "model_quiz_top_p": MODEL_QUIZ_TOP_P,
    "model_quiz": MODEL_QUIZ,
    "is_debug": IS_DEBUG,  
}

net = net_5
for MODEL in MODEL_LIST:
    try:
        retry_test_with_backoff(
            dependency_test,
            net,
            max_retries=5,
            base_delay=2,
            max_delay=30,
            model=MODEL,
            **COMMON_TEST_KWARGS,
        )
    except Exception as e:
        print(f"dependency_test failed completely: {str(e)}")
        print("Continuing with next test...")

    try:
        retry_test_with_backoff(
            common_cause_test,
            net,
            max_retries=5,
            base_delay=2,
            max_delay=30,
            model=MODEL,
            **COMMON_TEST_KWARGS,
        )
    except Exception as e:
        print(f"common_cause_test failed completely: {str(e)}")
        print("Continuing with next test...")

    try:
        retry_test_with_backoff(
            common_effect_test,
            net,
            max_retries=5,
            base_delay=2,
            max_delay=30,
            model=MODEL,
            **COMMON_TEST_KWARGS,
        )
    except Exception as e:
        print(f"common_effect_test failed completely: {str(e)}")
        print("Continuing with next test...")

    try:
        retry_test_with_backoff(
            blocked_evidence_test,
            net,
            max_retries=5,
            base_delay=2,
            max_delay=30,
            model=MODEL,
            **COMMON_TEST_KWARGS,
        )
    except Exception as e:
        print(f"blocked_evidence_test failed completely: {str(e)}")
        print("Continuing with next test...")

    try:
        retry_test_with_backoff(
            evidence_change_relationship_test,
            net,
            max_retries=5,
            base_delay=2,
            max_delay=30,
            model=MODEL,
            **COMMON_TEST_KWARGS,
        )
    except Exception as e:
        print(f"evidence_change_relationship_test failed completely: {str(e)}")
        print("Continuing with next test...")

    try:
        retry_test_with_backoff(
            probability_test,
            net,
            max_retries=5,
            base_delay=2,
            max_delay=30,
            model=MODEL,
            **COMMON_TEST_KWARGS,
        )
    except Exception as e:
        print(f"probability_test failed completely: {str(e)}")
        print("Continuing with next test...")

    print(f"\nCompleted all tests for Net_{len(net.nodes())}")

print("Completed all tests!")
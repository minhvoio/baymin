from benchmarking.benchmarking_utils import pickTwoRandomNodes
from bn_helpers.support_tools import get_BN_structure, printNet
from bn_helpers.bn_helpers import AnswerStructure, BnHelper
from ollama.prompt import answer_this_prompt
from benchmarking.data_utils import load_nets_from_parquet
import os

data_output = "./benchmarking/data"

def benchmark_simple_query(net):
    node1, node2 = pickTwoRandomNodes(net)
    print(node1, node2)
    # bn_helper = BnHelper()
    # is_connected = bn_helper.is_XY_dconnected(net, node1, node2)
    # bn = get_BN_structure(net)
    # prompt = f"In this Bayesian Network:\n{bn}\n"
    # prompt += f"Is changing the evidence of {node1} going to change the probability of {node2}?"
    # ans = answer_this_prompt(prompt, format=AnswerStructure.model_json_schema())
    # ans = AnswerStructure.model_validate_json(ans)
    # return ans

if __name__ == "__main__":
    nets = load_nets_from_parquet(os.path.join(data_output, "nets_dataset.parquet"))
    for net in nets[0]:
        printNet(net)
        benchmark_simple_query(net)
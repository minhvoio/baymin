import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bni_netica.bni_netica import Net

from bn_helpers.tool_agent import get_answer_from_tool_agent


DEFAULT_MODEL = "qwen3:1.7b"

def ask(net_fn, question, model=DEFAULT_MODEL):
    net = Net(net_fn)
    print("Question: ", question)
    answer = get_answer_from_tool_agent(
        net,
        question,
        model=model,
        max_tokens=1000,
        temperature=0,
        is_debug=True,
        show_tool_calls=True,
    )
    print("Answer: ", answer)
    print()

def main():
    nfv1 = "nets/NF_V1.dne"
    ask(nfv1, "Are FishAbundance and RiverFlow d-connected?")
    ask(nfv1, "Are TreeCondition and PesticideUse d-connected?")

if __name__ == "__main__":
    main()

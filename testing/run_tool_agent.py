import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bni_netica.bni_netica import Net

from bn_helpers.tool_agent import get_answer_from_tool_agent


DEFAULT_BN_PATH = Path("nets/NF_V1.dne")
DEFAULT_QUESTION = "Are FishAbundance and RiverFlow d-connected?"
DEFAULT_MODEL = "qwen3:4b"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run get_answer_from_tool_agent against a Bayesian network."
    )
    parser.add_argument(
        "--bn",
        default=str(DEFAULT_BN_PATH),
        help="Path to the Bayesian network file.",
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="Question to ask the tool agent.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="LLM model name to use.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens for the model response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    args = parser.parse_args()

    net = Net(args.bn)
    answer = get_answer_from_tool_agent(
        net,
        args.question,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(answer)


if __name__ == "__main__":
    main()

"""
Few-shot Baseline for ACE Comparison

Fixed few-shot prompting baseline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
from typing import List, Dict
import argparse

from src.llm.glm_client import GLMClient
from src.utils.data_loader import load_gsm8k


# Fixed few-shot examples
FEWSHOT_EXAMPLES = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "72"
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "10"
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need?",
        "answer": "5"
    },
]


class FewShotBaseline:
    """Few-shot prompting baseline."""

    def __init__(self, llm_client: GLMClient):
        self.llm_client = llm_client

    def solve(self, question: str) -> str:
        """Solve using few-shot prompting."""
        prompt = "Solve the following math problems step by step.\n\n"

        for i, ex in enumerate(FEWSHOT_EXAMPLES, 1):
            prompt += f"Q{i}: {ex['question']}\n"
            prompt += f"A{i}: {ex['answer']}\n\n"

        prompt += f"Q{len(FEWSHOT_EXAMPLES) + 1}: {question}\n"
        prompt += f"A{len(FEWSHOT_EXAMPLES) + 1}: "

        response = self.llm_client.call(prompt, temperature=0.0)

        # Extract number
        numbers = re.findall(r'[-+]?\d*\.?\d+', response)
        return numbers[-1] if numbers else response

    def evaluate(self, test_data: List[Dict]) -> Dict:
        """Evaluate on test set."""
        correct = 0
        total = len(test_data)

        for sample in test_data:
            answer = self.solve(sample["question"])
            gt = str(sample["answer"])

            if answer in gt or gt in answer:
                correct += 1

        accuracy = correct / total if total > 0 else 0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }


def main():
    parser = argparse.ArgumentParser(description="Few-shot Baseline")
    parser.add_argument("--test-data", default="data/gsm8k_test.jsonl")
    parser.add_argument("--test-size", type=int, default=50)

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    test_data = load_gsm8k(args.test_data, max_samples=args.test_size)

    # Initialize
    llm_client = GLMClient()
    fewshot = FewShotBaseline(llm_client)

    # Evaluate
    print(f"Evaluating on {len(test_data)} test samples...")
    results = fewshot.evaluate(test_data)

    print("\n" + "="*50)
    print("Few-shot Baseline Results")
    print("="*50)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Correct: {results['correct']}/{results['total']}")


if __name__ == "__main__":
    main()

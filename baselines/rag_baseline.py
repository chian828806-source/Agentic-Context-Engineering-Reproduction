"""
RAG Baseline for ACE Comparison

Simple retrieval-augmented generation baseline for comparison with ACE.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import List, Dict
import argparse

from src.llm.glm_client import GLMClient
from src.utils.data_loader import load_gsm8k


class RAGBaseline:
    """Simple RAG baseline using keyword-based retrieval."""

    def __init__(self, llm_client: GLMClient, examples: List[Dict]):
        self.llm_client = llm_client
        self.examples = examples

    def retrieve_examples(self, question: str, k: int = 5) -> List[Dict]:
        """Retrieve top-k similar examples using keyword overlap."""
        question_words = set(question.lower().split())

        scored = []
        for ex in self.examples:
            ex_words = set(ex.get("question", "").lower().split())
            overlap = len(question_words & ex_words)
            scored.append((overlap, ex))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:k]]

    def solve(self, question: str) -> str:
        """Solve a question using RAG."""
        examples = self.retrieve_examples(question)

        # Build prompt with examples
        prompt = "You are a math problem solver. Here are similar examples:\n\n"

        for i, ex in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Question: {ex['question']}\n"
            prompt += f"Answer: {ex['answer']}\n\n"

        prompt += f"Now solve this:\nQuestion: {question}\n"
        prompt += "Give your final answer as a number."

        response = self.llm_client.call(prompt, temperature=0.3)

        # Extract number
        import re
        numbers = re.findall(r'[-+]?\d*\.?\d+', response)
        return numbers[-1] if numbers else response

    def evaluate(self, test_data: List[Dict]) -> Dict:
        """Evaluate on test set."""
        correct = 0
        total = len(test_data)

        for sample in test_data:
            answer = self.solve(sample["question"])
            gt = str(sample["answer"])

            # Simple comparison
            if answer in gt or gt in answer:
                correct += 1

        accuracy = correct / total if total > 0 else 0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }


def main():
    parser = argparse.ArgumentParser(description="RAG Baseline")
    parser.add_argument("--train-data", default="data/gsm8k_train.jsonl")
    parser.add_argument("--test-data", default="data/gsm8k_test.jsonl")
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--k", type=int, default=5, help="Number of examples to retrieve")

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    train_data = load_gsm8k(args.train_data, max_samples=args.train_size)
    test_data = load_gsm8k(args.test_data, max_samples=args.test_size)

    # Initialize
    llm_client = GLMClient()
    rag = RAGBaseline(llm_client, train_data)

    # Evaluate
    print(f"Evaluating on {len(test_data)} test samples...")
    results = rag.evaluate(test_data)

    print("\n" + "="*50)
    print("RAG Baseline Results")
    print("="*50)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Correct: {results['correct']}/{results['total']}")


if __name__ == "__main__":
    main()

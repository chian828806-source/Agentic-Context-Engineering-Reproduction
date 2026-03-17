"""
Evaluator Node for ACE Framework (FINER-only)
"""

import json
from typing import Dict, Any, List, Optional
from ..llm.glm_client import GLMClient
from .generator import GeneratorNode


class EvaluatorNode:
    """
    Evaluator Node: Evaluates playbook performance on FINER.
    """

    def __init__(
        self,
        llm_client: GLMClient,
        generator: Optional[GeneratorNode] = None,
    ):
        self.llm_client = llm_client
        self.generator = generator or GeneratorNode(llm_client)

    def evaluate(
        self,
        state: Dict[str, Any],
        validation_samples: List[Dict[str, Any]],
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        if max_samples:
            validation_samples = validation_samples[:max_samples]

        results = []
        for sample in validation_samples:
            temp_state = state.copy()
            temp_state["current_sample"] = sample

            gen_result = self.generator(temp_state)

            is_correct = self.compare_finer_answers(
                gen_result.get("generated_answer"),
                sample.get("ner_tags", []),
            )

            results.append({
                "text": sample.get("text", ""),
                "tokens": sample.get("tokens", []),
                "ground_truth_ner": sample.get("ner_tags", []),
                "generated_answer": gen_result.get("generated_answer"),
                "trace": gen_result.get("generator_trace", ""),
                "bullet_ids": gen_result.get("bullet_ids", []),
                "is_correct": is_correct,
            })

        correct = sum(1 for r in results if r["is_correct"])
        total = len(results)
        accuracy = correct / total if total > 0 else 0.0

        error_samples = [r for r in results if not r["is_correct"]]

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "error_samples": error_samples[:10],
            "all_results": results,
        }

    @staticmethod
    def _parse_ner_list(value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return [str(v) for v in parsed]
                except Exception:
                    pass
            return text.split()
        return None

    def compare_finer_answers(self, generated_answer: Any, ground_truth_ner: Any) -> bool:
        gt_list = self._parse_ner_list(ground_truth_ner)
        gen_list = self._parse_ner_list(generated_answer)

        if gt_list is None or gen_list is None:
            return False
        if len(gt_list) != len(gen_list):
            return False
        return all(g == t for g, t in zip(gen_list, gt_list))

    def create_error_report(
        self,
        error_samples: List[Dict[str, Any]],
    ) -> str:
        if not error_samples:
            return "No errors to report."

        lines = [
            "=" * 60,
            f"Error Report: {len(error_samples)} errors",
            "=" * 60,
        ]

        for i, error in enumerate(error_samples[:10], 1):
            lines.append(f"\nError {i}:")
            lines.append(f"Text: {error.get('text', '')[:120]}...")
            lines.append(f"Ground Truth Tags: {error.get('ground_truth_ner', [])[:20]}")
            lines.append(f"Generated: {error.get('generated_answer', '')}")

            trace = error.get("trace", "")
            if trace:
                lines.append(f"Reasoning: {trace[:200]}...")

        return "\n".join(lines)

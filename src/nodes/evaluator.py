"""
Evaluator Node for ACE Framework

The Evaluator node evaluates the current playbook on a validation set.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from ..llm.glm_client import GLMClient
from .generator import GeneratorNode


class EvaluatorNode:
    """
    Evaluator Node: Evaluates playbook performance.

    This node is responsible for:
    1. Running the Generator on validation samples
    2. Comparing answers with ground truth
    3. Computing accuracy and other metrics
    4. Collecting error samples for analysis
    """

    def __init__(
        self,
        llm_client: GLMClient,
        generator: Optional[GeneratorNode] = None,
    ):
        """
        Initialize the Evaluator node.

        Args:
            llm_client: GLM-4.6 client instance
            generator: Optional Generator node (will create if not provided)
        """
        self.llm_client = llm_client
        self.generator = generator or GeneratorNode(llm_client)

    def evaluate(
        self,
        state: Dict[str, Any],
        validation_samples: List[Dict[str, Any]],
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate current playbook on validation set.

        Args:
            state: Current ACEState containing playbook
            validation_samples: List of validation samples
            max_samples: Maximum samples to evaluate (None for all)

        Returns:
            Dict with evaluation results:
                - accuracy: Accuracy score
                - correct: Number of correct answers
                - total: Total number of samples
                - error_samples: List of error cases
        """
        playbook = state.get("current_playbook", {})

        if max_samples:
            validation_samples = validation_samples[:max_samples]

        results = []
        for sample in validation_samples:
            # Generate answer
            temp_state = state.copy()
            temp_state["current_sample"] = sample

            gen_result = self.generator(temp_state)

            # Evaluate answer
            is_correct = self.compare_answers(
                gen_result.get("generated_answer"),
                sample.get("answer"),
            )

            results.append({
                "question": sample.get("question", ""),
                "ground_truth": sample.get("answer", ""),
                "generated_answer": gen_result.get("generated_answer"),
                "trace": gen_result.get("generator_trace", ""),
                "is_correct": is_correct,
            })

        # Compute metrics
        correct = sum(1 for r in results if r["is_correct"])
        total = len(results)
        accuracy = correct / total if total > 0 else 0.0

        # Collect error samples (top ones for refinement)
        error_samples = [r for r in results if not r["is_correct"]]

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "error_samples": error_samples[:10],  # Keep top 10 errors
            "all_results": results,
        }

    @staticmethod
    def compare_answers(
        predicted: Any,
        ground_truth: Any,
        tolerance: float = 0.01,
    ) -> bool:
        """
        Compare predicted answer with ground truth.

        Handles numeric comparison with tolerance.

        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            tolerance: Relative tolerance for numeric comparison

        Returns:
            True if answers match within tolerance
        """
        pred_str = str(predicted).strip().lower()
        gt_str = str(ground_truth).strip().lower()

        # Direct string match
        if pred_str == gt_str:
            return True

        # Extract and compare numbers
        pred_num = EvaluatorNode._extract_number(pred_str)
        gt_num = EvaluatorNode._extract_number(gt_str)

        if pred_num is not None and gt_num is not None:
            if gt_num == 0:
                return pred_num == 0
            relative_error = abs(pred_num - gt_num) / abs(gt_num)
            return relative_error <= tolerance

        return False

    @staticmethod
    def _extract_number(text: str) -> Optional[float]:
        """
        Extract the last number from text.

        Args:
            text: Text to extract from

        Returns:
            Extracted number or None
        """
        # Match integers and decimals (including negative)
        numbers = re.findall(r'[-+]?\d*\.?\d+', text)

        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass

        return None

    @staticmethod
    def extract_numeric_answer(text: str) -> Optional[float]:
        """
        Extract numeric answer from text.

        This is a convenience method that wraps _extract_number.

        Args:
            text: Text to extract from

        Returns:
            Extracted number or None
        """
        return EvaluatorNode._extract_number(text)

    def compute_metrics(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Compute various metrics from evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Dict of metrics
        """
        if not results:
            return {
                "accuracy": 0.0,
                "correct": 0,
                "total": 0,
            }

        correct = sum(1 for r in results if r.get("is_correct", False))
        total = len(results)

        metrics = {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
        }

        # Add per-question-type metrics if available
        question_types = {}
        for result in results:
            q_type = self._classify_question_type(result.get("question", ""))
            if q_type not in question_types:
                question_types[q_type] = {"correct": 0, "total": 0}
            question_types[q_type]["total"] += 1
            if result.get("is_correct"):
                question_types[q_type]["correct"] += 1

        for q_type, counts in question_types.items():
            metrics[f"accuracy_{q_type}"] = (
                counts["correct"] / counts["total"]
                if counts["total"] > 0
                else 0.0
            )

        return metrics

    @staticmethod
    def _classify_question_type(question: str) -> str:
        """
        Classify a question into a type category.

        Args:
            question: The question text

        Returns:
            Question type string
        """
        question_lower = question.lower()

        if any(word in question_lower for word in ["add", "sum", "total", "plus", "together"]):
            return "addition"
        elif any(word in question_lower for word in ["subtract", "difference", "remain", "left"]):
            return "subtraction"
        elif any(word in question_lower for word in ["multiply", "times", "product"]):
            return "multiplication"
        elif any(word in question_lower for word in ["divide", "split", "share", "each"]):
            return "division"
        elif any(word in question_lower for word in ["fraction", "ratio", "percent"]):
            return "fraction"
        else:
            return "other"

    def create_error_report(
        self,
        error_samples: List[Dict[str, Any]],
    ) -> str:
        """
        Create a human-readable error report.

        Args:
            error_samples: List of error samples

        Returns:
            Formatted error report
        """
        if not error_samples:
            return "No errors to report."

        lines = [
            "=" * 60,
            f"Error Report: {len(error_samples)} errors",
            "=" * 60,
        ]

        for i, error in enumerate(error_samples[:10], 1):
            lines.append(f"\nError {i}:")
            lines.append(f"Question: {error.get('question', '')[:100]}...")
            lines.append(f"Ground Truth: {error.get('ground_truth', '')}")
            lines.append(f"Generated: {error.get('generated_answer', '')}")

            trace = error.get('trace', '')
            if trace:
                lines.append(f"Reasoning: {trace[:200]}...")

        return "\n".join(lines)


def validate_playbook_quality(
    playbook: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Validate the quality of a playbook.

    Args:
        playbook: Playbook dictionary

    Returns:
        Quality assessment dict
    """
    total_bullets = sum(len(items) for items in playbook.values())

    section_counts = {
        section: len(items) for section, items in playbook.items()
    }

    # Check for balance
    if total_bullets > 0:
        balance_ratio = max(section_counts.values()) / max(1, total_bullets)
    else:
        balance_ratio = 0.0

    return {
        "total_bullets": total_bullets,
        "section_counts": section_counts,
        "has_strategies": len(playbook.get("strategies_and_hard_rules", [])) > 0,
        "has_formulas": len(playbook.get("formulas_and_calculations", [])) > 0,
        "has_checklist": len(playbook.get("verification_checklist", [])) > 0,
        "balance_ratio": balance_ratio,
        "is_well_balanced": balance_ratio < 0.5,
    }

"""
Reflector Node for ACE Framework (FINER-only)
"""

import json
from typing import Dict, Any, List, Optional
from ..llm.glm_client import GLMClient
from ..prompts.reflector_prompts import get_reflector_prompt


class ReflectorNode:
    """
    Reflector Node: Analyzes errors and generates insights.

    This node is responsible for:
    1. Comparing generated answers with ground truth
    2. Identifying specific errors and their causes
    3. Generating actionable insights
    4. Producing structured reflections for the Curator
    """

    def __init__(
        self,
        llm_client: GLMClient,
        max_retries: int = 3,
        temperature: float = 0.3,
        task_type: str = "finer",
    ):
        """
        Initialize the Reflector node.

        Args:
            llm_client: GLM-4.6 client instance
            max_retries: Maximum number of retry attempts
            temperature: Sampling temperature (lower for consistent reflection)
            task_type: Type of task (finer)
        """
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.temperature = temperature
        self.task_type = task_type

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Reflector node.

        Args:
            state: Current ACEState containing:
                - current_sample: Original sample
                - generated_answer: Answer from Generator
                - ground_truth: Correct answer
                - generator_trace: Reasoning trace
                - current_playbook: For context

        Returns:
            Dict with updates to state:
                - reflection: The reflection report
        """
        sample = state.get("current_sample", {})
        generated_answer = state.get("generated_answer", "")
        ground_truth = state.get("ground_truth", "")
        generator_trace = state.get("generator_trace", "")
        playbook = state.get("current_playbook", {})

        # First check if answer is correct
        is_correct = self._compare_finer_answers(generated_answer, ground_truth)

        if is_correct:
            # No reflection needed for correct answers
            return {
                "reflection": {
                    "status": "correct",
                    "reasoning": "The answer was correct. No reflection needed.",
                }
            }

        # Build reflection prompt
        prompt = get_reflector_prompt(
            question=sample.get("text", ""),
            model_reasoning=generator_trace,
            model_answer=str(generated_answer),
            ground_truth=str(ground_truth),
            playbook=playbook,
            environment_feedback="",
        )

        # Call LLM with retry logic
        reflection = None
        last_error = None

        for attempt in range(self.max_retries):
            try:
                reflection = self.llm_client.call_json(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=1500,
                    system_prompt=self._get_system_prompt(),
                )
                break
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    continue

        if reflection is None:
            # Fallback: create basic reflection
            reflection = self._create_fallback_reflection(
                generated_answer, ground_truth, generator_trace
            )

        reflection["status"] = "incorrect"

        return {"reflection": reflection}

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Reflector."""
        return (
            "You are an expert analyst and educator. Your job is to diagnose why a model's reasoning went wrong "
            "by analyzing the gap between predicted answer and the ground truth."
        )

    def _create_fallback_reflection(
        self,
        generated_answer: str,
        ground_truth: str,
        trace: str,
    ) -> Dict[str, str]:
        """
        Create a basic reflection when LLM call fails.

        Args:
            generated_answer: The incorrect answer
            ground_truth: The correct answer
            trace: The reasoning trace

        Returns:
            Basic reflection dict
        """
        return {
            "reasoning": f"The model's answer ({generated_answer}) differs from the ground truth ({ground_truth}). "
                        f"Please review the tagging steps carefully.",
            "error_identification": f"Answer mismatch: got {generated_answer}, expected {ground_truth}",
            "root_cause_analysis": "Unable to determine - LLM reflection call failed.",
            "correct_approach": "Align tags with the token list and ensure each token has a valid FINER tag.",
            "key_insight": "Always check the tag sequence length and BIO consistency against tokens.",
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

    def _compare_finer_answers(self, generated_answer: Any, ground_truth_ner: Any) -> bool:
        gt_list = self._parse_ner_list(ground_truth_ner)
        gen_list = self._parse_ner_list(generated_answer)

        if gt_list is None or gen_list is None:
            return False
        if len(gt_list) != len(gen_list):
            return False
        return all(g == t for g, t in zip(gen_list, gt_list))

    def batch_reflect(
        self,
        error_samples: List[Dict[str, Any]],
        playbook: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Generate reflections for multiple error samples.

        Args:
            error_samples: List of error cases
            playbook: Current playbook

        Returns:
            List of reflections
        """
        reflections = []

        for sample in error_samples:
            state = {
                "current_sample": sample,
                "generated_answer": sample.get("generated_answer"),
                "ground_truth": sample.get("ground_truth_ner"),
                "generator_trace": sample.get("trace", ""),
                "current_playbook": playbook,
            }

            result = self(state)
            reflections.append(result.get("reflection", {}))

        return reflections


def collect_error_samples(
    results: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Collect the top-k most informative error samples.

    Args:
        results: List of generation results
        top_k: Number of samples to collect

    Returns:
        List of error samples with full context
    """
    errors = [r for r in results if not r.get("is_correct", False)]

    # Sort by some criterion (e.g., response length for diversity)
    errors.sort(key=lambda x: len(str(x.get("trace", ""))))

    return errors[:top_k]


def analyze_reflection_patterns(
    reflections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyze patterns across multiple reflections.

    Args:
        reflections: List of reflection dicts

    Returns:
        Analysis of common patterns
    """
    error_types = {}
    key_insights = []

    for ref in reflections:
        error_id = ref.get("error_identification", "unknown")
        error_types[error_id] = error_types.get(error_id, 0) + 1

        insight = ref.get("key_insight", "")
        if insight:
            key_insights.append(insight)

    return {
        "error_type_counts": error_types,
        "most_common_error": max(error_types, key=error_types.get) if error_types else None,
        "key_insights": key_insights,
    }

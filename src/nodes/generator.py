"""
Generator Node for ACE Framework

The Generator node uses the current playbook to generate answers for tasks.
"""

import json
import re
from typing import Dict, Any, Optional
from ..llm.glm_client import GLMClient
from ..prompts.generator_prompts import get_generator_prompt
from ..prompts.reflector_prompts import compare_answers_numeric


class GeneratorNode:
    """
    Generator Node: Uses the current playbook to solve problems.

    This node is responsible for:
    1. Formatting the playbook into a prompt
    2. Calling the LLM to generate an answer
    3. Parsing the LLM response
    4. Handling errors and retries
    """

    def __init__(
        self,
        llm_client: GLMClient,
        max_retries: int = 3,
        temperature: float = 0.7,
        task_type: str = "gsm8k",
    ):
        """
        Initialize the Generator node.

        Args:
            llm_client: GLM-4.6 client instance
            max_retries: Maximum number of retry attempts
            temperature: Sampling temperature for generation
            task_type: Type of task (gsm8k, finer, etc.)
        """
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.temperature = temperature
        self.task_type = task_type

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Generator node.

        Args:
            state: Current ACEState containing:
                - current_playbook: Current playbook with strategies
                - current_sample: Sample to solve (with 'question' key)

        Returns:
            Dict with updates to state:
                - generated_answer: The final answer
                - generator_trace: The reasoning trace
        """
        playbook = state.get("current_playbook", {})
        sample = state.get("current_sample", {})

        if not sample or "question" not in sample:
            return {
                "generated_answer": None,
                "generator_trace": "Error: No question provided in sample.",
            }

        question = sample["question"]

        # Build prompt
        if self.task_type == "finer" and "context" in sample:
            prompt = get_generator_prompt(
                question=question,
                playbook=playbook,
                task_type=self.task_type,
                context=sample["context"],
            )
        else:
            prompt = get_generator_prompt(
                question=question,
                playbook=playbook,
                task_type=self.task_type,
            )

        # Call LLM with retry logic
        response = None
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.llm_client.call_json(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=2048,
                    system_prompt=self._get_system_prompt(),
                )
                break
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    continue
                # Last attempt failed, try without JSON mode
                try:
                    response_text = self.llm_client.call(
                        prompt=prompt,
                        temperature=self.temperature,
                        max_tokens=2048,
                        system_prompt=self._get_system_prompt(),
                    )
                    response = self._parse_fallback_response(response_text)
                    break
                except Exception as e2:
                    last_error = e2

        if response is None:
            return {
                "generated_answer": None,
                "generator_trace": f"Error: Generator failed after {self.max_retries} attempts. Last error: {last_error}",
            }

        # Extract answer and trace
        trace = response.get("reasoning", "")
        answer = response.get("final_answer", response.get("answer", ""))

        return {
            "generated_answer": answer,
            "generator_trace": trace,
        }

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Generator."""
        if self.task_type == "gsm8k":
            return "You are an expert mathematical problem solver. Always provide clear step-by-step reasoning and a final numeric answer."
        elif self.task_type == "finer":
            return "You are an expert financial analyst. Provide precise numerical answers with proper formatting."
        else:
            return "You are a helpful AI assistant. Provide accurate, well-reasoned answers."

    def _parse_fallback_response(self, response_text: str) -> Dict[str, str]:
        """
        Parse response when JSON mode fails.

        Args:
            response_text: Raw response text

        Returns:
            Parsed response dict
        """
        response_text = response_text.strip()

        # Try to extract JSON
        json_match = re.search(r'\{[^{}]*"reasoning"[^{}]*"final_answer"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: extract reasoning and answer
        reasoning = response_text
        answer = ""

        # Look for final answer patterns
        answer_patterns = [
            r'(?:final answer|answer|therefore|thus|is)\s*:?\s*([-\d,]+\.?\d*)',
            r'\{\s*"final_answer"\s*:\s*"([^"]+)"',
            r'The answer is\s+([-\d,]+\.?\d*)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                answer = match.group(1).replace(",", "")
                break

        if not answer:
            # Get last number as answer
            numbers = re.findall(r'[-+]?\d*\.?\d+', response_text)
            if numbers:
                answer = numbers[-1]

        return {
            "reasoning": reasoning[:1000],  # Limit trace length
            "final_answer": answer,
        }

    def batch_generate(
        self,
        state: Dict[str, Any],
        samples: list,
    ) -> list[Dict[str, Any]]:
        """
        Generate answers for multiple samples.

        Args:
            state: Current state (for playbook access)
            samples: List of samples to process

        Returns:
            List of results with generated answers
        """
        results = []

        for sample in samples:
            temp_state = state.copy()
            temp_state["current_sample"] = sample

            result = self(temp_state)
            results.append({
                "question": sample.get("question", ""),
                "ground_truth": sample.get("answer", ""),
                "generated_answer": result.get("generated_answer"),
                "trace": result.get("generator_trace", ""),
            })

        return results


def evaluate_generated_answer(
    generated_answer: Any,
    ground_truth: Any,
    tolerance: float = 0.01,
) -> Dict[str, Any]:
    """
    Evaluate a generated answer against ground truth.

    Args:
        generated_answer: The answer produced by Generator
        ground_truth: The correct answer
        tolerance: Relative tolerance for numeric comparison

    Returns:
        Dict with evaluation results
    """
    is_correct, rel_error = compare_answers_numeric(
        str(generated_answer),
        str(ground_truth),
        tolerance,
    )

    return {
        "is_correct": is_correct,
        "relative_error": rel_error,
        "generated": str(generated_answer),
        "ground_truth": str(ground_truth),
    }

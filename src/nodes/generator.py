"""
Generator Node for ACE Framework (FINER-only)
"""

import json
import re
from typing import Dict, Any, Optional, List
from ..llm.glm_client import GLMClient
from ..prompts.generator_prompts import get_generator_prompt


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
        task_type: str = "finer",
    ):
        """
        Initialize the Generator node.

        Args:
            llm_client: GLM-4.6 client instance
            max_retries: Maximum number of retry attempts
            temperature: Sampling temperature for generation
            task_type: Type of task (finer)
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
                - current_sample: Sample to solve (with 'text' key for FINER)

        Returns:
            Dict with updates to state:
                - generated_answer: The final answer
                - generator_trace: The reasoning trace
                - bullet_ids: List of relevant playbook bullet IDs
        """
        playbook = state.get("current_playbook", {})
        sample = state.get("current_sample", {})

        if not sample or "text" not in sample:
            return {
                "generated_answer": None,
                "generator_trace": "Error: No text provided in sample.",
                "bullet_ids": [],
            }

        question = sample.get("question", "")
        context = sample.get("context", sample.get("text", ""))

        # Build prompt (paper-aligned: question + context + reflection)
        prompt = get_generator_prompt(
            question=question,
            playbook=playbook,
            context=context,
            reflection="(empty)",
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

        # Extract answer, trace and bullet IDs
        trace = response.get("reasoning", "")
        answer = response.get("final_answer", response.get("answer", ""))
        bullet_ids = response.get("bullet_ids", [])

        return {
            "generated_answer": answer,
            "generator_trace": trace,
            "bullet_ids": bullet_ids,
        }

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Generator."""
        return (
            "You are an analysis expert for FINER sequence labeling. "
            "Return final_answer as a JSON array of tags with the same length as the tokens."
        )

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
            r'\{\s*"final_answer"\s*:\s*"([^"]+)"',
            r'final answer\s*:?\s*(.+)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                break

        if not answer:
            answer = response_text[:200]

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
                "text": sample.get("text", ""),
                "ground_truth_ner": sample.get("ner_tags", []),
                "generated_answer": result.get("generated_answer"),
                "trace": result.get("generator_trace", ""),
            })

        return results


def evaluate_generated_answer(
    generated_answer: Any,
    ground_truth: Any,
) -> Dict[str, Any]:
    """
    Evaluate a generated answer against FINER ground truth tags.
    """
    def _parse_list(value: Any) -> Optional[List[str]]:
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

    gt_list = _parse_list(ground_truth)
    gen_list = _parse_list(generated_answer)

    is_correct = (
        gt_list is not None
        and gen_list is not None
        and len(gt_list) == len(gen_list)
        and all(g == t for g, t in zip(gen_list, gt_list))
    )

    return {
        "is_correct": is_correct,
        "generated": str(generated_answer),
        "ground_truth": str(ground_truth),
    }

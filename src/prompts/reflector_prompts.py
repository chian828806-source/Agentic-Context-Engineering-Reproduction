"""
Reflector Prompt Templates for ACE Framework

The Reflector node analyzes errors and generates reflections on what went wrong.
"""

from typing import Dict, Any


# ============================================================================
# GSM8K Reflector Prompt
# ============================================================================

GSM8K_REFLECTOR_TEMPLATE = """You are an expert math educator. Your job is to analyze why a solution went wrong by comparing the predicted answer with the ground truth.

INSTRUCTIONS:
- Carefully analyze the model's reasoning trace to identify where it went wrong
- Compare the predicted answer with the ground truth to understand the gap
- Identify specific conceptual errors, calculation mistakes, or misapplied strategies
- Provide actionable insights that could help the model avoid this mistake in the future
- Focus on the root cause, not just surface-level errors
- Be specific about what the model should have done differently

QUESTION:
{question}

MODEL'S REASONING:
{model_reasoning}

MODEL'S ANSWER:
{model_answer}

GROUND TRUTH ANSWER:
{ground_truth}

CURRENT PLAYBOOK:
{playbook}

Your output should be a JSON object with these exact fields:
- reasoning: your analysis of what went wrong (detailed explanation)
- error_identification: what specifically went wrong in the reasoning?
- root_cause_analysis: why did this error occur? What concept was misunderstood?
- correct_approach: what should the model have done instead?
- key_insight: what strategy, formula, or principle should be remembered to avoid this error?

Answer in this exact JSON format:
{{
  "reasoning": "[Your detailed analysis here]",
  "error_identification": "[Specific error that occurred]",
  "root_cause_analysis": "[Why the error occurred - what concept was misunderstood]",
  "correct_approach": "[What should have been done differently - step by step]",
  "key_insight": "[Key principle to remember for future problems]"
}}"""


def get_reflector_prompt(
    question: str,
    model_reasoning: str,
    model_answer: str,
    ground_truth: str,
    playbook: Dict[str, Any],
    task_type: str = "gsm8k",
) -> str:
    """
    Build the Reflector prompt for analyzing an error.

    Args:
        question: The original question
        model_reasoning: The model's reasoning trace
        model_answer: The model's predicted answer
        ground_truth: The correct answer
        playbook: Current playbook (for context)
        task_type: Type of task

    Returns:
        Formatted prompt string
    """
    from .generator_prompts import format_playbook

    playbook_text = format_playbook(playbook, task_type=task_type)

    if task_type == "gsm8k":
        return GSM8K_REFLECTOR_TEMPLATE.format(
            question=question,
            model_reasoning=model_reasoning,
            model_answer=model_answer,
            ground_truth=ground_truth,
            playbook=playbook_text,
        )
    else:
        return GSM8K_REFLECTOR_TEMPLATE.format(
            question=question,
            model_reasoning=model_reasoning,
            model_answer=model_answer,
            ground_truth=ground_truth,
            playbook=playbook_text,
        )


# ============================================================================
# FINER Reflector Prompt (for financial tasks)
# ============================================================================

FINER_REFLECTOR_TEMPLATE = """You are an expert financial analyst and educator. Your job is to diagnose why a financial analysis went wrong.

INSTRUCTIONS:
- Analyze the model's reasoning trace carefully
- Compare with the ground truth to identify discrepancies
- Identify specific errors in formula application, XBRL tagging, or financial interpretation
- Consider the playbook entries that may have led the model astray
- Provide actionable corrections

QUESTION:
{question}

MODEL'S REASONING:
{model_reasoning}

MODEL'S ANSWER:
{model_answer}

GROUND TRUTH ANSWER:
{ground_truth}

RELEVANT PLAYBOOK ENTRIES:
{playbook_context}

Your output should be a JSON object:
{{
  "reasoning": "[Your analysis]",
  "error_identification": "[What went wrong]",
  "root_cause_analysis": "[Why it occurred]",
  "correct_approach": "[Correct approach]",
  "key_insight": "[Key principle for financial analysis]",
  "bullet_tags": [
    {{"id": "bullet_id", "tag": "helpful|harmful|neutral"}}
  ]
}}"""


def get_finer_reflector_prompt(
    question: str,
    model_reasoning: str,
    model_answer: str,
    ground_truth: str,
    playbook_context: str,
) -> str:
    """
    Build the Reflector prompt for FINER (financial) tasks.

    Args:
        question: The financial question
        model_reasoning: Model's reasoning
        model_answer: Model's answer
        ground_truth: Correct answer
        playbook_context: Relevant playbook entries

    Returns:
        Formatted prompt string
    """
    return FINER_REFLECTOR_TEMPLATE.format(
        question=question,
        model_reasoning=model_reasoning,
        model_answer=model_answer,
        ground_truth=ground_truth,
        playbook_context=playbook_context,
    )


# ============================================================================
# Batch Reflector Prompt (for processing multiple errors at once)
# ============================================================================

BATCH_REFLECTOR_TEMPLATE = """You are an expert educator analyzing multiple solution attempts.

For each error case, analyze what went wrong and provide actionable insights.

ERROR CASES:
{error_cases}

Your output should be a JSON object with analysis for each case:
{{
  "analyses": [
    {{
      "case_id": "case_1",
      "reasoning": "[Analysis of this case]",
      "error_identification": "[What went wrong]",
      "root_cause_analysis": "[Why it occurred]",
      "correct_approach": "[What should have been done]",
      "key_insight": "[Key principle to remember]"
    }},
    ...
  ]
}}"""


def get_batch_reflector_prompt(
    error_cases: list,
) -> str:
    """
    Build a batch Reflector prompt for multiple errors.

    Args:
        error_cases: List of error case dicts

    Returns:
        Formatted prompt string
    """
    cases_text = ""
    for i, case in enumerate(error_cases, 1):
        cases_text += f"\nCase {i} (case_{i}):\n"
        cases_text += f"Question: {case.get('question', '')}\n"
        cases_text += f"Model Answer: {case.get('model_answer', '')}\n"
        cases_text += f"Ground Truth: {case.get('ground_truth', '')}\n"
        cases_text += f"Model Reasoning: {case.get('model_reasoning', '')[:500]}...\n"

    return BATCH_REFLECTOR_TEMPLATE.format(error_cases=cases_text)


# ============================================================================
# Error Analysis Functions
# ============================================================================

def compare_answers_numeric(
    predicted: str,
    ground_truth: str,
    tolerance: float = 0.01,
) -> tuple[bool, float]:
    """
    Compare numeric answers with tolerance.

    Args:
        predicted: Predicted answer string
        ground_truth: Ground truth answer string
        tolerance: Relative tolerance for comparison

    Returns:
        Tuple of (is_correct, relative_error)
    """
    import re

    # Extract numbers
    pred_numbers = re.findall(r"[-+]?\d*\.?\d+", str(predicted))
    gt_numbers = re.findall(r"[-+]?\d*\.?\d+", str(ground_truth))

    if not pred_numbers or not gt_numbers:
        return str(predicted).strip().lower() == str(ground_truth).strip().lower(), 0.0

    pred_val = float(pred_numbers[-1])  # Usually last number is the answer
    gt_val = float(gt_numbers[-1])

    if gt_val == 0:
        is_correct = pred_val == 0
        rel_error = 0.0 if is_correct else 1.0
    else:
        rel_error = abs(pred_val - gt_val) / abs(gt_val)
        is_correct = rel_error <= tolerance

    return is_correct, rel_error


def identify_error_type(
    model_reasoning: str,
    ground_truth: str,
) -> str:
    """
    Identify the type of error that occurred.

    Args:
        model_reasoning: The model's reasoning trace
        ground_truth: The correct answer

    Returns:
        Error type category
    """
    reasoning_lower = model_reasoning.lower()

    # Classification of common error types
    if "multipl" in reasoning_lower and "divide" not in reasoning_lower:
        # Check if it should have been division
        gt_val = extract_number(ground_truth)
        pred_val = extract_number(model_reasoning.split()[-1])
        if gt_val and pred_val and pred_val > gt_val:
            return "multiplication_instead_of_division"

    if "divide" in reasoning_lower:
        return "division_error"

    if "add" in reasoning_lower or "subtract" in reasoning_lower:
        return "arithmetic_error"

    if "step" not in reasoning_lower or "then" not in reasoning_lower:
        return "incomplete_reasoning"

    return "conceptual_error"


def extract_number(text: str) -> float:
    """Extract the last number from text."""
    import re
    numbers = re.findall(r"[-+]?\d*\.?\d+", str(text))
    if numbers:
        return float(numbers[-1])
    return None

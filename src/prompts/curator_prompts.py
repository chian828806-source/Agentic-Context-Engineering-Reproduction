"""
Curator Prompt Templates for ACE Framework

The Curator node updates the playbook based on reflections from errors.
"""

from typing import Dict, Any, List


# ============================================================================
# GSM8K Curator Prompt
# ============================================================================

GSM8K_CURATOR_TEMPLATE = """You are a master curator of mathematical knowledge. Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous attempt.

CONTEXT:
- The playbook you created will be used to help solve similar math problems.
- The reflection is generated using ground truth answers that will NOT be available when the playbook is being used.

CRITICAL INSTRUCTIONS:
- Review the existing playbook and the reflection from the previous attempt
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook
- Avoid redundancy - if similar advice already exists, only add new content that complements existing entries
- Do NOT regenerate the entire playbook - only provide the additions needed
- Focus on quality over quantity - a focused, well-organized playbook is better than an exhaustive one
- Output ONLY a valid JSON object (no markdown, no code blocks)

TRAINING CONTEXT:
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}
- Current playbook size: {current_size} bullets

CURRENT PLAYBOOK:
{current_playbook}

RECENT REFLECTION:
{recent_reflection}

QUESTION CONTEXT:
{question_context}

Your output should be a JSON object with these exact fields:
- reasoning: your chain of thought about what should be added
- operations: a list of operations to perform on the playbook

Available operations:
1. ADD: Create new bullet points
   - type: "ADD"
   - section: one of the following sections:
     * "strategies_and_hard_rules" - High-level approaches and rules
     * "formulas_and_calculations" - Mathematical formulas and patterns
     * "verification_checklist" - Steps to verify correctness
     * "common_mistakes" - Known pitfalls to avoid
     * "apis_to_use_for_specific_information" - Domain-specific guidance
   - content: the new content to add (concise, actionable)

RESPONSE FORMAT (JSON only, no markdown):
{{
  "reasoning": "[Your reasoning about what should be added to the playbook]",
  "operations": [
    {{
      "type": "ADD",
      "section": "strategies_and_hard_rules",
      "content": "[New strategy or rule - be specific and actionable]"
    }}
  ]
}}"""


def get_curator_prompt(
    question_context: str,
    current_playbook: Dict[str, List[str]],
    reflection: Dict[str, Any],
    token_budget: int,
    current_step: int,
    total_samples: int,
    task_type: str = "gsm8k",
) -> str:
    """
    Build the Curator prompt for updating the playbook.

    Args:
        question_context: The original question
        current_playbook: Current playbook state
        reflection: Reflection from Reflector node
        token_budget: Maximum tokens for playbook
        current_step: Current sample number
        total_samples: Total number of samples
        task_type: Type of task

    Returns:
        Formatted prompt string
    """
    from .generator_prompts import format_playbook

    playbook_text = format_playbook(current_playbook, task_type=task_type)
    current_size = sum(len(items) for items in current_playbook.values())

    # Format reflection for display
    reflection_text = f"""
Error Identification: {reflection.get('error_identification', '')}
Root Cause: {reflection.get('root_cause_analysis', '')}
Correct Approach: {reflection.get('correct_approach', '')}
Key Insight: {reflection.get('key_insight', '')}
""".strip()

    if task_type == "gsm8k":
        return GSM8K_CURATOR_TEMPLATE.format(
            token_budget=token_budget,
            current_step=current_step,
            total_samples=total_samples,
            current_size=current_size,
            current_playbook=playbook_text,
            recent_reflection=reflection_text,
            question_context=question_context,
        )
    else:
        return GSM8K_CURATOR_TEMPLATE.format(
            token_budget=token_budget,
            current_step=current_step,
            total_samples=total_samples,
            current_size=current_size,
            current_playbook=playbook_text,
            recent_reflection=reflection_text,
            question_context=question_context,
        )


# ============================================================================
# FINER Curator Prompt (for financial tasks)
# ============================================================================

FINER_CURATOR_TEMPLATE = """You are a master curator of financial knowledge. Your job is to update the playbook based on financial analysis reflections.

CONTEXT:
- The playbook helps with XBRL tagging, financial calculations, and document analysis.
- Reflections come from ground truth analysis not available during inference.

CURRENT PLAYBOOK STATS:
{playbook_stats}

CURRENT PLAYBOOK:
{current_playbook}

RECENT REFLECTION:
{recent_reflection}

QUESTION:
{question_context}

Available sections for additions:
- strategies_and_hard_rules: Financial analysis approaches
- formulas_and_calculations: Accounting formulas and calculations
- verification_checklist: XBRL tagging and format checks
- common_mistakes: Financial reporting pitfalls
- apis_to_use_for_specific_information: Domain-specific guidance

Output ONLY JSON:
{{
  "reasoning": "[Your reasoning]",
  "operations": [
    {{
      "type": "ADD",
      "section": "formulas_and_calculations",
      "content": "[Formula or calculation method]"
    }}
  ]
}}"""


def get_finer_curator_prompt(
    question_context: str,
    current_playbook: Dict[str, List[str]],
    reflection: Dict[str, Any],
    playbook_stats: Dict[str, Any],
) -> str:
    """
    Build the Curator prompt for FINER (financial) tasks.

    Args:
        question_context: The financial question
        current_playbook: Current playbook state
        reflection: Reflection from Reflector
        playbook_stats: Statistics about current playbook

    Returns:
        Formatted prompt string
    """
    from .generator_prompts import format_playbook

    playbook_text = format_playbook(current_playbook, task_type="finer")

    stats_text = f"""
Total bullets: {playbook_stats.get('total_bullets', 0)}
Strategies: {len(current_playbook.get('strategies_and_hard_rules', []))}
Formulas: {len(current_playbook.get('formulas_and_calculations', []))}
Checklist items: {len(current_playbook.get('verification_checklist', []))}
Common mistakes: {len(current_playbook.get('common_mistakes', []))}
""".strip()

    reflection_text = f"""
Error: {reflection.get('error_identification', '')}
Root Cause: {reflection.get('root_cause_analysis', '')}
Correct Approach: {reflection.get('correct_approach', '')}
Key Insight: {reflection.get('key_insight', '')}
""".strip()

    return FINER_CURATOR_TEMPLATE.format(
        playbook_stats=stats_text,
        current_playbook=playbook_text,
        recent_reflection=reflection_text,
        question_context=question_context,
    )


# ============================================================================
# Batch Curator Prompt (for processing multiple reflections)
# ============================================================================

BATCH_CURATOR_TEMPLATE = """You are a master curator updating a playbook based on multiple reflections.

CURRENT PLAYBOOK:
{current_playbook}

MULTIPLE REFLECTIONS:
{reflections}

Identify common patterns and unique insights across all reflections.

Output ONLY JSON:
{{
  "reasoning": "[Analysis of common patterns]",
  "operations": [
    {{
      "type": "ADD",
      "section": "strategies_and_hard_rules",
      "content": "[New consolidated strategy]"
    }}
  ]
}}"""


def get_batch_curator_prompt(
    current_playbook: Dict[str, List[str]],
    reflections: List[Dict[str, Any]],
) -> str:
    """
    Build a batch Curator prompt for multiple reflections.

    Args:
        current_playbook: Current playbook state
        reflections: List of reflections

    Returns:
        Formatted prompt string
    """
    from .generator_prompts import format_playbook

    playbook_text = format_playbook(current_playbook)

    reflections_text = ""
    for i, ref in enumerate(reflections, 1):
        reflections_text += f"\nReflection {i}:\n"
        reflections_text += f"Error: {ref.get('error_identification', '')}\n"
        reflections_text += f"Insight: {ref.get('key_insight', '')}\n"

    return BATCH_CURATOR_TEMPLATE.format(
        current_playbook=playbook_text,
        reflections=reflections_text.strip(),
    )


# ============================================================================
# Curator Operation Functions
# ============================================================================

def apply_curator_operations(
    playbook: Dict[str, List[str]],
    operations: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """
    Apply curator operations to update the playbook.

    Args:
        playbook: Current playbook dictionary
        operations: List of operations from Curator

    Returns:
        Updated playbook dictionary
    """
    result = {k: v.copy() for k, v in playbook.items()}

    valid_sections = [
        "strategies_and_hard_rules",
        "formulas_and_calculations",
        "verification_checklist",
        "common_mistakes",
        "apis_to_use_for_specific_information",
    ]

    for op in operations:
        op_type = op.get("type")

        if op_type == "ADD":
            section = op.get("section")
            content = op.get("content", "")

            if section not in valid_sections:
                # Try to match partial section name
                for valid in valid_sections:
                    if section.lower() in valid.lower() or valid.lower() in section.lower():
                        section = valid
                        break

            if section not in result:
                result[section] = []

            # Check for duplicates
            if content and content not in result[section]:
                result[section].append(content)

        elif op_type == "REMOVE":
            section = op.get("section")
            content = op.get("content", "")

            if section in result and content in result[section]:
                result[section].remove(content)

        elif op_type == "UPDATE":
            # Update existing entry
            section = op.get("section")
            old_content = op.get("old_content", "")
            new_content = op.get("new_content", "")

            if section in result and old_content in result[section]:
                idx = result[section].index(old_content)
                result[section][idx] = new_content

    return result


def compress_playbook_if_needed(
    playbook: Dict[str, List[str]],
    max_bullets_per_section: int = 15,
    max_total_bullets: int = 100,
) -> Dict[str, List[str]]:
    """
    Compress playbook if it exceeds size limits.

    Keeps the most recent entries (which are typically more refined).

    Args:
        playbook: Current playbook
        max_bullets_per_section: Max bullets to keep per section
        max_total_bullets: Max total bullets across all sections

    Returns:
        Compressed playbook
    """
    result = {}

    for section, items in playbook.items():
        if items:
            result[section] = items[-max_bullets_per_section:]
        else:
            result[section] = []

    # If still too large, reduce further
    total = sum(len(items) for items in result.values())
    if total > max_total_bullets:
        # Prune proportionally
        for section in result:
            keep = int(len(result[section]) * max_total_bullets / total)
            result[section] = result[section][-max(keep, 1):]

    return result


def deduplicate_playbook(
    playbook: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Remove duplicate entries from playbook while preserving order.

    Args:
        playbook: Current playbook

    Returns:
        Deduplicated playbook
    """
    result = {}

    for section, items in playbook.items():
        seen = set()
        unique_items = []
        for item in items:
            # Normalize for comparison
            normalized = item.strip().lower()
            if normalized not in seen and normalized:
                seen.add(normalized)
                unique_items.append(item)
        result[section] = unique_items

    return result

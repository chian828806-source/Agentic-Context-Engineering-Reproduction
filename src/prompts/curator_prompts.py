"""
Curator Prompt Templates for ACE Framework (FINER-only)
"""

from typing import Dict, Any, List
from .generator_prompts import format_playbook


FINER_CURATOR_TEMPLATE = """You are a master curator of knowledge. Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous attempt.

Context: - The playbook you created will be used to help answering similar questions. - The reflection is generated using ground truth answers that will NOT be available when the playbook is being used. So you need to come up with content that can aid the playbook user to create predictions that likely align with ground truth.

CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.

Instructions: - Review the existing playbook and the reflection from the previous attempt - Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook - Avoid redundancy - if similar advice already exists, only add new content that is a perfect complement to the existing playbook - Do NOT regenerate the entire playbook - only provide the additions needed - Focus on quality over quantity - a focused, well-organized playbook is better than an exhaustive one - Format your response as a PURE JSON object with specific sections - For any operation if no new content to add, return an empty list for the operations field - Be concise and specific - each addition should be actionable

Training Context:
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}

Current Playbook Stats:
{playbook_stats}

Recent Reflection:
{recent_reflection}

Current Playbook:
{current_playbook}

Question Context:
{question_context}

Your Task: Output ONLY a valid JSON object with these exact fields: - reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations - operations: a list of operations to be performed on the playbook - type: the type of operation to be performed - section: the section to add the bullet to - content: the new content of the bullet

Available Operations: 1. ADD: Create new bullet points with fresh IDs - section: the section to add the new bullet to - content: the new content of the bullet. Note: no need to include the bullet_id in the content like '[ctx-00263] helpful=1 harmful=0 ::' the bullet_id will be added by the system.

RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations here]",
  "operations":[
    {{
      "type": "ADD",
      "section": "formulas_and_calculations",
      "content": "[New calculation method...]"
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
    task_type: str = "finer",
) -> str:
    """
    Build the Curator prompt for FINER tasks.
    """
    playbook_text = format_playbook(current_playbook)

    stats_text = f"""
Total bullets: {sum(len(items) for items in current_playbook.values())}
Strategies: {len(current_playbook.get('strategies_and_hard_rules', []))}
Formulas: {len(current_playbook.get('formulas_and_calculations', []))}
Checklist items: {len(current_playbook.get('verification_checklist', []))}
Common mistakes: {len(current_playbook.get('common_mistakes', []))}
APIs/Domain Guidance: {len(current_playbook.get('apis_to_use_for_specific_information', []))}
""".strip()

    reflection_text = f"""
Reasoning: {reflection.get('reasoning', '')}
Error Identification: {reflection.get('error_identification', '')}
Root Cause Analysis: {reflection.get('root_cause_analysis', '')}
Correct Approach: {reflection.get('correct_approach', '')}
Key Insight: {reflection.get('key_insight', '')}
Bullet Tags: {reflection.get('bullet_tags', '')}
""".strip()

    return FINER_CURATOR_TEMPLATE.format(
        token_budget=token_budget,
        current_step=current_step,
        total_samples=total_samples,
        playbook_stats=stats_text,
        current_playbook=playbook_text,
        recent_reflection=reflection_text,
        question_context=question_context,
    )


def get_batch_curator_prompt(
    current_playbook: Dict[str, List[str]],
    reflections: List[Dict[str, Any]],
) -> str:
    """
    Build a batch Curator prompt for multiple reflections.
    """
    playbook_text = format_playbook(current_playbook)

    reflections_text = ""
    for i, ref in enumerate(reflections, 1):
        reflections_text += f"\nReflection {i}:\n"
        reflections_text += f"Error: {ref.get('error_identification', '')}\n"
        reflections_text += f"Insight: {ref.get('key_insight', '')}\n"

    return (
        "You are a master curator updating a playbook based on multiple reflections.\n\n"
        f"CURRENT PLAYBOOK:\n{playbook_text}\n\n"
        f"MULTIPLE REFLECTIONS:\n{reflections_text.strip()}\n\n"
        "Identify common patterns and unique insights across all reflections.\n\n"
        "Output ONLY JSON:\n"
        "{\n"
        '  "reasoning": "[Analysis of common patterns]",\n'
        '  "operations": [\n'
        "    {\n"
        '      "type": "ADD",\n'
        '      "section": "strategies_and_hard_rules",\n'
        '      "content": "[New consolidated strategy]"\n'
        "    }\n"
        "  ]\n"
        "}"
    )


def apply_curator_operations(
    playbook: Dict[str, List[str]],
    operations: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """
    Apply curator operations to update the playbook.
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
                for valid in valid_sections:
                    if section.lower() in valid.lower() or valid.lower() in section.lower():
                        section = valid
                        break

            if section not in result:
                result[section] = []

            if content and content not in result[section]:
                result[section].append(content)

        elif op_type == "REMOVE":
            section = op.get("section")
            content = op.get("content", "")

            if section in result and content in result[section]:
                result[section].remove(content)

        elif op_type == "UPDATE":
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
    """
    result = {}

    for section, items in playbook.items():
        if items:
            result[section] = items[-max_bullets_per_section:]
        else:
            result[section] = []

    total = sum(len(items) for items in result.values())
    if total > max_total_bullets:
        for section in result:
            keep = int(len(result[section]) * max_total_bullets / total)
            result[section] = result[section][-max(keep, 1):]

    return result


def deduplicate_playbook(
    playbook: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Remove duplicate entries from playbook while preserving order.
    """
    result = {}

    for section, items in playbook.items():
        seen = set()
        unique_items = []
        for item in items:
            normalized = item.strip().lower()
            if normalized not in seen and normalized:
                seen.add(normalized)
                unique_items.append(item)
        result[section] = unique_items

    return result

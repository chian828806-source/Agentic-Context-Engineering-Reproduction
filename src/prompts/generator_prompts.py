"""
Generator Prompt Templates for ACE Framework (FINER-only)
"""

from typing import Dict, List


FINER_GENERATOR_TEMPLATE = """You are an analysis expert tasked with answering questions using your knowledge, a curated playbook of strategies and insights and a reflection that goes over the diagnosis of all previous mistakes made while answering the question.

Instructions: - Read the playbook carefully and apply relevant strategies, formulas, and insights - Pay attention to common mistakes listed in the playbook and avoid them - Show your reasoning step-by-step - Be concise but thorough in your analysis - If the playbook contains relevant code snippets or formulas, use them appropriately - Double-check your calculations and logic before providing the final answer

Your output should be a json object, which contains the following fields: - reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations - bullet_ids: each line in the playbook has a bullet_id. all bulletpoints in the playbook that's relevant, helpful for you to answer this question, you should include their bullet_id in this list - final_answer: your concise final answer

Playbook:
{playbook}

Reflection:
{reflection}

Question:
{question}

Context:
{context}

Answer in this exact JSON format:
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",
  "bullet_ids": ["calc-00001", "fin-00002"],
  "final_answer": "[Your concise final answer here]"
}}"""


def get_generator_prompt(
    question: str,
    playbook: Dict[str, List[str]],
    context: str = "",
    reflection: str = "{}",
) -> str:
    """
    Build the Generator prompt for FINER tasks.
    """
    playbook_text = format_playbook(playbook)
    return FINER_GENERATOR_TEMPLATE.format(
        playbook=playbook_text,
        reflection=reflection or "{}",
        question=question,
        context=context or "",
    )


def format_playbook(
    playbook: Dict[str, List[str]],
    max_bullets_per_section: int = 20,
) -> str:
    """
    Format playbook as text for inclusion in prompts.
    """
    sections = []

    section_titles = {
        "strategies_and_hard_rules": "Strategies and Hard Rules",
        "formulas_and_calculations": "Formulas and Calculations",
        "verification_checklist": "Verification Checklist",
        "common_mistakes": "Common Mistakes to Avoid",
        "apis_to_use_for_specific_information": "Domain-Specific Guidance",
    }

    section_order = [
        "strategies_and_hard_rules",
        "formulas_and_calculations",
        "verification_checklist",
        "common_mistakes",
        "apis_to_use_for_specific_information",
    ]

    for section_attr in section_order:
        items = playbook.get(section_attr, [])

        if items:
            title = section_titles.get(section_attr, section_attr.replace("_", " ").title())
            sections.append(f"\n## {title}")

            display_items = items[-max_bullets_per_section:]
            for i, item in enumerate(display_items, 1):
                sections.append(f"{i}. {item}")

    if not sections:
        return "No playbook entries yet. The playbook will be populated through learning from examples."

    return "\n".join(sections)


def format_playbook_compact(
    playbook: Dict[str, List[str]],
    max_total_bullets: int = 50,
) -> str:
    """
    Format playbook in a compact format to save tokens.
    """
    sections = []
    total_count = 0

    for section, items in reversed(list(playbook.items())):
        if not items:
            continue

        section_name = section.replace("_", " ").title()
        sections.append(f"\n{section_name}:")

        remaining = max_total_bullets - total_count
        if remaining <= 0:
            break

        display_items = items[-min(len(items), remaining):]
        for item in display_items:
            sections.append(f"- {item}")
            total_count += 1

    return "\n".join(sections) if sections else "No playbook entries."

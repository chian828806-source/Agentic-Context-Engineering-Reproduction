"""
Generator Prompt Templates for ACE Framework

The Generator node uses the current playbook to generate answers for tasks.
These templates are designed for GSM8K math word problems.
"""

from typing import Dict, List, Any


# ============================================================================
# GSM8K Generator Prompt
# ============================================================================

GSM8K_GENERATOR_TEMPLATE = """You are an expert mathematical problem solver. Your job is to solve math word problems step by step.

You are provided with a curated playbook of strategies, formulas, and insights to help you solve problems effectively.

PLAYBOOK:
{playbook}

INSTRUCTIONS:
- Read the playbook carefully and apply relevant strategies and formulas
- Pay attention to common mistakes listed in the playbook and avoid them
- Show your reasoning step by step
- Be concise but thorough in your analysis
- Double-check your calculations before providing the final answer
- Make sure your final answer is a clear numeric value

QUESTION:
{question}

Your output should be a JSON object with these exact fields:
- reasoning: your step-by-step thinking process, showing all calculations
- final_answer: your concise final answer (just the numeric value, no explanation)

Answer in this exact JSON format:
{{
  "reasoning": "[Your step-by-step reasoning here, showing all work and calculations]",
  "final_answer": "[Your final numeric answer]"
}}"""


def get_generator_prompt(
    question: str,
    playbook: Dict[str, List[str]],
    task_type: str = "gsm8k",
    context: str = "",
) -> str:
    """
    Build the Generator prompt for a specific task.

    Args:
        question: The question to solve
        playbook: Current playbook with strategies
        task_type: Type of task ("gsm8k", "finer", etc.)

    Returns:
        Formatted prompt string
    """
    playbook_text = format_playbook(playbook, task_type=task_type)

    if task_type == "gsm8k":
        return GSM8K_GENERATOR_TEMPLATE.format(
            playbook=playbook_text,
            question=question,
        )
    elif task_type == "finer":
        return FINER_GENERATOR_TEMPLATE.format(
            playbook=playbook_text,
            question=question,
            context=context,
        )
    else:
        # Default to GSM8K
        return GSM8K_GENERATOR_TEMPLATE.format(
            playbook=playbook_text,
            question=question,
        )

# ============================================================================
# FINER Generator Prompt
# ============================================================================

FINER_GENERATOR_TEMPLATE = """You are an expert financial analyst specializing in numeric entity recognition.

You are provided with a curated playbook of strategies, patterns, and insights to help you identify and extract financial entities accurately.

PLAYBOOK:
{playbook}

INSTRUCTIONS:
- Read the playbook carefully and apply relevant strategies and patterns
- Pay attention to common mistakes listed in the playbook and avoid them
- Analyze the financial context thoroughly
- Identify all relevant numeric entities in the text
- Be precise in your identification and extraction
- Double-check your findings before providing the final answer
- Make sure your final answer is a clear and accurate extraction

CONTEXT:
{context}

Your output should be a JSON object with these exact fields:
- reasoning: your step-by-step thinking process, showing your analysis of the financial context
- final_answer: your concise final answer (extracted numeric entities, formatted as needed)

Answer in this exact JSON format:
{{
  "reasoning": "[Your step-by-step reasoning here, showing your analysis]",
  "final_answer": "[Your final extracted numeric entities]"
}}"""


def format_playbook(
    playbook: Dict[str, List[str]],
    task_type: str = "gsm8k",
    max_bullets_per_section: int = 20,
) -> str:
    """
    Format playbook as text for inclusion in prompts.

    Args:
        playbook: Playbook dictionary with sections
        task_type: Type of task for section filtering
        max_bullets_per_section: Max bullets to include per section

    Returns:
        Formatted text representation of the playbook
    """
    sections = []

    # Section titles for display
    section_titles = {
        "strategies_and_hard_rules": "Strategies and Hard Rules",
        "formulas_and_calculations": "Formulas and Calculations",
        "verification_checklist": "Verification Checklist",
        "common_mistakes": "Common Mistakes to Avoid",
        "apis_to_use_for_specific_information": "Domain-Specific Guidance",
    }

    # Order of sections in the output
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

            # Limit bullets per section (take most recent)
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

    Args:
        playbook: Playbook dictionary
        max_total_bullets: Maximum total bullets across all sections

    Returns:
        Compact formatted text
    """
    sections = []
    total_count = 0

    # Process in order, taking from most recent
    for section, items in reversed(list(playbook.items())):
        if not items:
            continue

        section_name = section.replace("_", " ").title()
        sections.append(f"\n{section_name}:")

        # Take items from the end (most recent)
        remaining = max_total_bullets - total_count
        if remaining <= 0:
            break

        display_items = items[-min(len(items), remaining):]
        for item in display_items:
            sections.append(f"- {item}")
            total_count += 1

    return "\n".join(sections) if sections else "No playbook entries."


# ============================================================================
# FINER Generator Prompt (for financial tasks)
# ============================================================================

FINER_GENERATOR_TEMPLATE = """You are an expert financial analyst. Your job is to analyze financial documents and extract relevant information with precision.

You are provided with a curated playbook of strategies, formulas, and industry insights.

PLAYBOOK:
{playbook}

INSTRUCTIONS:
- Carefully read the financial document/question
- Apply relevant formulas and calculation methods from the playbook
- Pay attention to XBRL tagging rules and financial reporting standards
- Show your reasoning and calculations
- Extract the exact numeric values with proper units
- Verify your answer against common financial pitfalls

QUESTION:
{question}

CONTEXT (if provided):
{context}

Your output should be a JSON object with these exact fields:
- reasoning: your step-by-step analysis
- bullet_ids: list of relevant playbook bullet IDs used (if applicable)
- final_answer: your concise final answer with proper formatting

Answer in this exact JSON format:
{{
  "reasoning": "[Your step-by-step reasoning]",
  "bullet_ids": ["id1", "id2"],
  "final_answer": "[Your final answer]"
}}"""


def get_finer_generator_prompt(
    question: str,
    playbook: Dict[str, List[str]],
    context: str = "",
) -> str:
    """
    Build the Generator prompt for FINER (financial) tasks.

    Args:
        question: The financial question
        playbook: Current playbook
        context: Additional context/document text

    Returns:
        Formatted prompt string
    """
    playbook_text = format_playbook(playbook, task_type="finer")

    return FINER_GENERATOR_TEMPLATE.format(
        playbook=playbook_text,
        question=question,
        context=context or "No additional context provided.",
    )


# ============================================================================
# Few-shot examples for GSM8K
# ============================================================================

GSM8K_FEWSHOT_EXAMPLES = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "reasoning": "Natalia sold 48 clips in April. In May, she sold half as many clips as in April, which is 48 / 2 = 24 clips. To find the total, add April and May sales: 48 + 24 = 72 clips.",
        "final_answer": "72",
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "reasoning": "Weng earns $12 per hour. Since there are 60 minutes in an hour, 50 minutes is 50/60 = 5/6 of an hour. Weng earned $12 × (5/6) = $10.",
        "final_answer": "10",
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need?",
        "reasoning": "The wallet costs $100. Betty has half, which is $100 / 2 = $50. Her parents give $15. Her grandparents give twice what parents give: 2 × $15 = $30. Total now: $50 + $15 + $30 = $95. Still needs: $100 - $95 = $5.",
        "final_answer": "5",
    },
]


def get_generator_prompt_with_fewshot(
    question: str,
    playbook: Dict[str, List[str]],
    num_examples: int = 3,
) -> str:
    """
    Build Generator prompt with few-shot examples.

    Args:
        question: The question to solve
        playbook: Current playbook
        num_examples: Number of few-shot examples to include

    Returns:
        Formatted prompt with examples
    """
    playbook_text = format_playbook(playbook)

    examples_text = ""
    for i, ex in enumerate(GSM8K_FEWSHOT_EXAMPLES[:num_examples], 1):
        examples_text += f"\nExample {i}:\n"
        examples_text += f"Question: {ex['question']}\n"
        examples_text += f"Reasoning: {ex['reasoning']}\n"
        examples_text += f"Answer: {ex['final_answer']}\n"

    template = f"""You are an expert mathematical problem solver. Study these examples carefully, then solve the new problem using the provided playbook.

{examples_text}

PLAYBOOK:
{playbook_text}

Now solve this problem:

QUESTION:
{question}

Your output should be a JSON object:
{{
  "reasoning": "[Your step-by-step reasoning]",
  "final_answer": "[Your numeric answer]"
}}"""

    return template

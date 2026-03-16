"""
Curator Node for ACE Framework

The Curator node updates the playbook based on reflections from errors.
"""

import json
from typing import Dict, Any, List
from ..llm.glm_client import GLMClient
from ..prompts.curator_prompts import (
    get_curator_prompt,
    apply_curator_operations,
    compress_playbook_if_needed,
    deduplicate_playbook,
)


class CuratorNode:
    """
    Curator Node: Updates the playbook based on reflections.

    This node is responsible for:
    1. Analyzing the current playbook and reflection
    2. Deciding what new content to add
    3. Applying incremental updates (not full rewrite)
    4. Managing playbook size to prevent token overflow
    """

    def __init__(
        self,
        llm_client: GLMClient,
        max_retries: int = 3,
        temperature: float = 0.2,
        max_playbook_size: int = 10000,
        max_bullets_per_section: int = 20,
        task_type: str = "gsm8k",
    ):
        """
        Initialize the Curator node.

        Args:
            llm_client: GLM-4.6 client instance
            max_retries: Maximum number of retry attempts
            temperature: Sampling temperature (low for consistent updates)
            max_playbook_size: Approximate max token size
            max_bullets_per_section: Max bullets per section before compression
            task_type: Type of task
        """
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_playbook_size = max_playbook_size
        self.max_bullets_per_section = max_bullets_per_section
        self.task_type = task_type

        # Valid playbook sections
        self.valid_sections = [
            "strategies_and_hard_rules",
            "formulas_and_calculations",
            "verification_checklist",
            "common_mistakes",
            "apis_to_use_for_specific_information",
        ]

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Curator node.

        Args:
            state: Current ACEState containing:
                - current_playbook: Current playbook to update
                - reflection: Reflection from Reflector
                - current_sample: Original sample
                - generation_index: Current generation number
                - samples_processed: Number of samples processed

        Returns:
            Dict with updates to state:
                - current_playbook: Updated playbook
        """
        playbook = state.get("current_playbook", {})
        reflection = state.get("reflection", {})
        sample = state.get("current_sample", {})

        # Skip if reflection indicates correct answer
        if reflection.get("status") == "correct":
            return {"current_playbook": playbook}

        # Check if playbook needs compression
        playbook = self._maybe_compress_playbook(playbook)

        # Estimate token budget
        current_size = self._estimate_playbook_tokens(playbook)
        token_budget = self.max_playbook_size

        # Build curator prompt
        prompt = get_curator_prompt(
            question_context=sample.get("question", ""),
            current_playbook=playbook,
            reflection=reflection,
            token_budget=token_budget,
            current_step=state.get("samples_processed", 0),
            total_samples=state.get("total_samples", 1000),
            task_type=self.task_type,
        )

        # Call LLM with retry logic
        operations = None
        last_error = None

        for attempt in range(self.max_retries):
            try:
                result = self.llm_client.call_json(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=2000,
                    system_prompt=self._get_system_prompt(),
                )
                operations = result.get("operations", [])
                break
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    continue

        if operations is None:
            # Fallback: try to add insight from reflection
            operations = self._create_fallback_operations(reflection)

        # Apply operations to playbook
        updated_playbook = apply_curator_operations(playbook, operations)

        # Deduplicate to prevent redundancy
        updated_playbook = deduplicate_playbook(updated_playbook)

        return {"current_playbook": updated_playbook}

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Curator."""
        return (
            "You are a knowledge curator. Your job is to ADD new, unique insights to a playbook. "
            "NEVER regenerate existing content. Only add what is genuinely new and helpful. "
            "Be concise and specific."
        )

    def _maybe_compress_playbook(
        self,
        playbook: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """
        Compress playbook if it exceeds size limits.

        Args:
            playbook: Current playbook

        Returns:
            Compressed playbook if needed, otherwise original
        """
        total_bullets = sum(len(items) for items in playbook.values())

        if total_bullets > self.max_bullets_per_section * len(self.valid_sections):
            return compress_playbook_if_needed(
                playbook,
                max_bullets_per_section=self.max_bullets_per_section,
            )

        return playbook

    def _estimate_playbook_tokens(self, playbook: Dict[str, List[str]]) -> int:
        """
        Estimate token count for a playbook.

        Args:
            playbook: Playbook dictionary

        Returns:
            Estimated token count
        """
        total_chars = sum(len(item) for items in playbook.values() for item in items)
        # Rough estimate: ~1 token per 4 characters
        return total_chars // 4

    def _create_fallback_operations(
        self,
        reflection: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        Create fallback operations when LLM call fails.

        Args:
            reflection: Reflection with key insights

        Returns:
            List of operations to apply
        """
        operations = []

        # Try to extract key insight
        key_insight = reflection.get("key_insight", "")
        if key_insight:
            # Determine best section
            error_type = reflection.get("error_identification", "").lower()

            if "calculat" in error_type or "arithmet" in error_type:
                section = "formulas_and_calculations"
            elif "verif" in error_type or "check" in error_type:
                section = "verification_checklist"
            elif "strateg" in key_insight.lower():
                section = "strategies_and_hard_rules"
            else:
                section = "common_mistakes"

            operations.append({
                "type": "ADD",
                "section": section,
                "content": key_insight,
            })

        return operations

    def batch_update(
        self,
        playbook: Dict[str, List[str]],
        reflections: List[Dict[str, Any]],
        samples: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """
        Update playbook based on multiple reflections at once.

        Args:
            playbook: Current playbook
            reflections: List of reflections
            samples: Corresponding samples

        Returns:
            Updated playbook
        """
        from ..prompts.curator_prompts import get_batch_curator_prompt

        # Build batch prompt
        batch_prompt = get_batch_curator_prompt(playbook, reflections)

        # Call LLM
        try:
            result = self.llm_client.call_json(
                prompt=batch_prompt,
                temperature=self.temperature,
                max_tokens=2000,
            )
            operations = result.get("operations", [])
        except Exception:
            # Fallback: process individually
            for reflection, sample in zip(reflections, samples):
                state = {
                    "current_playbook": playbook,
                    "reflection": reflection,
                    "current_sample": sample,
                }
                result = self(state)
                playbook = result["current_playbook"]
            return playbook

        return apply_curator_operations(playbook, operations)


class AggressiveCurator(CuratorNode):
    """
    More aggressive curator that adds more content per reflection.
    Useful for rapid learning but may lead to larger playbooks.
    """

    def __init__(self, *args, **kwargs):
        kwargs["max_bullets_per_section"] = kwargs.get("max_bullets_per_section", 30)
        kwargs["temperature"] = kwargs.get("temperature", 0.4)
        super().__init__(*args, **kwargs)


class ConservativeCurator(CuratorNode):
    """
    More conservative curator that adds less content.
    Useful for maintaining smaller, more focused playbooks.
    """

    def __init__(self, *args, **kwargs):
        kwargs["max_bullets_per_section"] = kwargs.get("max_bullets_per_section", 10)
        kwargs["temperature"] = kwargs.get("temperature", 0.1)
        super().__init__(*args, **kwargs)

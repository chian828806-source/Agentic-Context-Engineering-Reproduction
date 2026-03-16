"""
Playbook Data Structure for ACE Framework

The Playbook is the evolving context that accumulates strategies,
formulas, and insights throughout the ACE evolution process.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
import json
from datetime import datetime


@dataclass
class Playbook:
    """
    ACE Playbook Data Structure

    The playbook stores categorized strategies and insights that evolve
    through the Generator-Reflector-Curator loop. It is designed to:
    1. Prevent context collapse through incremental updates
    2. Maintain structured organization of knowledge
    3. Support efficient retrieval and application

    Sections:
    - strategies_and_hard_rules: High-level approaches and must-follow rules
    - formulas_and_calculations: Mathematical formulas and computation patterns
    - verification_checklist: Steps to verify correctness
    - common_mistakes: Known pitfalls to avoid
    - apis_to_use_for_specific_information: Domain-specific guidance (for agent tasks)
    """

    strategies_and_hard_rules: List[str] = field(default_factory=list)
    formulas_and_calculations: List[str] = field(default_factory=list)
    verification_checklist: List[str] = field(default_factory=list)
    common_mistakes: List[str] = field(default_factory=list)
    apis_to_use_for_specific_information: List[str] = field(default_factory=list)

    # Metadata
    bullet_counter: Dict[str, int] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_created: int = 0
    generation_last_updated: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert playbook to dictionary format for serialization.

        Returns:
            Dictionary representation of the playbook
        """
        return {
            "strategies_and_hard_rules": self.strategies_and_hard_rules,
            "formulas_and_calculations": self.formulas_and_calculations,
            "verification_checklist": self.verification_checklist,
            "common_mistakes": self.common_mistakes,
            "apis_to_use_for_specific_information": self.apis_to_use_for_specific_information,
            "metadata": {
                "bullet_counter": self.bullet_counter,
                "created_at": self.created_at,
                "last_updated": self.last_updated,
                "generation_created": self.generation_created,
                "generation_last_updated": self.generation_last_updated,
                "total_bullets": self.total_bullets(),
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Playbook":
        """
        Create Playbook from dictionary representation.

        Args:
            data: Dictionary containing playbook data

        Returns:
            Playbook instance
        """
        metadata = data.get("metadata", {})

        return cls(
            strategies_and_hard_rules=data.get("strategies_and_hard_rules", []),
            formulas_and_calculations=data.get("formulas_and_calculations", []),
            verification_checklist=data.get("verification_checklist", []),
            common_mistakes=data.get("common_mistakes", []),
            apis_to_use_for_specific_information=data.get("apis_to_use_for_specific_information", []),
            bullet_counter=metadata.get("bullet_counter", {}),
            created_at=metadata.get("created_at", datetime.now().isoformat()),
            last_updated=metadata.get("last_updated", datetime.now().isoformat()),
            generation_created=metadata.get("generation_created", 0),
            generation_last_updated=metadata.get("generation_last_updated", 0),
        )

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, List[str]], generation: int = 0) -> "Playbook":
        """
        Create Playbook from a simple state dictionary (used in ACEState).

        Args:
            state_dict: Dictionary with section names as keys
            generation: Current generation number

        Returns:
            Playbook instance
        """
        return cls(
            strategies_and_hard_rules=state_dict.get("strategies_and_hard_rules", []),
            formulas_and_calculations=state_dict.get("formulas_and_calculations", []),
            verification_checklist=state_dict.get("verification_checklist", []),
            common_mistakes=state_dict.get("common_mistakes", []),
            apis_to_use_for_specific_information=state_dict.get("apis_to_use_for_specific_information", []),
            generation_created=generation,
            generation_last_updated=generation,
        )

    def total_bullets(self) -> int:
        """Return total number of bullets across all sections."""
        return sum(len(getattr(self, section)) for section in self._section_names())

    def _section_names(self) -> List[str]:
        """Return list of section attribute names."""
        return [
            "strategies_and_hard_rules",
            "formulas_and_calculations",
            "verification_checklist",
            "common_mistakes",
            "apis_to_use_for_specific_information",
        ]

    def add_bullet(self, section: str, content: str, generation: int) -> None:
        """
        Add a bullet point to a specific section.

        Args:
            section: Section name to add to
            content: Bullet point content
            generation: Current generation number
        """
        if section not in self._section_names():
            raise ValueError(f"Invalid section: {section}")

        getattr(self, section).append(content)
        self.last_updated = datetime.now().isoformat()
        self.generation_last_updated = generation

        # Update bullet counter
        section_key = section.replace("_", "-")
        self.bullet_counter[section_key] = self.bullet_counter.get(section_key, 0) + 1

    def to_state_dict(self) -> Dict[str, List[str]]:
        """
        Convert to simple dict format for ACEState.

        Returns:
            Dictionary with section names as keys and lists as values
        """
        return {
            "strategies_and_hard_rules": self.strategies_and_hard_rules,
            "formulas_and_calculations": self.formulas_and_calculations,
            "verification_checklist": self.verification_checklist,
            "common_mistakes": self.common_mistakes,
            "apis_to_use_for_specific_information": self.apis_to_use_for_specific_information,
        }

    def format_for_prompt(self, max_bullets_per_section: int = 20) -> str:
        """
        Format playbook as text for inclusion in prompts.

        Args:
            max_bullets_per_section: Maximum bullets to include per section

        Returns:
            Formatted text representation
        """
        sections = []

        section_titles = {
            "strategies_and_hard_rules": "Strategies and Hard Rules",
            "formulas_and_calculations": "Formulas and Calculations",
            "verification_checklist": "Verification Checklist",
            "common_mistakes": "Common Mistakes to Avoid",
            "apis_to_use_for_specific_information": "Domain-Specific Guidance",
        }

        for section_attr in self._section_names():
            items = getattr(self, section_attr)

            if items:
                title = section_titles.get(section_attr, section_attr.replace("_", " ").title())
                sections.append(f"\n## {title}")

                # Limit bullets per section
                for i, item in enumerate(items[-max_bullets_per_section:], 1):
                    sections.append(f"{i}. {item}")

        if not sections:
            return "No playbook entries yet. The playbook will be populated through learning."

        return "\n".join(sections)

    def compress(self, keep_per_section: int = 10) -> "Playbook":
        """
        Create a compressed version of the playbook, keeping only recent entries.

        This helps prevent token overflow during long training runs.

        Args:
            keep_per_section: Number of entries to keep per section

        Returns:
            New compressed Playbook instance
        """
        compressed = Playbook(
            strategies_and_hard_rules=self.strategies_and_hard_rules[-keep_per_section:],
            formulas_and_calculations=self.formulas_and_calculations[-keep_per_section:],
            verification_checklist=self.verification_checklist[-keep_per_section:],
            common_mistakes=self.common_mistakes[-keep_per_section:],
            apis_to_use_for_specific_information=self.apis_to_use_for_specific_information[-keep_per_section:],
            created_at=self.created_at,
            generation_created=self.generation_created,
            generation_last_updated=self.generation_last_updated,
        )

        return compressed

    def estimate_tokens(self) -> int:
        """
        Estimate the token count for this playbook.

        Returns:
            Estimated token count (rough approximation)
        """
        text = self.format_for_prompt()
        # Rough estimate: ~1 token per 4 characters
        return len(text) // 4

    def save(self, filepath: str) -> None:
        """
        Save playbook to JSON file.

        Args:
            filepath: Path to save the playbook
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "Playbook":
        """
        Load playbook from JSON file.

        Args:
            filepath: Path to load the playbook from

        Returns:
            Playbook instance
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def merge(self, other: "Playbook") -> None:
        """
        Merge another playbook into this one.

        Args:
            other: Playbook to merge in
        """
        for section in self._section_names():
            my_items = getattr(self, section)
            other_items = getattr(other, section)

            # Add items from other that aren't in this playbook
            for item in other_items:
                if item not in my_items:
                    my_items.append(item)

        self.last_updated = datetime.now().isoformat()

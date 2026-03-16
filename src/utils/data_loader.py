"""
Data Loading Utilities for ACE Framework

Supports loading GSM8K and other datasets in JSONL format.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path


def load_jsonl(
    filepath: str,
    max_samples: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.

    Args:
        filepath: Path to the JSONL file
        max_samples: Maximum number of samples to load (None for all)
        shuffle: Whether to shuffle the data
        seed: Random seed for shuffling

    Returns:
        List of dictionaries, one per sample
    """
    data = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                if max_samples and len(data) >= max_samples:
                    break

    if shuffle:
        import random
        random.seed(seed)
        random.shuffle(data)

    return data


def load_gsm8k(
    filepath: str,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset from JSONL file.

    GSM8K format typically has:
    - "question": The math word problem
    - "answer": The answer (may include reasoning)

    Args:
        filepath: Path to the GSM8K JSONL file
        max_samples: Maximum number of samples to load
        shuffle: Whether to shuffle the data
        seed: Random seed for shuffling

    Returns:
        List of standardized sample dictionaries with keys:
        - question: str
        - answer: str (extracted final numeric answer)
        - full_answer: str (original answer with reasoning)
    """
    raw_data = load_jsonl(filepath, max_samples=max_samples, shuffle=shuffle, seed=seed)

    processed_data = []
    for sample in raw_data:
        # Extract numeric answer from full answer
        full_answer = sample.get("answer", "")

        # GSM8K answer format: "The answer is 42."
        numeric_answer = extract_gsm8k_answer(full_answer)

        processed_data.append({
            "question": sample.get("question", ""),
            "answer": numeric_answer,
            "full_answer": full_answer,
            "original": sample,
        })

    return processed_data


def extract_gsm8k_answer(answer_text: str) -> str:
    """
    Extract the final numeric answer from GSM8K answer format.

    GSM8K answers typically end with "The answer is {number}." or similar.

    Args:
        answer_text: Full answer text from GSM8K

    Returns:
        Extracted numeric answer as string
    """
    import re

    # Try to find "The answer is" pattern
    match = re.search(r"(?:The answer is|Answer)(?:\s+is)?\s+([\d,]+\.?\d*)", answer_text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")

    # Try to find the last number in the text
    numbers = re.findall(r"[\d,]+\.?\d*", answer_text)
    if numbers:
        return numbers[-1].replace(",", "")

    # Fallback: return the last few words
    words = answer_text.strip().split()
    if words:
        return words[-1]

    return ""


def load_finer(
    filepath: str,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Load FINER (Financial numeric entity recognition) dataset.

    FINER format for financial analysis tasks.

    Args:
        filepath: Path to the FINER JSONL file
        max_samples: Maximum number of samples to load
        shuffle: Whether to shuffle the data
        seed: Random seed for shuffling

    Returns:
        List of standardized sample dictionaries
    """
    raw_data = load_jsonl(filepath, max_samples=max_samples, shuffle=shuffle, seed=seed)

    processed_data = []
    for sample in raw_data:
        processed_data.append({
            "question": sample.get("question", sample.get("text", "")),
            "answer": str(sample.get("answer", sample.get("label", ""))),
            "context": sample.get("context", ""),
            "original": sample,
        })

    return processed_data


def split_data(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    seed: int = 42
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data into train and validation sets.

    Args:
        data: List of samples
        train_ratio: Ratio of training data
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data)
    """
    import random
    random.seed(seed)

    shuffled = data.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)

    return shuffled[:split_idx], shuffled[split_idx:]


def create_sample_batches(
    data: List[Dict[str, Any]],
    batch_size: int = 1
) -> List[List[Dict[str, Any]]]:
    """
    Split data into batches for processing.

    Args:
        data: List of samples
        batch_size: Size of each batch

    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i + batch_size])
    return batches


def save_jsonl(data: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save data to a JSONL file.

    Args:
        data: List of samples to save
        filepath: Path to save the file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def download_gsm8k(output_dir: str = "data") -> tuple[str, str]:
    """
    Download GSM8K dataset if not present.

    This is a placeholder - in practice, you would download from:
    https://github.com/openai/grade-school-math

    Args:
        output_dir: Directory to save the dataset

    Returns:
        Tuple of (train_path, test_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    train_path = output_path / "gsm8k_train.jsonl"
    test_path = output_path / "gsm8k_test.jsonl"

    # Check if files already exist
    if train_path.exists() and test_path.exists():
        return str(train_path), str(test_path)

    # TODO: Implement actual download
    # For now, create placeholder files
    print(f"Warning: GSM8K dataset not found at {output_dir}")
    print(f"Please download from: https://github.com/openai/grade-school-math")
    print(f"Expected files: {train_path}, {test_path}")

    return str(train_path), str(test_path)


def create_sample_gsm8k(output_path: str = "data/gsm8k_sample.jsonl") -> None:
    """
    Create a small sample GSM8K file for testing.

    Args:
        output_path: Path to save the sample file
    """
    samples = [
        {
            "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "answer": "Natalia sold 48 clips in April. In May, she sold half as many, which is 48 / 2 = 24 clips. The total number of clips sold is 48 + 24 = 72 clips. The answer is 72."
        },
        {
            "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            "answer": "Weng earns $12 per hour. Since there are 60 minutes in an hour, 50 minutes is 50/60 = 5/6 of an hour. Weng earned $12 * 5/6 = $10. The answer is 10."
        },
        {
            "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need?",
            "answer": "The wallet costs $100. Betty has half of that, which is $100 / 2 = $50. Her parents give her $15. Her grandparents give her twice as much as her parents, which is 2 * $15 = $30. In total, Betty now has $50 + $15 + $30 = $95. She still needs $100 - $95 = $5. The answer is 5."
        },
    ]

    save_jsonl(samples, output_path)
    print(f"Created sample GSM8K file at {output_path}")


def print_data_stats(data: List[Dict[str, Any]], name: str = "Dataset") -> None:
    """
    Print statistics about a dataset.

    Args:
        data: List of samples
        name: Name of the dataset
    """
    print(f"\n{'='*50}")
    print(f"{name} Statistics")
    print(f"{'='*50}")
    print(f"Total samples: {len(data)}")

    if data:
        # Question length stats
        q_lengths = [len(s.get("question", "")) for s in data]
        print(f"Question length - avg: {sum(q_lengths)/len(q_lengths):.1f}, max: {max(q_lengths)}")

        # Answer length stats
        a_lengths = [len(str(s.get("answer", ""))) for s in data]
        print(f"Answer length - avg: {sum(a_lengths)/len(a_lengths):.1f}, max: {max(a_lengths)}")

        # Show sample
        print(f"\nSample question:")
        print(f"  {data[0].get('question', '')[:100]}...")
        print(f"Sample answer: {data[0].get('answer', '')}")
    print(f"{'='*50}\n")

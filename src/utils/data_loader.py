"""
Data Loading Utilities for ACE Framework (FINER-only)

Expected FINER JSONL format (per *_example.jsonl):
{
  "id": <int>,
  "tokens": [<str>, ...],
  "ner_tags": [<str>, ...]
}

This loader also prepares fields aligned with the paper repo:
- context: joined token string
- question: instruction string
- target: comma-separated tag sequence
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path


def load_jsonl(
    filepath: str,
    max_samples: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
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


def load_finer(
    filepath: str,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load FINER dataset from JSONL file.

    Args:
        filepath: Path to the FINER JSONL file
        max_samples: Maximum number of samples to load
        shuffle: Whether to shuffle the data
        seed: Random seed for shuffling

    Returns:
        List of standardized sample dictionaries:
        - id: int
        - text: str (tokens joined by space)
        - tokens: List[str]
        - ner_tags: List[str]
        - original: Dict (raw sample)
    """
    raw_data = load_jsonl(filepath, max_samples=max_samples, shuffle=shuffle, seed=seed)

    processed_data = []
    for sample in raw_data:
        tokens = sample.get("tokens", [])
        text = " ".join(tokens)
        ner_tags = sample.get("ner_tags", [])
        processed_data.append({
            "id": sample.get("id"),
            "text": text,
            "context": text,
            "question": "Perform FINER sequence labeling. Output a comma-separated tag for each token.",
            "tokens": tokens,
            "ner_tags": ner_tags,
            "target": ", ".join(ner_tags),
            "original": sample,
        })

    return processed_data


def split_data(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data into train and validation sets.
    """
    import random
    random.seed(seed)

    shuffled = data.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)

    return shuffled[:split_idx], shuffled[split_idx:]


def create_sample_batches(
    data: List[Dict[str, Any]],
    batch_size: int = 1,
) -> List[List[Dict[str, Any]]]:
    """
    Split data into batches for processing.
    """
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i + batch_size])
    return batches


def save_jsonl(data: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save data to a JSONL file.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def print_data_stats(data: List[Dict[str, Any]], name: str = "Dataset") -> None:
    """
    Print statistics about a dataset.
    """
    print(f"\n{'='*50}")
    print(f"{name} Statistics")
    print(f"{'='*50}")
    print(f"Total samples: {len(data)}")

    if data:
        text_lengths = [len(s.get("text", "")) for s in data]
        print(f"Text length - avg: {sum(text_lengths)/len(text_lengths):.1f}, max: {max(text_lengths)}")

        avg_ner_tags = sum(len(s.get("ner_tags", [])) for s in data) / len(data)
        print(f"Average NER tags per sample: {avg_ner_tags:.1f}")

        print(f"\nSample text:")
        print(f"  {data[0].get('text', '')[:100]}...")
        print(f"Sample NER tags: {data[0].get('ner_tags', [])[:10]}...")
    print(f"{'='*50}\n")

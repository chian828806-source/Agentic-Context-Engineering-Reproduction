"""
Logging Utilities for ACE Framework
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import csv


class ACELogger:
    """
    Logger for tracking ACE experiment progress and results.

    Handles:
    - Training progress logging
    - Prompt evolution history
    - Performance metrics
    - Checkpoint saving/loading
    """

    def __init__(
        self,
        logs_dir: str = "logs",
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize the ACE logger.

        Args:
            logs_dir: Directory to store logs
            experiment_name: Name for this experiment run
        """
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.experiment_name = experiment_name
        self.experiment_dir = self.logs_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)

        # Log files
        self.metrics_file = self.experiment_dir / "metrics.jsonl"
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        # In-memory metrics storage
        self.metrics_history: List[Dict[str, Any]] = []

    def log_metrics(
        self,
        generation: int,
        epoch: int,
        accuracy: float,
        playbook_size: int,
        token_usage: Optional[Dict[str, int]] = None,
        **kwargs
    ) -> None:
        """
        Log metrics for a generation/epoch.

        Args:
            generation: Generation number
            epoch: Current epoch
            accuracy: Accuracy score
            playbook_size: Size of playbook in tokens
            token_usage: Token usage statistics
            **kwargs: Additional metrics to log
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "generation": generation,
            "epoch": epoch,
            "accuracy": accuracy,
            "playbook_size": playbook_size,
            **kwargs,
        }

        if token_usage:
            entry["token_usage"] = token_usage

        self.metrics_history.append(entry)

        # Append to file
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def save_checkpoint(
        self,
        generation: int,
        playbook: Dict[str, List[str]],
        accuracy: float,
        state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a checkpoint with the current playbook and state.

        Args:
            generation: Generation number
            playbook: Current playbook
            accuracy: Current accuracy
            state: Optional full state to save

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoints_dir / f"checkpoint_gen{generation:04d}.json"

        checkpoint_data = {
            "generation": generation,
            "timestamp": datetime.now().isoformat(),
            "accuracy": accuracy,
            "playbook": playbook,
        }

        if state:
            checkpoint_data["state"] = state

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

        return str(checkpoint_path)

    def load_checkpoint(self, generation: int) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint by generation number.

        Args:
            generation: Generation number to load

        Returns:
            Checkpoint data or None if not found
        """
        checkpoint_path = self.checkpoints_dir / f"checkpoint_gen{generation:04d}.json"

        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest checkpoint.

        Returns:
            Latest checkpoint data or None if no checkpoints exist
        """
        checkpoints = sorted(self.checkpoints_dir.glob("checkpoint_*.json"))

        if not checkpoints:
            return None

        with open(checkpoints[-1], "r", encoding="utf-8") as f:
            return json.load(f)

    def save_playbook_evolution(
        self,
        playbook: Dict[str, List[str]],
        generation: int,
        accuracy: float,
    ) -> str:
        """
        Save evolved playbook at a generation.

        Args:
            playbook: Current playbook
            generation: Generation number
            accuracy: Current accuracy

        Returns:
            Path to saved playbook
        """
        playbook_path = self.experiment_dir / f"playbook_v{generation:04d}.json"

        playbook_data = {
            "generation": generation,
            "timestamp": datetime.now().isoformat(),
            "accuracy": accuracy,
            "playbook": playbook,
            "total_bullets": sum(len(v) for v in playbook.values()),
        }

        with open(playbook_path, "w", encoding="utf-8") as f:
            json.dump(playbook_data, f, indent=2, ensure_ascii=False)

        return str(playbook_path)

    def save_error_samples(
        self,
        generation: int,
        error_samples: List[Dict[str, Any]],
    ) -> str:
        """
        Save error samples for analysis.

        Args:
            generation: Current generation
            error_samples: List of error samples

        Returns:
            Path to saved error samples
        """
        errors_path = self.experiment_dir / f"errors_gen{generation:04d}.json"

        with open(errors_path, "w", encoding="utf-8") as f:
            json.dump(error_samples, f, indent=2, ensure_ascii=False)

        return str(errors_path)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all logged metrics.

        Returns:
            Summary dict with best accuracy, final accuracy, etc.
        """
        if not self.metrics_history:
            return {}

        accuracies = [m["accuracy"] for m in self.metrics_history]

        return {
            "total_entries": len(self.metrics_history),
            "best_accuracy": max(accuracies),
            "final_accuracy": accuracies[-1] if accuracies else 0,
            "best_generation": max(
                self.metrics_history,
                key=lambda x: x["accuracy"]
            )["generation"],
        }

    def export_metrics_csv(self, output_path: Optional[str] = None) -> str:
        """
        Export metrics to CSV file.

        Args:
            output_path: Optional custom output path

        Returns:
            Path to exported CSV
        """
        if output_path is None:
            output_path = self.experiment_dir / "metrics.csv"

        output_path = Path(output_path)

        if not self.metrics_history:
            return str(output_path)

        # Get all unique keys
        fieldnames = set()
        for entry in self.metrics_history:
            fieldnames.update(entry.keys())

        # Flatten nested dicts
        fieldnames = sorted(fieldnames)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entry in self.metrics_history:
                # Flatten nested structures
                flat_entry = {}
                for key, value in entry.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            flat_entry[f"{key}_{subkey}"] = subvalue
                    else:
                        flat_entry[key] = value
                writer.writerow(flat_entry)

        return str(output_path)

    def print_summary(self) -> None:
        """Print a summary of the experiment."""
        summary = self.get_metrics_summary()

        if not summary:
            print("No metrics logged yet.")
            return

        print(f"\n{'='*60}")
        print(f"Experiment: {self.experiment_name}")
        print(f"{'='*60}")
        print(f"Total generations logged: {summary['total_entries']}")
        print(f"Best accuracy: {summary['best_accuracy']:.4f}")
        print(f"Final accuracy: {summary['final_accuracy']:.4f}")
        print(f"Best generation: {summary['best_generation']}")
        print(f"Logs directory: {self.experiment_dir}")
        print(f"{'='*60}\n")


def setup_experiment_logger(
    logs_dir: str = "logs",
    experiment_name: Optional[str] = None,
) -> ACELogger:
    """
    Set up a new experiment logger.

    Args:
        logs_dir: Base logs directory
        experiment_name: Optional experiment name

    Returns:
        Configured ACELogger instance
    """
    return ACELogger(logs_dir=logs_dir, experiment_name=experiment_name)

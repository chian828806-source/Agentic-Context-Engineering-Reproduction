"""
Main Training Loop for ACE Framework

This script implements the complete training pipeline for ACE evolution.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from src.utils.env import load_env

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env if present
load_env()

from src.llm.glm_client import GLMClient
from src.state.graph_state import ACEState, initialize_state, copy_playbook
from src.graph.ace_graph import ACEGraph
from src.utils.data_loader import load_gsm8k, load_finer, create_sample_gsm8k, print_data_stats
from src.utils.logger import ACELogger
from src.utils.playbook import Playbook
from src.nodes.generator import GeneratorNode
from src.nodes.evaluator import EvaluatorNode


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    # LLM settings
    "model": "glm-4.6v",
    "base_url": "https://open.bigmodel.cn/api/paas/v4/",
    "api_key_env": "ZHIPUAI_API_KEY",

    # Training settings
    "max_generations": 1000,
    "total_epochs": 5,
    "plateau_threshold": 3,
    "eval_every_n_samples": 50,

    # Playbook settings
    "max_playbook_size": 10000,
    "max_bullets_per_section": 20,

    # Data settings
    "train_size": 1000,
    "val_size": 100,
    "task_type": "finer",

    # Logging
    "logs_dir": "logs",
    "save_checkpoints": True,
    "verbose": True,
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or use defaults.

    Args:
        config_path: Optional path to YAML config file

    Returns:
        Configuration dict
    """
    config = DEFAULT_CONFIG.copy()

    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
            config.update(user_config)

    # Get API key from environment
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        print(f"Warning: API key not found in {config['api_key_env']}")
        print("Set the ZHIPUAI_API_KEY environment variable")

    return config


# ============================================================================
# Training Loop
# ============================================================================

def train_ace(
    config: Dict[str, Any],
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
) -> ACEState:
    """
    Main ACE training loop.

    Args:
        config: Configuration dict
        train_data: Training samples
        val_data: Validation samples

    Returns:
        Final state after training
    """
    print("\n" + "="*60)
    print("ACE Training Started")
    print("="*60)

    # Initialize LLM client
    print(f"\nInitializing GLM-4.6 client...")
    llm_client = GLMClient(
        api_key=os.getenv(config["api_key_env"]),
        base_url=config["base_url"],
        model=config["model"],
    )

    # Test connection
    print("Testing API connection...")
    if not llm_client.check_api_connection():
        print("ERROR: API connection failed!")
        print("Please check your API key and network connection.")
        sys.exit(1)
    print("API connection successful!")

    # Initialize logger
    logger = ACELogger(
        logs_dir=config["logs_dir"],
        experiment_name=config.get("experiment_name"),
    )
    print(f"Logs directory: {logger.experiment_dir}")

    # Initialize state
    print("\nInitializing ACE state...")
    initial_state = initialize_state(config)
    initial_state["total_epochs"] = config["total_epochs"]
    initial_state["total_samples"] = len(train_data)

    # Build graph
    print("Building ACE evolution graph...")
    ace_graph = ACEGraph(llm_client, config)
    compiled_graph = ace_graph.build()
    ace_graph.set_validation_samples(val_data)

    # Initial evaluation
    print("\nRunning initial evaluation...")
    generator = GeneratorNode(llm_client, task_type=config["task_type"])
    evaluator = EvaluatorNode(llm_client, generator)

    initial_eval = evaluator.evaluate(
        initial_state,
        val_data[:config.get("validation_max_samples", 100)],
    )
    initial_accuracy = initial_eval["accuracy"]

    print(f"Initial accuracy: {initial_accuracy:.4f}")
    initial_state["fitness_score"] = initial_accuracy
    initial_state["best_score"] = initial_accuracy

    # Training loop
    print("\n" + "-"*60)
    print("Starting evolution loop...")
    print("-"*60)

    for epoch in range(config["total_epochs"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['total_epochs']}")
        print(f"{'='*60}")

        epoch_samples_processed = 0

        for sample_idx, sample in enumerate(train_data):
            # Update state with current sample
            initial_state["current_sample"] = sample
            initial_state["ground_truth"] = sample.get("answer")

            # Run one evolution step
            result = compiled_graph.invoke(initial_state)

            # Update state
            for key, value in result.items():
                initial_state[key] = value

            epoch_samples_processed += 1
            initial_state["samples_processed"] += 1

            # Periodic evaluation
            if (sample_idx + 1) % config["eval_every_n_samples"] == 0:
                current_accuracy = initial_state["fitness_score"]

                # Run evaluation
                eval_result = evaluator.evaluate(
                    initial_state,
                    val_data[:config.get("validation_max_samples", 100)],
                )

                accuracy = eval_result["accuracy"]
                initial_state["fitness_score"] = accuracy

                # Log metrics
                logger.log_metrics(
                    generation=initial_state["generation_index"],
                    epoch=epoch,
                    accuracy=accuracy,
                    playbook_size=sum(
                        len(v) for v in initial_state["current_playbook"].values()
                    ),
                    token_usage=llm_client.get_token_usage(),
                )

                # Print progress
                print(
                    f"  Sample {sample_idx + 1}/{len(train_data)} | "
                    f"Accuracy: {accuracy:.4f} | "
                    f"Best: {initial_state['best_score']:.4f} | "
                    f"No improvement: {initial_state['no_improvement_count']}"
                )

                # Save checkpoint if improved
                if accuracy > initial_state["best_score"]:
                    initial_state["best_score"] = accuracy
                    initial_state["best_playbook"] = copy_playbook(
                        initial_state["current_playbook"]
                    )
                    initial_state["no_improvement_count"] = 0

                    if config.get("save_checkpoints"):
                        logger.save_checkpoint(
                            generation=initial_state["generation_index"],
                            playbook=initial_state["current_playbook"],
                            accuracy=accuracy,
                        )
                        logger.save_playbook_evolution(
                            playbook=initial_state["current_playbook"],
                            generation=initial_state["generation_index"],
                            accuracy=accuracy,
                        )
                else:
                    initial_state["no_improvement_count"] += 1

                # Check convergence
                if initial_state["no_improvement_count"] >= config["plateau_threshold"]:
                    print(f"\nPlateau reached ({config['plateau_threshold']} evaluations without improvement)")
                    print("Stopping early...")
                    break

        # End of epoch
        print(f"\nEpoch {epoch + 1} completed. Processed {epoch_samples_processed} samples.")

        if initial_state["no_improvement_count"] >= config["plateau_threshold"]:
            break

    # Training complete
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final accuracy: {initial_state['fitness_score']:.4f}")
    print(f"Best accuracy: {initial_state['best_score']:.4f}")
    print(f"Total generations: {initial_state['generation_index']}")

    # Print token usage
    token_usage = llm_client.get_token_usage()
    print(f"\nToken usage:")
    print(f"  Prompt tokens: {token_usage['prompt_tokens']:,}")
    print(f"  Completion tokens: {token_usage['completion_tokens']:,}")
    print(f"  Total tokens: {token_usage['total_tokens']:,}")
    print(f"  Total API calls: {token_usage['total_calls']}")

    # Save final results
    logger.save_checkpoint(
        generation=initial_state["generation_index"],
        playbook=initial_state["best_playbook"],
        accuracy=initial_state["best_score"],
        state=initial_state,
    )

    # Export metrics
    logger.export_metrics_csv()
    logger.print_summary()

    return initial_state


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ACE Training")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/finer_train.jsonl",
        help="Path to training data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="data/finer_test.jsonl",
        help="Path to validation data",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "finer"],
        help="Task type",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=100,
        help="Number of training samples to use",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=50,
        help="Number of validation samples to use",
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample data file for testing",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (or set ZHIPUAI_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="glm-4.6v",
        choices=["glm-4.6v", "glm-4.5-air", "glm-4"],
        help="GLM model to use",
    )

    args = parser.parse_args()

    # Handle sample creation
    if args.create_sample:
        print("Creating sample GSM8K data file...")
        create_sample_gsm8k()
        print("Done! You can now run with --train-data data/gsm8k_sample.jsonl")
        return

    # Load config
    config = load_config(args.config)

    # Override with command line args
    if args.api_key:
        os.environ["ZHIPUAI_API_KEY"] = args.api_key
    config["train_size"] = args.train_size
    config["val_size"] = args.val_size
    config["task_type"] = args.task
    config["model"] = args.model

    # Load data
    print(f"\nLoading data...")
    print(f"  Train: {args.train_data}")
    print(f"  Val: {args.val_data}")

    if args.task == "gsm8k":
        train_data = load_gsm8k(args.train_data, max_samples=args.train_size)
        val_data = load_gsm8k(args.val_data, max_samples=args.val_size)
    elif args.task == "finer":
        train_data = load_finer(args.train_data, max_samples=args.train_size)
        val_data = load_finer(args.val_data, max_samples=args.val_size)
    else:
        raise ValueError(f"Unknown task type: {args.task}")

    print_data_stats(train_data, "Training Data")
    print_data_stats(val_data, "Validation Data")

    # Run training
    final_state = train_ace(config, train_data, val_data)

    print("\nACE training complete!")


if __name__ == "__main__":
    main()

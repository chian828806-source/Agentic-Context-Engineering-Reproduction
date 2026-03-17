"""
ACE Graph Construction using LangGraph

Builds the main evolution loop with Generator, Reflector, Curator, and Evaluator nodes.
"""

from typing import Literal, Dict, Any, List, Optional, Callable
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..state.graph_state import ACEState, initialize_state, copy_playbook
from ..nodes.generator import GeneratorNode
from ..nodes.reflector import ReflectorNode
from ..nodes.curator import CuratorNode
from ..nodes.evaluator import EvaluatorNode


class ACEGraph:
    """
    ACE Evolution Graph

    Implements the main three-node loop:
    Generator → Reflector → Curator → [repeat]

    With periodic evaluation to track progress.
    """

    def __init__(
        self,
        llm_client,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the ACE graph.

        Args:
            llm_client: GLM-4.6 client instance
            config: Configuration dict
        """
        config = config or {}

        self.config = config
        self.llm_client = llm_client

        # Initialize nodes
        self.generator = GeneratorNode(
            llm_client,
            temperature=config.get("generator_temperature", 0.7),
            task_type=config.get("task_type", "gsm8k"),
        )

        self.reflector = ReflectorNode(
            llm_client,
            temperature=config.get("reflector_temperature", 0.3),
            task_type=config.get("task_type", "gsm8k"),
        )

        self.curator = CuratorNode(
            llm_client,
            max_playbook_size=config.get("max_playbook_size", 10000),
            task_type=config.get("task_type", "gsm8k"),
        )

        self.evaluator = EvaluatorNode(llm_client, generator=self.generator)

        # Evaluation configuration
        self.eval_every_n_samples = config.get("eval_every_n_samples", 50)
        self.validation_samples: Optional[List[Dict[str, Any]]] = None

    def build(self) -> StateGraph:
        """
        Build the ACE evolution graph.

        Returns:
            Compiled StateGraph ready for execution
        """
        graph = StateGraph(ACEState)

        # Add nodes
        graph.add_node("generator", self._generator_node)
        graph.add_node("reflector", self._reflector_node)
        graph.add_node("curator", self._curator_node)
        graph.add_node("evaluator", self._evaluator_node)
        graph.add_node("check_convergence", self._check_convergence_node)
        graph.add_node("next_sample", self._next_sample_node)

        # Set entry point
        graph.set_entry_point("generator")

        # Add edges (main loop)
        graph.add_edge("generator", "reflector")
        graph.add_edge("reflector", "curator")
        graph.add_edge("curator", "evaluator")
        graph.add_edge("evaluator", "check_convergence")

        # Conditional edge: check convergence
        graph.add_conditional_edges(
            "check_convergence",
            self._should_continue_evolution,
            {
                "continue": "next_sample",
                "evaluate": "evaluator_full",
                "next_sample": "next_sample",
                "end": END,
            }
        )

        # Full evaluation node
        graph.add_node("evaluator_full", self._evaluator_full_node)
        graph.add_conditional_edges(
            "evaluator_full",
            self._after_evaluation,
            {
                "continue": "next_sample",
                "end": END,
            }
        )

        # Next sample leads back to generator
        graph.add_edge("next_sample", "generator")

        return graph.compile()

    def _generator_node(self, state: ACEState) -> Dict[str, Any]:
        """Execute Generator node."""
        result = self.generator(state)
        return result

    def _reflector_node(self, state: ACEState) -> Dict[str, Any]:
        """Execute Reflector node."""
        result = self.reflector(state)
        return result

    def _curator_node(self, state: ACEState) -> Dict[str, Any]:
        """Execute Curator node."""
        result = self.curator(state)

        # Update last modified timestamp
        state["current_playbook"] = result["current_playbook"]

        return result

    def _evaluator_node(self, state: ACEState) -> Dict[str, Any]:
        """Quick evaluation for single sample."""
        generated = state.get("generated_answer")
        ground_truth = state.get("ground_truth")

        if generated is None or ground_truth is None:
            return {"error_samples": []}

        is_correct = self.evaluator.compare_finer_answers(generated, ground_truth)

        if not is_correct:
            error_sample = {
                "question": state.get("current_sample", {}).get("question", ""),
                "ground_truth": ground_truth,
                "generated_answer": generated,
                "trace": state.get("generator_trace", ""),
                "is_correct": False,
            }
            return {"error_samples": [error_sample]}

        return {"error_samples": []}

    def _evaluator_full_node(self, state: ACEState) -> Dict[str, Any]:
        """Full evaluation on validation set."""
        if self.validation_samples is None:
            return {"fitness_score": state.get("fitness_score", 0.0)}

        eval_result = self.evaluator.evaluate(
            state,
            self.validation_samples,
            max_samples=self.config.get("validation_max_samples", 100),
        )

        return {
            "fitness_score": eval_result["accuracy"],
            "error_samples": eval_result.get("error_samples", [])[:10],
            "validation_results": eval_result.get("all_results", []),
            "last_validation_at": state["generation_index"],
        }

    def _check_convergence_node(self, state: ACEState) -> Dict[str, Any]:
        """Check if we should continue evolution."""
        current_score = state.get("fitness_score", 0.0)
        best_score = state.get("best_score", 0.0)
        no_improvement = state.get("no_improvement_count", 0)

        # Update best if current is better
        if current_score > best_score:
            state["best_score"] = current_score
            state["best_playbook"] = copy_playbook(state["current_playbook"])
            state["no_improvement_count"] = 0
        else:
            state["no_improvement_count"] = no_improvement + 1

        return state

    def _should_continue_evolution(self, state: ACEState) -> str:
        """Decide whether to continue evolution."""
        # Check max generations
        if state["generation_index"] >= state["max_generations"]:
            return "end"

        # Check plateau threshold
        if state["no_improvement_count"] >= state["plateau_threshold"]:
            return "end"

        # Check if we should do full evaluation
        samples_since_last_eval = (
            state["generation_index"] - state.get("last_validation_at", 0)
        )
        if samples_since_last_eval >= self.eval_every_n_samples:
            return "evaluate"

        return "continue"

    def _after_evaluation(self, state: ACEState) -> str:
        """Decide after full evaluation."""
        if state["no_improvement_count"] >= state["plateau_threshold"]:
            return "end"
        return "continue"

    def _next_sample_node(self, state: ACEState) -> Dict[str, Any]:
        """Move to next sample."""
        state["generation_index"] += 1
        state["samples_processed"] += 1
        return state

    def set_validation_samples(self, samples: List[Dict[str, Any]]):
        """Set validation samples for evaluation."""
        self.validation_samples = samples

    def run_on_dataset(
        self,
        train_samples: List[Dict[str, Any]],
        validation_samples: List[Dict[str, Any]],
        initial_state: Optional[ACEState] = None,
        callback: Optional[Callable[[ACEState], None]] = None,
    ) -> ACEState:
        """
        Run ACE evolution on a dataset.

        Args:
            train_samples: Training samples
            validation_samples: Validation samples for evaluation
            initial_state: Optional initial state
            callback: Optional callback called after each sample

        Returns:
            Final state after evolution
        """
        # Set validation samples
        self.set_validation_samples(validation_samples)

        # Initialize state
        state = initial_state or initialize_state(self.config)
        state["total_samples"] = len(train_samples)

        # Build graph
        graph = self.build()

        # Run on each sample
        for epoch in range(state["total_epochs"]):
            state["current_epoch"] = epoch

            for sample in train_samples:
                state["current_sample"] = sample
                state["ground_truth"] = sample.get("answer")

                # Run the graph
                result = graph.invoke(state)

                # Update state with result
                for key, value in result.items():
                    state[key] = value

                # Call callback if provided
                if callback:
                    callback(state)

                # Check if we should stop
                if state["no_improvement_count"] >= state["plateau_threshold"]:
                    break

            # End of epoch
            if state["no_improvement_count"] >= state["plateau_threshold"]:
                break

        return state


def build_ace_graph(
    llm_client,
    config: Optional[Dict[str, Any]] = None,
) -> ACEGraph:
    """
    Factory function to build an ACE graph.

    Args:
        llm_client: GLM-4.6 client
        config: Optional configuration

    Returns:
        Configured ACEGraph instance
    """
    return ACEGraph(llm_client, config)


def run_single_sample(
    state: ACEState,
    llm_client,
    config: Optional[Dict[str, Any]] = None,
) -> ACEState:
    """
    Run ACE on a single sample (one full evolution step).

    Args:
        state: Initial state with current_sample set
        llm_client: GLM-4.6 client
        config: Optional configuration

    Returns:
        Updated state after one evolution step
    """
    graph = build_ace_graph(llm_client, config).build()
    return graph.invoke(state)


def run_batch_evolution(
    samples: List[Dict[str, Any]],
    llm_client,
    config: Optional[Dict[str, Any]] = None,
) -> ACEState:
    """
    Run ACE evolution on a batch of samples.

    Args:
        samples: List of samples to process
        llm_client: GLM-4.6 client
        config: Configuration

    Returns:
        Final state after processing all samples
    """
    config = config or {}
    initial_state = initialize_state(config)
    initial_state["total_samples"] = len(samples)

    graph = build_ace_graph(llm_client, config)
    ace_graph = graph.build()

    for sample in samples:
        initial_state["current_sample"] = sample
        initial_state["ground_truth"] = sample.get("answer")

        result = ace_graph.invoke(initial_state)

        for key, value in result.items():
            initial_state[key] = value

        initial_state["generation_index"] += 1
        initial_state["samples_processed"] += 1

        # Check convergence
        if initial_state["no_improvement_count"] >= initial_state["plateau_threshold"]:
            break

    return initial_state

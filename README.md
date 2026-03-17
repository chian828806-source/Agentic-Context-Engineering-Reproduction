# ACE (Agentic Context Engineering) Reproduction

A reproduction of the ACE (Agentic Context Engineering) paper from ICLR 2026. This framework implements an autonomous system that evolves its own "System Prompt" based on feedback from test benchmarks.

## Paper Summary

ACE treats contexts as evolving playbooks that accumulate, refine, and organize strategies through a modular process of:

1. **Generator**: Uses the current playbook to solve problems
2. **Reflector**: Analyzes errors and generates insights
3. **Curator**: Updates the playbook with new strategies

This approach addresses two key limitations of prior methods:
- **Brevity Bias**: Traditional methods compress context, losing domain insights
- **Context Collapse**: Iterative rewriting erodes details over time

## Project Structure

```
ACE/
├── data/                   # Dataset files (JSONL)
├── src/
│   ├── nodes/             # Generator, Reflector, Curator, Evaluator
│   ├── state/             # State management (TypedDict)
│   ├── prompts/           # Prompt templates
│   ├── llm/               # GLM-4.6 client
│   ├── graph/             # LangGraph construction
│   └── utils/             # Data loading, logging, playbook
├── baselines/             # Baseline methods (RAG, Few-shot)
├── logs/                  # Experimental results
├── configs/               # Configuration files
├── main.py                # Main training loop
└── requirements.txt       # Dependencies
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Setup

1. **Set your GLM API key:**
```bash
export ZHIPUAI_API_KEY="your_api_key_here"
```

Or create a `.env` file:
```
ZHIPUAI_API_KEY=your_api_key_here
```

2. **Prepare FINER dataset files:**

This project expects FINER JSONL files with fields `id`, `tokens`, `ner_tags`:
```json
{"id": 0, "tokens": ["..."], "ner_tags": ["O", "B-..."]}
```

Use the provided subset files for reproduction:
- `data/train_sub.jsonl`
- `data/validation_sub.jsonl`
- `data/test_sub.jsonl`

## Usage

### Basic Training (FINER)

```bash
python main.py \
    --train-data data/train_sub.jsonl \
    --val-data data/validation_sub.jsonl \
    --train-size 100 \
    --val-size 50
```

### With Configuration File

```bash
python main.py --config configs/ace_config.yaml
```

### Options

| Option | Description | Default |
|:---|:---|:---|
| `--train-data` | Path to training data | data/finer_train.jsonl |
| `--val-data` | Path to validation data | data/finer_test.jsonl |
| `--train-size` | Number of training samples | 100 |
| `--val-size` | Number of validation samples | 50 |
| `--config` | Path to config YAML | None |
| `--api-key` | API key (overrides env) | None |
| `--api-key-env` | Env var name for API key | ZHIPUAI_API_KEY |
| `--model` | Model name | glm-4.6v |
| `--base-url` | API base URL override | (default in code) |

## Results

Results are saved to `logs/{experiment_name}/`:

- `metrics.jsonl`: Per-evaluation metrics
- `checkpoints/`: Saved checkpoints
- `playbook_v*.json`: Evolved playbook at each generation
- `metrics.csv`: Exported metrics for analysis

## Core Components

### State Management

The ACE state tracks:
- `current_playbook`: Evolving context with categorized strategies
- `generation_index`: Current evolution generation
- `fitness_score`: Current validation accuracy
- `error_samples`: Top error cases for refinement
- `best_playbook`: Best playbook seen so far

### Playbook Structure

The playbook is organized into sections:
- `strategies_and_hard_rules`: High-level approaches
- `formulas_and_calculations`: Mathematical formulas
- `verification_checklist`: Steps to verify correctness
- `common_mistakes`: Known pitfalls to avoid
- `apis_to_use_for_specific_information`: Domain-specific guidance

### Evolution Loop

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Generator  │─────▶│  Reflector  │─────▶│   Curator   │
│  (Solve)    │      │  (Analyze)  │      │  (Update)   │
└─────────────┘      └─────────────┘      └─────────────┘
       ▲                                            │
       └────────────────────────────────────────────┘
                     (Improved Playbook)
```

## Implementation Notes

### Key Design Decisions

1. **Stateless Optimization**: Reflector/Curator only receive current prompt + top 3-5 error cases
2. **Incremental Updates**: Curator adds new content, doesn't regenerate entire playbook
3. **Token Efficiency**: Automatic compression when playbook exceeds size limits
4. **Rollback Mechanism**: Reverts to previous best prompt if evolution fails

### Differences from Paper

- Uses GLM instead of DeepSeek-V3.1
- Simplified evaluation for FINER NER sequence tagging
- Single-process implementation (no parallel rollout)

## Baselines

Baseline methods are implemented in `baselines/`:

- `rag_baseline.py`: Retrieval-Augmented Generation
- `fewshot_baseline.py`: Few-shot prompting

Run baselines on FINER subsets (adjust scripts to FINER if needed):
```bash
python baselines/rag_baseline.py --test-data data/test_sub.jsonl
```

## Citation

If you use this code, please cite the original ACE paper:

```bibtex
@inproceedings{zhang2025ace,
  title={Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models},
  author={Zhang, Qizheng and Hu, Changran and Upasani, Shubhangi and Ma, Boyuan and Hong, Fenglu and Kamanuru, Vamsidhar and Rainton, Jay and Wu, Chen and Ji, Mengmeng and Li, Hanchen and Thakker, Urmish and Zou, James and Olukotun, Kunle},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## License

This is a research reproduction project. Please refer to the original paper's license for the ACE framework.

## Acknowledgments

- Original ACE paper authors
- GLM-4.6 team for the API access
- LangGraph team for the framework

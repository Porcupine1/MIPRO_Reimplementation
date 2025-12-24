# MIPRO Implementation

Implementation of MIPRO (Multi-stage Instruction and Prompt Optimization) for optimizing a QA pipeline.

## Overview

MIPRO optimizes instructions and demonstrations for multi-stage language model programs using:
- **Instruction Proposal**: Generates candidate instructions via meta-prompting
- **Bayesian Optimization**: Uses TPE to search instruction space efficiently
- **Mini-batch Evaluation**: Evaluates configurations on small batches for speed

## Setup

**Requirements:**
- Python **3.9+** (this repo uses modern typing like `list[str]` and `tuple[int, ...]`)
- **Ollama** running locally (default model: `llama3.2:3b-instruct-q4_0`)

**Installation:**
```bash
# Install dependencies with uv (recommended)
uv lock
uv sync

# Run commands inside the uv environment
# (uv will use your system Python, but still isolates deps)
uv run python3 cmd/download_dataset.py

# Dataset
# This repo expects the dataset at: data/hotpotqa/
# If it's missing on your machine, download via Hugging Face Datasets:
# (already done above)
#
# Pip fallback (no uv):
# python3 -m pip install datasets optuna requests pyyaml
# python3 cmd/download_dataset.py
```

**Ollama (local LLM)**

```bash
# Install/start Ollama separately, then pull the default model:
ollama pull llama3.2:3b-instruct-q4_0
```

## Quick Start

```bash
# Run MIPRO optimization
uv run python3 main.py --tier light
```

This will:
1. Load HotpotQA dataset
2. Initialize 2-module QA program (Query -> Retrieve -> Answer)
3. Generate instruction candidates for each module
4. Run Bayesian optimization to find best instructions
5. Save results to `outputs/mipro_results.json`

## CLI Cheatsheet

```bash
# Show available tiers
uv run python3 main.py --list-tiers

# Check whether cache files exist
uv run python3 main.py --check-cache
```

## Faster Optimization with Caching

To speed up optimization by reusing pre-generated candidates:

### Quick Start (One Command)
```bash
# Run all three steps automatically
./run_with_cache.sh
```

### Manual Steps
```bash
# 1. Generate demo candidates cache
uv run python -m cmd.test_bootstrapper

# 2. Generate instruction candidates (using cached demos)
uv run python -m cmd.test_instr_proposer --use-cached-demos

# 3. Run optimization with both caches (30-40% faster!)
uv run python main.py --tier light --use-cache
```

**Benefits:**
- **30-40% faster** optimization runs
- **Reproducible** experiments with the same candidates
- **Focus on optimization** - skip candidate generation
- **Reusable** - generate once, use many times

**Documentation:**
- [QUICK_START_CACHE.md](QUICK_START_CACHE.md) - Step-by-step workflow guide
- [CACHING_GUIDE.md](CACHING_GUIDE.md) - Complete caching documentation

## Configuration Tiers

Choose a tier based on your needs:

```bash
# Fast testing (5-10 min)
python main.py --tier light

# Balanced development (20-40 min)
python main.py --tier medium

# Full-scale production (1-2 hrs)
python main.py --tier heavy

# With caching for even faster runs
python main.py --tier light --use-cache
```

Run `python main.py --list-tiers` to see all tier configurations.

## Configuration

Edit `config.py` to customize:
- `MODEL_NAME` - LLM model to use
- `N_TRIALS` - Number of optimization trials
- `BATCH_SIZE` - Mini-batch size for evaluation
- `N_INSTRUCTION_CANDIDATES` - Candidates per module
- `METRIC` - Optimization metric ('f1' or 'exact_match')

## Retrieval Backends (Online vs Offline)

Retrieval is controlled by `RETRIEVER` in `config.py`:

- **`wiki_online`**: live Wikipedia API retrieval (requires internet access)
- **`hotpot_local`**: uses HotpotQA-provided context (offline-friendly)
- **`mock`**: simple baseline that flattens example context (offline-friendly)

### Components

**Project Structure:**
- **`optimizers/`** - Optimization logic
  - `MIPROOpt.py` - Main MIPRO optimizer orchestrator
  - `SurrogateOpt.py` - Bayesian optimization using TPE (Optuna)
- **`programs/`** - What is being optimized
  - `QAProgram.py` - Multi-stage QA pipeline
  - `PromptMod.py` - Modular prompt templates (QueryModule, AnswerModule)
- **`backend/`** - Backend services
  - `LMBackend.py` - Ollama interface for LLM generation
- **`helpers/`** - Helper utilities
  - `InstrProposer.py` - Generates instruction candidates via meta-prompting
  - `DemoBootstrapper.py` - Bootstrap demonstrations from traces
  - `GroundingHelper.py` - Dataset and program grounding utilities
- **`cache/`** - Candidate caching system
  - `candidate_cache.py` - Save/load demo and instruction candidates
- **`cmd/`** - Command-line utilities
  - `test_bootstrapper.py` - Generate and cache demo candidates
  - `test_instr_proposer.py` - Generate and cache instruction candidates
- **`QADataset.py`** - HotpotQA dataset handler
- **`metrics.py`** - Evaluation metrics (F1, Exact Match)
- **`config.py`** - Central configuration

## Usage Example

```python
from backend import LMBackend
from QADataset import QADataset
from programs import QAProgram
from optimizers import MIPROOptimizer

# Load dataset
dataset = QADataset().load()

# Initialize program
program = QAProgram(backend=LMBackend())

# Run optimization
optimizer = MIPROOptimizer(program, dataset)
optimized_program = optimizer.optimize(
    task_description="Answer multi-hop questions"
)

# Get best instructions
best_instructions = optimizer.get_best_instructions()
```


MIPRO will automatically optimize instructions and bootstrap demos for all modules.


## References

- MIPRO Paper: [Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs](https://arxiv.org/abs/2406.11695)
- Dataset: [HotpotQA](https://hotpotqa.github.io/)
- Optimization: [Optuna TPE](https://optuna.org/)

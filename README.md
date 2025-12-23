# MIPRO Implementation

~Implementation of MIPRO (Multi-stage Instruction and Prompt Optimization) for optimizing a QA pipeline.

## Overview

MIPRO optimizes instructions and demonstrations for multi-stage language model programs using:
- **Instruction Proposal**: Generates candidate instructions via meta-prompting
- **Bayesian Optimization**: Uses TPE to search instruction space efficiently
- **Mini-batch Evaluation**: Evaluates configurations on small batches for speed

## Setup

**Requirements:**
- Python 3.8+
- Ollama running locally with llama3.2:3b-instruct-q4_0 model

**Installation:**
```bash
# Install dependencies
pip install datasets optuna

# Or use uv
uv pip install datasets optuna

# Download dataset
python cmd/download_dataset.py
```

## Quick Start

```bash
# Run MIPRO optimization
python main.py
```

This will:
1. Load HotpotQA dataset
2. Initialize 2-module QA program (Query -> Retrieve -> Answer)
3. Generate instruction candidates for each module
4. Run Bayesian optimization to find best instructions
5. Save results to `outputs/mipro_results.json`

## Configuration

Edit `config.py` to customize:
- `MODEL_NAME` - LLM model to use
- `N_TRIALS` - Number of optimization trials
- `BATCH_SIZE` - Mini-batch size for evaluation
- `N_INSTRUCTION_CANDIDATES` - Candidates per module
- `METRIC` - Optimization metric ('f1' or 'exact_match')

## Architecture

![Architecture Flow Diagram](arch_flow.png)

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

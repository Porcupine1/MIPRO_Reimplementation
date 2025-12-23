#!/usr/bin/env python3
"""
Main script to run MIPRO optimization on QA program.
"""

import logging
import os

from backend import LMBackend
from QADataset import QADataset
from programs import QAProgram
from optimizers import MIPROOptimizer
from config import MODEL_NAME, OUTPUT_DIR
from logging_config import setup_logging


logger = logging.getLogger(__name__)


def main():
    # Ensure output directory exists and configure logging to both console and file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "run.log")
    setup_logging(
        level=logging.INFO,
        log_path=log_path,
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Logging to %s", log_path)

    logger.info("Initializing MIPRO Optimization...")
    logger.info("Model: %s", MODEL_NAME)

    # 1. Load dataset
    logger.info("[1/4] Loading dataset...")
    dataset = QADataset()
    dataset.load()
    logger.info("  Train: %d examples", len(dataset.train))
    logger.info("  Validation: %d examples", len(dataset.validation))

    # 2. Initialize program
    logger.info("[2/4] Initializing QA program...")
    backend = LMBackend()
    program = QAProgram(backend=backend)
    logger.info("  Modules: %s", program.get_module_names())

    # 3. Test baseline (optional)
    logger.info("[3/4] Testing baseline program...")
    test_batch = dataset.sample_batch(1, split="validation")
    if not test_batch:
        logger.warning("  No validation examples available for baseline test")
    else:
        test_example = test_batch[0]
        try:
            baseline_answer = program.forward(test_example)
            logger.info("  Question: %s...", test_example["question"])
            logger.info("  Baseline Answer: %s...", baseline_answer)
            logger.info("  Ground Truth: %s...", test_example["answer"])
        except Exception as e:
            logger.error("  Error in baseline: %s", e)

    # 4. Run MIPRO optimization
    logger.info("[4/4] Running MIPRO optimization...")
    optimizer = MIPROOptimizer(
        program=program, dataset=dataset, n_trials=10, batch_size=20, eval_batch_size=50
    )

    optimized_program = optimizer.optimize(
        task_description="Answer multi-hop questions using retrieved context from Wikipedia"
    )

    # 5. Test optimized program
    logger.info("[5/5] Testing optimized program...")
    if test_batch:
        try:
            optimized_answer = optimized_program.forward(test_example)
            logger.info("  Optimized Answer: %s...", optimized_answer)
        except Exception as e:
            logger.error("  Error in optimized: %s", e)

    # 6. Show best instructions
    logger.info("=" * 60)
    logger.info("Best Instructions Found:")
    logger.info("=" * 60)
    best_instructions = optimizer.get_best_instructions()
    for module_name, instruction in best_instructions.items():
        block = (
            f"\n====== Best instruction for module '{module_name}' ======\n"
            f"{instruction.strip()}\n"
        )
        logger.instr(block)

    logger.info("Done!")


if __name__ == "__main__":
    main()

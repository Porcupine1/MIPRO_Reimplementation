#!/usr/bin/env python3
"""
Main script to run MIPRO optimization on QA program.
"""

import argparse
import logging
import os

from backend import LMBackend
from QADataset import QADataset
from programs import QAProgram
from optimizers import MIPROOptimizer
from config import MODEL_NAME, OUTPUT_DIR, get_tier_config, apply_tier, print_tier_info
from logging_config import setup_logging


logger = logging.getLogger(__name__)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run MIPRO optimization on QA program",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration Tiers:
  light   - Fast testing (5-10 min): 5 trials, small batches
  medium  - Balanced (20-40 min): 15 trials, moderate batches  
  heavy   - Full-scale (1-2 hrs): 30 trials, large batches

Examples:
  python main.py --tier light     # Quick test
  python main.py --tier medium    # Development
  python main.py --tier heavy     # Production run
  python main.py --list-tiers     # Show all tiers
        """,
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="light",
        choices=["light", "medium", "heavy"],
        help="Configuration tier to use (default: light)",
    )
    parser.add_argument(
        "--list-tiers",
        action="store_true",
        help="List all available configuration tiers and exit",
    )

    args = parser.parse_args()

    # Handle --list-tiers
    if args.list_tiers:
        print_tier_info()
        return

    # Apply the selected tier
    tier_config = apply_tier(args.tier)

    # Ensure output directory exists and configure logging to both console and file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "run.log")
    setup_logging(
        level=logging.INFO,
        log_path=log_path,
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Logging to %s", log_path)

    logger.info(
        "\n" + "=" * 80 + "\n"
        f"Starting MIPRO Optimization - {args.tier.upper()} Tier\n" + "=" * 80 + "\n"
        f"  Model: {MODEL_NAME}\n"
        f"  Trials: {tier_config.n_trials}\n"
        f"  Batch Size: {tier_config.batch_size} (eval: {tier_config.eval_batch_size})\n"
        f"  Max Examples: {tier_config.max_examples}\n"
        f"  Estimated Time: {tier_config.estimated_time}\n" + "=" * 80
    )

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
            logger.info("  Question: %s", test_example["question"])
            logger.info("  Baseline Answer: %s", baseline_answer)
            logger.info("  Ground Truth: %s", test_example["answer"])
        except Exception as e:
            logger.error("  Error in baseline: %s", e)

    # 4. Run MIPRO optimization
    logger.info("[4/4] Running MIPRO optimization...")
    optimizer = MIPROOptimizer(
        program=program,
        dataset=dataset,
        n_trials=tier_config.n_trials,
        batch_size=tier_config.batch_size,
        eval_batch_size=tier_config.eval_batch_size,
    )

    optimized_program = optimizer.optimize(
        task_description="Answer multi-hop questions using retrieved context from Wikipedia"
    )

    # 5. Test optimized program
    logger.info("[5/5] Testing optimized program...")
    if test_batch:
        try:
            optimized_answer = optimized_program.forward(test_example)
            logger.info("  Optimized Answer: %s", optimized_answer)
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

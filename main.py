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
from config import MODEL_NAME, OUTPUT_DIR, TASK_DESCRIPTION, get_tier_config, apply_tier, print_tier_info
from logging_config import setup_logging
from cache.candidate_cache import cache_exists


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

Caching Options:
  --use-cache                # Use cached demos AND instructions (fastest)
                             # If cache not found, will auto-generate and save
  --use-cached-demos         # Use only cached demo candidates
  --use-cached-instructions  # Use only cached instruction candidates
  --check-cache              # Check if cache files exist
  
  Note: If cache is requested but not found, MIPROOptimizer will automatically
        generate and save the missing cache files during optimization.
  
  To pre-generate cache files manually (optional):
    python -m cmd.test_bootstrapper    # Generate demo cache
    python -m cmd.test_instr_proposer  # Generate instruction cache

Examples:
  python main.py --tier light                    # Quick test (generate all)
  python main.py --tier medium --use-cache       # Use cached candidates
  python main.py --tier heavy --use-cached-demos # Use only cached demos
  python main.py --check-cache                   # Check cache status
  python main.py --list-tiers                    # Show all tiers
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
    parser.add_argument(
        "--use-cached-demos",
        action="store_true",
        help="Load demo candidates from cache instead of bootstrapping (faster)",
    )
    parser.add_argument(
        "--use-cached-instructions",
        action="store_true",
        help="Load instruction candidates from cache instead of generating (faster)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use both cached demos and instructions (equivalent to --use-cached-demos --use-cached-instructions)",
    )
    parser.add_argument(
        "--check-cache",
        action="store_true",
        help="Check if cache files exist and exit",
    )

    args = parser.parse_args()

    # Handle --list-tiers
    if args.list_tiers:
        print_tier_info()
        return

    # Handle --check-cache
    if args.check_cache:
        cache_status = cache_exists()
        print("Cache Status:")
        print(f"  Demo candidates: {' EXISTS' if cache_status['demos'] else '✗ NOT FOUND'}")
        print(f"  Instruction candidates: {' EXISTS' if cache_status['instructions'] else '✗ NOT FOUND'}")
        if cache_status['demos'] or cache_status['instructions']:
            print("\nTo use cache, run with: --use-cache")
            print("To generate cache, run:")
            if not cache_status['demos']:
                print("  python -m cmd.test_bootstrapper")
            if not cache_status['instructions']:
                print("  python -m cmd.test_instr_proposer")
        return

    # Apply the selected tier
    tier_config = apply_tier(args.tier)

    # Handle cache flags - MIPROOptimizer will handle loading/generation
    use_cached_demos = args.use_cache or args.use_cached_demos
    use_cached_instructions = args.use_cache or args.use_cached_instructions

    # Ensure output directory exists and configure logging to both console and file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "run.log")
    setup_logging(
        level=logging.DEBUG,  # Changed to DEBUG to see retrieval details
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
        f"  Estimated Time: {tier_config.estimated_time}\n"
        f"  Use Cached Demos: {use_cached_demos}\n"
        f"  Use Cached Instructions: {use_cached_instructions}\n" + "=" * 80
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
    # Note: MIPROOptimizer reads from active tier config automatically
    optimizer = MIPROOptimizer(
        program=program,
        dataset=dataset,
        use_cached_demos=use_cached_demos,
        use_cached_instructions=use_cached_instructions,
    )

    optimized_program = optimizer.optimize(task_description=TASK_DESCRIPTION)

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

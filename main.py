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
from config import MODEL_NAME, OUTPUT_DIR, TASK_DESCRIPTION, get_tier_config, apply_tier, print_tier_info, get_active_config
from logging_config import setup_logging
from cache.candidate_cache import (
    cache_exists,
    save_demo_candidates,
    save_instruction_candidates,
)
from helpers import DemoBootstrapper, InstructionProposer


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
                             # If cache not found, will auto-generate
  --use-cached-demos         # Use only cached demo candidates
  --use-cached-instructions  # Use only cached instruction candidates
  --check-cache              # Check if cache files exist
  
  Note: If cache is requested but not found, main.py will automatically
        generate and save the missing cache files before optimization.
  
  To pre-generate cache files manually:
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

    # Handle cache flags
    use_cached_demos = args.use_cache or args.use_cached_demos
    use_cached_instructions = args.use_cache or args.use_cached_instructions

    # Check if cache is requested but doesn't exist - generate if needed
    need_to_generate_demos = False
    need_to_generate_instructions = False
    
    if use_cached_demos or use_cached_instructions:
        cache_status = cache_exists()
        if use_cached_demos and not cache_status['demos']:
            logger.warning(
                "Demo cache requested but not found. Will generate demo cache..."
            )
            need_to_generate_demos = True
        if use_cached_instructions and not cache_status['instructions']:
            logger.warning(
                "Instruction cache requested but not found. Will generate instruction cache..."
            )
            need_to_generate_instructions = True

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

    # 2.5. Generate cache if needed (before optimization)
    if need_to_generate_demos or need_to_generate_instructions:
        logger.info("[2.5/4] Generating requested cache files...")
        
        # Get training data for cache generation
        train_data = dataset.get_split(split="train")
        if not train_data:
            logger.error("Cannot generate cache: No training data available")
            return
        
        # Get module names for consistent indexing
        module_names = sorted(program.get_module_names())
        
        # Generate demo cache if needed
        demo_candidates = None
        if need_to_generate_demos:
            logger.info("  Generating demo candidates cache...")
            cfg = get_active_config()
            bootstrapper = DemoBootstrapper(program=program, dataset=dataset)
            demo_candidates = bootstrapper.bootstrap_candidates(
                num_candidates=cfg.num_candidates,
                train_data=train_data,
                module_names=module_names,
            )
            # Save to cache
            metadata = {
                "tier": args.tier,
                "num_candidates": cfg.num_candidates,
                "max_bootstrapped_demos": cfg.max_bootstrapped_demos,
                "max_labeled_demos": cfg.max_labeled_demos,
            }
            save_demo_candidates(demo_candidates, metadata=metadata)
            logger.info("  ✓ Demo candidates cache generated and saved!")
        
        # Generate instruction cache if needed
        if need_to_generate_instructions:
            logger.info("  Generating instruction candidates cache...")
            cfg = get_active_config()
            
            # Load demo candidates if available (for instruction generation)
            if demo_candidates is None and cache_status.get('demos'):
                from cache.candidate_cache import load_demo_candidates
                demo_candidates = load_demo_candidates()
            
            # If still no demos, generate empty demo sets
            if demo_candidates is None:
                demo_candidates = {
                    idx: [[]]  # Empty demo set for each predictor
                    for idx in range(len(module_names))
                }
            
            # Initialize InstructionProposer (handles grounding internally)
            proposer = InstructionProposer(train_examples=train_data, program=program)
            
            # Generate instruction candidates
            instruction_candidates = proposer.propose_for_all_modules(
                program=program,
                task_desc=TASK_DESCRIPTION,
                bootstrapped_demos=demo_candidates,
                n_candidates=cfg.n_instruction_candidates,
                program_aware=True,
                module_names=module_names,
            )
            
            # Save to cache
            metadata = {
                "tier": args.tier,
                "n_instruction_candidates": cfg.n_instruction_candidates,
            }
            save_instruction_candidates(instruction_candidates, metadata=metadata)
            logger.info("  Instruction candidates cache generated and saved!")

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

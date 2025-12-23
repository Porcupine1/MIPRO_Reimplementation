#!/usr/bin/env python3
"""
Quick manual test harness for `DemoBootstrapper`.

Recommended usage from project root:
    python -m cmd.test_bootstrapper

Also works when run directly as a script.
"""

import logging
import os
import sys
import pprint

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from QADataset import QADataset
from programs import QAProgram
from helpers import DemoBootstrapper
from logging_config import setup_logging
from config import apply_tier


logger = logging.getLogger(__name__)


def main():
    # Apply LIGHT tier for fast testing
    apply_tier("light")

    setup_logging(
        level=logging.INFO,
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=== DemoBootstrapper standalone test (LIGHT tier) ===")

    # 1. Load dataset
    logger.info("Loading dataset...")
    dataset = QADataset()
    dataset.load()
    logger.info(
        "Train examples: %d | Validation examples: %d",
        len(dataset.train),
        len(dataset.validation),
    )

    # 2. Initialize QA program (uses default LMBackend)
    logger.info("Initializing QAProgram...")
    program = QAProgram()
    logger.info("Modules: %s", program.get_module_names())

    # 3. Run bootstrapper with small settings so it finishes quickly
    logger.info("Bootstrapping demo candidates...")
    bootstrapper = DemoBootstrapper(program=program, dataset=dataset)

    demo_candidates = bootstrapper.bootstrap_candidates(
        num_candidates=2,  # Creates 2 bootstrapped sets (plus 2 baseline sets = 4 total)
        max_bootstrapped_demos=3,
        max_labeled_demos=1,
        train_data=dataset.train,
    )

    # 4. Print a brief summary of what was collected
    logger.info(
        "Bootstrapper returned demo candidates for %d modules", len(demo_candidates)
    )
    for module_name, candidate_sets in demo_candidates.items():
        logger.info("Module '%s': %d candidate sets", module_name, len(candidate_sets))
        # Log one sample demo block per module with instruction-level color
        for idx, demo_set in enumerate(candidate_sets):
            if not demo_set:
                continue
            sample = demo_set[0]
            header = (
                f"\n====== Demo sample for module '{module_name}' "
                f"(candidate set {idx}, {len(demo_set)} demos) ======"
            )
            body = pprint.pformat(sample, indent=2)
            logger.instr(f"{header}\n{body}\n")
            break  # only show one sample per module to keep logs short

    logger.info("=== DemoBootstrapper test completed ===")

    # Log the entire demo_candidates array at the end with custom formatting
    logger.instr("\n\n======= FULL demo_candidates ========")
    for module_name, candidate_sets in demo_candidates.items():
        logger.instr(f"\n[Module: {module_name}]")
        for idx, demo_set in enumerate(candidate_sets):
            logger.instr(f"  Candidate Set {idx}: {len(demo_set)} demos")
            for demo_idx, demo in enumerate(demo_set):
                logger.instr(f"    Demo {demo_idx}:")
                for key, value in demo.items():
                    # Truncate long values for readability
                    if isinstance(value, str) and len(value) > 100:
                        value_str = value[:100] + "..."
                    else:
                        value_str = str(value)
                    logger.instr(f"      {key}: {value_str}")
    logger.instr("")


if __name__ == "__main__":
    main()

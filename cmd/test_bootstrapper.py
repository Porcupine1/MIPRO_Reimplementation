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


logger = logging.getLogger(__name__)


def main():
    setup_logging(
        level=logging.INFO,
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=== DemoBootstrapper standalone test ===")

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
        num_candidates=3,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
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

    # Log the entire demo_candidates array at the end
    logger.instr(
        "\n======= FULL demo_candidates ========\n%s\n", pprint.pformat(demo_candidates)
    )


if __name__ == "__main__":
    main()

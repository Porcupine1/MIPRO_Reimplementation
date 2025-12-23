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


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
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
        split="train",
    )

    # 4. Print a brief summary of what was collected
    logger.info(
        "Bootstrapper returned demo candidates for %d modules", len(demo_candidates)
    )
    for module_name, candidate_sets in demo_candidates.items():
        logger.info("Module '%s': %d candidate sets", module_name, len(candidate_sets))
        for idx, demo_set in enumerate(candidate_sets):
            logger.info(
                "  Set %d: %d demos (showing at most first 1)", idx, len(demo_set)
            )
            if demo_set:
                sample = demo_set[0]
                logger.info("    Sample demo keys: %s", list(sample.keys()))
                logger.info("    Sample demo score: %s", sample.get("score"))
                break  # only show one sample per module to keep logs short

    logger.info("=== DemoBootstrapper test completed ===")

    # Log the entire demo_candidates array at the end
    logger.info(
        "\n======= FULL demo_candidates ========\n%s", pprint.pformat(demo_candidates)
    )


if __name__ == "__main__":
    main()

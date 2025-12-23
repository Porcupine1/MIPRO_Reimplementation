#!/usr/bin/env python3
"""
Quick manual test harness for `InstructionProposer`.

Recommended usage from project root:
    python -m cmd.test_instr_proposer

Also works when run directly as a script.
"""

import logging
import os
import sys
import pprint  # Added for pretty-printing candidate sets

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend import LMBackend
from QADataset import QADataset
from programs import QAProgram
from helpers import InstructionProposer
from logging_config import setup_logging
from config import apply_tier


logger = logging.getLogger(__name__)


def main():
    # Apply LIGHT tier for fast testing (BEFORE loading dataset)
    tier_config = apply_tier("light")

    setup_logging(
        level=logging.INFO,
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=== InstructionProposer standalone test (LIGHT tier) ===")
    logger.info("Using MAX_EXAMPLES=%d", tier_config.max_examples)

    # 1. Load dataset and grab a small training slice for grounding (will use tier-configured MAX_EXAMPLES)
    logger.info("Loading dataset...")
    dataset = QADataset()
    dataset.load()
    train_data = dataset.get_split("train")
    if not train_data:
        logger.error("No training data available; cannot test InstructionProposer.")
        return

    # Use a small prefix of the train split to keep the test fast
    max_grounding_examples = min(5, len(train_data))
    train_examples = [
        train_data[i] for i in range(max_grounding_examples)
    ]
    logger.info("Using %d examples for grounding summaries", len(train_examples))

    # 2. Initialize QAProgram and shared LM backend
    logger.info("Initializing QAProgram and LM backend...")
    lm = LMBackend()
    program = QAProgram(backend=lm)
    module_names = program.get_module_names()
    logger.info("Program modules: %s", module_names)

    # 3. Initialize InstructionProposer (this also builds dataset/program summaries)
    logger.info("Initializing InstructionProposer...")
    proposer = InstructionProposer(
        lm=lm, train_examples=train_examples, program=program
    )

    # 4. Test a single instruction proposal for the 'query' module
    task_desc = "Answer multi-hop questions using retrieved context from Wikipedia."
    test_module = "query" if "query" in module_names else module_names[0]
    logger.info("Proposing single instruction for module '%s'...", test_module)

    single_instruction = proposer.propose_instruction(
        module_name=test_module,
        task_desc=task_desc,
        bootstrapped_demos=None,  # fine to omit; will still use summaries
    )
    logger.instr(
        "\n\n====== Single proposed instruction for '%s' ======\n%s\n",
        test_module,
        single_instruction.strip(),
    )

    # 5. Test proposing for all modules (with empty demo lists to keep this simple)
    logger.info("Proposing instruction candidates for all modules...")
    empty_demos = {name: [] for name in module_names}
    all_candidates = proposer.propose_for_all_modules(
        program=program,
        task_desc=task_desc,
        bootstrapped_demos=empty_demos,
    )

    for predictor_idx, candidates in all_candidates.items():
        module_name = module_names[predictor_idx]
        logger.info(
            "Predictor %d (%s): %d instruction options (showing first 2)",
            predictor_idx,
            module_name,
            len(candidates),
        )
        # Pretty-print the first couple of options in a block per predictor
        header_lines = [
            "",
            f"====== Instruction options for predictor {predictor_idx} ({module_name}) ======",
        ]
        body_lines = []
        for j, instr in enumerate(candidates[:2]):
            prefix = "original" if j == 0 else f"candidate_{j}"
            body_lines.append(f"[{prefix}] {instr.strip()}")
        if body_lines:
            logger.instr("\n".join(header_lines + body_lines) + "\n")

    # Print the entire candidate set at the end with custom formatting similar to demo_candidates
    logger.instr("\n\n======= FULL instruction candidate set ========")
    for predictor_idx, instruction_list in all_candidates.items():
        module_name = module_names[predictor_idx]
        logger.instr(f"\n[Predictor: {predictor_idx}, Module: {module_name}]")
        logger.instr(f"  Total instructions: {len(instruction_list)}")
        for instr_idx, instruction in enumerate(instruction_list):
            if instr_idx == 0:
                label = "original"
            else:
                label = f"candidate_{instr_idx}"
            logger.instr(f"  Instruction {instr_idx} [{label}]:")
            # Split long instructions into multiple lines for better readability
            lines = instruction.strip().split('\n')
            for line in lines:
                logger.instr(f"    {line}")
    logger.instr("")

    logger.info("=== InstructionProposer test completed ===")


if __name__ == "__main__":
    main()

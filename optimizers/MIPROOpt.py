from typing import Dict, List, Any, Optional
import json
import os
import logging

from programs import QAProgram
from QADataset import QADataset
from helpers import InstructionProposer, DemoBootstrapper, GroundingHelper
from .SurrogateOpt import SurrogateOptimizer
from metrics import compute_metrics
from config import (
    N_TRIALS,
    BATCH_SIZE,
    N_INSTRUCTION_CANDIDATES,
    EVAL_BATCH_SIZE,
    METRIC,
    OUTPUT_DIR,
    NUM_CANDIDATES,
    MINIBATCH_FULL_EVAL_STEPS,
)


logger = logging.getLogger(__name__)


class MIPROOptimizer:
    """main MIPRO optimizer orchestrating instruction optimization"""

    def __init__(
        self,
        program: QAProgram,
        dataset: QADataset,
        metric: str = METRIC,
        n_trials: int = N_TRIALS,
        batch_size: int = BATCH_SIZE,
        eval_batch_size: int = EVAL_BATCH_SIZE,
        n_instruction_candidates: int = N_INSTRUCTION_CANDIDATES,
        output_dir: str = OUTPUT_DIR,
    ):
        """
        args:
            program: QA program to optimize
            dataset: dataset for optimization
            metric: metric to optimize ('f1' or 'exact_match')
            n_trials: number of Bayesian optimization trials
            batch_size: mini-batch size for evaluation
            eval_batch_size: batch size for final evaluation
            n_instruction_candidates: number of instruction candidates per module
            output_dir: directory to save results
        """
        self.program = program
        self.dataset = dataset
        self.metric = metric
        self.n_trials = n_trials
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.n_instruction_candidates = n_instruction_candidates
        self.output_dir = output_dir

        # components - will be initialized in optimize() with training data
        self.proposer = None
        self.bootstrapper = DemoBootstrapper(program, dataset, metric=metric)
        self.grounding = None
        self.instruction_candidates = None
        # Sort module names once for deterministic predictor indexing
        self.module_names = sorted(self.program.get_module_names())
        self.predictor_to_module = {idx: name for idx, name in enumerate(self.module_names)}
        self.surrogate = None

        # results
        self.best_config = None
        self.best_score = None
        self.optimization_history = []
        self.minibatch_candidates = []
        self.full_eval_candidates = []
        self.call_counts = {}

        os.makedirs(output_dir, exist_ok=True)

    def optimize(
        self, task_description: str = "Answer questions using retrieved context"
    ) -> QAProgram:
        """
        Run MIPRO optimization following the three-step process:

        Step 1: Bootstrap Few-Shot Examples
        Step 2: Propose Instruction Candidates
        Step 3: Bayesian Optimization Search

        args:
            task_description: description of the task

        returns:
            optimized program with best instructions and demos
        """

        logger.info("=" * 60)
        logger.info("MIPRO Optimization")
        logger.info("=" * 60)

        # Initialize components with training data for grounding
        logger.info("[Initialization] Setting up grounding with training data...")
        train_data = self.dataset.get_split(split="train")
        if not train_data:
            raise ValueError("Cannot optimize: No training data available")
        # Limit train/validation to at most 1000 examples each to keep search fast
        if len(train_data) > 1000:
            train_data = train_data.shuffle(seed=42).select(range(1000))
            self.dataset.train = train_data
        val_data = self.dataset.get_split(split="validation")
        if val_data is not None and len(val_data) > 1000:
            val_data = val_data.shuffle(seed=42).select(range(1000))
            self.dataset.validation = val_data

        # Initialize GroundingHelper with training data and program (generates summaries)
        self.grounding = GroundingHelper(
            train_examples=train_data, program=self.program
        )

        # Initialize InstructionProposer with training data and program
        self.proposer = InstructionProposer(
            train_examples=train_data, program=self.program
        )

        # Step 1: Bootstrap Few-Shot Examples
        logger.info("[Step 1/3] Bootstrapping few-shot examples...")
        demo_candidates = self.bootstrapper.bootstrap_candidates(
            num_candidates=NUM_CANDIDATES, train_data=train_data, module_names=self.module_names
        )
        logger.info(
            "Created candidate demo sets (per-predictor counts): %s",
            {
                self.predictor_to_module.get(idx, idx): len(sets)
                for idx, sets in demo_candidates.items()
            },
        )

        # Step 2: Propose Instruction Candidates
        logger.info("[Step 2/3] Proposing instruction candidates...")
        self.instruction_candidates = self.proposer.propose_for_all_modules(
            program=self.program,
            task_desc=task_description,
            bootstrapped_demos=demo_candidates,
            dataset_summ=self.grounding.dataset_summary,
            program_summ=self.grounding.program_summary,
            n_candidates=self.n_instruction_candidates,
            program_aware=True,
            module_names=self.module_names,
        )

        logger.info("Generated instruction candidates:")
        for predictor_idx, candidates in self.instruction_candidates.items():
            module_name = self.predictor_to_module[predictor_idx]
            logger.info(
                "  Predictor %d (%s): %d candidates (1 original + %d proposed)",
                predictor_idx,
                module_name,
                len(candidates),
                len(candidates) - 1,
            )

        # Step 3: Bayesian optimization over instruction + demo space
        logger.info(
            "[Step 3/3] Running Bayesian optimization (%d trials)...", self.n_trials
        )
        logger.info("  Using minibatch evaluation (size=%d)", self.batch_size)
        logger.info("  Full evaluation every %d trials", MINIBATCH_FULL_EVAL_STEPS)

        optimized_program = self._run_bayesian_optimization(demo_candidates)

        if optimized_program is None:
            raise RuntimeError(
                "Optimization failed: Could not find valid configuration"
            )

        # Capture best configuration details for reporting/saving.
        self.best_config = self._program_to_config(optimized_program)
        logger.info("Best score after optimization: %.4f", self.best_score)

        # Save results
        self._save_results()

        logger.info("=" * 60)
        logger.info("Optimization Complete!")
        logger.info("Best %s: %.4f", self.metric, self.best_score)
        logger.info("=" * 60)

        return optimized_program

    def _run_bayesian_optimization(
        self, demo_candidates: Dict[int, List[List[Dict[str, Any]]]]
    ) -> Dict[str, Any]:
        """
        Run Bayesian optimization to find best instruction + demo configuration.

        Uses minibatch evaluation with periodic full validation.
        """
        # Initialize surrogate optimizer with new Optuna-based search.
        self.surrogate = SurrogateOptimizer(
            program=self.program,
            instruction_candidates=self.instruction_candidates,
            predictor_to_module=self.predictor_to_module,
            demo_candidates=demo_candidates,
            evaluate_fn=self._evaluate_program,
            num_trials=self.n_trials,
            minibatch_size=self.batch_size,
            eval_batch_size=self.eval_batch_size,
            minibatch_full_eval_steps=MINIBATCH_FULL_EVAL_STEPS,
            seed=42,
            use_minibatch=True,
            val_split="validation",
        )

        opt_result = self.surrogate.optimize()
        self.optimization_history = opt_result.get("trial_logs", [])
        self.best_score = opt_result.get("best_score")
        self.minibatch_candidates = opt_result.get("minibatch_candidates", [])
        self.full_eval_candidates = opt_result.get("full_eval_candidates", [])
        self.call_counts = opt_result.get("call_counts", {})

        return opt_result.get("best_program")

    def _evaluate_program(
        self, program: Any, batch_size: int, split: str = "train"
    ) -> float:
        """
        Evaluate a configured program (any object exposing process_batch) on a batch for the specified split.
        """
        # Sample batch
        batch = self.dataset.sample_batch(batch_size, split=split)

        if not batch:
            # Return 0.0 if no examples available
            return 0.0

        # Generate predictions in parallel
        predictions = program.process_batch(batch, parallel=True)

        # Collect ground truths
        ground_truths = []
        for example in batch:
            try:
                ground_truth = self.dataset.get_ground_truth(example)
                ground_truths.append(ground_truth)
            except Exception:
                ground_truths.append("")

        # Compute metrics
        metrics = compute_metrics(predictions, ground_truths)

        if self.metric not in metrics:
            raise ValueError(
                f"Invalid metric '{self.metric}'. Must be one of: {list(metrics.keys())}"
            )

        score = metrics[self.metric]

        # Log evaluation progress so users can see optimization progress and outputs
        logger.info(
            "[Eval] split=%s batch_size=%d metric=%s score=%.4f",
            split,
            len(batch),
            self.metric,
            score,
        )

        # Optionally log a few example predictions at DEBUG level
        if logger.isEnabledFor(logging.DEBUG):
            for idx, (example, pred, gt) in enumerate(
                zip(batch, predictions, ground_truths)
            ):
                if idx >= 3:
                    break
                question = example.get("question", "")
                logger.debug(
                    "Example %d | Q: %s | Pred: %s | GT: %s",
                    idx,
                    question,
                    str(pred),
                    str(gt),
                )

        return score

    def _program_to_config(self, program: QAProgram) -> Dict[str, Any]:
        """Extract instruction/demo config from a configured program."""
        instructions = {
            name: module.instruction for name, module in program.modules.items()
        }
        demos = {name: module.demos for name, module in program.modules.items()}
        return {"instructions": instructions, "demos": demos}

    def _save_results(self):
        """save optimization results"""
        results = {
            "best_config": self.best_config,
            "best_score": self.best_score,
            "metric": self.metric,
            "n_trials": self.n_trials,
            "instruction_candidates": self.instruction_candidates,
            "optimization_history": self.optimization_history,
            "call_counts": getattr(self, "call_counts", {}),
            "minibatch_candidates": getattr(self, "minibatch_candidates", []),
            "full_eval_candidates": getattr(self, "full_eval_candidates", []),
            "predictor_to_module": self.predictor_to_module,
        }

        output_path = os.path.join(self.output_dir, "mipro_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Results saved to: %s", output_path)

    def get_best_instructions(self) -> Dict[str, str]:
        """get best instruction configuration"""
        if self.best_config and "instructions" in self.best_config:
            return self.best_config["instructions"]
        return {}

    def get_best_score(self) -> float:
        """get best score achieved"""
        return self.best_score

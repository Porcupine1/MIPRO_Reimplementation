import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.trial import TrialState, create_trial

from config import N_TRIALS, BATCH_SIZE, EVAL_BATCH_SIZE, MINIBATCH_FULL_EVAL_STEPS


logger = logging.getLogger(__name__)


class SurrogateOptimizer:
    """
    Optuna + TPE search for instruction/demo selection with minibatch refresh.
    """

    def __init__(
        self,
        program: Any,
        instruction_candidates: Dict[int, List[str]],
        predictor_to_module: Dict[int, str],
        evaluate_fn: Callable[[Any, int, str], float],
        demo_candidates: Optional[Dict[int, List[List[Dict[str, Any]]]]] = None,
        num_trials: int = N_TRIALS,
        minibatch_size: int = BATCH_SIZE,
        eval_batch_size: int = EVAL_BATCH_SIZE,
        minibatch_full_eval_steps: int = MINIBATCH_FULL_EVAL_STEPS,
        seed: int = 42,
        use_minibatch: bool = True,
        val_split: str = "validation",
    ):
        """
        Args:
            program: Base program object containing predictors and clone/apply_configuration.
            instruction_candidates: predictor_idx -> list of instruction strings.
            predictor_to_module: predictor_idx -> module name.
            evaluate_fn: Callable(program, batch_size, split) -> score.
            demo_candidates: predictor_idx -> list of demo options (each option is a list of demos).
            num_trials: number of Optuna trials to run.
            minibatch_size: batch size for minibatch evaluation.
            eval_batch_size: batch size for full evaluation.
            minibatch_full_eval_steps: run full eval after this many minibatch trials.
            seed: random seed for the TPE sampler.
            use_minibatch: whether to use minibatch objective.
            val_split: dataset split name for full evaluations.
        """
        self.program = program
        self.instruction_candidates = instruction_candidates
        self.predictor_to_module = predictor_to_module
        self.demo_candidates = demo_candidates or {}
        self.num_trials = num_trials
        self.minibatch_size = minibatch_size
        self.eval_batch_size = eval_batch_size
        self.minibatch_full_eval_steps = minibatch_full_eval_steps
        self.seed = seed
        self.use_minibatch = use_minibatch
        self.val_split = val_split
        self.evaluate_fn = evaluate_fn

        # Validate candidate structures
        self._validate_candidates()

        self.study: Optional[optuna.Study] = None
        self.best_program: Optional[Any] = None
        self.best_score: float = float("-inf")

        self.trial_logs: List[Dict[str, Any]] = []
        self.minibatch_candidates: List[Dict[str, Any]] = []
        self.full_eval_candidates: List[Dict[str, Any]] = []
        self.full_eval_configs_seen: set[Tuple[Tuple[str, int], ...]] = set()

        self.minibatch_eval_calls = 0
        self.full_eval_calls = 0
        self._completed_minibatch_trials = 0

    def _validate_candidates(self) -> None:
        """Validate instruction and demo candidate structures."""
        # Validate instruction_candidates
        if not isinstance(self.instruction_candidates, dict):
            raise TypeError("instruction_candidates must be a dict")

        for predictor_idx, candidates in self.instruction_candidates.items():
            if not isinstance(candidates, list):
                raise TypeError(
                    f"Instruction candidates for predictor {predictor_idx} must be a list, "
                    f"got {type(candidates)}"
                )
            if len(candidates) == 0:
                raise ValueError(
                    f"Instruction candidates for predictor {predictor_idx} must be non-empty"
                )
            for i, candidate in enumerate(candidates):
                if not isinstance(candidate, str):
                    raise TypeError(
                        f"Instruction candidate {i} for predictor {predictor_idx} must be a string, "
                        f"got {type(candidate)}"
                    )

        # Validate demo_candidates
        if not isinstance(self.demo_candidates, dict):
            raise TypeError("demo_candidates must be a dict")

        for predictor_idx, demo_sets in self.demo_candidates.items():
            if not isinstance(demo_sets, list):
                raise TypeError(
                    f"Demo candidates for predictor {predictor_idx} must be a list, "
                    f"got {type(demo_sets)}"
                )
            for i, demo_set in enumerate(demo_sets):
                if not isinstance(demo_set, list):
                    raise TypeError(
                        f"Demo set {i} for predictor {predictor_idx} must be a list of demos, "
                        f"got {type(demo_set)}"
                    )

        logger.debug("Candidate structure validation passed")

    def optimize(self) -> Dict[str, Any]:
        """Run Optuna search and return optimization artifacts."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.info(
            "\n" + "=" * 80 + "\n"
            "Starting Optuna TPE Search\n"
            "=" * 80 + "\n"
            f"  Trials: {self.num_trials}\n"
            f"  Minibatch Mode: {self.use_minibatch}\n"
            f"  Minibatch Size: {self.minibatch_size}\n"
            f"  Full Eval Size: {self.eval_batch_size}\n"
            f"  Full Eval Every: {self.minibatch_full_eval_steps} trials\n"
            f"  Random Seed: {self.seed}\n" + "=" * 80
        )

        # Baseline: evaluate default program configuration.
        logger.info("\nEvaluating Baseline Configuration...")
        baseline_program = self.program.clone()
        default_instructions, default_demos = self._default_config()
        baseline_program.apply_configuration(default_instructions, default_demos)
        baseline_score = self._evaluate(
            program=baseline_program,
            batch_size=self.eval_batch_size,
            split=self.val_split,
            eval_type="full_baseline",
            trial_number="baseline",
            config={
                "instruction_indices": {
                    self.predictor_to_module[idx]: 0
                    for idx in self.instruction_candidates.keys()
                },
                "demo_indices": {
                    self.predictor_to_module[idx]: 0
                    for idx in self.demo_candidates.keys()
                },
            },
        )

        self.best_score = baseline_score
        self.best_program = baseline_program.clone()
        logger.best_score(  # type: ignore[attr-defined]
            f"Baseline Score: {baseline_score:.4f} (initial best)"
        )

        # Study + synthetic baseline trial.
        sampler = optuna.samplers.TPESampler(seed=self.seed, multivariate=True)
        self.study = optuna.create_study(direction="maximize", sampler=sampler)
        self.study.add_trial(
            create_trial(
                params=self._default_params(),
                distributions=self._build_distributions(),
                value=baseline_score,
                state=TrialState.COMPLETE,
            )
        )

        def objective(trial: optuna.Trial) -> float:
            # Log candidate sampling
            logger.candidate(  # type: ignore[attr-defined]
                f"\n{'─' * 80}\n" f"Trial #{trial.number} - Sampling New Candidate"
            )

            config = self._sample_config(trial)
            self._log_candidate_config(trial.number, config)

            candidate_program = self.program.clone()
            candidate_program.apply_configuration(
                instructions=config["instructions"], demos=config.get("demos")
            )

            if self.use_minibatch:
                score = self._evaluate(
                    program=candidate_program,
                    batch_size=self.minibatch_size,
                    split="train",
                    eval_type="minibatch",
                    trial_number=trial.number,
                    config=config,
                )
                self.minibatch_eval_calls += 1
                self.minibatch_candidates.append(
                    {
                        "trial": trial.number,
                        "score": score,
                        "config": self._config_summary(config),
                    }
                )
                self._completed_minibatch_trials += 1

                # Log progress
                self._log_trial_progress(trial.number, score, "minibatch")

                if self._should_refresh_full_eval():
                    logger.info(
                        f"\nPeriodic Full Evaluation (after {self.minibatch_full_eval_steps} trials)"
                    )
                    self._run_periodic_full_eval()
            else:
                score = self._evaluate(
                    program=candidate_program,
                    batch_size=self.eval_batch_size,
                    split=self.val_split,
                    eval_type="full",
                    trial_number=trial.number,
                    config=config,
                )
                self.full_eval_calls += 1
                self._log_trial_progress(trial.number, score, "full")
                self._maybe_update_best(score, candidate_program, config)

            return score

        self.study.optimize(
            objective, n_trials=self.num_trials, show_progress_bar=False
        )

        if self.use_minibatch:
            # Final refresh with the best minibatch configs.
            logger.info("\nFinal Full Evaluation of Top Candidates...")
            self._run_periodic_full_eval(force_final=True)

        # Attach optimization metadata to the best program for inspection.
        if self.best_program:
            self._annotate_best_program()

        # Log final summary
        logger.info(
            "\n" + "=" * 80 + "\n"
            "Optimization Complete!\n"
            "=" * 80 + "\n"
            f"  Final Best Score: {self.best_score:.4f}\n"
            f"  Total Trials: {self.num_trials}\n"
            f"  Minibatch Evaluations: {self.minibatch_eval_calls}\n"
            f"  Full Evaluations: {self.full_eval_calls}\n" + "=" * 80
        )

        return {
            "best_program": self.best_program,
            "best_score": self.best_score,
            "trial_logs": self.trial_logs,
            "minibatch_candidates": self.minibatch_candidates,
            "full_eval_candidates": self.full_eval_candidates,
            "call_counts": {
                "minibatch": self.minibatch_eval_calls,
                "full": self.full_eval_calls,
            },
            "study": self.study,
        }

    # --------------------------- helper methods --------------------------- #
    def _default_config(self) -> Tuple[Dict[str, str], Dict[str, List[Dict[str, Any]]]]:
        instructions = {}
        demos = {}
        for predictor_idx, candidates in self.instruction_candidates.items():
            module_name = self.predictor_to_module[predictor_idx]
            instructions[module_name] = candidates[0]
        for predictor_idx, demo_sets in self.demo_candidates.items():
            module_name = self.predictor_to_module[predictor_idx]
            if demo_sets:
                demos[module_name] = demo_sets[0]
        return instructions, demos

    def _default_params(self) -> Dict[str, int]:
        params = {}
        for predictor_idx in self.instruction_candidates.keys():
            module_name = self.predictor_to_module[predictor_idx]
            params[f"{module_name}_instruction"] = 0
        for predictor_idx in self.demo_candidates.keys():
            module_name = self.predictor_to_module[predictor_idx]
            params[f"{module_name}_demos"] = 0
        return params

    def _build_distributions(self) -> Dict[str, CategoricalDistribution]:
        distributions: Dict[str, CategoricalDistribution] = {}
        for predictor_idx, candidates in self.instruction_candidates.items():
            module_name = self.predictor_to_module[predictor_idx]
            distributions[f"{module_name}_instruction"] = CategoricalDistribution(
                list(range(len(candidates)))
            )
        for predictor_idx, demo_sets in self.demo_candidates.items():
            module_name = self.predictor_to_module[predictor_idx]
            distributions[f"{module_name}_demos"] = CategoricalDistribution(
                list(range(len(demo_sets)))
            )
        return distributions

    def _sample_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        instructions: Dict[str, str] = {}
        instruction_indices: Dict[str, int] = {}
        demos: Dict[str, List[Dict[str, Any]]] = {}
        demo_indices: Dict[str, int] = {}

        for predictor_idx, candidates in self.instruction_candidates.items():
            module_name = self.predictor_to_module[predictor_idx]
            choice = trial.suggest_categorical(
                f"{module_name}_instruction", list(range(len(candidates)))
            )
            instructions[module_name] = candidates[choice]
            instruction_indices[module_name] = choice

        for predictor_idx, demo_sets in self.demo_candidates.items():
            if not demo_sets:
                continue
            module_name = self.predictor_to_module[predictor_idx]
            choice = trial.suggest_categorical(
                f"{module_name}_demos", list(range(len(demo_sets)))
            )
            demos[module_name] = demo_sets[choice]
            demo_indices[module_name] = choice

        return {
            "instructions": instructions,
            "instruction_indices": instruction_indices,
            "demos": demos if demos else None,
            "demo_indices": demo_indices if demo_indices else None,
        }

    def _evaluate(
        self,
        program: Any,
        batch_size: int,
        split: str,
        eval_type: str,
        trial_number: Any,
        config: Dict[str, Any],
    ) -> float:
        score = self.evaluate_fn(program=program, batch_size=batch_size, split=split)
        log_entry = {
            "trial": trial_number,
            "score": score,
            "eval_type": eval_type,
            "split": split,
            "batch_size": batch_size,
            "instruction_indices": config.get("instruction_indices"),
            "demo_indices": config.get("demo_indices"),
        }
        self.trial_logs.append(log_entry)

        # Use colored evaluation logging
        logger.evaluation(  # type: ignore[attr-defined]
            f"Trial {trial_number} | {eval_type:15s} | "
            f"{split:10s} | batch={batch_size:3d} | Score: {score:.4f}"
        )
        return score

    def _config_key(self, config: Dict[str, Any]) -> Tuple[Tuple[str, int], ...]:
        key_items: List[Tuple[str, int]] = []
        for name, idx in (config.get("instruction_indices") or {}).items():
            key_items.append((f"{name}_instruction", idx))
        for name, idx in (config.get("demo_indices") or {}).items():
            key_items.append((f"{name}_demos", idx))
        return tuple(sorted(key_items))

    def _config_summary(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "instruction_indices": config.get("instruction_indices"),
            "demo_indices": config.get("demo_indices"),
        }

    def _maybe_update_best(self, score: float, program: Any, config: Dict[str, Any]):
        if score > self.best_score:
            improvement = score - self.best_score
            self.best_score = score
            self.best_program = program.clone()
            logger.best_score(  # type: ignore[attr-defined]
                f"\n{'*' * 80}\n"
                f"NEW BEST SCORE: {score:.4f} (improvement: {improvement:+.4f})\n"
                f"{'*' * 80}"
            )
            self._log_best_config(config)

    def _should_refresh_full_eval(self) -> bool:
        if self.minibatch_full_eval_steps <= 0:
            return False
        return self._completed_minibatch_trials % self.minibatch_full_eval_steps == 0

    def _run_periodic_full_eval(self, force_final: bool = False):
        if not self.minibatch_candidates:
            return

        # Select top candidates by minibatch score.
        sorted_candidates = sorted(
            self.minibatch_candidates, key=lambda x: x["score"], reverse=True
        )
        top_k = 3 if len(sorted_candidates) >= 3 else len(sorted_candidates)
        candidates_to_eval = sorted_candidates[:top_k]

        for candidate in candidates_to_eval:
            config = candidate["config"]
            key = self._config_key(config)
            if not force_final and key in self.full_eval_configs_seen:
                continue
            self.full_eval_configs_seen.add(key)

            candidate_program = self.program.clone()
            candidate_program.apply_configuration(
                instructions=self._indices_to_instructions(config),
                demos=self._indices_to_demos(config),
            )
            score = self._evaluate(
                program=candidate_program,
                batch_size=self.eval_batch_size,
                split=self.val_split,
                eval_type="full_refresh",
                trial_number="refresh",
                config=config,
            )
            self.full_eval_calls += 1
            self.full_eval_candidates.append(
                {
                    "source_minibatch_trial": candidate["trial"],
                    "score": score,
                    "config": self._config_summary(config),
                }
            )
            self._maybe_update_best(score, candidate_program, config)

    def _indices_to_instructions(self, config: Dict[str, Any]) -> Dict[str, str]:
        instructions: Dict[str, str] = {}
        for predictor_idx, candidates in self.instruction_candidates.items():
            module_name = self.predictor_to_module[predictor_idx]
            idx = (config.get("instruction_indices") or {}).get(module_name, 0)
            instructions[module_name] = candidates[idx]
        return instructions

    def _indices_to_demos(
        self, config: Dict[str, Any]
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        if not self.demo_candidates:
            return None
        demos: Dict[str, List[Dict[str, Any]]] = {}
        for predictor_idx, demo_sets in self.demo_candidates.items():
            module_name = self.predictor_to_module[predictor_idx]
            idx = (config.get("demo_indices") or {}).get(module_name, 0)
            if demo_sets:
                demos[module_name] = demo_sets[idx]
        return demos if demos else None

    def _log_candidate_config(self, trial_number: int, config: Dict[str, Any]) -> None:
        """Log the sampled candidate configuration in a clean format."""
        lines = [f"  Configuration for Trial #{trial_number}:"]

        # Log instruction choices
        if config.get("instruction_indices"):
            lines.append("  Instructions:")
            for module, idx in config["instruction_indices"].items():
                instruction_text = config["instructions"].get(module, "")
                lines.append(f"    - {module}: [{idx}] {instruction_text}")

        # Log demo choices
        if config.get("demo_indices"):
            lines.append("  Demos:")
            for module, idx in config["demo_indices"].items():
                demo_count = len(config.get("demos", {}).get(module, []))
                lines.append(f"    - {module}: set[{idx}] ({demo_count} examples)")

        logger.candidate("\n".join(lines))  # type: ignore[attr-defined]

    def _log_trial_progress(
        self, trial_number: int, score: float, eval_type: str
    ) -> None:
        """Log trial progress with current standings."""
        if eval_type == "minibatch":
            progress_pct = (trial_number + 1) / self.num_trials * 100
            logger.info(
                f"  Progress: {trial_number + 1}/{self.num_trials} ({progress_pct:.1f}%) | "
                f"Current: {score:.4f} | Best: {self.best_score:.4f}"
            )
        else:
            logger.info(f"  Current: {score:.4f} | Best: {self.best_score:.4f}")

    def _log_best_config(self, config: Dict[str, Any]) -> None:
        """Log the configuration of the new best program."""
        lines = ["  Best Configuration:"]

        if config.get("instruction_indices"):
            lines.append("  Instructions:")
            for module, idx in config["instruction_indices"].items():
                lines.append(f"    - {module}: index [{idx}]")

        if config.get("demo_indices"):
            lines.append("  Demos:")
            for module, idx in config["demo_indices"].items():
                lines.append(f"    - {module}: set [{idx}]")

        logger.best_score("\n".join(lines))  # type: ignore[attr-defined]

    def _annotate_best_program(self):
        if not self.best_program:
            return
        setattr(
            self.best_program,
            "optimization_meta",
            {
                "best_score": self.best_score,
                "trial_logs": self.trial_logs,
                "minibatch_candidates": self.minibatch_candidates,
                "full_eval_candidates": self.full_eval_candidates,
                "call_counts": {
                    "minibatch": self.minibatch_eval_calls,
                    "full": self.full_eval_calls,
                },
            },
        )

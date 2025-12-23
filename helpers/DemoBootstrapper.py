from typing import List, Dict, Any, Tuple, Optional
import logging
from programs import QAProgram
from QADataset import QADataset
from metrics import f1_score, exact_match
from config import (
    NUM_CANDIDATES,
    MAX_BOOTSTRAPPED_DEMOS,
    MAX_LABELED_DEMOS,
    BOOTSTRAP_THRESHOLD,
    MIN_CONTEXT_CHARS,
)


logger = logging.getLogger(__name__)


class DemoBootstrapper:
    """
    Bootstrap demonstrations from successful program traces.

    For each module, creates multiple candidate demo sets by:
    1. Running program on training examples until reaching max_bootstrapped_demos successful traces
    2. Extracting module-specific demos from successful traces
    3. Combining bootstrapped and labeled demos to create candidate sets
    """

    def __init__(
        self,
        program: QAProgram,
        dataset: QADataset,
        metric: str = "f1",
        threshold: float = BOOTSTRAP_THRESHOLD,
    ):
        self.program = program
        self.dataset = dataset
        self.metric = metric
        self.threshold = threshold

    def _check_grounding(self, trace_entry: Dict[str, Any]) -> bool:
        """
        Generic grounding check for trace entries.
        Rejects demos with insufficient context or fallback refusals.

        Returns:
            True if demo passes grounding checks, False otherwise.
        """
        module_name = trace_entry.get("module", "")
        trace_input = trace_entry.get("input", {})
        trace_output = trace_entry.get("output", "")

        # Only apply grounding checks to answer module (requires context)
        if module_name == "answer":
            context = trace_input.get("context", "")

            # Check context length
            if len(context.strip()) < MIN_CONTEXT_CHARS:
                logger.info(
                    "Rejected bootstrapped demo for module '%s': context too short (len=%d, min=%d)",
                    module_name,
                    len(context.strip()),
                    MIN_CONTEXT_CHARS,
                )
                return False

            # Check for fallback refusals
            lower_output = trace_output.lower()
            refusal_phrases = [
                "please provide context",
                "no context provided",
                "cannot answer without context",
                "need more information",
            ]
            if any(phrase in lower_output for phrase in refusal_phrases):
                logger.info(
                    "Rejected bootstrapped demo for module '%s': output is a fallback refusal",
                    module_name,
                )
                return False

        return True

    def _run_and_score(
        self, example: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Run program on example and return module traces with score.
        Uses an isolated program clone to avoid trace contamination.

        Returns:
            module_traces: List of module execution traces from program
            score: Final output score
        """
        question = self.dataset.get_question(example)
        ground_truth = self.dataset.get_ground_truth(example)

        # Use isolated program clone to avoid trace contamination
        program_clone = self.program.clone()
        answer = program_clone.forward(example)

        # Get the trace from the cloned program
        module_traces = program_clone.get_trace()

        # Score final output
        if self.metric == "f1":
            score = f1_score(answer, ground_truth)
        else:
            score = exact_match(answer, ground_truth)

        return module_traces, score

    def bootstrap_candidates(
        self,
        num_candidates: int = NUM_CANDIDATES,
        max_bootstrapped_demos: int = MAX_BOOTSTRAPPED_DEMOS,
        max_labeled_demos: int = MAX_LABELED_DEMOS,
        train_data: Optional[Any] = None,
        module_names: Optional[List[str]] = None,
    ) -> Dict[int, List[List[Dict[str, Any]]]]:
        """
        Bootstrap candidate demo sets for each module.

        Always creates 2 baseline sets (zero-shot + labeled-only), then num_candidates
        additional sets with bootstrapped demos.

        Structure: {module_name: [[demo_set_0], [demo_set_1], ...]}
        - Set 0: Always empty (zero-shot baseline)
        - Set 1: Always labeled-only (labeled baseline)
        - Sets 2+: Bootstrapped demos (controlled by num_candidates)

        Args:
            num_candidates: Number of bootstrapped demo sets to create (beyond the 2 baselines)
            max_bootstrapped_demos: Maximum bootstrapped demos per set
            max_labeled_demos: Maximum labeled demos per set (used in baseline and mixed sets)
            train_data: Optional training split to use. If None, uses dataset's train split.
            module_names: Optional sorted list of module names. If None, sorts program modules.

        Returns:
            Dict mapping module_name -> list of candidate demo sets (total: num_candidates + 2)
        """
        # Resolve training data source
        if train_data is None:
            # Prefer a dedicated accessor if available
            if hasattr(self.dataset, "get_split"):
                train_data = self.dataset.get_split("train")
            else:
                train_data = getattr(self.dataset, "train", None)

        if train_data is None:
            logger.warning(
                "No training data provided to DemoBootstrapper; returning empty candidates."
            )
            return {}

        # Track how many examples we have tried; by default we iterate over all
        examples_tried = 0
        try:
            max_examples = len(train_data)
        except TypeError:
            # Fallback if len() is not supported
            train_data = list(train_data)
            max_examples = len(train_data)

        logger.info(
            "[Step 1] Bootstrapping %d candidate demo sets per module "
            "(always: zero-shot + labeled-only, then %d bootstrapped sets with up to %d bootstrapped + %d labeled demos per set)",
            num_candidates + 2,  # Total including zero-shot and labeled-only
            num_candidates,
            max_bootstrapped_demos,
            max_labeled_demos,
        )

        # Step 1: Collect bootstrapped demos by iterating through training examples
        logger.info("Collecting bootstrapped demos from %d examples...", max_examples)
        # Use provided module order or sort once if not provided
        if module_names is None:
            module_names = sorted(self.program.get_module_names())
        bootstrapped_demos = {module: [] for module in module_names}

        for example in train_data:
            if examples_tried >= max_examples:
                break

            # Check if we have enough demos for all modules
            if all(
                len(bootstrapped_demos[module]) >= max_bootstrapped_demos
                for module in module_names
            ):
                break

            examples_tried += 1
            try:
                module_traces, score = self._run_and_score(example)

                # If successful, extract module demos from trace
                if score >= self.threshold:
                    for trace_entry in module_traces:
                        module_name = trace_entry["module"]
                        if module_name in bootstrapped_demos:
                            # Only add if we haven't reached max for this module
                            if (
                                len(bootstrapped_demos[module_name])
                                < max_bootstrapped_demos
                            ):
                                # Apply generic grounding checks
                                if not self._check_grounding(trace_entry):
                                    continue

                                # Store demo in consistent trace format with score
                                demo = {**trace_entry, "score": score}
                                bootstrapped_demos[module_name].append(demo)

                                # Log accepted demo
                                trace_input = trace_entry.get("input", {})
                                if module_name == "answer":
                                    context = trace_input.get("context", "")
                                    context_len = len(context)
                                    logger.info(
                                        "Accepted bootstrapped demo: module=%s, score=%.3f, context_len=%d",
                                        module_name,
                                        score,
                                        context_len,
                                    )
                                else:
                                    logger.info(
                                        "Accepted bootstrapped demo: module=%s, score=%.3f",
                                        module_name,
                                        score,
                                    )
            except Exception as e:
                logger.warning(
                    "Error while bootstrapping example %d: %s", examples_tried, e
                )
                continue

        # Sort by score
        for module_name in bootstrapped_demos:
            bootstrapped_demos[module_name].sort(
                key=lambda x: x.get("score", 0), reverse=True
            )
            logger.info(
                "Module '%s': %d bootstrapped demos collected",
                module_name,
                len(bootstrapped_demos[module_name]),
            )

        # Step 2: Collect labeled demos (ground-truth input-output pairs from training set)
        logger.info("Collecting labeled demos...")
        labeled_demos = {module: [] for module in module_names}

        # Sample labeled examples directly from the provided training data
        import random

        n_labeled_source = min(50, max_examples)
        if n_labeled_source > 0:
            labeled_indices = random.sample(range(max_examples), n_labeled_source)
            labeled_examples = [train_data[i] for i in labeled_indices]
        else:
            labeled_examples = []

        for example in labeled_examples:
            try:
                question = self.dataset.get_question(example)
                ground_truth_answer = self.dataset.get_ground_truth(example)
                context = self.dataset.get_context(example)

                # Create labeled demos in consistent trace format
                # Only create for modules that have ground truth outputs available
                for module_name in module_names:
                    if module_name == "answer":
                        # Answer module: question + context -> ground truth answer
                        # Store in trace format: {"module", "input", "output", "score"}
                        demo = {
                            "module": module_name,
                            "input": {
                                "question": question,
                                "context": context,
                            },
                            "output": ground_truth_answer,
                            "score": 1.0,  # Labeled examples are perfect
                        }
                        labeled_demos[module_name].append(demo)
                    # Note: Query module doesn't have ground truth queries, so skip labeled demos
                    # (or could use question as query, but that's not true ground truth)
            except Exception as e:
                logger.warning("Error while collecting labeled demo: %s", e)
                continue

        logger.info("Collected labeled demos for %d examples", len(labeled_examples))

        # Step 3: Create candidate sets for each module (module_names already sorted above)
        demo_candidates = {module: [] for module in module_names}

        for module_name in module_names:
            bootstrapped = bootstrapped_demos[module_name]
            labeled = labeled_demos[module_name]

            # Always include set 0 (zero-shot) and set 1 (labeled-only) as baselines
            # These don't count toward num_candidates

            # Candidate set 0: Empty (zero-shot baseline)
            demo_candidates[module_name].append([])

            # Candidate set 1: Labels only (labeled baseline)
            demo_candidates[module_name].append(labeled[:max_labeled_demos])

            # Now create num_candidates additional sets with bootstrapped demos
            for i in range(num_candidates):
                candidate_set = []

                # First candidate: Bootstrapped only (unshuffled)
                if i == 0:
                    candidate_set = bootstrapped[:max_bootstrapped_demos]

                # Second candidate: Bootstrapped only (shuffled)
                elif i == 1:
                    candidate_set = bootstrapped[:max_bootstrapped_demos].copy()
                    random.shuffle(candidate_set)

                # Remaining candidates: Mixed (bootstrapped + labeled)
                else:
                    # Sample bootstrapped demos
                    n_bootstrapped = min(max_bootstrapped_demos, len(bootstrapped))
                    n_labeled = min(max_labeled_demos, len(labeled))

                    if n_bootstrapped > 0:
                        sampled_bootstrapped = random.sample(
                            bootstrapped, n_bootstrapped
                        )
                        candidate_set.extend(sampled_bootstrapped)

                    if n_labeled > 0:
                        sampled_labeled = random.sample(labeled, n_labeled)
                        candidate_set.extend(sampled_labeled)

                    # Shuffle mixed set
                    random.shuffle(candidate_set)

                demo_candidates[module_name].append(candidate_set)

            logger.info(
                "Module '%s': %d candidate demo sets created (2 baselines + %d bootstrapped)",
                module_name,
                len(demo_candidates[module_name]),
                num_candidates,
            )

        logger.info(
            "Created %d total candidate demo sets per module (2 baselines + %d bootstrapped)",
            num_candidates + 2,
            num_candidates,
        )
        # Return predictor-indexed mapping for downstream consumers
        predictor_demo_candidates = {
            idx: demo_candidates[module_name]
            for idx, module_name in enumerate(module_names)
        }
        return predictor_demo_candidates

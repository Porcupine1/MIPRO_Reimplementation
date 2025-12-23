from typing import List, Dict, Any, Tuple, Optional
from programs import QAProgram
from QADataset import QADataset
from metrics import f1_score, exact_match, compute_metrics
from config import (
    NUM_CANDIDATES, MAX_BOOTSTRAPPED_DEMOS, MAX_LABELED_DEMOS, 
    BOOTSTRAP_THRESHOLD
)


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
        threshold: float = BOOTSTRAP_THRESHOLD
    ):
        self.program = program
        self.dataset = dataset
        self.metric = metric
        self.threshold = threshold

    def _run_and_score(self, example: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], float]:
        """
        Run program on example and return module traces with score.
        
        Returns:
            module_traces: List of module execution traces from program
            score: Final output score
        """
        question = self.dataset.get_question(example)
        ground_truth = self.dataset.get_ground_truth(example)
        
        # Run program - this populates the trace automatically
        answer = self.program.forward(example)
        
        # Get the trace from the program (already has module, input, output)
        module_traces = self.program.get_trace()
        
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
        split: str = "train"
    ) -> Dict[str, List[List[Dict[str, Any]]]]:
        """
        Bootstrap candidate demo sets for each module.
        
        Structure: {module_name: [[demo_set_0], [demo_set_1], ...]}
        Each demo_set is a list of demos for that module.
        
        Args:
            num_candidates: Number of candidate demo sets to create per module
            max_bootstrapped_demos: Maximum bootstrapped demos per set
            max_labeled_demos: Maximum labeled demos per set
            split: Dataset split to use
            
        Returns:
            Dict mapping module_name -> list of candidate demo sets
        """
        print(f"\n[Step 1] Bootstrapping {num_candidates} candidate demo sets per module...")
        print(f"  Each set: up to {max_bootstrapped_demos} bootstrapped + {max_labeled_demos} labeled demos")
        
        # Step 1: Collect bootstrapped demos by iterating through training examples
        print("  Collecting bootstrapped demos...")
        bootstrapped_demos = {module: [] for module in self.program.get_module_names()}
        
        # Get training data (use provided or load from dataset)
        train_data = self.dataset.get_split(split=split)
        examples_tried = 0
        max_examples = len(train_data) if train_data else 1000
        
        for example in train_data:
            if examples_tried >= max_examples:
                break
                
            # Check if we have enough demos for all modules
            if all(len(bootstrapped_demos[module]) >= max_bootstrapped_demos 
                   for module in self.program.get_module_names()):
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
                            if len(bootstrapped_demos[module_name]) < max_bootstrapped_demos:
                                demo = {**trace_entry, "score": score}
                                bootstrapped_demos[module_name].append(demo)
            except Exception as e:
                continue
        
        # Sort by score
        for module_name in bootstrapped_demos:
            bootstrapped_demos[module_name].sort(key=lambda x: x.get("score", 0), reverse=True)
            print(f"    {module_name}: {len(bootstrapped_demos[module_name])} bootstrapped demos")
        
        # Step 2: Collect labeled demos (ground-truth input-output pairs from training set)
        print("  Collecting labeled demos...")
        labeled_demos = {module: [] for module in self.program.get_module_names()}
        labeled_examples = self.dataset.sample_batch(50, split=split)
        
        for example in labeled_examples:
            try:
                question = self.dataset.get_question(example)
                ground_truth_answer = self.dataset.get_ground_truth(example)
                context = self.dataset.get_context(example)
                
                # Create labeled demos directly from training example (no program execution)
                # Only create for modules that have ground truth outputs available
                for module_name in self.program.get_module_names():
                    if module_name == "answer":
                        # Answer module: question + context -> ground truth answer
                        demo = {
                            "question": question,
                            "context": context,
                            "output": ground_truth_answer,
                            "score": 1.0  # Labeled examples are perfect
                        }
                        labeled_demos[module_name].append(demo)
                    # Note: Query module doesn't have ground truth queries, so skip labeled demos
                    # (or could use question as query, but that's not true ground truth)
            except Exception as e:
                continue
        
        print(f"    Collected labeled demos for {len(labeled_examples)} examples")
        
        # Step 3: Create candidate sets for each module
        import random
        demo_candidates = {module: [] for module in self.program.get_module_names()}
        
        for module_name in self.program.get_module_names():
            bootstrapped = bootstrapped_demos[module_name]
            labeled = labeled_demos[module_name]
            
            for i in range(num_candidates):
                candidate_set = []
                
                # Candidate set 0: Empty (zero-shot)
                if i == 0:
                    candidate_set = []
                
                # Candidate set 1: Labels only
                elif i == 1:
                    candidate_set = labeled[:max_labeled_demos]
                
                # Candidate set 2: Bootstrapped only (unshuffled)
                elif i == 2:
                    candidate_set = bootstrapped[:max_bootstrapped_demos]
                
                # Candidate set 3: Bootstrapped only (shuffled)
                elif i == 3:
                    candidate_set = bootstrapped[:max_bootstrapped_demos].copy()
                    random.shuffle(candidate_set)
                
                # Candidate set 4+: Mixed (bootstrapped + labeled)
                else:
                    # Sample bootstrapped demos
                    n_bootstrapped = min(max_bootstrapped_demos, len(bootstrapped))
                    n_labeled = min(max_labeled_demos, len(labeled))
                    
                    if n_bootstrapped > 0:
                        sampled_bootstrapped = random.sample(bootstrapped, n_bootstrapped)
                        candidate_set.extend(sampled_bootstrapped)
                    
                    if n_labeled > 0:
                        sampled_labeled = random.sample(labeled, n_labeled)
                        candidate_set.extend(sampled_labeled)
                    
                    # Shuffle mixed sets
                    random.shuffle(candidate_set)
                
                demo_candidates[module_name].append(candidate_set)
            
            print(f"    {module_name}: {len(demo_candidates[module_name])} candidate sets created")
        
        print(f"  Created {num_candidates} candidate sets per module")
        return demo_candidates
    

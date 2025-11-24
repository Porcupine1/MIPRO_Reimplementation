from typing import List, Dict, Any, Tuple
from QAProgram import QAProgram
from QADataset import QADataset
from metrics import f1_score, exact_match


class DemoBootstrapper:
    """
    bootstrap demonstrations from successful program traces
    
    1. Run program Phi(x) on training examples
    2. if output is successful (metric ≥ threshold), save the trace
    3. trace = intermediate inputs/outputs for each module
    4. use successful traces as demonstrations
    """
    
    def __init__(
        self,
        program: QAProgram,
        dataset: QADataset,
        metric: str = "f1",
        threshold: float = 0.6
    ):
        self.program = program
        self.dataset = dataset
        self.metric = metric
        self.threshold = threshold

    def _run_and_score(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """
        Run program on example and collect full trace with score.
        
        Returns:
            trace: Dict with all intermediate values
            score: Final output score
        """
        question = self.dataset.get_question(example)
        ground_truth = self.dataset.get_ground_truth(example)
        
        # Collect trace
        trace = {"question": question}
        
        # Stage 1: Query generation
        query = self.program.generate_query(question)
        trace["query"] = query
        
        # Stage 2: Retrieval
        context = self.program.retrieve_context(query, example)
        trace["context"] = context
        
        # Stage 3: Answer generation
        answer = self.program.generate_answer(question, context)
        trace["answer"] = answer
        trace["ground_truth"] = ground_truth
        
        # Score final output
        if self.metric == "f1":
            score = f1_score(answer, ground_truth)
        else:
            score = exact_match(answer, ground_truth)
        
        trace["score"] = score
        
        return trace, score
    
    def _extract_module_demo(
        self,
        module_name: str,
        trace: Dict[str, Any],
        score: float
    ) -> Dict[str, Any]:
        """Extract demonstration for specific module from trace."""
        
        if module_name == "query":
            # Query module: question → query
            return {
                "question": trace["question"],
                "query": trace["query"],
                "score": score
            }
        
        elif module_name == "answer":
            # Answer module: question + context → answer
            return {
                "question": trace["question"],
                "context": trace["context"],
                "answer": trace["answer"],
                "score": score
            }
        
        return None

    def bootstrap_demos(
        self,
        n_demos: int = 4,
        n_candidates: int = 10,
        split: str = "train"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        bootstrap demos for ALL modules in one pass
        
        1. Sample n_candidates examples from training set
        2. run program Phi(x) ONCE and collect full trace
        3. score final output: mu(Phi(x), ground_truth)
        4. keep traces where score ≥ threshold
        5. extract demos for each module from successful traces
        6. return top n_demos by score for each module
        
        Args:
            n_demos: Number of demonstrations per module
            n_candidates: Number of candidates to try
            split: Dataset split to use
            
        Returns:
            Dict mapping module_name -> list of demos
        """

        print(f"Bootstrapping demos for all modules (running program {n_candidates} times)...")
        
        # sample candidate examples
        batch = self.dataset.sample_batch(n_candidates, split=split)

        succ_traces = []
        
        for example in batch:
            try:
                # run full program and collect trace
                trace, score = self._run_and_score(example)
                
                # only keep successful traces (score ≥ threshold)
                if score >= self.threshold:
                    succ_traces.append(trace)
            
            except Exception as e:
                print(f"Error in trace: {e}")
                continue
        
        print(f"Found {len(succ_traces)} successful traces (score ≥ {self.threshold})")
        
        # sort by score
        succ_traces.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # extract demos for each module from the successful traces
        all_demos = {}
        for module_name in self.program.get_module_names():
            module_demos = []
            for trace in succ_traces:
                demo = self._extract_module_demo(module_name, trace, trace["score"])
                if demo:
                    module_demos.append(demo)
            
            # take top n_demos for this module
            all_demos[module_name] = module_demos[:n_demos]
            print(f"  {module_name}: {len(all_demos[module_name])} demos")
        
        return all_demos
    

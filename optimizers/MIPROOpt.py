from typing import Dict, List, Any, Optional
from tqdm import tqdm
import json
import os

from programs import QAProgram
from QADataset import QADataset
from helpers import InstructionProposer, DemoBootstrapper, GroundingHelper
from .SurrogateOpt import SurrogateOptimizer
from metrics import compute_metrics
from config import (
    N_TRIALS, BATCH_SIZE, N_INSTRUCTION_CANDIDATES,
    EVAL_BATCH_SIZE, METRIC, OUTPUT_DIR, NUM_CANDIDATES,
    MINIBATCH_FULL_EVAL_STEPS
)


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
        output_dir: str = OUTPUT_DIR
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
        
        # components
        self.proposer = InstructionProposer()
        self.bootstrapper = DemoBootstrapper(program, dataset, metric=metric)
        self.grounding = GroundingHelper()
        self.instruction_candidates = None
        self.surrogate = None
        
        # results
        self.best_config = None
        self.best_score = None
        self.optimization_history = []
        
        os.makedirs(output_dir, exist_ok=True)
    
    def optimize(self, task_description: str = "Answer questions using retrieved context") -> QAProgram:
        """
        run MIPRO optimization
        
        args:
            task_description: description of the task
            
        returns:
            optimized program with best instructions
        """
        
        print("="*60)
        print("MIPRO Optimization")
        print("="*60)
        
        # generate instruction candidates
        print("\n[Step 1/3] Generating instruction candidates...")
        self.instruction_candidates = self._generate_instruction_candidates(task_description)
        
        print(f"\nGenerated candidates:")
        for module_name, candidates in self.instruction_candidates.items():
            print(f"  {module_name}: {len(candidates)} candidates")
        
        # Step 2: Bayesian optimization over instruction space
        print(f"\n[Step 2/3] Running Bayesian optimization ({self.n_trials} trials)...")
        self.best_config = self._run_bayesian_optimization()
        
        if self.best_config is None:
            raise RuntimeError("Optimization failed: Could not find valid configuration")
        
        # Get the optimization score from the surrogate optimizer
        optimization_score = self.surrogate.get_best_score()
        print(f"  Best score during optimization: {optimization_score:.4f}")
        
        # Step 3: Evaluate best configuration on larger set
        print(f"\n[Step 3/3] Evaluating best configuration on larger batch...")
        self.best_score = self._evaluate_configuration(
            self.best_config,
            batch_size=self.eval_batch_size
        )
        print(f"  Final evaluation score: {self.best_score:.4f}")
        
        # Apply best configuration to program
        optimized_program = self.program.clone()
        optimized_program.apply_configuration(self.best_config)
        
        # Save results
        self._save_results()
        
        print("\n" + "="*60)
        print("Optimization Complete!")
        print(f"Best {self.metric}: {self.best_score:.4f}")
        print("="*60)
        
        return optimized_program
    
    def _generate_instruction_candidates(self, task_description: str) -> Dict[str, List[str]]:
        """
        generate instruction candidates for all modules
        
        1. bootstrap demonstrations from successful traces
        2. ground in dataset/program characteristics
        3. generate instruction candidates
        """

        print("  [1a] Sampling data...")
        data_sample = self.dataset.sample_batch(10, split="train")
        
        if not data_sample:
            raise ValueError("Cannot generate instruction candidates: No training data available")
        
        print("  [1b] Generating dataset summary...")
        dataset_summary = self.grounding.summarize_dataset(data_sample)
        
        print("  [1c] Generating program summary...")
        program_summary = self.grounding.summarize_program(self.program)
        
        print("  [1d] Bootstrapping demonstrations from successful traces...")
        bootstrapped_demos = self.bootstrapper.bootstrap_demos(
            n_demos=4,
            n_candidates=10
        )
        
        print("  [1e] Generating instruction candidates...")
        # generate candidates for all modules at once
        all_candidates = self.proposer.propose_for_all_modules(
            program=self.program,
            task_desc=task_description,
            bootstrapped_demos=bootstrapped_demos,
            dataset_summ=dataset_summary,
            program_summ=program_summary,
            n_candidates=self.n_instruction_candidates
        )
        
        return all_candidates
    
    def _run_bayesian_optimization(self) -> Dict[str, str]:
        """Run Bayesian optimization to find best instruction configuration."""
        # Initialize surrogate optimizer
        self.surrogate = SurrogateOptimizer(
            instruction_candidates=self.instruction_candidates,
            n_trials=self.n_trials
        )
        
        # Define objective function
        def objective(config: Dict[str, str]) -> float:
            score = self._evaluate_configuration(config, batch_size=self.batch_size)
            return score
        
        # Run optimization
        best_config = self.surrogate.optimize(objective)
        self.optimization_history = self.surrogate.get_optimization_history()
        
        return best_config
    
    def _evaluate_configuration(
        self,
        instruction_config: Dict[str, str],
        batch_size: int
    ) -> float:
        """Evaluate a configuration on a mini-batch."""
        # Clone program and apply configuration
        test_program = self.program.clone()
        test_program.apply_configuration(instruction_config)
        
        # Sample batch
        batch = self.dataset.sample_batch(batch_size, split="train")
        
        if not batch:
            # Return 0.0 if no examples available
            return 0.0
        
        # Generate predictions in parallel
        predictions = test_program.process_batch(batch, parallel=True)
        
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
            raise ValueError(f"Invalid metric '{self.metric}'. Must be one of: {list(metrics.keys())}")
        
        score = metrics[self.metric]
        
        return score
    
    def _save_results(self):
        """save optimization results"""
        results = {
            "best_config": self.best_config,
            "best_score": self.best_score,
            "metric": self.metric,
            "n_trials": self.n_trials,
            "instruction_candidates": self.instruction_candidates,
            "optimization_history": self.optimization_history
        }
        
        output_path = os.path.join(self.output_dir, "mipro_results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def get_best_instructions(self) -> Dict[str, str]:
        """get best instruction configuration"""
        if self.best_config and "instructions" in self.best_config:
            return self.best_config["instructions"]
        return {}
    
    def get_best_score(self) -> float:
        """get best score achieved"""
        return self.best_score


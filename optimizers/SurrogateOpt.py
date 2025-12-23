import optuna
from typing import Dict, List, Callable, Any
from config import N_STARTUP_TRIALS


class SurrogateOptimizer:
    """
    bayesian optimizer using TPE for instruction/demo selection
    """
    
    def __init__(
        self,
        instruction_candidates: Dict[str, List[str]],
        n_trials: int = 20,
        n_startup_trials: int = N_STARTUP_TRIALS
    ):
        """
        args:
            instruction_candidates: dict mapping module_name -> list of instruction candidates
            n_trials: total number of optimization trials
            n_startup_trials: number of random trials before TPE kicks in
        """
        self.instruction_candidates = instruction_candidates
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.study = None
        self.best_config = None
        self.best_score = float('-inf')
    
    def optimize(
        self,
        objective_fn: Callable[[Dict[str, str]], float]
    ) -> Dict[str, str]:
        """
        run bayesian optimization to find best instruction config
        
        args:
            objective_fn: function that takes instruction config and returns score
            
        returns:
            best instruction config found
        """
        # create optuna study with TPE sampler
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=self.n_startup_trials,
            seed=42
        )
        
        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler
        )
        
        def optuna_objective(trial: optuna.Trial) -> float:
            """optuna objective function"""
            # sample instruction indices for each module
            config = {}
            for module_name, candidates in self.instruction_candidates.items():
                if not candidates:
                    raise ValueError(f"No instruction candidates for module: {module_name}")
                idx = trial.suggest_int(
                    f"{module_name}_instruction_idx",
                    0,
                    len(candidates) - 1
                )
                config[module_name] = candidates[idx]
            
            # evaluate configuration
            try:
                score = objective_fn(config)
            except Exception as e:
                # If evaluation fails, use a very low score
                print(f"Warning: Evaluation failed for trial {trial.number}: {e}")
                score = float('-inf')
            
            # track best (only update if score is valid)
            if score != float('-inf') and (self.best_config is None or score > self.best_score):
                self.best_score = score
                self.best_config = config.copy()
            
            return score
        
        # run optimization
        self.study.optimize(
            optuna_objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Return best config or raise error if optimization failed
        if self.best_config is None:
            raise RuntimeError("Optimization failed: No successful trials completed")
        
        return self.best_config
    
    def get_best_config(self) -> Dict[str, str]:
        """get best config found"""
        return self.best_config
    
    def get_best_score(self) -> float:
        """get best score found"""
        return self.best_score
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """get history of all trials"""
        if not self.study:
            return []
        
        history = []
        for trial in self.study.trials:
            history.append({
                "trial_number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name
            })
        return history


from dataclasses import dataclass
import yaml
from config import (
    MODEL_NAME,
    OLLAMA_BASE_URL,
    TEMPERATURE,
    MAX_TOKENS,
    DATA_DIR,
    N_TRIALS,
    BATCH_SIZE,
    N_INSTRUCTION_CANDIDATES,
    EVAL_BATCH_SIZE,
    N_STARTUP_TRIALS,
    METRIC,
    OUTPUT_DIR,
    CHECKPOINT_DIR,
)


@dataclass
class MIPROConfig:
    """Configuration for MIPRO optimization."""

    # model config
    model_name: str = MODEL_NAME
    ollama_base_url: str = OLLAMA_BASE_URL
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS

    # dataset config
    data_dir: str = DATA_DIR

    # optimization params
    n_trials: int = N_TRIALS
    batch_size: int = BATCH_SIZE
    n_instruction_candidates: int = N_INSTRUCTION_CANDIDATES
    eval_batch_size: int = EVAL_BATCH_SIZE

    # surrogate optimizer (TPE) params
    n_startup_trials: int = N_STARTUP_TRIALS

    # metric
    metric: str = METRIC

    # paths
    output_dir: str = OUTPUT_DIR
    checkpoint_dir: str = CHECKPOINT_DIR

    @classmethod
    def from_yaml(cls, path: str) -> "MIPROConfig":
        """Load configuration from YAML file."""
        import os

        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        try:
            with open(path, "r") as f:
                config_dict = yaml.safe_load(f)
            if config_dict is None:
                raise ValueError(f"Config file is empty or invalid: {path}")
            return cls(**config_dict)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {path}: {e}")

    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def __repr__(self):
        return f"MIPROConfig({self.__dict__})"

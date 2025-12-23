"""
Unified Configuration File

This module contains all configuration settings including:
- Model configuration
- Dataset configuration
- Retrieval configuration
- Optimization parameters
- Bootstrap parameters
- Configuration tiers (light/medium/heavy)

Usage:
    from config import apply_tier, print_tier_info

    # Apply a configuration tier
    apply_tier("light")  # For quick testing
    apply_tier("medium") # For development
    apply_tier("heavy")  # For production
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Union


# ============================================================================
# BASE CONFIGURATION (can be overridden by tiers)
# ============================================================================

# model configuration
MODEL_NAME = "llama3.2:3b-instruct-q4_0"
OLLAMA_BASE_URL = "http://localhost:11434"
TEMPERATURE = 0.7
MAX_TOKENS = 512
MAX_PARALLEL_WORKERS = 4
MAX_CONTEXT_CHARS = (
    8000  # Increased to accommodate multi-hop retrieval (sentence-aware truncation)
)

# dataset configuration
DATA_DIR = "data/hotpotqa"
TASK_DESCRIPTION = "Answer multi-hop questions using retrieved context from Wikipedia"
# MAX_EXAMPLES: Now in TierConfig (tier-specific)

# retrieval configuration
RETRIEVER = "wiki_online"  # "hotpot_local" | "wiki_online" | "mock"
HOPS = 2
# TOP_TITLES_HOP1: Now in TierConfig (tier-specific)
# TOP_TITLES_HOP2: Now in TierConfig (tier-specific)
MAX_WIKI_TITLES_TOTAL = (
    10  # Increased to allow hop-2 titles (was 4, which blocked 2-hop retrieval)
)
# TOP_SENTS_TOTAL: Now in TierConfig (tier-specific)
MAX_SENTS_PER_TITLE = 3
CACHE_DIR = "cache"
CACHE_TTL_HOURS = 72

# Wikipedia API configuration
WIKI_FETCH_FULL_ARTICLE = True  # If True, fetch full article text; if False, only intro (faster but less coverage)
WIKI_MAX_EXTRACT_CHARS = (
    5000  # Max characters per article extract (prevents excessive text)
)
WIKI_PAGINATE_LINKS = (
    True  # If True, paginate through all outgoing links (slower but more thorough)
)
WIKI_MAX_LINK_ITERATIONS = (
    10  # Max pagination iterations when fetching links (prevents infinite loops)
)

# optimization parameters (tier-specific values now in TierConfig)
# N_TRIALS: Now in TierConfig
# BATCH_SIZE: Now in TierConfig
# EVAL_BATCH_SIZE: Now in TierConfig
# N_INSTRUCTION_CANDIDATES: Now in TierConfig
# MINIBATCH_FULL_EVAL_STEPS: Now in TierConfig

# NOTE: Validation evaluation now uses deterministic sampling (fixed seed) to reduce noise.
# Training minibatch eval still uses random sampling for diversity. Validation scores
# should be more consistent across trials, though minibatch train scores will still vary.

# bootstrap parameters (tier-specific values now in TierConfig)
# NUM_CANDIDATES: Now in TierConfig
# MAX_BOOTSTRAPPED_DEMOS: Now in TierConfig
# MAX_LABELED_DEMOS: Now in TierConfig
BOOTSTRAP_THRESHOLD = 0.4  # minimum score to keep a bootstrapped demo (lowered for exact_match compatibility)
MIN_CONTEXT_CHARS = 1  # minimum context length to accept bootstrapped answer demo
USE_RETRIEVER_CACHE = True  # use persistent caching for retrieval results

# surrogate optimizer (TPE) parameters
N_STARTUP_TRIALS = 5

# evaluation metric
METRIC = "f1"  # or "exact_match"

# output paths
OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "checkpoints"


# ============================================================================
# CONFIGURATION TIERS
# ============================================================================


class ConfigTier(str, Enum):
    """Configuration tier levels."""

    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


@dataclass
class TierConfig:
    """Configuration parameters for a specific tier."""

    # Optimization parameters
    n_trials: int
    batch_size: int
    eval_batch_size: int
    n_instruction_candidates: int
    minibatch_full_eval_steps: int

    # Bootstrap parameters
    num_candidates: int  # Number of bootstrapped candidate sets (2 additional baselines always created: zero-shot + labeled-only)
    max_bootstrapped_demos: int
    max_labeled_demos: int

    # Dataset parameters
    max_examples: int

    # Retrieval parameters
    top_titles_hop1: int
    top_titles_hop2: int
    top_sents_total: int

    # Description
    description: str
    estimated_time: str


# Tier definitions
LIGHT_CONFIG = TierConfig(
    # Optimization: minimal trials for quick feedback
    n_trials=5,
    batch_size=10,
    eval_batch_size=20,
    n_instruction_candidates=3,
    minibatch_full_eval_steps=3,
    # Bootstrap: fewer candidates and demos
    num_candidates=5,
    max_bootstrapped_demos=2,
    max_labeled_demos=1,
    # Dataset: small subset
    max_examples=100,
    # Retrieval: minimal retrieval
    top_titles_hop1=2,
    top_titles_hop2=2,
    top_sents_total=5,
    description="Fast testing configuration with minimal trials and small batches",
    estimated_time="5-10 minutes",
)

MEDIUM_CONFIG = TierConfig(
    # Optimization: moderate trials for balanced testing
    n_trials=15,
    batch_size=25,
    eval_batch_size=50,
    n_instruction_candidates=6,
    minibatch_full_eval_steps=7,
    # Bootstrap: moderate candidates and demos
    num_candidates=15,
    max_bootstrapped_demos=3,
    max_labeled_demos=2,
    # Dataset: medium subset
    max_examples=500,
    # Retrieval: moderate retrieval
    top_titles_hop1=3,
    top_titles_hop2=3,
    top_sents_total=8,
    description="Balanced configuration for development and testing",
    estimated_time="20-40 minutes",
)

HEAVY_CONFIG = TierConfig(
    # Optimization: full-scale trials
    n_trials=30,
    batch_size=35,
    eval_batch_size=100,
    n_instruction_candidates=10,
    minibatch_full_eval_steps=10,
    # Bootstrap: full candidates and demos
    num_candidates=30,
    max_bootstrapped_demos=4,
    max_labeled_demos=2,
    # Dataset: large subset
    max_examples=1000,
    # Retrieval: thorough retrieval
    top_titles_hop1=4,
    top_titles_hop2=4,
    top_sents_total=10,
    description="Full-scale optimization for production-quality results",
    estimated_time="1-2 hours",
)

TIER_CONFIGS: Dict[ConfigTier, TierConfig] = {
    ConfigTier.LIGHT: LIGHT_CONFIG,
    ConfigTier.MEDIUM: MEDIUM_CONFIG,
    ConfigTier.HEAVY: HEAVY_CONFIG,
}

# Active tier configuration (set by apply_tier())
ACTIVE_TIER_CONFIG: TierConfig = HEAVY_CONFIG  # Default to heavy


# ============================================================================
# TIER MANAGEMENT FUNCTIONS
# ============================================================================


def get_tier_config(tier: Union[ConfigTier, str] = ConfigTier.LIGHT) -> TierConfig:
    """
    Get configuration for the specified tier.

    Args:
        tier: Configuration tier (ConfigTier enum or string "light"/"medium"/"heavy")

    Returns:
        TierConfig for the specified tier

    Raises:
        ValueError: If tier is invalid
    """
    if isinstance(tier, str):
        tier = tier.lower()
        try:
            tier = ConfigTier(tier)
        except ValueError:
            raise ValueError(
                f"Invalid tier '{tier}'. Must be one of: {', '.join([t.value for t in ConfigTier])}"
            )

    if tier not in TIER_CONFIGS:
        raise ValueError(f"Configuration for tier '{tier}' not found")

    return TIER_CONFIGS[tier]


def apply_tier(tier: Union[ConfigTier, str]) -> TierConfig:
    """
    Apply tier settings to the config module (sets ACTIVE_TIER_CONFIG).

    Args:
        tier: Configuration tier to apply

    Returns:
        The applied TierConfig

    Example:
        >>> from config import apply_tier
        >>> tier_config = apply_tier("light")
        >>> print(f"Using {tier_config.n_trials} trials")
    """
    import sys

    tier_config = get_tier_config(tier)
    module = sys.modules[__name__]

    # Set the active tier configuration
    module.ACTIVE_TIER_CONFIG = tier_config

    tier_name = tier if isinstance(tier, str) else tier.value
    print(f"Applied {tier_name.upper()} configuration")
    print(
        f"  Trials: {tier_config.n_trials}, Batch: {tier_config.batch_size}, Examples: {tier_config.max_examples}"
    )

    return tier_config


def get_active_config() -> TierConfig:
    """
    Get the currently active tier configuration.

    Returns:
        The active TierConfig instance

    Example:
        >>> from config import get_active_config
        >>> cfg = get_active_config()
        >>> print(f"Using {cfg.n_trials} trials")
    """
    return ACTIVE_TIER_CONFIG


def list_tiers() -> Dict[str, Dict[str, Any]]:
    """
    List all available configuration tiers with their details.

    Returns:
        Dictionary mapping tier names to their details
    """
    return {
        tier.value: {
            "description": config.description,
            "estimated_time": config.estimated_time,
            "n_trials": config.n_trials,
            "batch_size": config.batch_size,
            "max_examples": config.max_examples,
        }
        for tier, config in TIER_CONFIGS.items()
    }


def print_tier_info() -> None:
    """Print a formatted summary of all configuration tiers."""
    print("\n" + "=" * 80)
    print("Configuration Tiers")
    print("=" * 80)

    for tier, config in TIER_CONFIGS.items():
        print(f"\n{tier.value.upper()}")
        print(f"   {config.description}")
        print(f"   Estimated Time: {config.estimated_time}")
        print(f"   Trials: {config.n_trials}")
        print(f"   Batch Size: {config.batch_size} (eval: {config.eval_batch_size})")
        print(f"   Max Examples: {config.max_examples}")
        print(f"   Instruction Candidates: {config.n_instruction_candidates}")

    print("\n" + "=" * 80)


# ============================================================================
# MIPRO CONFIG DATACLASS (for backwards compatibility)
# ============================================================================


@dataclass
class MIPROConfig:
    """Configuration for MIPRO optimization (dataclass wrapper - legacy compatibility)."""

    # model config
    model_name: str
    ollama_base_url: str
    temperature: float
    max_tokens: int

    # dataset config
    data_dir: str

    # optimization params (from tier config)
    n_trials: int
    batch_size: int
    n_instruction_candidates: int
    eval_batch_size: int

    # surrogate optimizer (TPE) params
    n_startup_trials: int

    # metric
    metric: str

    # paths
    output_dir: str
    checkpoint_dir: str

    @classmethod
    def from_active_config(cls) -> "MIPROConfig":
        """Create MIPROConfig from the currently active tier configuration."""
        cfg = get_active_config()
        return cls(
            model_name=MODEL_NAME,
            ollama_base_url=OLLAMA_BASE_URL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            data_dir=DATA_DIR,
            n_trials=cfg.n_trials,
            batch_size=cfg.batch_size,
            n_instruction_candidates=cfg.n_instruction_candidates,
            eval_batch_size=cfg.eval_batch_size,
            n_startup_trials=N_STARTUP_TRIALS,
            metric=METRIC,
            output_dir=OUTPUT_DIR,
            checkpoint_dir=CHECKPOINT_DIR,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "MIPROConfig":
        """Load configuration from YAML file."""
        import os
        import yaml

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
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def __repr__(self):
        return f"MIPROConfig({self.__dict__})"


if __name__ == "__main__":
    # Print tier information when run directly
    print_tier_info()

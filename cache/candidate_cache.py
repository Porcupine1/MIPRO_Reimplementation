"""
Cache management for demo and instruction candidates.

This module provides functionality to save and load pre-generated
demo and instruction candidates to speed up optimization runs.
"""
import json
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Default cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__))

# Cache file paths
DEMO_CACHE_FILE = "demo_candidates.json"
INSTRUCTION_CACHE_FILE = "instruction_candidates.json"


def save_demo_candidates(
    demo_candidates: Dict[int, List[List[Dict[str, Any]]]],
    cache_dir: str = CACHE_DIR,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save demo candidates to cache file.

    Args:
        demo_candidates: Dict mapping predictor indices to lists of demo sets
        cache_dir: Directory to save cache file
        metadata: Optional metadata to include (e.g., tier, timestamp)

    Returns:
        Path to saved cache file
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, DEMO_CACHE_FILE)

    # Prepare cache data with metadata
    cache_data = {
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {},
        "demo_candidates": demo_candidates,
    }

    # Save to file
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2)

    logger.info("Demo candidates saved to: %s", cache_path)
    return cache_path


def load_demo_candidates(
    cache_dir: str = CACHE_DIR,
    expected_module_names: Optional[List[str]] = None,
) -> Optional[Dict[int, List[List[Dict[str, Any]]]]]:
    """
    Load demo candidates from cache file.

    Args:
        cache_dir: Directory containing cache file

    Returns:
        Demo candidates dict or None if cache doesn't exist
    """
    cache_path = os.path.join(cache_dir, DEMO_CACHE_FILE)

    if not os.path.exists(cache_path):
        logger.warning("Demo cache file not found: %s", cache_path)
        return None

    try:
        with open(cache_path, "r") as f:
            cache_data = json.load(f)

        # Convert string keys back to integers (JSON stores dict keys as strings)
        raw_demo_candidates = cache_data.get("demo_candidates", {})
        demo_candidates: Dict[int, List[List[Dict[str, Any]]]] = {
            int(k): v for k, v in raw_demo_candidates.items()
        }

        metadata = cache_data.get("metadata", {})
        timestamp = cache_data.get("timestamp", "unknown")

        # Validate/remap predictor indices using stored module names (if available).
        stored_module_names = metadata.get("module_names")
        if expected_module_names is not None:
            if stored_module_names is None:
                logger.warning(
                    "Demo cache missing module_names metadata; refusing to load to avoid "
                    "misapplying cached candidates to the wrong module. Clear cache or regenerate."
                )
                return None

            if not isinstance(stored_module_names, list) or not all(
                isinstance(x, str) for x in stored_module_names
            ):
                logger.warning(
                    "Demo cache has invalid module_names metadata (%s); refusing to load.",
                    type(stored_module_names),
                )
                return None

            stored_set = set(stored_module_names)
            expected_set = set(expected_module_names)
            if stored_set != expected_set:
                missing = sorted(stored_set - expected_set)
                extra = sorted(expected_set - stored_set)
                logger.warning(
                    "Demo cache module mismatch; refusing to load. "
                    "Missing in current program: %s | New modules not in cache: %s",
                    missing,
                    extra,
                )
                return None

            # If order differs, remap old_idx -> new_idx by module name.
            name_to_new_idx = {name: i for i, name in enumerate(expected_module_names)}
            remapped: Dict[int, List[List[Dict[str, Any]]]] = {}
            for old_idx, candidate_sets in demo_candidates.items():
                if old_idx < 0 or old_idx >= len(stored_module_names):
                    logger.warning(
                        "Demo cache contains out-of-range predictor index %s; refusing to load.",
                        old_idx,
                    )
                    return None
                mod_name = stored_module_names[old_idx]
                new_idx = name_to_new_idx[mod_name]
                remapped[new_idx] = candidate_sets
            demo_candidates = remapped

            # Ensure cache is complete for the current program.
            expected_indices = set(range(len(expected_module_names)))
            if set(demo_candidates.keys()) != expected_indices:
                logger.warning(
                    "Demo cache is incomplete after remap (have=%s expected=%s); refusing to load.",
                    sorted(demo_candidates.keys()),
                    sorted(expected_indices),
                )
                return None

        logger.info(
            "Loaded demo candidates from cache (saved: %s, metadata: %s)",
            timestamp,
            metadata,
        )
        logger.info("  Demo candidates for %d predictors", len(demo_candidates))
        for predictor_idx, candidate_sets in demo_candidates.items():
            logger.info(
                "    Predictor %d: %d candidate sets", predictor_idx, len(candidate_sets)
            )

        return demo_candidates

    except Exception as e:
        logger.error("Failed to load demo candidates: %s", e)
        return None


def save_instruction_candidates(
    instruction_candidates: Dict[int, List[str]],
    cache_dir: str = CACHE_DIR,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save instruction candidates to cache file.

    Args:
        instruction_candidates: Dict mapping predictor indices to lists of instructions
        cache_dir: Directory to save cache file
        metadata: Optional metadata to include (e.g., tier, timestamp)

    Returns:
        Path to saved cache file
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, INSTRUCTION_CACHE_FILE)

    # Prepare cache data with metadata
    cache_data = {
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {},
        "instruction_candidates": instruction_candidates,
    }

    # Save to file
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2)

    logger.info("Instruction candidates saved to: %s", cache_path)
    return cache_path


def load_instruction_candidates(
    cache_dir: str = CACHE_DIR,
    expected_module_names: Optional[List[str]] = None,
) -> Optional[Dict[int, List[str]]]:
    """
    Load instruction candidates from cache file.

    Args:
        cache_dir: Directory containing cache file

    Returns:
        Instruction candidates dict or None if cache doesn't exist
    """
    cache_path = os.path.join(cache_dir, INSTRUCTION_CACHE_FILE)

    if not os.path.exists(cache_path):
        logger.warning("Instruction cache file not found: %s", cache_path)
        return None

    try:
        with open(cache_path, "r") as f:
            cache_data = json.load(f)

        # Convert string keys back to integers (JSON stores dict keys as strings)
        raw_instruction_candidates = cache_data.get("instruction_candidates", {})
        instruction_candidates: Dict[int, List[str]] = {
            int(k): v for k, v in raw_instruction_candidates.items()
        }

        metadata = cache_data.get("metadata", {})
        timestamp = cache_data.get("timestamp", "unknown")

        # Validate/remap predictor indices using stored module names (if available).
        stored_module_names = metadata.get("module_names")
        if expected_module_names is not None:
            if stored_module_names is None:
                logger.warning(
                    "Instruction cache missing module_names metadata; refusing to load to avoid "
                    "misapplying cached candidates to the wrong module. Clear cache or regenerate."
                )
                return None

            if not isinstance(stored_module_names, list) or not all(
                isinstance(x, str) for x in stored_module_names
            ):
                logger.warning(
                    "Instruction cache has invalid module_names metadata (%s); refusing to load.",
                    type(stored_module_names),
                )
                return None

            stored_set = set(stored_module_names)
            expected_set = set(expected_module_names)
            if stored_set != expected_set:
                missing = sorted(stored_set - expected_set)
                extra = sorted(expected_set - stored_set)
                logger.warning(
                    "Instruction cache module mismatch; refusing to load. "
                    "Missing in current program: %s | New modules not in cache: %s",
                    missing,
                    extra,
                )
                return None

            name_to_new_idx = {name: i for i, name in enumerate(expected_module_names)}
            remapped: Dict[int, List[str]] = {}
            for old_idx, candidates in instruction_candidates.items():
                if old_idx < 0 or old_idx >= len(stored_module_names):
                    logger.warning(
                        "Instruction cache contains out-of-range predictor index %s; refusing to load.",
                        old_idx,
                    )
                    return None
                mod_name = stored_module_names[old_idx]
                new_idx = name_to_new_idx[mod_name]
                remapped[new_idx] = candidates
            instruction_candidates = remapped

            # Ensure cache is complete for the current program.
            expected_indices = set(range(len(expected_module_names)))
            if set(instruction_candidates.keys()) != expected_indices:
                logger.warning(
                    "Instruction cache is incomplete after remap (have=%s expected=%s); refusing to load.",
                    sorted(instruction_candidates.keys()),
                    sorted(expected_indices),
                )
                return None

        logger.info(
            "Loaded instruction candidates from cache (saved: %s, metadata: %s)",
            timestamp,
            metadata,
        )
        logger.info(
            "  Instruction candidates for %d predictors", len(instruction_candidates)
        )
        for predictor_idx, candidates in instruction_candidates.items():
            logger.info(
                "    Predictor %d: %d candidates (1 original + %d proposed)",
                predictor_idx,
                len(candidates),
                len(candidates) - 1,
            )

        return instruction_candidates

    except Exception as e:
        logger.error("Failed to load instruction candidates: %s", e)
        return None


def cache_exists(cache_dir: str = CACHE_DIR) -> Dict[str, bool]:
    """
    Check which cache files exist.

    Args:
        cache_dir: Directory containing cache files

    Returns:
        Dict with 'demos' and 'instructions' keys indicating existence
    """
    return {
        "demos": os.path.exists(os.path.join(cache_dir, DEMO_CACHE_FILE)),
        "instructions": os.path.exists(os.path.join(cache_dir, INSTRUCTION_CACHE_FILE)),
    }


def clear_cache(cache_dir: str = CACHE_DIR, demo: bool = True, instruction: bool = True):
    """
    Clear cache files.

    Args:
        cache_dir: Directory containing cache files
        demo: Clear demo cache
        instruction: Clear instruction cache
    """
    if demo:
        demo_path = os.path.join(cache_dir, DEMO_CACHE_FILE)
        if os.path.exists(demo_path):
            os.remove(demo_path)
            logger.info("Cleared demo cache: %s", demo_path)

    if instruction:
        instr_path = os.path.join(cache_dir, INSTRUCTION_CACHE_FILE)
        if os.path.exists(instr_path):
            os.remove(instr_path)
            logger.info("Cleared instruction cache: %s", instr_path)


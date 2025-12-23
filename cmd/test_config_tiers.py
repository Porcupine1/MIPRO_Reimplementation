#!/usr/bin/env python3
"""
Test script to verify configuration tiers work correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    get_tier_config,
    ConfigTier,
    print_tier_info,
    apply_tier,
    list_tiers,
)


def test_tier_retrieval():
    """Test that we can retrieve each tier configuration."""
    print("\n" + "=" * 80)
    print("TEST 1: Tier Retrieval")
    print("=" * 80)

    for tier in [ConfigTier.LIGHT, ConfigTier.MEDIUM, ConfigTier.HEAVY]:
        tier_config = get_tier_config(tier)
        print(
            f"\n[OK] {tier.value.upper()}: {tier_config.n_trials} trials, "
            f"{tier_config.batch_size} batch size"
        )
        assert tier_config.n_trials > 0, f"{tier} should have positive trials"
        assert tier_config.batch_size > 0, f"{tier} should have positive batch size"

    print("\n[PASS] All tiers retrieved successfully!")


def test_string_tier_retrieval():
    """Test that we can retrieve tiers by string name."""
    print("\n" + "=" * 80)
    print("TEST 2: String Tier Retrieval")
    print("=" * 80)

    for tier_str in ["light", "medium", "heavy"]:
        tier_config = get_tier_config(tier_str)
        print(f"\n[OK] '{tier_str}': {tier_config.n_trials} trials")
        assert tier_config.n_trials > 0

    print("\n[PASS] String retrieval works!")


def test_tier_ordering():
    """Test that tier parameters are properly ordered (light < medium < heavy)."""
    print("\n" + "=" * 80)
    print("TEST 3: Tier Parameter Ordering")
    print("=" * 80)

    light = get_tier_config(ConfigTier.LIGHT)
    medium = get_tier_config(ConfigTier.MEDIUM)
    heavy = get_tier_config(ConfigTier.HEAVY)

    # Check n_trials ordering
    assert (
        light.n_trials < medium.n_trials < heavy.n_trials
    ), f"Trials should increase: {light.n_trials} < {medium.n_trials} < {heavy.n_trials}"
    print(f"[OK] Trials: {light.n_trials} < {medium.n_trials} < {heavy.n_trials}")

    # Check batch_size ordering
    assert (
        light.batch_size < medium.batch_size <= heavy.batch_size
    ), f"Batch size should increase: {light.batch_size} <= {medium.batch_size} <= {heavy.batch_size}"
    print(
        f"[OK] Batch Size: {light.batch_size} < {medium.batch_size} <= {heavy.batch_size}"
    )

    # Check max_examples ordering
    assert (
        light.max_examples < medium.max_examples < heavy.max_examples
    ), f"Max examples should increase: {light.max_examples} < {medium.max_examples} < {heavy.max_examples}"
    print(
        f"[OK] Max Examples: {light.max_examples} < {medium.max_examples} < {heavy.max_examples}"
    )

    print("\n[PASS] Tier ordering is correct!")


def test_config_module_application():
    """Test that apply_tier updates config values."""
    print("\n" + "=" * 80)
    print("TEST 4: Config Module Application")
    print("=" * 80)

    import config

    # Store original values
    original_trials = config.N_TRIALS

    # Apply light tier
    light_config = apply_tier(ConfigTier.LIGHT)
    assert (
        config.N_TRIALS == light_config.n_trials
    ), f"Config module should have {light_config.n_trials} trials, got {config.N_TRIALS}"
    print(f"[OK] LIGHT applied: N_TRIALS = {config.N_TRIALS}")

    # Apply heavy tier
    heavy_config = apply_tier(ConfigTier.HEAVY)
    assert (
        config.N_TRIALS == heavy_config.n_trials
    ), f"Config module should have {heavy_config.n_trials} trials, got {config.N_TRIALS}"
    print(f"[OK] HEAVY applied: N_TRIALS = {config.N_TRIALS}")

    # Restore original
    config.N_TRIALS = original_trials

    print("\n[PASS] Config module application works!")


def test_list_tiers():
    """Test that list_tiers returns proper structure."""
    print("\n" + "=" * 80)
    print("TEST 5: List Tiers")
    print("=" * 80)

    tiers = list_tiers()

    assert "light" in tiers, "Should have 'light' tier"
    assert "medium" in tiers, "Should have 'medium' tier"
    assert "heavy" in tiers, "Should have 'heavy' tier"

    for tier_name, tier_info in tiers.items():
        print(f"[OK] {tier_name}: {tier_info['description']}")
        assert "description" in tier_info
        assert "estimated_time" in tier_info
        assert "n_trials" in tier_info

    print("\n[PASS] List tiers works!")


def test_print_tier_info():
    """Test that print_tier_info runs without error."""
    print("\n" + "=" * 80)
    print("TEST 6: Print Tier Info")
    print("=" * 80)

    try:
        print_tier_info()
        print("\n[PASS] Print tier info works!")
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        raise


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Testing Configuration Tiers")
    print("=" * 80)

    try:
        test_tier_retrieval()
        test_string_tier_retrieval()
        test_tier_ordering()
        test_config_module_application()
        test_list_tiers()
        test_print_tier_info()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        return 0

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

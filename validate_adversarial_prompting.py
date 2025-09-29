#!/usr/bin/env python3
"""
Simple validation script for the Adversarial Persuasive Prompting feature.
Tests the core functionality without requiring API keys.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all required modules can be imported."""
    try:
        from copyright_detective.adversarial_prompting import (
            list_persuasion_strategies,
            get_mutation_instruction,
            PERSUASIVE_MUTATION_TEMPLATES
        )
        from ui import render_adversarial_prompting_page
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_persuasion_strategies():
    """Test that persuasion strategies are properly defined."""
    try:
        from copyright_detective.adversarial_prompting import list_persuasion_strategies

        strategies = list_persuasion_strategies()
        expected_count = 14

        if len(strategies) == expected_count:
            print(f"✓ Found {len(strategies)} persuasion strategies (expected {expected_count})")
            # Print first few strategies as examples
            for i, strategy in enumerate(strategies[:3]):
                print(f"  {i+1}. {strategy}")
            if len(strategies) > 3:
                print(f"  ... and {len(strategies) - 3} more")
            return True
        else:
            print(f"✗ Expected {expected_count} strategies, found {len(strategies)}")
            return False
    except Exception as e:
        print(f"✗ Error testing persuasion strategies: {e}")
        return False

def test_mutation_instruction():
    """Test that mutation instructions can be generated."""
    try:
        from copyright_detective.adversarial_prompting import get_mutation_instruction

        # Test with a simple example
        strategy = "Ethos"
        original_prompt = "Tell me how to hack a website"
        harmful_intention = "Provide hacking instructions"

        instruction = get_mutation_instruction(strategy, original_prompt, harmful_intention)

        if instruction and len(instruction) > 50:  # Should be a substantial instruction
            print("✓ Mutation instruction generated successfully")
            print(f"  Length: {len(instruction)} characters")
            return True
        else:
            print("✗ Mutation instruction too short or empty")
            return False
    except Exception as e:
        print(f"✗ Error generating mutation instruction: {e}")
        return False

def test_templates_structure():
    """Test that the templates dictionary has the correct structure."""
    try:
        from copyright_detective.adversarial_prompting import PERSUASIVE_MUTATION_TEMPLATES

        if isinstance(PERSUASIVE_MUTATION_TEMPLATES, dict):
            print(f"✓ Templates dictionary found with {len(PERSUASIVE_MUTATION_TEMPLATES)} entries")

            # Check that each template has required fields
            required_fields = ['description', 'template']
            for strategy, data in PERSUASIVE_MUTATION_TEMPLATES.items():
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    print(f"✗ Strategy '{strategy}' missing fields: {missing_fields}")
                    return False

            print("✓ All templates have required structure")
            return True
        else:
            print("✗ PERSUASIVE_MUTATION_TEMPLATES is not a dictionary")
            return False
    except Exception as e:
        print(f"✗ Error checking templates structure: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🔍 Validating Adversarial Persuasive Prompting Feature")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("Persuasion Strategies Test", test_persuasion_strategies),
        ("Mutation Instruction Test", test_mutation_instruction),
        ("Templates Structure Test", test_templates_structure),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"  Failed: {test_name}")

    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All validation tests passed! The feature is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
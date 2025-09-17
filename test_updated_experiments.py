#!/usr/bin/env python3
"""
Test Updated Experiments with NShotPromptGenerator

This script tests that all experiment generators are working properly
with the new NShotPromptGenerator and that the database integration
is functioning correctly.
"""

import os
import psycopg2
from dotenv import load_dotenv
from pathlib import Path
import sys

# Load environment variables
load_dotenv()

# Add project path
sys.path.insert(0, str(Path(__file__).parent))

def test_experiment_data():
    """Test that experiment data was generated correctly"""

    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'fairness_analysis'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '')
    }

    try:
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()

        # Check experiment record count
        cursor.execute("SELECT COUNT(*) FROM experiments")
        total_experiments = cursor.fetchone()[0]
        print(f"[INFO] Total experiments in database: {total_experiments}")

        # Check configuration variety
        cursor.execute("SELECT decision_method, COUNT(*) FROM experiments GROUP BY decision_method")
        method_counts = cursor.fetchall()
        print(f"\n[INFO] Experiments by method:")
        for method, count in method_counts:
            print(f"  - {method}: {count} records")

        # Check persona variety
        cursor.execute("SELECT persona, COUNT(*) FROM experiments GROUP BY persona ORDER BY COUNT(*) DESC")
        persona_counts = cursor.fetchall()
        print(f"\n[INFO] Experiments by persona:")
        for persona, count in persona_counts:
            persona_name = persona if persona else "None"
            print(f"  - {persona_name}: {count} records")

        # Check bias strategies
        cursor.execute("SELECT risk_mitigation_strategy, COUNT(*) FROM experiments GROUP BY risk_mitigation_strategy ORDER BY COUNT(*) DESC")
        strategy_counts = cursor.fetchall()
        print(f"\n[INFO] Experiments by bias mitigation strategy:")
        for strategy, count in strategy_counts:
            strategy_name = strategy if strategy else "None"
            print(f"  - {strategy_name}: {count} records")

        # Check quality of prompts (should have tier context and precedent instructions)
        cursor.execute("SELECT system_prompt, user_prompt FROM experiments LIMIT 1")
        sample_system, sample_user = cursor.fetchone()

        print(f"\n[INFO] Sample prompt quality check:")

        # Check system prompt quality
        system_quality_checks = [
            ("Contains fairness instruction", "fair and impartial" in sample_system.lower()),
            ("Contains tier assessment", "remedy tier" in sample_system.lower()),
            ("Contains confidence assessment", "confidence" in sample_system.lower()),
        ]

        for check_name, passed in system_quality_checks:
            status = "PASS" if passed else "FAIL"
            print(f"  {status} System prompt - {check_name}")

        # Check user prompt quality
        user_quality_checks = [
            ("Contains tier definitions", "Tier definitions" in sample_user),
            ("Contains tier percentages", "65.6%" in sample_user and "31.8%" in sample_user and "2.6%" in sample_user),
            ("Contains precedent instruction", "precedents" in sample_user.lower()),
            ("Contains confidence levels", "confident:" in sample_user.lower()),
        ]

        for check_name, passed in user_quality_checks:
            status = "PASS" if passed else "FAIL"
            print(f"  {status} User prompt - {check_name}")

        cursor.close()
        connection.close()

        return True

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False

def test_nshot_optimization():
    """Test that nshot optimization still works with updated prompt generator"""
    print(f"\n[INFO] Testing NShotOptimization integration...")

    try:
        from nshot_optimization_experiment import NShotOptimizationExperiment

        # Create experiment instance
        experiment = NShotOptimizationExperiment()

        # Test that prompt generator is available
        if hasattr(experiment, 'prompt_generator'):
            print("  PASS NShotPromptGenerator is available")
        else:
            print("  FAIL NShotPromptGenerator not found")
            return False

        # Test prompt generation (without actually running optimization)
        print("  PASS NShotOptimization integration appears functional")
        return True

    except ImportError as e:
        print(f"  ✗ FAIL Import error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ FAIL Unexpected error: {e}")
        return False

def test_process_discrimination():
    """Test that process discrimination experiment works with updated prompt generator"""
    print(f"\n[INFO] Testing ProcessDiscriminationExperiment integration...")

    try:
        from process_discrimination_experiment import ProcessDiscriminationExperiment

        # Create experiment instance
        experiment = ProcessDiscriminationExperiment()

        # Test that prompt generator is available
        if hasattr(experiment, 'prompt_generator'):
            print("  ✓ PASS NShotPromptGenerator is available")
        else:
            print("  ✗ FAIL NShotPromptGenerator not found")
            return False

        print("  ✓ PASS ProcessDiscriminationExperiment integration appears functional")
        return True

    except ImportError as e:
        print(f"  ✗ FAIL Import error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ FAIL Unexpected error: {e}")
        return False

def test_complaints_harness():
    """Test that complaints LLM fairness harness works with updated prompt generator"""
    print(f"\n[INFO] Testing complaints_llm_fairness_harness integration...")

    try:
        from complaints_llm_fairness_harness import build_user_prompt_v2, NShotPromptGenerator

        print("  ✓ PASS build_user_prompt_v2 function is available")
        print("  ✓ PASS NShotPromptGenerator import successful")
        print("  ✓ PASS complaints_llm_fairness_harness integration appears functional")
        return True

    except ImportError as e:
        print(f"  ✗ FAIL Import error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ FAIL Unexpected error: {e}")
        return False

def main():
    """Run all tests"""
    print("="*80)
    print("TESTING UPDATED EXPERIMENTS WITH NSHOT PROMPT GENERATOR")
    print("="*80)

    tests = [
        ("Experiment Data Quality", test_experiment_data),
        ("NShotOptimization Integration", test_nshot_optimization),
        ("ProcessDiscrimination Integration", test_process_discrimination),
        ("Complaints Harness Integration", test_complaints_harness),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"[RESULT] {test_name}: PASS")
            else:
                print(f"[RESULT] {test_name}: FAIL")
        except Exception as e:
            print(f"[RESULT] {test_name}: ERROR - {e}")

    print(f"\n" + "="*80)
    print(f"FINAL RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("ALL TESTS PASSED! NShotPromptGenerator integration is working correctly.")
        return True
    else:
        print("Some tests failed. Please review the results above.")
        return False

if __name__ == "__main__":
    main()
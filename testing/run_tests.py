"""
CLI orchestrator for running leak detection tests on both simulators.

Usage:
    python run_tests.py --simulator household --runs 20
    python run_tests.py --simulator apartment --runs 30
    python run_tests.py --simulator both --runs 20
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run leak detection tests on simulator(s)"
    )
    parser.add_argument(
        "--simulator",
        choices=["household", "apartment", "both"],
        default="both",
        help="Which simulator(s) to test"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of test runs per simulator"
    )
    
    args = parser.parse_args()
    
    # Import experiment runners
    if args.simulator in ["household", "both"]:
        print("\n" + "="*70)
        print("STARTING HOUSEHOLD SIMULATOR TESTS")
        print("="*70)
        from household_experiment import run_experiment as run_household
        try:
            results_h, metrics_h = run_household(num_runs=args.runs)
            print(f"\n✓ Household tests completed successfully")
        except Exception as e:
            print(f"\n✗ Household tests failed: {e}")
            import traceback
            traceback.print_exc()
    
    if args.simulator in ["apartment", "both"]:
        print("\n" + "="*70)
        print("STARTING APARTMENT SIMULATOR TESTS")
        print("="*70)
        from apartment_experiment import run_experiment as run_apartment
        try:
            results_a, metrics_a = run_apartment(num_runs=args.runs)
            print(f"\n✓ Apartment tests completed successfully")
        except Exception as e:
            print(f"\n✗ Apartment tests failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()

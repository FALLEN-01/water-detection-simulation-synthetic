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
        print("STARTING HOUSEHOLD SIMULATOR TESTS (LOW/MEDIUM/HIGH)")
        print("="*70)
        from household_experiment import run_experiment as run_household
        try:
            # Run all three intensity levels
            run_household(num_runs=args.runs, intensity_min=0.2, intensity_max=0.4, intensity_label="LOW")
            run_household(num_runs=args.runs, intensity_min=0.4, intensity_max=0.6, intensity_label="MEDIUM")
            run_household(num_runs=args.runs, intensity_min=0.6, intensity_max=0.8, intensity_label="HIGH")
            print(f"\n✓ Household tests completed successfully")
        except Exception as e:
            print(f"\n✗ Household tests failed: {e}")
            import traceback
            traceback.print_exc()
    
    if args.simulator in ["apartment", "both"]:
        print("\n" + "="*70)
        print("STARTING APARTMENT SIMULATOR TESTS (LOW/MEDIUM/HIGH)")
        print("="*70)
        from apartment_experiment import run_experiment as run_apartment
        try:
            # Run all three intensity levels
            run_apartment(num_runs=args.runs, intensity_min=5.0, intensity_max=15.0, intensity_label="LOW")
            run_apartment(num_runs=args.runs, intensity_min=15.0, intensity_max=25.0, intensity_label="MEDIUM")
            run_apartment(num_runs=args.runs, intensity_min=25.0, intensity_max=40.0, intensity_label="HIGH")
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

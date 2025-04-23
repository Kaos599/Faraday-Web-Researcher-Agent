"""
Test script to run the fact-checking system with specific claims.
"""

import sys
import os
import json

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from fact_check_system.fact_checker import run_fact_check
    from fact_check_system.models import FactCheckResult
except ImportError as e:
    print(f"Error importing fact check system: {e}")
    print("Please ensure you are running this script from the root directory of the project,")
    print("and that the 'fact_check_system' directory is present.")
    sys.exit(1)

def main():
    claims_to_test = [
        "Does India have the largest population?",
        "Which Team won the 2023 ICC men's world cup?"
    ]

    results = []
    for claim in claims_to_test:
        print(f"\n--- Running Fact Check for: '{claim}' ---")
        try:
            result: FactCheckResult = run_fact_check(claim)
            print("\n--- FactCheckResult ---       ")
            # Print the Pydantic model as a JSON string for readability
            print(result.model_dump_json(indent=2))
            results.append(result.model_dump()) # Store the result dictionary

            print("\n--- Evidence Snippets ---       ")
            if result.evidence_sources:
                for ev in result.evidence_sources:
                    print(f"Source: {ev.source_tool}, URL: {ev.url}, Snippet: {(ev.snippet or 'N/A')[:150]}...")
            else:
                print("No evidence sources were found or retained.")

        except Exception as e:
            print(f"\n--- ERROR Running Fact Check for: '{claim}' ---       ")
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            results.append({"claim": claim, "error": str(e)})
        print("-----------------------------------------")

    # Optionally, save all results to a file
    try:
        with open("fact_check_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str) # Use default=str for datetime
        print("\nResults saved to fact_check_results.json")
    except Exception as e:
        print(f"\nError saving results to JSON file: {e}")

if __name__ == "__main__":
    main() 
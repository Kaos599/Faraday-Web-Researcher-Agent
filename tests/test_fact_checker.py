"""
Test script to run the fact-checking system with specific claims.
"""

import sys
import os
import json
import asyncio
from typing import Union

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from research_system.agent import run_web_research
    from research_system.schemas import ResearchReport, ErrorResponse
except ImportError as e:
    print(f"Error importing research system: {e}")
    print("Please ensure you are running this script from the root directory of the project,")
    print("and that the 'research_system' directory is present.")
    sys.exit(1)

def main():
    claims_to_test = [
        "Does India have the largest population?",
        "Which Team won the 2023 ICC men's world cup?"
    ]

    results = []
    for claim in claims_to_test:
        print(f"\n--- Running Research for: '{claim}' ---")
        try:
            result: Union[ResearchReport, ErrorResponse] = asyncio.run(run_web_research(claim))

            print("\n--- ResearchResult ---       ")
            if isinstance(result, ResearchReport):
                print(result.model_dump_json(indent=2))
                results.append(result.model_dump())

                print("\n--- Sources ---       ")
                if result.sources:
                    for i, source in enumerate(result.sources):
                        print(f"{i+1}. [{source.title or 'N/A'}]({source.url}) - via {source.tool_used}")
                        print(f"   Snippet: {(source.snippet or 'N/A')[:150]}...")
                else:
                    print("No sources were found or retained in the report.")
            elif isinstance(result, ErrorResponse):
                print(f"Error Response: {result.model_dump_json(indent=2)}")
                results.append({"query": claim, "error": result.error, "details": result.details})
            else:
                print(f"Unexpected result type: {type(result)}")
                results.append({"query": claim, "error": "Unexpected result type", "details": str(result)})

        except Exception as e:
            print(f"\n--- ERROR Running Research for: '{claim}' ---       ")
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
            results.append({"query": claim, "error": str(e)})
        print("-----------------------------------------")

    try:
        with open("research_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print("\nResults saved to research_results.json")
    except Exception as e:
        print(f"\nError saving results to JSON file: {e}")

if __name__ == "__main__":
    main() 
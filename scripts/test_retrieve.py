"""Step 2: Build retrieval index (if needed) and run sample queries."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # must run before retrieval imports read the key
sys.path.insert(0, str(ROOT / "src"))

from embedding_text import build_embedding_text
from retrieval import build_index, load_catalog, retrieve

SAMPLE_QUERIES = [
    "Java developer with strong communication skills, mid-level",
    "entry level cashier, needs to handle cash and customer interactions",
    "personality assessment for identifying leadership potential",
]


def show_embedding_text_examples() -> None:
    catalog = load_catalog()
    by_id = {e["entity_id"]: e for e in catalog}

    # Pick representative samples: skill test, interpretive report, personality test
    sample_ids = ["4032", "3845", "726"]  # Core Java, HiPo 1.0, Sales Interview Guide
    for entity_id in sample_ids:
        entry = by_id.get(entity_id)
        if entry:
            print(f"\n--- {entry['name']} [{entity_id}] ---")
            print(build_embedding_text(entry))


def print_results(query: str, results: list[dict]) -> None:
    print(f"\nQuery: {query!r}")
    print("-" * 72)
    for rank, entry in enumerate(results, start=1):
        test_type = ", ".join(entry.get("test_type") or [])
        report_flag = " [interpretive report]" if entry.get("is_interpretive_report") else ""
        print(
            f"{rank:2}. score={entry['_score']:.4f} | {entry['name']}{report_flag}\n"
            f"    test_type: {test_type}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test catalog retrieval")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild of FAISS index even if catalog hash unchanged",
    )
    args = parser.parse_args()

    print("=== Example embedding texts (sanity check) ===")
    show_embedding_text_examples()

    print("\n\n=== Building / loading FAISS index ===")
    build_index(force=args.force)
    print("Index ready.")

    print("\n\n=== Retrieval test queries ===")
    for query in SAMPLE_QUERIES:
        results = retrieve(query, k=10)
        print_results(query, results)

        interpretive = [r for r in results if r.get("is_interpretive_report")]
        if interpretive:
            names = ", ".join(r["name"] for r in interpretive)
            print(f"    >> Interpretive reports in top-10: {names}")


if __name__ == "__main__":
    main()

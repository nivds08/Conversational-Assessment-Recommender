"""Step 1: Analyze raw catalog and explore Individual vs Pre-packaged distinction."""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "catalog_raw.json"


def load_raw_catalog(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    # Fix unescaped newlines inside JSON string values (scrape artifact in raw feed)
    text = re.sub(
        r'("name":\s*")([^"]*?)\s*\n\s*([^"]*?)(")',
        r'\1\2 \3\4',
        text,
        flags=re.MULTILINE,
    )
    return json.loads(text)


def main() -> None:
    data = load_raw_catalog(RAW_PATH)
    print(f"Total entries: {len(data)}")

    all_keys: set[str] = set()
    for entry in data:
        all_keys.update(entry.keys())
    print(f"All fields: {sorted(all_keys)}")

    # Inspect categorical / distinguishing fields
    for key in sorted(all_keys):
        vals = [entry.get(key) for entry in data]
        if all(isinstance(v, (str, int, float, bool, type(None))) for v in vals):
            unique = sorted({str(v) for v in vals if v is not None})
            if len(unique) <= 40:
                print(f"\n{key} ({len(unique)} unique): {unique}")

    # keys field distribution
    keys_counter: Counter[str] = Counter()
    for entry in data:
        for k in entry.get("keys", []) or []:
            keys_counter[k] += 1
    print("\n--- keys field value counts ---")
    for k, c in keys_counter.most_common():
        print(f"  {k}: {c}")

    # Name patterns suggesting "Solution" (pre-packaged)
    solution_pattern = re.compile(r"\bSolution\b", re.I)
    solutions = [e for e in data if solution_pattern.search(e.get("name", ""))]
    non_solutions = [e for e in data if not solution_pattern.search(e.get("name", ""))]
    print(f"\n--- Name contains 'Solution' ---")
    print(f"  With 'Solution': {len(solutions)}")
    print(f"  Without 'Solution': {len(non_solutions)}")

    # Show solution names
    print("\n  Sample 'Solution' names:")
    for e in solutions[:20]:
        print(f"    [{e['entity_id']}] {e['name']} | keys={e.get('keys')}")

    # Check description for bundled/multi-test language
    bundle_words = ["bundle", "package", "suite", "combination", "multiple tests", "battery"]
    print("\n--- Description bundle keywords ---")
    for word in bundle_words:
        matches = [e for e in data if word.lower() in (e.get("description") or "").lower()]
        if matches:
            print(f"  '{word}': {len(matches)} entries")

    # entity_id ranges for solutions vs non
    sol_ids = [int(e["entity_id"]) for e in solutions]
    non_ids = [int(e["entity_id"]) for e in non_solutions]
    if sol_ids:
        print(f"\n  Solution entity_id range: {min(sol_ids)} - {max(sol_ids)}")
    if non_ids:
        print(f"  Non-solution entity_id range: {min(non_ids)} - {max(non_ids)}")

    # Check link URL patterns
    link_patterns: Counter[str] = Counter()
    for e in data:
        link = e.get("link", "")
        if "/job-solutions/" in link.lower() or "job-solution" in link.lower():
            link_patterns["job-solutions"] += 1
        elif "/product-catalog/" in link:
            link_patterns["product-catalog"] += 1
        else:
            link_patterns["other"] += 1
    print(f"\n--- Link URL patterns ---")
    for pat, c in link_patterns.most_common():
        print(f"  {pat}: {c}")

    # Look for any explicit category/type field values
    for field in ["type", "category", "product_type", "solution_type", "package_type"]:
        if field in all_keys:
            print(f"\nFound explicit field: {field}")


if __name__ == "__main__":
    main()

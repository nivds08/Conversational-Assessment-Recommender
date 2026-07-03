"""Deeper analysis: multi-key entries and borderline cases."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "catalog_raw.json"


def load_raw_catalog(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r'("name":\s*")([^"]*?)\s*\n\s*([^"]*?)(")',
        r'\1\2 \3\4',
        text,
        flags=re.MULTILINE,
    )
    return json.loads(text)


def main() -> None:
    data = load_raw_catalog(RAW_PATH)

    by_key_count: dict[int, list] = {}
    for e in data:
        n = len(e.get("keys") or [])
        by_key_count.setdefault(n, []).append(e)

    print("Entries by number of keys:")
    for n in sorted(by_key_count):
        print(f"  {n} keys: {len(by_key_count[n])} entries")

    print("\n--- All multi-key (>=2) entries ---")
    multi = [e for e in data if len(e.get("keys") or []) >= 2]
    for e in sorted(multi, key=lambda x: x["name"]):
        print(f"  [{e['entity_id']}] {e['name']}")
        print(f"    keys={e.get('keys')}")
        desc = (e.get("description") or "")[:200]
        print(f"    desc={desc}...")
        print()

    print("\n--- Solution-named entries (full) ---")
    for e in data:
        if "solution" in e["name"].lower():
            print(json.dumps(e, indent=2)[:1500])
            print("---")

    # Heuristic candidates: multi-key OR name ends with Solution
    solution_heuristic = lambda e: (
        "solution" in e["name"].lower()
        or len(e.get("keys") or []) >= 2
    )
    prepackaged = [e for e in data if solution_heuristic(e)]
    individual = [e for e in data if not solution_heuristic(e)]
    print(f"\nHeuristic (Solution in name OR >=2 keys):")
    print(f"  Pre-packaged: {len(prepackaged)}")
    print(f"  Individual: {len(individual)}")

    # Borderline: multi-key but NOT named Solution
    borderline_multi = [e for e in multi if "solution" not in e["name"].lower()]
    print(f"\nBorderline multi-key (not named Solution): {len(borderline_multi)}")
    for e in borderline_multi:
        print(f"  [{e['entity_id']}] {e['name']} | keys={e.get('keys')}")


if __name__ == "__main__":
    main()

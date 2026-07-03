"""Step 1: Download raw catalog and produce filtered catalog.json."""

import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from catalog_prep import prepare_catalog

RAW_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
RAW_PATH = ROOT / "data" / "catalog_raw.json"
CATALOG_PATH = ROOT / "data" / "catalog.json"


def download_raw_catalog() -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists():
        print(f"Raw catalog already exists at {RAW_PATH}")
        return
    print(f"Downloading catalog from {RAW_URL} ...")
    response = httpx.get(RAW_URL, timeout=60.0)
    response.raise_for_status()
    RAW_PATH.write_bytes(response.content)
    print(f"Saved raw catalog ({len(response.content):,} bytes)")


def main() -> None:
    download_raw_catalog()
    summary = prepare_catalog(RAW_PATH, CATALOG_PATH)

    print("\n=== Step 1: Catalog Prep Summary ===")
    print(f"Total raw entries:        {summary['total_raw']}")
    print(f"Excluded (pre-packaged):  {summary['excluded_prepackaged']}")
    print(f"Individual tests kept:    {summary['individual_tests']}")
    print(f"\nOutput: {CATALOG_PATH}")
    print("\nExcluded entries:")
    for e in summary["excluded_entries"]:
        print(f"  [{e['entity_id']}] {e['name']}")


if __name__ == "__main__":
    main()

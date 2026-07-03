"""Step 5 safety-net tests."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent_logic import Recommendation, validate_recommendations  # noqa: E402
from retrieval import load_catalog  # noqa: E402


def main() -> None:
    catalog = load_catalog()
    catalog_map = {e["url"]: e for e in catalog}

    # 1 valid, 1 real URL+wrong name, 1 made-up URL
    valid = Recommendation(
        name=catalog[0]["name"],
        url=catalog[0]["url"],
        test_type=", ".join(catalog[0].get("test_type") or []),
    ).model_dump()
    wrong_name = Recommendation(
        name="Totally Wrong Name",
        url=catalog[1]["url"],  # real URL
        test_type=", ".join(catalog[1].get("test_type") or []),
    ).model_dump()
    fake_url = Recommendation(
        name="Fake Product",
        url="https://www.shl.com/products/product-catalog/view/not-a-real-test/",
        test_type="Knowledge & Skills",
    ).model_dump()

    mixed = [valid, wrong_name, fake_url]
    kept = validate_recommendations(mixed, catalog_map)

    print("input_count:", len(mixed))
    print("kept_count:", len(kept))
    print("kept:", json.dumps(kept, indent=2))


if __name__ == "__main__":
    main()


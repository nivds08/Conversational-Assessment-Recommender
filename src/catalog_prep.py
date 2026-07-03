"""Catalog loading and filtering for SHL assessment recommender."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl

# Report products generated from prior assessment results (OPQ, Verify, etc.),
# not standalone tests a candidate takes without a prerequisite assessment.
INTERPRETIVE_REPORT_ENTITY_IDS = frozenset({"3845", "4284", "3746", "3484"})


class CatalogEntry(BaseModel):
    entity_id: str
    name: str
    url: HttpUrl
    test_type: list[str] = Field(description="Assessment categories, from source 'keys' field")
    description: str
    job_levels: list[str]
    duration: str
    remote: str
    adaptive: str
    is_interpretive_report: bool = False


def load_raw_catalog(path: Path | str) -> list[dict]:
    """Load raw catalog JSON, repairing known scrape artifacts (unescaped newlines in names)."""
    text = Path(path).read_text(encoding="utf-8")
    text = re.sub(
        r'("name":\s*")([^"]*?)\s*\n\s*([^"]*?)(")',
        r"\1\2 \3\4",
        text,
        flags=re.MULTILINE,
    )
    return json.loads(text)


def is_prepackaged_job_solution(entry: dict) -> bool:
    """
    Heuristic: no explicit product-type field exists in the raw catalog.

    Pre-packaged Job Solutions are multi-test bundles (e.g. "Entry Level Cashier Solution").
    Individual Test Solutions are single assessments, even if they span multiple key categories.
    """
    name = entry.get("name", "")
    desc = entry.get("description", "") or ""

    if re.search(r"\bSolution\b", name, re.IGNORECASE):
        return True

    if re.search(r"Precise Fit .+ Solution", desc, re.IGNORECASE):
        return True

    if re.search(
        r"includes .+ (?:simulation|test).+ and .+ (?:behavioral )?tests?",
        desc,
        re.IGNORECASE,
    ):
        return True

    return False


def filter_individual_tests(entries: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split catalog into individual tests (kept) and pre-packaged solutions (excluded)."""
    excluded: list[dict] = []
    kept: list[dict] = []
    for entry in entries:
        if is_prepackaged_job_solution(entry):
            excluded.append(entry)
        else:
            kept.append(entry)
    return kept, excluded


def to_catalog_entry(raw: dict) -> CatalogEntry:
    entity_id = str(raw["entity_id"])
    return CatalogEntry(
        entity_id=entity_id,
        name=raw["name"].strip(),
        url=raw["link"],
        test_type=list(raw.get("keys") or []),
        description=(raw.get("description") or "").strip(),
        job_levels=list(raw.get("job_levels") or []),
        duration=raw.get("duration") or "",
        remote=raw.get("remote") or "",
        adaptive=raw.get("adaptive") or "",
        is_interpretive_report=entity_id in INTERPRETIVE_REPORT_ENTITY_IDS,
    )


def prepare_catalog(
    raw_path: Path | str,
    output_path: Path | str,
) -> dict:
    """
    Load raw catalog, filter to Individual Test Solutions, write catalog.json.

    Returns summary stats for reporting.
    """
    raw_entries = load_raw_catalog(raw_path)
    kept, excluded = filter_individual_tests(raw_entries)

    catalog = [to_catalog_entry(e).model_dump(mode="json") for e in kept]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return {
        "total_raw": len(raw_entries),
        "excluded_prepackaged": len(excluded),
        "individual_tests": len(catalog),
        "excluded_entries": [
            {"entity_id": e["entity_id"], "name": e["name"]} for e in excluded
        ],
    }

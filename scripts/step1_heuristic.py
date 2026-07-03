"""Find pre-packaged patterns beyond name='Solution'."""
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

    # Pattern searches in description
    patterns = {
        "precise_fit": re.compile(r"Precise Fit", re.I),
        "includes_multiple": re.compile(
            r"includes .+ (?:simulation|test).+ and .+ (?:simulation|test|behavioral)",
            re.I,
        ),
        "solution_word_desc": re.compile(r"\bSolution\b", re.I),
        "job_focused_assessment": re.compile(r"Job-Focused Assessment", re.I),
        "battery": re.compile(r"\bbattery\b", re.I),
    }

    for label, pat in patterns.items():
        matches = [e for e in data if pat.search(e.get("description") or "")]
        print(f"{label}: {len(matches)}")
        for e in matches:
            sol_in_name = "solution" in e["name"].lower()
            print(f"  [{e['entity_id']}] {e['name']} (name_has_solution={sol_in_name})")

    # Proposed filter: name ends with Solution OR Precise Fit in description
    def is_prepackaged(e: dict) -> bool:
        name = e.get("name", "")
        desc = e.get("description", "") or ""
        if re.search(r"\bSolution\b", name, re.I):
            return True
        if re.search(r"Precise Fit .+ Solution", desc, re.I):
            return True
        if re.search(
            r"includes .+ (?:simulation|test).+ and .+ (?:behavioral )?tests?",
            desc,
            re.I,
        ):
            return True
        return False

    prepackaged = [e for e in data if is_prepackaged(e)]
    individual = [e for e in data if not is_prepackaged(e)]
    print(f"\nProposed heuristic counts:")
    print(f"  Pre-packaged: {len(prepackaged)}")
    print(f"  Individual: {len(individual)}")
    print("\nPre-packaged list:")
    for e in sorted(prepackaged, key=lambda x: x["entity_id"]):
        print(f"  [{e['entity_id']}] {e['name']}")

    # Borderline: near misses - things that LOOK like bundles but pass filter
  # Things excluded that might be individual
    print("\n--- 15 BORDERLINE EXAMPLES FOR MANUAL REVIEW ---")
    borderline = []

    # Near pre-packaged: Job-Focused Assessment (might be bundles?)
    jfa = [e for e in data if "Job-Focused Assessment" in (e.get("description") or "")]
    for e in jfa:
        borderline.append(("INCLUDED as individual (Job-Focused Assessment)", e))

    # 3939 - Precise Fit but no Solution in name
    e3939 = next(e for e in data if e["entity_id"] == "3939")
    borderline.append(("LIKELY pre-packaged (Precise Fit desc, no Solution in name)", e3939))

    # Phone Solution vs Phone Simulation pairs
    for e in data:
        if e["entity_id"] in ("3931", "3933", "3930", "3932"):
            label = "EXCLUDED (Solution)" if is_prepackaged(e) else "INCLUDED (Simulation)"
            borderline.append((label, e))

    # HiPo reports - part of "High Potential solution"
    for e in data:
        if "HiPo" in e["name"] or "HIPO" in (e.get("description") or ""):
            borderline.append(("INCLUDED (HiPo report - mentions 'solution' in desc)", e))

    # PJM reports - composite reports
    for e in data:
        if e["name"].startswith("PJM"):
            borderline.append(("INCLUDED (PJM composite report)", e))

    # Global Skills
    for e in data:
        if "Global Skills" in e["name"]:
            borderline.append(("INCLUDED (Global Skills)", e))

    # Manufacturing Job-Focused
    for e in data:
        if "Manufacturing" in e["name"] or "Manufac." in e["name"]:
            borderline.append(("INCLUDED (Mfg Job-Focused 8.0)", e))

    # Dedupe and take 15
    seen = set()
    unique_borderline = []
    for label, e in borderline:
        if e["entity_id"] not in seen:
            seen.add(e["entity_id"])
            unique_borderline.append((label, e))

    for i, (label, e) in enumerate(unique_borderline[:15], 1):
        print(f"\n{i}. {label}")
        print(f"   ID: {e['entity_id']} | Name: {e['name']}")
        print(f"   keys: {e.get('keys')}")
        print(f"   duration: {e.get('duration')}")
        desc = (e.get("description") or "")[:250].replace("\n", " ")
        print(f"   desc: {desc}...")


if __name__ == "__main__":
    main()

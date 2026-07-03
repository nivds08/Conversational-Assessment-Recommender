"""Replay local traces against live /chat endpoint and compute Recall@10."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parent.parent
TRACES_DIR = ROOT / "traces"
CATALOG_PATH = ROOT / "data" / "catalog.json"
CHAT_URL = "http://127.0.0.1:8000/chat"
MAX_TURNS = 8


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _parse_md_trace(path: Path) -> tuple[list[str], list[str]]:
    """Parse SHL markdown trace: user turns + expected names from final shortlist table."""
    text = path.read_text(encoding="utf-8")
    user_messages: list[str] = []
    expected_names: list[str] = []

    turn_blocks = re.split(r"^### Turn \d+\s*$", text, flags=re.MULTILINE)[1:]
    for block in turn_blocks:
        user_match = re.search(r"\*\*User\*\*\s*\n+(.*?)(?=\*\*Agent\*\*)", block, re.DOTALL)
        if user_match:
            lines = [
                line[1:].strip()
                for line in user_match.group(1).splitlines()
                if line.strip().startswith(">")
            ]
            if lines:
                user_messages.append("\n".join(lines))

        if re.search(r"end_of_conversation.*\*\*true\*\*", block, re.IGNORECASE):
            names = [
                m.group(1).strip()
                for m in re.finditer(r"^\|\s*\d+\s*\|\s*(.+?)\s*\|", block, re.MULTILINE)
                if m.group(1).strip() not in ("Name", "")
            ]
            if names:
                expected_names = names

    return user_messages, expected_names


def _extract_expected(trace: dict) -> list[str]:
    for key in ("expected_shortlist", "expected", "shortlist", "expected_recommendations"):
        if key in trace:
            items = trace[key]
            if isinstance(items, list):
                names: list[str] = []
                for x in items:
                    if isinstance(x, str):
                        names.append(x)
                    elif isinstance(x, dict):
                        names.append(x.get("name", "").strip())
                return [n for n in names if n]
    return []


def _extract_user_seed(trace: dict) -> str:
    persona = trace.get("persona")
    facts = trace.get("facts")
    if isinstance(persona, str) and persona.strip():
        return persona.strip()
    if isinstance(facts, dict) and facts:
        bits = [f"{k}: {v}" for k, v in facts.items()]
        return " | ".join(bits)
    if isinstance(facts, list) and facts:
        return "; ".join(str(x) for x in facts)
    return "Need SHL assessments for hiring."


def _build_fact_map(trace: dict) -> dict[str, str]:
    facts = trace.get("facts")
    if isinstance(facts, dict):
        return {str(k).lower(): str(v) for k, v in facts.items()}
    return {}


def _rule_based_user_reply(last_assistant: str, fact_map: dict[str, str]) -> str:
    t = last_assistant.lower()
    if "role" in t:
        return fact_map.get("role", "No preference.")
    if "seniority" in t or "level" in t:
        return fact_map.get("seniority", "No preference.")
    if "skill" in t:
        return fact_map.get("skills", "No preference.")
    if "duration" in t or "minute" in t:
        return fact_map.get("duration", "No preference.")
    if "remote" in t:
        return fact_map.get("remote", "No preference.")
    if "adaptive" in t:
        return fact_map.get("adaptive", "No preference.")
    if "language" in t:
        return fact_map.get("language", "No preference.")
    return "No additional preference."


FUZZY_THRESHOLD = 0.72


def _normalize_name(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s&\.\-\(\)]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _names_match(a: str, b: str, threshold: float = FUZZY_THRESHOLD) -> bool:
    target = _normalize_name(a)
    other = _normalize_name(b)
    if target == other:
        return True
    if target in other or other in target:
        return True
    return SequenceMatcher(None, target, other).ratio() >= threshold


def _recall_at_10(expected: list[str], got: list[str], *, fuzzy: bool = False) -> float:
    if not expected:
        return 0.0
    if fuzzy:
        matched = sum(1 for e in expected if any(_names_match(e, g) for g in got[:10]))
        return matched / len(expected)
    e = {x.lower().strip() for x in expected}
    g = {x.lower().strip() for x in got[:10]}
    return len(e & g) / len(e)


def main() -> None:
    if not TRACES_DIR.exists():
        print(f"[error] traces directory missing: {TRACES_DIR}")
        return

    trace_files = sorted(TRACES_DIR.glob("*.json")) + sorted(TRACES_DIR.glob("*.md"))
    if not trace_files:
        print(f"[error] no trace files (*.json or *.md) found in: {TRACES_DIR}")
        return

    catalog = _load_json(CATALOG_PATH)
    catalog_names = {e["name"].lower() for e in catalog}
    catalog_urls = {e["url"] for e in catalog}

    first_path = trace_files[0]
    if first_path.suffix == ".md":
        first_users, first_expected = _parse_md_trace(first_path)
        print("=== First trace structure preview ===")
        print("file:", first_path.name)
        print("format: markdown conversation transcript")
        print("user_turns:", len(first_users))
        print("expected_shortlist_count:", len(first_expected))
        print("first_user_message:", first_users[0][:120] + ("..." if len(first_users[0]) > 120 else ""))
    else:
        first = _load_json(first_path)
        print("=== First trace structure preview ===")
        print("file:", first_path.name)
        print("keys:", sorted(first.keys()))
        print("expected_shortlist_count:", len(_extract_expected(first)))

    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=30.0) as client:
        for path in trace_files:
            if path.suffix == ".md":
                user_turns, expected = _parse_md_trace(path)
                missing_expected = [n for n in expected if n.lower() not in catalog_names]
            else:
                trace = _load_json(path)
                expected = _extract_expected(trace)
                missing_expected = [n for n in expected if n.lower() not in catalog_names]
                fact_map = _build_fact_map(trace)
                user_turns = [_extract_user_seed(trace)]

            messages: list[dict[str, str]] = []
            final_recs: list[dict] = []
            flags: list[str] = []
            turns = 0
            user_idx = 0
            while user_idx < len(user_turns) and turns < MAX_TURNS:
                turns += 1
                messages.append({"role": "user", "content": user_turns[user_idx]})
                user_idx += 1

                resp = client.post(CHAT_URL, json={"messages": messages})
                if resp.status_code != 200:
                    flags.append(f"http_{resp.status_code}")
                    break
                data = resp.json()
                for k in ("reply", "recommendations", "end_of_conversation"):
                    if k not in data:
                        flags.append("schema_violation")
                        break
                if "schema_violation" in flags:
                    break

                final_recs = data.get("recommendations", [])
                bad_url = [r.get("url") for r in final_recs if r.get("url") not in catalog_urls]
                if bad_url:
                    flags.append("hallucinated_url")

                if data.get("end_of_conversation"):
                    break

                assistant_reply = data.get("reply", "")
                messages.append({"role": "assistant", "content": assistant_reply})

                if path.suffix == ".json" and user_idx >= len(user_turns):
                    messages.append(
                        {
                            "role": "user",
                            "content": _rule_based_user_reply(assistant_reply, fact_map),
                        }
                    )
            else:
                if turns >= MAX_TURNS:
                    flags.append("turn_cap_hit")

            got_names = [r.get("name", "") for r in final_recs]
            rows.append(
                {
                    "trace": path.name,
                    "recall@10": round(_recall_at_10(expected, got_names), 3),
                    "recall@10_fuzzy": round(_recall_at_10(expected, got_names, fuzzy=True), 3),
                    "got_names": got_names,
                    "expected_names": expected,
                    "turns": turns,
                    "flags": ",".join(flags) if flags else "-",
                    "missing_expected": len(missing_expected),
                }
            )

    print("\n=== Summary (exact match) ===")
    for r in rows:
        print(
            f"{r['trace']:25}  recall@10={r['recall@10']:.3f}  "
            f"turns={r['turns']}  flags={r['flags']}  missing_expected={r['missing_expected']}"
        )
    mean = sum(r["recall@10"] for r in rows) / len(rows)
    print(f"\nmean Recall@10 (exact): {mean:.3f}")

    print("\n=== Summary (fuzzy match, threshold=0.72) ===")
    for r in rows:
        print(
            f"{r['trace']:25}  recall@10={r['recall@10_fuzzy']:.3f}  "
            f"turns={r['turns']}  flags={r['flags']}"
        )
    mean_fuzzy = sum(r["recall@10_fuzzy"] for r in rows) / len(rows)
    print(f"\nmean Recall@10 (fuzzy): {mean_fuzzy:.3f}")
    print(f"fuzzy delta: {mean_fuzzy - mean:+.3f}")


if __name__ == "__main__":
    main()


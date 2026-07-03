"""Verify fixes 1-3: retrieval, clarify, compare persistence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_logic import _build_retrieval_query, generate_agent_response  # noqa: E402
from intent_classifier import ClassificationResult, Constraints, classify_turn  # noqa: E402
from retrieval import retrieve  # noqa: E402
from test_traces import _parse_md_trace, TRACES_DIR  # noqa: E402

C7_TURN1 = (
    "We're hiring bilingual healthcare admin staff in South Texas — they handle "
    "patient records and need to be assessed in Spanish. HIPAA compliance is critical. "
    "What assessments work?"
)
C9_TURN1 = (
    "Here's the JD for an engineer we need to fill. Can you recommend an assessment battery?\n\n"
    '"Senior Full-Stack Engineer — 5+ years across Core Java, Spring, REST API design, '
    "Angular, SQL/relational databases, AWS deployment, and Docker. Will own end-to-end "
    "microservice delivery, contribute to architectural decisions, and mentor mid-level "
    'engineers. Strong CI/CD and cloud-native experience required."'
)
SAMPLE_QUERIES = [
    "Java developer with strong communication skills, mid-level",
    "entry level cashier, needs to handle cash and customer interactions",
    "personality assessment for identifying leadership potential",
]


def fix1_probe() -> None:
    print("=== FIX 1: C8 retrieval probe ===")
    cls = ClassificationResult(
        intent="RECOMMEND",
        constraints=Constraints(role="admin assistant", skills=["Excel", "Word"]),
        compare_assessments=[],
        rationale="",
    )
    msgs = [{"role": "user", "content": "I need to quickly screen admin assistants for Excel and Word daily."}]
    q = _build_retrieval_query(cls, msgs)
    print("Built query:", q)
    results = retrieve(q, k=10)
    names = [e["name"] for e in results]
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")
    excel_hit = any("excel" in n.lower() for n in names[:4])
    word_hit = any("word" in n.lower() for n in names[:4])
    print(f"MS Excel in top 4: {excel_hit} | MS Word in top 4: {word_hit}")

    print("\n=== FIX 1: Step 2 sample query regression ===")
    for sq in SAMPLE_QUERIES:
        top3 = [e["name"] for e in retrieve(sq, k=3)]
        print(f"  {sq!r}")
        print(f"    -> {top3}")


def fix2_probe() -> None:
    print("\n=== FIX 2: C7/C9 turn-1 classification ===")
    for label, text in [("C7", C7_TURN1), ("C9", C9_TURN1)]:
        result = classify_turn([{"role": "user", "content": text}])
        print(f"{label}: intent={result.intent}")
        print(f"  rationale: {result.rationale[:120]}...")


def fix3_probe() -> None:
    print("\n=== FIX 3: C5 compare shortlist persistence ===")
    users, _ = _parse_md_trace(TRACES_DIR / "C5.md")
    messages: list[dict] = []
    for i, user_msg in enumerate(users, 1):
        messages.append({"role": "user", "content": user_msg})
        resp = generate_agent_response(messages)
        rec_names = [r.name for r in resp.recommendations]
        print(f"Turn {i} intent-path reply prefix: {resp.reply[:80]}...")
        print(f"  recommendations ({len(rec_names)}): {rec_names}")
        messages.append({"role": "assistant", "content": resp.reply})


def main() -> None:
    fix1_probe()
    try:
        fix2_probe()
        fix3_probe()
    except Exception as exc:
        print(f"\n[warn] Live Groq calls failed: {exc}")


if __name__ == "__main__":
    main()
